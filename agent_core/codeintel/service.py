from __future__ import annotations

import asyncio
import base64
from contextlib import closing
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, Callable, Iterable
from agent_core.codeintel.cache import CodeCache

from agent_core.codeintel.budget import BudgetExceeded, QueryBudget
from agent_core.codeintel.catalog import FileCatalog
from agent_core.codeintel.config import CodeIntelConfig
from agent_core.codeintel.indexer import update_pending
from agent_core.codeintel.models import ChangeSet, CodeHit, SearchPage, SearchRequest, StaleEvidence
from agent_core.codeintel.paths import glob_match
from agent_core.codeintel.snapshots import contained, read_snapshot
from agent_core.codeintel.store import CodeStore
from agent_core.file_lock import FileLock

logger = logging.getLogger(__name__)


def _like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class CodeIntelligenceService:
    """Lazy local service. Queries never implicitly enumerate or rebuild the repo."""

    def __init__(self, workspace: str | Path, config: CodeIntelConfig | None = None,
                 *, database: Path | None = None, allowed: Callable[[str], bool] | None = None) -> None:
        self.workspace = Path(workspace).resolve()
        self.config = config or CodeIntelConfig()
        self.store = CodeStore(self.workspace, database)
        self.catalog = FileCatalog(self.store)
        self._initialized = False
        self._lock = threading.RLock()
        self._background: asyncio.Task[Any] | None = None
        self._observer: Any = None
        self._closed = False
        self.watch_state = "not_started"
        self.allowed = allowed
        self.policy_key = ""
        self.cache = CodeCache(self.config.cache_entries)

    def _initialize(self) -> None:
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self.store.initialize()
                    self._initialized = True

    async def _run(self, work: Callable[[], Any], budget: QueryBudget) -> Any:
        task = asyncio.create_task(asyncio.to_thread(work))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            budget.cancelled.set()
            # Do not leave a cancelled query consuming IO in a detached thread.
            try:
                await task
            except (BudgetExceeded, TimeoutError):
                pass
            raise

    async def status(self) -> dict[str, Any]:
        def work() -> dict[str, Any]:
            self._initialize()
            return {**self.store.status(), "watcher": self.watch_state}
        return await asyncio.to_thread(work)

    async def catalog_ready(self) -> bool:
        def work() -> bool:
            self._initialize()
            with self.store.connect() as db:
                return self.store.get(db, "catalog_complete") == "1"
        return await asyncio.to_thread(work)

    async def has_pending_work(self) -> bool:
        def work() -> bool:
            self._initialize()
            with self.store.connect() as db:
                return self.store.get(db, "catalog_complete") != "1" or db.execute("SELECT 1 FROM pending LIMIT 1").fetchone() is not None
        return await asyncio.to_thread(work)

    async def glob_paths(self, arguments: dict[str, Any], *, scope: Any = None,
                         allowed: Callable[[str], bool] | None = None) -> Any:
        from agent_core.models import ToolResult
        budget = QueryBudget(self.config, scope)
        def work() -> ToolResult:
            self._initialize()
            base = contained(self.workspace, str(arguments.get("path", ".")))
            relative = base.relative_to(self.workspace).as_posix()
            pattern = str(arguments["pattern"])
            limit = max(1, min(int(arguments.get("max_results", 200)), 2000))
            shown: list[str] = []
            reasons: list[str] = []
            versions: dict[str, Any] = {}
            with self.store.connect() as db:
                clauses, values = [], []
                if relative != ".":
                    clauses.append("path LIKE ? ESCAPE '\\'")
                    values.append(_like(relative) + "/%")
                sql = "SELECT path FROM files" + (" WHERE " + " AND ".join(clauses) if clauses else "")
                sql += " ORDER BY mtime_ns DESC,path LIMIT ?"
                db.set_progress_handler(lambda: int(time.monotonic() >= budget.deadline or budget.cancelled.is_set()), 1000)
                try:
                    for row in db.execute(sql, [*values, self.config.max_candidates + 1]):
                        budget.consume(candidates=1)
                        raw = str(row[0])
                        path = (self.workspace / raw).relative_to(base).as_posix()
                        match = glob_match(path, pattern)
                        if not match or (allowed is not None and not allowed(raw)) or (self.allowed is not None and not self.allowed(raw)):
                            continue
                        if len(shown) >= limit:
                            reasons.append("max_results")
                            break
                        try:
                            _, version = read_snapshot(self.workspace, raw, budget)
                        except (OSError, ValueError, StaleEvidence):
                            reasons.append(f"unavailable:{raw}")
                            continue
                        budget.output(raw + json.dumps(version.to_dict()))
                        shown.append(raw)
                        versions[raw] = version.to_dict()
                except (BudgetExceeded, sqlite3.OperationalError) as exc:
                    if isinstance(exc, sqlite3.OperationalError) and "interrupt" not in str(exc):
                        raise
                    reasons.append(str(exc))
                finally:
                    db.set_progress_handler(None, 0)
                if db.execute("SELECT 1 FROM pending LIMIT 1").fetchone():
                    reasons.append("pending_updates")
            content = "\n".join(shown) or "No files matched."
            if reasons:
                content += "\n[partial listing: " + ", ".join(reasons[:5]) + "; remaining scope not checked]"
            return ToolResult("glob", content, metadata={"matches": len(shown), "truncated": bool(reasons),
                "file_versions": versions, "coverage": {"scope": relative, "complete": not reasons,
                "stop_reasons": reasons}, "usage": budget.usage()})
        return await self._run(work, budget)

    async def ensure_index(self, *, scope: Any = None, reconcile: bool = False,
                           audit: bool = False) -> dict[str, Any]:
        budget = QueryBudget(self.config, scope)

        def work() -> dict[str, Any]:
            self._initialize()
            reason = None
            try:
                with self._lock, FileLock(self.store.path.with_suffix(".writer.lock"), timeout=max(0.01, budget.deadline - time.monotonic())):
                    budget.check()
                    from agent_core.codeintel.changes import replay_commits
                    replay_commits(self.store)
                    disk_bytes = sum(p.stat().st_size for p in
                                     (self.store.path, Path(str(self.store.path) + "-wal")) if p.exists())
                    if disk_bytes > self.config.max_index_bytes:
                        raise BudgetExceeded("index_disk_quota")
                    if reconcile:
                        self.catalog.start_reconcile()
                    if audit:
                        with self.store.connect() as db:
                            db.execute("INSERT OR IGNORE INTO pending SELECT path,'audit' FROM files")
                    # Commit progress in bounded chunks: a whole slice in one transaction
                    # held the SQLite write lock long enough to fail concurrent writers.
                    def drain() -> None:
                        nonlocal reason
                        while reason is None:
                            more = False
                            with self.store.batch():
                                try:
                                    more = update_pending(self.store, budget, self.allowed, chunk=128)
                                except (BudgetExceeded, TimeoutError) as exc:
                                    reason = str(exc)
                            if not more:
                                return
                    drain()
                    if reason is None:
                        self.catalog.advance(budget)
                    drain()
            except (BudgetExceeded, TimeoutError) as exc:
                reason = str(exc)
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc):
                    raise
                reason = "writer_busy"
            return {**self.store.status(), "stop_reason": reason, "usage": budget.usage()}

        return await self._run(work, budget)

    async def publish_changes(self, changes: ChangeSet) -> None:
        def work() -> None:
            self._initialize()
            paths = []
            for raw in changes.paths:
                try:
                    paths.append(contained(self.workspace, raw).relative_to(self.workspace).as_posix())
                except ValueError:
                    continue
            with self._lock, FileLock(self.store.path.with_suffix(".writer.lock"), timeout=self.config.query_seconds):
                self.store.enqueue(changes.event_id, tuple(paths), changes.source)
        await asyncio.to_thread(work)

    async def read_region(self, path: str, start: int = 1, limit: int = 200, *,
                          expected_version: dict[str, Any] | None = None, scope: Any = None) -> CodeHit:
        budget = QueryBudget(self.config, scope)
        def work() -> CodeHit:
            relative = contained(self.workspace, path).relative_to(self.workspace).as_posix()
            if self.allowed is not None and not self.allowed(relative):
                raise PermissionError("code path is restricted by read policy")
            data, version = read_snapshot(self.workspace, path, budget)
            if expected_version and (expected_version.get("sha256") != version.sha256
                                     or expected_version.get("worktree_id") != version.worktree_id
                                     or expected_version.get("path") != version.path):
                raise StaleEvidence("read version changed; retrieve current evidence")
            lines = data.decode("utf-8").splitlines()
            first = max(1, start)
            chosen: list[str] = []
            truncated = False
            for line in lines[first - 1:first - 1 + max(1, limit)]:
                try:
                    budget.output(line + "\n")
                except BudgetExceeded:
                    truncated = True
                    break
                chosen.append(line)
            if truncated:
                chosen.append("[... truncated: context output budget reached; request a smaller limit ...]")
            from agent_core.codeintel.backends import module
            return CodeHit(version.path, first, first + len(chosen) - 1, "\n".join(chosen), version, module=module(version.path))
        return await self._run(work, budget)

    async def search(self, request: SearchRequest, *, scope: Any = None,
                     allowed: Callable[[str], bool] | None = None) -> SearchPage:
        budget = QueryBudget(self.config, scope)
        def stage(modules: tuple[str, ...] | None = None) -> SearchPage:
            page = self._search(request, budget, allowed, resolved_modules=modules)
            if request.kind == "auto" and request.query.isidentifier() and not page.hits and request.cursor is None:
                try:
                    budget.check()
                    fallback = self._search(request, budget, allowed, resolved_kind="text", resolved_modules=modules)
                    fallback.coverage.checked = page.coverage.checked + fallback.coverage.checked
                    for reason in page.coverage.reasons:
                        fallback.coverage.add_reason(reason)
                    for advisory in page.coverage.advisories:
                        if advisory not in fallback.coverage.advisories:
                            fallback.coverage.advisories.append(advisory)
                    fallback.coverage.complete = fallback.coverage.complete and not fallback.coverage.reasons
                    return fallback
                except BudgetExceeded as exc:
                    page.coverage.add_reason(str(exc))
            return page
        def work() -> SearchPage:
            page = stage()
            if request.expand_scope and request.modules and not page.hits and request.cursor is None:
                try:
                    budget.check()
                    wider = stage(())
                    wider.coverage.checked = page.coverage.checked + wider.coverage.checked
                    wider.usage["scope_expansions"] = 1
                    return wider
                except BudgetExceeded as exc:
                    page.coverage.add_reason(str(exc))
            return page
        return await self._run(work, budget)

    def _search(self, request: SearchRequest, budget: QueryBudget,
                allowed: Callable[[str], bool] | None, *, resolved_kind: str | None = None,
                resolved_modules: tuple[str, ...] | None = None) -> SearchPage:
        self._initialize()
        from agent_core.codeintel.changes import replay_commits
        # This is an indexed outbox lookup, not a repository scan. Replay writes are
        # idempotent; a busy writer (chunked build, CLI, publish) must not fail the
        # query — hits are hash-verified against current bytes and pending rows are
        # already reflected in coverage.
        replay_note = None
        try:
            replay_commits(self.store)
        except (sqlite3.OperationalError, TimeoutError, OSError, ValueError) as exc:
            replay_note = f"commit_replay_deferred:{type(exc).__name__}"
        path = contained(self.workspace, request.path).relative_to(self.workspace).as_posix()
        page = SearchPage()
        if replay_note is not None:
            page.coverage.add_reason(replay_note)
        modules = request.modules if resolved_modules is None else resolved_modules
        kind = resolved_kind or request.kind
        if kind == "auto":
            from agent_core.codeintel.backends import LANGUAGES
            kind = "path" if "/" in request.query or "*" in request.query or Path(request.query).suffix in LANGUAGES else "symbol" if request.query.isidentifier() else "text"
        if kind not in {"path", "text", "symbol", "references", "calls", "imports", "inherits", "package_dependency", "build_dependency"}:
            raise ValueError(f"unsupported query kind: {kind}")
        if len(request.query) > 4096:
            raise ValueError("query exceeds 4096 characters")
        matcher = re.compile(request.query if request.regex else re.escape(request.query), re.I if request.ignore_case else 0)
        signature = hashlib.sha256(json.dumps({**asdict(request), "cursor": None}, sort_keys=True).encode()).hexdigest()
        after_path, after_line = "", 0
        query_generation = None
        if request.cursor:
            try:
                cursor = json.loads(base64.urlsafe_b64decode(request.cursor))
                if cursor["query"] != signature:
                    raise ValueError("cursor belongs to a different query")
                after_path, after_line, query_generation = cursor["path"], cursor["line"], cursor["generation"]
                kind = cursor.get("kind", kind)
                if request.expand_scope:
                    modules = tuple(cursor.get("modules", modules))
                if kind not in {"path", "text", "symbol", "references", "calls", "imports", "inherits", "package_dependency", "build_dependency"}:
                    raise ValueError("invalid cursor strategy")
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError("invalid query cursor") from exc
        page.coverage.requested = [f"module:{name} under {path}" for name in modules] or [path]
        limit = max(1, min(request.limit, self.config.max_results))
        exhausted = True
        with self.store.connect() as db:
            db.execute("BEGIN")
            page.generation = int(self.store.get(db, "generation"))
            if query_generation is not None and query_generation != page.generation:
                # A moved index mid-pagination is a coverage fact, not a tool error:
                # return an empty page that tells the caller to restart from cursor=None.
                page.coverage.add_reason("generation_changed; restart pagination without cursor")
                page.usage = budget.usage()
                return page
            if self.store.get(db, "catalog_complete") != "1":
                page.coverage.add_reason("catalog_incomplete: run code index build")
            if db.execute("SELECT 1 FROM pending LIMIT 1").fetchone():
                page.coverage.add_reason("pending_updates")
            if self.store.get(db, "scan_error") != "0":
                page.coverage.add_reason("unreadable_catalog_entries")
            clauses, args = ["f.path >= ?"], [after_path]
            if path != ".":
                clauses.append("(f.path=? OR f.path LIKE ? ESCAPE '\\')")
                args.extend([path, _like(path) + "/%"])
            if modules:
                chosen = modules[:self.config.max_shards]
                clauses.append("f.module IN (" + ",".join("?" for _ in chosen) + ")")
                args.extend(chosen)
                if len(chosen) < len(modules):
                    page.coverage.add_reason("max_shards")
                    page.coverage.unchecked.extend(modules[len(chosen):])
            if request.language:
                clauses.append("f.language=?")
                args.append(request.language)
            join, extra = "", "0 AS match_line, 0 AS match_end, '' AS match_name, '' AS precision"
            relation_fields = "'' AS source_symbol,'' AS target_symbol"
            if kind == "path":
                if any(char in request.query for char in "*?["):
                    clauses.append("(f.path GLOB ? OR f.path GLOB ?)")
                    args.extend([request.query, request.query.removeprefix("**/")])
                else:
                    clauses.append("f.path LIKE ? ESCAPE '\\'")
                    args.append("%" + _like(request.query) + "%")
            elif kind == "symbol":
                join = "JOIN symbols s ON s.file_id=f.id"
                clauses.extend(["f.state='ready'", "(s.name=? OR s.qualified=?)"])
                args.extend([request.query, request.query])
                # Force selective symbol discovery before module/file joins. Without
                # this prefilter SQLite may visit every symbol in a module.
                clauses.append("f.id IN (SELECT file_id FROM symbols WHERE name=? OR qualified=?)")
                args.extend([request.query, request.query])
                extra = "s.line AS match_line,s.end_line AS match_end,s.qualified AS match_name,'syntax_definition' AS precision"
            elif kind in {"references", "calls", "imports", "inherits", "package_dependency", "build_dependency"}:
                join = "JOIN edges e ON e.file_id=f.id"
                column = "e.source" if request.direction == "outgoing" else "e.target"
                clauses.extend(["f.state='ready'", "e.kind=?", f"({column}=? OR {column} LIKE ? ESCAPE '\\')"])
                args.extend([kind, request.query, "%." + _like(request.query)])
                extra = "e.line AS match_line,e.line AS match_end,e.target AS match_name,e.precision AS precision"
                relation_fields = "e.source AS source_symbol,e.target AS target_symbol"
            elif kind == "text":
                clauses.append("f.state IN ('ready','text_only')")
                if not request.regex and not request.ignore_case and len(request.query) >= 3 and self.store.get(db, "fts") == "1":
                    clauses.append("f.id IN (SELECT rowid FROM text_fts WHERE text_fts MATCH ?)")
                    args.append('"' + request.query.replace('"', '""') + '"')
            # A bounded query can never imply absence in skipped/unsupported files.
            if db.execute("SELECT 1 FROM files WHERE state='skipped' LIMIT 1").fetchone():
                page.coverage.add_reason("excluded_unindexable_files")
            if kind not in {"path", "text"}:
                advisory = "python_syntax_only; dynamic and other-language relations unresolved"
                if advisory not in page.coverage.advisories:
                    page.coverage.advisories.append(advisory)
            sql = "SELECT f.path,f.hash,f.revision,f.module,f.state," + extra + "," + relation_fields + " FROM files f " + join
            sql += " WHERE " + " AND ".join(clauses) + " ORDER BY f.path,match_line LIMIT ?"
            args.append(self.config.max_candidates + 1)  # type: ignore[arg-type]
            db.set_progress_handler(lambda: int(time.monotonic() >= budget.deadline or budget.cancelled.is_set()
                                               or (budget.scope is not None and budget.scope.cancelled())), 1000)
            verified: dict[str, tuple[list[str], Any]] = {}
            regex_candidates: list[tuple[str, str, int, str]] = []
            try:
                for row in db.execute(sql, args):
                    budget.consume(candidates=1)
                    raw = str(row["path"])
                    if (allowed is not None and not allowed(raw)) or (self.allowed is not None and not self.allowed(raw)):
                        if "permission_filtered" not in page.coverage.reasons:
                            page.coverage.add_reason("permission_filtered")
                        continue
                    if kind == "path":
                        if any(char in request.query for char in "*?["):
                            # Component-aware semantics, same as the glob tool.
                            if not glob_match(raw, request.query):
                                continue
                        elif request.query not in raw:
                            continue
                    if kind == "text" and request.regex:
                        # Defer to one batched matcher process after candidate collection.
                        regex_candidates.append((raw, str(row["hash"]), int(row["revision"]), str(row["module"])))
                        continue
                    if raw not in verified:
                        try:
                            data, version = read_snapshot(self.workspace, raw, budget, revision=int(row["revision"]))
                            if kind != "path" and version.sha256 != row["hash"]:
                                page.coverage.add_reason(f"stale:{raw}")
                                continue
                            verified[raw] = (self.cache.lines(version.sha256, data), version)
                        except (OSError, ValueError, UnicodeError, StaleEvidence) as exc:
                            page.coverage.add_reason(f"unavailable:{raw}:{type(exc).__name__}")
                            continue
                    lines, version = verified[raw]
                    hits: Iterable[tuple[int, int, str]]
                    if kind == "path":
                        hits = [(0, 0, raw)]
                    elif kind == "text":
                        hits = ((i, i, line) for i, line in enumerate(lines, 1) if matcher.search(line))
                    else:
                        first, last = int(row["match_line"]), int(row["match_end"])
                        hits = [(first, last, "\n".join(lines[first - 1:min(last, first + 10)]))]
                    for first, last, text in hits:
                        budget.check()
                        if raw == after_path and first <= after_line:
                            continue
                        if len(page.hits) >= limit:
                            exhausted = False
                            raise BudgetExceeded("max_results")
                        hit = CodeHit(raw, first or None, last or None, text, version, kind,
                                      str(row["precision"]) or "literal", str(row["module"]),
                                      row["source_symbol"] or None, row["target_symbol"] or None)
                        budget.output(json.dumps(asdict(hit), ensure_ascii=False))
                        page.hits.append(hit)
                    # Cache only the current file, not the entire query's source text.
                    verified = {raw: verified[raw]}
            except BudgetExceeded as exc:
                exhausted = False
                page.coverage.add_reason(str(exc))
            except sqlite3.OperationalError as exc:
                if "interrupt" not in str(exc):
                    raise
                exhausted = False
                if budget.scope is not None and budget.scope.cancelled():
                    budget.scope.raise_if_cancelled()
                page.coverage.add_reason("sql_interrupted_or_deadline")
            finally:
                db.set_progress_handler(None, 0)
        if kind == "text" and request.regex and regex_candidates:
            from agent_core.codeintel.scanning import regex_files
            versions: dict[str, tuple[Any, str]] = {}

            def produce() -> Iterable[tuple[str, bytes]]:
                for raw, expected_hash, revision, module_name in regex_candidates:
                    try:
                        data, version = read_snapshot(self.workspace, raw, budget, revision=revision)
                    except (OSError, ValueError, UnicodeError, StaleEvidence) as exc:
                        page.coverage.add_reason(f"unavailable:{raw}:{type(exc).__name__}")
                        continue
                    if version.sha256 != expected_hash:
                        page.coverage.add_reason(f"stale:{raw}")
                        continue
                    versions[raw] = (version, module_name)
                    yield raw, data

            def add_hit(raw: str, first: int, text: str) -> None:
                if raw == after_path and first <= after_line:
                    return
                if len(page.hits) >= limit:
                    raise BudgetExceeded("max_results")
                version, module_name = versions[raw]
                hit = CodeHit(raw, first, first, text, version, kind, "literal", module_name)
                budget.output(json.dumps(asdict(hit), ensure_ascii=False))
                page.hits.append(hit)

            try:
                with closing(regex_files(request.query, produce, self.workspace, budget,
                                         request.ignore_case)) as matches:
                    for raw, first, text in matches:
                        budget.check()
                        add_hit(raw, first, text)
            except BudgetExceeded as exc:
                exhausted = False
                page.coverage.add_reason(str(exc))
            except OSError as exc:
                if page.hits:
                    # The matcher child died mid-stream; report partial hits.
                    exhausted = False
                    page.coverage.add_reason(f"regex_engine_failed:{type(exc).__name__}")
                else:
                    # No spawn-capable child: match in-process, where a pathological
                    # pattern is bounded by per-line budget checks but not mid-line.
                    try:
                        matcher = re.compile(request.query, re.I if request.ignore_case else 0)
                        for raw, data in produce():
                            for first, text in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                                budget.check()
                                if matcher.search(text):
                                    add_hit(raw, first, text)
                    except BudgetExceeded as inner:
                        exhausted = False
                        page.coverage.add_reason(str(inner))
        checked_scopes = [f"module:{name} under {path}" for name in modules[:self.config.max_shards]] or [path]
        page.coverage.checked = [value + " (indexed candidates; returned files hash-verified)" for value in checked_scopes]
        page.coverage.complete = exhausted and not page.coverage.reasons
        if not page.coverage.complete:
            page.coverage.unchecked.extend(value + " (remaining, pending, or unindexed candidates)" for value in checked_scopes)
        if not exhausted and page.hits:
            last_hit = page.hits[-1]
            page.cursor = base64.urlsafe_b64encode(json.dumps({"query": signature, "generation": page.generation,
                "path": last_hit.path, "line": last_hit.start_line or 0, "kind": kind, "modules": modules}).encode()).decode()
        page.usage = budget.usage()
        return page

    async def expand_relations(self, seeds: list[str], *, relation: str = "references", depth: int = 1,
                               scope: Any = None, direction: str = "incoming", path: str = ".",
                               modules: tuple[str, ...] = (), allowed: Callable[[str], bool] | None = None) -> SearchPage:
        budget = QueryBudget(self.config, scope)
        def work() -> SearchPage:
            result = SearchPage()
            frontier, visited = list(seeds), set()
            try:
                for _ in range(min(max(1, depth), self.config.max_depth)):
                    next_frontier = []
                    for seed in frontier:
                        if seed in visited:
                            continue
                        visited.add(seed)
                        page = self._search(SearchRequest(seed, kind=relation, path=path, modules=modules,
                                                          direction=direction), budget, allowed)
                        result.generation = page.generation
                        result.coverage.reasons.extend(page.coverage.reasons)
                        for hit in page.hits:
                            budget.consume(edges=1)
                            result.hits.append(hit)
                            if len(result.hits) >= self.config.max_results:
                                raise BudgetExceeded("max_results")
                            symbol = hit.source_symbol if direction == "incoming" else hit.target_symbol
                            if symbol:
                                next_frontier.append(symbol)
                    frontier = next_frontier
            except BudgetExceeded as exc:
                result.coverage.add_reason(str(exc))
            result.coverage.requested = seeds
            if depth > self.config.max_depth:
                result.coverage.add_reason("max_depth")
            result.coverage.checked = sorted(visited)
            result.coverage.unchecked = frontier
            result.coverage.complete = False  # syntactic graph is not a proof of complete resolution
            result.usage = budget.usage()
            return result
        return await self._run(work, budget)

    async def close(self) -> None:
        self._closed = True
        if self._background is not None:
            self._background.cancel()
            await asyncio.gather(self._background, return_exceptions=True)
            self._background = None
        if self._observer is not None:
            self._observer.stop()
            await asyncio.to_thread(self._observer.join, 2)
            self._observer = None
        self.catalog.close()
        await asyncio.to_thread(self.store.close)
