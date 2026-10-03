from __future__ import annotations

import asyncio
from contextlib import closing
import fnmatch
import json
import logging
from pathlib import Path
import re
from typing import Any, Iterable

from agent_core.codeintel.budget import BudgetExceeded, QueryBudget
from agent_core.codeintel.config import CodeIntelConfig
from agent_core.codeintel.paths import glob_match
from agent_core.codeintel.scanning import regex_files, walk_files
from agent_core.codeintel.snapshots import read_snapshot, worktree_id
from agent_core.models import ToolResult
from agent_core.tools.base import current_execution_context, current_execution_scope

logger = logging.getLogger("agent_core.tools.builtin")


def _config(tool: Any) -> CodeIntelConfig:
    session = getattr(tool, "_code_session", None)
    return getattr(session, "codeintel_config", None) or CodeIntelConfig()


async def indexed_search(tool: Any, arguments: dict[str, Any], session: Any) -> ToolResult | None:
    from agent_core.codeintel.runtime import get_service, remember_version
    from agent_core.codeintel.models import SearchRequest
    from agent_core.tools.codeintel import path_allowed
    # Legacy glob needs newest-first ordering; its catalog adapter handles this below.
    service = await get_service(session)
    if not await service.catalog_ready():
        # A bounded legacy scan keeps first use useful; say why the index was bypassed.
        result = await asyncio.to_thread(tool._invoke, arguments)
        config = getattr(session, "codeintel_config", None)
        if config is not None and config.maintain:
            note = "code index still building; results from a bounded live scan"
        else:
            note = "code index not built; results from a bounded live scan; run polaris code build"
        return ToolResult(result.name, result.content + f"\n[{note}]", ok=result.ok, metadata=result.metadata)
    if tool.name == "glob":
        return await service.glob_paths(arguments, scope=current_execution_scope(),
                                        allowed=lambda p: path_allowed(session, p, "glob"))
    page = await service.search(SearchRequest(str(arguments["pattern"]), kind="text",
        path=str(arguments.get("path", ".")), regex=bool(arguments.get("regex")),
        ignore_case=bool(arguments.get("ignore_case")), limit=int(arguments.get("max_results", 100))),
        scope=current_execution_scope(), allowed=lambda p: path_allowed(session, p, "search_text") and
        (not arguments.get("glob") or fnmatch.fnmatch(Path(p).name, str(arguments["glob"]))))
    body = "\n".join(f"{h.path}:{h.start_line}: {h.text.strip()}" for h in page.hits) or "No matches."
    if not page.coverage.complete:
        body += "\n[partial indexed search: " + ", ".join(page.coverage.reasons[:5]) + "; remaining scope not checked]"
    for hit in page.hits:
        remember_version(session, hit.path, hit.version.to_dict())
        session.code_working_set.add(hit)
    return ToolResult(tool.name, body, metadata={"matches": len(page.hits), "truncated": not page.coverage.complete,
        "codeintel": page.to_dict()})


def search(tool: Any, arguments: dict[str, Any]) -> ToolResult:
    pattern = str(arguments["pattern"])
    base = tool.resolve_workspace_path(arguments.get("path", "."))
    config = _config(tool)
    budget = QueryBudget(config, current_execution_scope())
    limit = max(1, min(int(arguments.get("max_results", 100)), config.max_results))
    try:
        re.compile(pattern if arguments.get("regex") else re.escape(pattern))
    except re.error as exc:
        return ToolResult(tool.name, f"Invalid regex: {exc}", ok=False, metadata={"error_type": "BadRegex"})
    if not base.exists():
        return ToolResult(tool.name, f"No such path: {base}", ok=False, metadata={"error_type": "NotFound"})
    results: list[str] = []
    versions: dict[str, Any] = {}
    reasons: list[str] = []
    checked: list[str] = []
    context = current_execution_context()
    logical_root = context.logical_workspace if context and context.logical_workspace else tool.workspace
    regex = bool(arguments.get("regex"))

    def record(rel: str, line_number: int, line: str, version: dict[str, Any]) -> None:
        budget.check()
        rendered = f"{rel}:{line_number}: {line.strip()}"
        budget.output(rendered)
        results.append(rendered)
        versions[rel] = version
        if len(results) >= limit:
            raise BudgetExceeded("max_results")

    def candidate_stream() -> Iterable[tuple[str, bytes, dict[str, Any]]]:
        for file in walk_files(tool.workspace, base, budget):
            if arguments.get("glob") and not fnmatch.fnmatch(file.name, str(arguments["glob"])):
                continue
            rel = file.relative_to(tool.workspace).as_posix()
            try:
                raw, version = read_snapshot(tool.workspace, rel, budget, allow_secret=base.is_file())
            except (OSError, ValueError) as exc:
                reasons.append(f"skipped:{rel}:{type(exc).__name__}")
                continue
            checked.append(rel)
            if b"\0" in raw[:1024]:
                reasons.append(f"binary:{rel}")
                continue
            yield rel, raw, {**version.to_dict(), "worktree_id": worktree_id(logical_root)}

    try:
        if not regex:
            needle = pattern.lower() if arguments.get("ignore_case") else pattern
            for rel, raw, versioned in candidate_stream():
                for line_number, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
                    if needle in (line.lower() if arguments.get("ignore_case") else line):
                        record(rel, line_number, line, versioned)
    except BudgetExceeded as exc:
        reasons.append(str(exc))
    if regex:
        # One batched matcher child over a lazily-fed stream: matching overlaps
        # with reading, and matches emitted before budget death are kept.
        stream_versions: dict[str, Any] = {}

        def produce() -> Iterable[tuple[str, bytes]]:
            for rel, raw, versioned in candidate_stream():
                stream_versions[rel] = versioned
                yield rel, raw

        try:
            with closing(regex_files(pattern, produce, tool.workspace, budget,
                                     bool(arguments.get("ignore_case")))) as matches:
                for rel, line_number, line in matches:
                    record(rel, line_number, line, stream_versions[rel])
        except BudgetExceeded as exc:
            reasons.append(str(exc))
        except OSError as exc:
            if results:
                # The matcher child died mid-stream; report partial matches.
                reasons.append(f"regex_engine_failed:{type(exc).__name__}")
            else:
                # No spawn-capable child: match in-process, where a pathological pattern
                # is bounded by per-line budget checks but cannot be killed mid-line.
                matcher = re.compile(pattern, re.I if arguments.get("ignore_case") else 0)
                try:
                    for rel, raw, versioned in candidate_stream():
                        for line_number, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
                            if matcher.search(line):
                                record(rel, line_number, line, versioned)
                except BudgetExceeded as inner:
                    reasons.append(str(inner))
    body = "\n".join(results) or "No matches."
    if reasons:
        body += "\n[partial search: " + ", ".join(reasons[:5]) + "; remaining scope not checked]"
    return ToolResult(tool.name, body, metadata={"matches": len(results), "truncated": bool(reasons),
        "file_versions": versions, "coverage": {"checked": checked, "requested": str(arguments.get("path", ".")),
        "complete": not reasons, "stop_reasons": reasons}, "usage": budget.usage()})


def glob(tool: Any, arguments: dict[str, Any]) -> ToolResult:
    pattern = str(arguments["pattern"])
    base = tool.resolve_workspace_path(arguments.get("path", "."))
    config = _config(tool)
    budget = QueryBudget(config, current_execution_scope())
    limit = max(1, min(int(arguments.get("max_results", 200)), config.max_results))
    if not base.exists():
        return ToolResult(tool.name, f"No such path: {base}", ok=False, metadata={"error_type": "NotFound"})
    candidates: list[tuple[int, str]] = []
    matched = 0
    reasons = []
    try:
        for file in walk_files(tool.workspace, base, budget):
            rel = file.relative_to(base).as_posix()
            if glob_match(rel, pattern):
                matched += 1
                context = current_execution_context()
                view = context.workspace_view if context else None
                physical = view.resolve_read_path(file.relative_to(tool.workspace).as_posix()) if view and hasattr(view, "resolve_read_path") else file
                candidates.append((physical.stat().st_mtime_ns, file.relative_to(tool.workspace).as_posix()))
                if len(candidates) > limit * 2:
                    candidates.sort(reverse=True)
                    del candidates[limit:]
    except (BudgetExceeded, OSError) as exc:
        reasons.append(str(exc))
    candidates.sort(reverse=True)
    shown = []
    versions = {}
    context = current_execution_context()
    logical_root = context.logical_workspace if context and context.logical_workspace else tool.workspace
    for _, path in candidates[:limit]:
        try:
            _, version = read_snapshot(tool.workspace, path, budget)
            value = {**version.to_dict(), "worktree_id": worktree_id(logical_root)}
            budget.output(path + json.dumps(value))
            versions[path] = value
            shown.append(path)
        except BudgetExceeded as exc:
            reasons.append(str(exc))
            break
        except (OSError, ValueError) as exc:
            reasons.append(f"unavailable:{path}:{type(exc).__name__}")
    truncated = bool(reasons) or matched > limit
    body = "\n".join(shown) or "No files matched."
    if truncated:
        body += "\n[partial listing: result or scan budget reached; remaining scope not checked]"
    return ToolResult(tool.name, body, metadata={"matches": len(shown), "truncated": truncated, "file_versions": versions,
        "coverage": {"complete": not truncated, "scope": str(arguments.get("path", ".")), "stop_reasons": reasons},
        "usage": budget.usage()})
