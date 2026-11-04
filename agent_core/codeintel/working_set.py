from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
import json
from pathlib import PurePosixPath, PureWindowsPath
import re
from typing import Any

from agent_core.codeintel import candidate_pool
from agent_core.codeintel.models import CodeHit
from agent_core.codeintel.strategy import ExplorationState
from agent_core.permission_safety import is_secret_path

MAX_CANDIDATES = 64


class TaskWorkingSet:
    """Bounded evidence references, independent of history and long-term memory.

    Candidate records merge text/symbol/relation/LSP evidence by normalized
    identity. They are evidence metadata only: they never carry raw source, raw
    queries or edit credentials, and a stale/invalid candidate cannot satisfy an
    edit precondition — editing still requires a fresh code_validate.
    """

    def __init__(self, max_entries: int = 128) -> None:
        self.max_entries = max_entries
        self.entries: OrderedDict[tuple[str, int | None, int | None], dict[str, Any]] = OrderedDict()
        self.candidates: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.exploration = ExplorationState()
        self.binding: tuple[str, str] | None = None

    def add(self, hit: CodeHit, *, reason: str = "retrieval") -> bool:
        key = (hit.path, hit.start_line, hit.end_line)
        previous = self.entries.pop(key, None)
        self.entries[key] = {"path": hit.path, "start_line": hit.start_line, "end_line": hit.end_line,
                             "version": asdict(hit.version), "module": hit.module, "reason": reason,
                             "freshness": "observed"}
        while len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)
        return previous is None or previous["version"] != asdict(hit.version) or previous.get("freshness") != "observed"

    def add_candidate(self, hit: CodeHit, *, origin: str, evidence_ref: str,
                      role: str | None = None) -> dict[str, Any]:
        """Merge one observation into the unified candidate pool; returns the record."""
        symbol = None
        if isinstance(hit.symbol, dict):
            symbol = {k: hit.symbol.get(k) for k in ("qualified_name", "kind") if hit.symbol.get(k)}
            if hit.start_line is not None and hit.end_line is not None:
                symbol["range"] = [hit.start_line, hit.end_line]
        line_range = (hit.start_line, hit.end_line) if hit.start_line is not None and hit.end_line is not None else None
        key = candidate_pool.candidate_id(hit.path, symbol, line_range)
        record = self.candidates.get(key)
        if record is None:
            record = candidate_pool.new_candidate(hit.path, asdict(hit.version), symbol=symbol,
                                                  line_range=line_range, origin=origin)
            self.candidates[key] = record
            while len(self.candidates) > MAX_CANDIDATES:
                self.candidates.popitem(last=False)
        candidate_pool.merge_evidence(record, evidence_ref=evidence_ref,
                                      note=f"{hit.version.sha256[:12]}@{hit.module}",
                                      role=role, expected_version=asdict(hit.version))
        return record

    def candidate_for(self, path: str, symbol: dict[str, Any] | None = None,
                      line_range: tuple[int, int] | None = None) -> dict[str, Any] | None:
        if symbol is not None or line_range is not None:
            return self.candidates.get(candidate_pool.candidate_id(path, symbol, line_range))
        for record in reversed(list(self.candidates.values())):
            if record["path"] == path:
                return record
        return None

    def record_validation(self, path: str, *, ok: bool, version: dict[str, Any],
                          anchor_sha256: str | None = None, detail: str | None = None,
                          symbol: dict[str, Any] | None = None) -> dict[str, Any]:
        record = self.candidate_for(path, symbol=symbol)
        if record is None:
            record = candidate_pool.new_candidate(path, version, symbol=symbol, origin="code_validate")
            self.candidates[record["id"]] = record
            while len(self.candidates) > MAX_CANDIDATES:
                self.candidates.popitem(last=False)
        candidate_pool.record_anchor_validation(record, ok=ok, version=version,
                                                anchor_sha256=anchor_sha256, detail=detail)
        return record

    def invalidate(self, paths: tuple[str, ...]) -> None:
        changed = set(paths)
        for key in list(self.entries):
            if key[0] in changed:
                self.entries[key]["freshness"] = "stale"
        for record in self.candidates.values():
            if record["path"] in changed:
                candidate_pool.invalidate(record, "committed paths changed")
        self.exploration.invalidate()

    def to_state(self) -> dict[str, Any]:
        if self.binding is None or not (self.entries or self.exploration.queries or self.candidates):
            return {}
        value: dict[str, Any] = {"v": 2 if self.candidates else 1, "workspace": self.binding[0],
                 "policy": self.binding[1], "evidence": list(self.entries.values()),
                 "queries": list(self.exploration.queries.values())}
        if self.candidates:
            value["candidates"] = list(self.candidates.values())
        # Keep metadata bounded even with unusually long source names.
        while len(json.dumps(value, ensure_ascii=False).encode()) > 128 * 1024:
            if value.get("candidates"):
                value["candidates"].pop(0)
                continue
            records = value["queries"] if value["queries"] else value["evidence"]
            if not records:
                raise ValueError("exploration binding exceeds budget")
            records.pop(0)
        return json.loads(json.dumps(value))

    def restore(self, value: dict[str, Any], *, allowed) -> None:
        if not isinstance(value, dict) or value.get("v") not in (1, 2) or len(json.dumps(value, ensure_ascii=False).encode()) > 128 * 1024:
            raise ValueError("invalid exploration state")
        if (value.get("workspace"), value.get("policy")) != self.binding:
            return
        evidence = value.get("evidence")
        if not isinstance(evidence, list) or len(evidence) > self.max_entries:
            raise ValueError("invalid persisted evidence")
        for entry in evidence:
            if not isinstance(entry, dict):
                raise ValueError("invalid evidence reference")
            path, version = entry.get("path"), entry.get("version")
            if (not isinstance(path, str) or not path or len(path) > 4096 or PurePosixPath(path).is_absolute()
                    or PureWindowsPath(path).drive or "\\" in path or ":" in path
                    or any(p in {"", ".", ".."} for p in path.split("/")) or is_secret_path(path)
                    or not isinstance(version, dict) or version.get("path") != path
                    or version.get("worktree_id") != self.binding[0]
                    or not isinstance(version.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", version["sha256"]) is None
                    or type(version.get("revision")) is not int or version["revision"] < 0):
                raise ValueError("invalid evidence identity")
            if any(entry.get(k) is not None and (type(entry[k]) is not int or entry[k] < 1) for k in ("start_line", "end_line")):
                raise ValueError("invalid evidence range")
            if any(not isinstance(entry.get(k), str) or len(entry[k]) > 4096 for k in ("module", "reason")):
                raise ValueError("invalid evidence annotation")
            if allowed(path):
                record = {k: entry[k] for k in ("path", "start_line", "end_line", "module", "reason")}
                record["version"] = {k: version[k] for k in ("worktree_id", "path", "sha256", "revision")}
                record["freshness"] = "needs_revalidation"
                self.entries[(path, entry["start_line"], entry["end_line"])] = record
        candidates = value.get("candidates", [])
        if not isinstance(candidates, list) or len(candidates) > MAX_CANDIDATES:
            raise ValueError("invalid persisted candidates")
        for record in candidates:
            if not candidate_pool.validate_record(record):
                raise ValueError("invalid persisted candidate record")
            path = record["path"]
            if (PurePosixPath(path).is_absolute() or PureWindowsPath(path).drive or "\\" in path or ":" in path
                    or any(p in {"", ".", ".."} for p in path.split("/")) or is_secret_path(path)
                    or record["expected_version"]["worktree_id"] != self.binding[0]):
                raise ValueError("invalid persisted candidate identity")
            if allowed(path):
                restored = json.loads(json.dumps(record))
                # Restored candidates are historical: they must be revalidated before any use.
                restored["validation_state"] = "needs_revalidation"
                restored["invalidation_reason"] = "restored from history"
                self.candidates[restored["id"]] = restored
        self.exploration.restore(value.get("queries", []))

    def modules(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(e["module"]) for e in reversed(list(self.entries.values()))))

    def render_references(self, max_bytes: int = 4000) -> str:
        lines = ["Code evidence references (historical; revalidate with code_context before editing):"]
        used = len(lines[0].encode())
        evidence_budget = max_bytes - min(1600, max_bytes // 2) if self.exploration.queries else max_bytes
        for entry in reversed(list(self.entries.values())):
            line = f"{entry['path']}:{entry['start_line']}-{entry['end_line']} sha256={entry['version']['sha256']} freshness={entry.get('freshness', 'historical')}"
            size = len(line.encode()) + 1
            if used + size > evidence_budget:
                break
            lines.append(line)
            used += size
        for record in reversed(list(self.candidates.values())[-16:]):
            line = (f"candidate {record['path']} role={record['role']} state={record['validation_state']}"
                    f" evidence={len(record['evidence_refs'])} conflicts={len(record['contradicting'])}")
            if record.get("invalidation_reason"):
                line += f" reason={record['invalidation_reason'][:120]}"
            size = len(line.encode()) + 1
            if used + size > max_bytes:
                break
            lines.append(line)
            used += size
        for query in reversed(list(self.exploration.queries.values())[-8:]):
            line = json.dumps({k: query[k] for k in ("query_id", "kind", "no_gain_repeats", "next_action", "freshness")})
            if used + len(line.encode()) + 1 > max_bytes:
                break
            lines.append(line)
            used += len(line.encode()) + 1
        return "\n".join(lines) if len(lines) > 1 else ""
