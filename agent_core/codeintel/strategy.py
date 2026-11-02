"""Bounded exploration observations; guidance never replaces a fresh tool query."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
from typing import Any

from agent_core.codeintel.models import SearchPage, SearchRequest


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class ExplorationState:
    def __init__(self, max_entries: int = 64) -> None:
        self.max_entries = max_entries
        self.queries: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def record(self, request: SearchRequest, page: SearchPage, *, tool: str = "code_search",
               depth: int = 1, new_evidence: int = 0) -> dict[str, Any]:
        # Include the input cursor: a legitimate continuation is not repetition.
        # Persist fingerprints, not raw query strings or potentially large cursors.
        key = fingerprint([tool, asdict(request), depth])
        result = fingerprint([[(h.path, h.start_line, h.end_line, h.version.sha256, h.kind,
                               h.source_symbol, h.target_symbol) for h in page.hits],
                              asdict(page.coverage), page.cursor, page.scan_complete])
        previous = self.queries.pop(key, {})
        unchanged = (previous.get("freshness") == "observed" and previous.get("generation") == page.generation
                     and previous.get("result") == result and new_evidence == 0)
        action = ("continue_cursor" if page.cursor else "refresh_or_narrow_scope" if not page.coverage.complete
                  else "read_candidate_context" if page.hits else "change_query_or_expand_scope")
        entry = {"query_id": key, "result": result, "kind": request.kind, "path": request.path[:512],
                 "modules": [m[:512] for m in request.modules[:16]],
                 "checked": [s[:512] for s in page.coverage.checked[:16]],
                 "scope_truncated": (len(request.path) > 512 or len(request.modules) > 16 or len(page.coverage.checked) > 16 or
                                     any(len(s) > 512 for s in (*request.modules, *page.coverage.checked))),
                 "generation": page.generation, "attempts": min(1_000_000, previous.get("attempts", 0) + 1),
                 "no_gain_repeats": min(1_000_000, previous.get("no_gain_repeats", 0) + 1) if unchanged else 0,
                 "new_evidence": new_evidence, "hit_count": len(page.hits),
                 "coverage_complete": page.coverage.complete, "next_action": action, "freshness": "observed"}
        self.queries[key] = entry
        while len(self.queries) > self.max_entries:
            self.queries.popitem(last=False)
        return {**{k: v for k, v in entry.items() if k != "result"},
                "advisory": "Repeated query produced no new evidence; change strategy." if unchanged
                else "Coverage is observational; re-read candidates before editing."}

    def invalidate(self) -> None:
        for query in self.queries.values():
            query["freshness"] = "stale"

    def restore(self, records: list[dict[str, Any]]) -> None:
        if not isinstance(records, list) or len(records) > self.max_entries:
            raise ValueError("invalid exploration query history")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("invalid exploration query")
            if any(not isinstance(record.get(k), str) or len(record[k]) != 64 or
                   any(c not in "0123456789abcdef" for c in record[k]) for k in ("query_id", "result")):
                raise ValueError("invalid exploration fingerprint")
            if any(type(record.get(k)) is not int or not 0 <= record[k] <= 1_000_000 for k in
                   ("attempts", "no_gain_repeats", "new_evidence", "hit_count")):
                raise ValueError("invalid exploration counts")
            if (type(record.get("generation")) is not int or record["generation"] < 0
                    or type(record.get("coverage_complete")) is not bool
                    or any(not isinstance(record.get(k), str) or len(record[k]) > 512 for k in ("kind", "path", "next_action"))):
                raise ValueError("invalid exploration fields")
            for key in ("modules", "checked"):
                if not isinstance(record.get(key, []), list) or len(record.get(key, [])) > 16 or any(
                        not isinstance(s, str) or len(s) > 512 for s in record.get(key, [])):
                    raise ValueError("invalid exploration scope")
            if type(record.get("scope_truncated", False)) is not bool:
                raise ValueError("invalid exploration scope disclosure")
            # Whitelist persisted fields. Unknown fields cannot become prompt data.
            self.queries[record["query_id"]] = {k: record[k] for k in
                ("query_id", "result", "kind", "path", "generation", "attempts", "no_gain_repeats",
                 "new_evidence", "hit_count", "coverage_complete", "next_action")}
            self.queries[record["query_id"]]["freshness"] = "needs_revalidation"
            self.queries[record["query_id"]].update(modules=record.get("modules", []), checked=record.get("checked", []),
                                                   scope_truncated=record.get("scope_truncated", False))
