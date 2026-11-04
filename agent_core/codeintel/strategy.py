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


class RetrievalController:
    """Stage router for bounded retrieval: orders stages, converges repeats, never loops.

    Decisions are advisory records for the caller's normal tool flow; the controller
    itself executes nothing, so permissions, locks, budgets, logging and sandboxing all
    stay with the ordinary tool pipeline. Hard gates: repeated no-gain queries are
    deduped, total stages per plan are capped, and one shared budget covers every stage.
    """

    MAX_STAGES = 16

    def __init__(self) -> None:
        self.stages = 0
        self.seen: set[str] = set()
        self.decisions: list[dict[str, Any]] = []

    def _record(self, decision: str, reason: str, query: str = "") -> dict[str, Any]:
        entry = {"stage_seq": self.stages, "decision": decision, "reason": reason[:256]}
        if query:
            entry["query_fingerprint"] = fingerprint(query)[:16]
        self.decisions.append(entry)
        return entry

    def stages_for(self, query: str) -> list[str]:
        """Initial stage route: explicit paths scope first; identifiers hit symbols then
        text; natural language goes lexical. Semantic retrieval is deferred by design."""
        if self.stages >= self.MAX_STAGES:
            self._record("stop", "stage budget exhausted", query)
            return []
        path_shaped = "/" in query or "*" in query or "." in query.rsplit("/", 1)[-1]
        if path_shaped:
            route = ["path", "text"]
        elif query.isidentifier():
            route = ["symbol", "text"]
        else:
            route = ["text"]
        self._record("route", f"{route[0]}-first", query)
        return route

    def allow_execution(self, request: SearchRequest, *, tool: str = "code_search") -> bool:
        """An identical (tool, request) already executed in this controller's scope
        converges instead of re-executing; across calls, searches always re-run."""
        key = fingerprint([tool, asdict(request), 1])
        if key in self.seen:
            self._record("dedupe", "identical query already executed in this plan; converging")
            return False
        self.seen.add(key)
        return True

    def proceed(self, page: SearchPage) -> bool:
        """Whether to continue to the next stage after a page: stop on complete zero-hit
        coverage (nothing more exists in this scope) or a missing cursor on a finished scan."""
        if self.stages >= self.MAX_STAGES:
            self._record("stop", "stage budget exhausted")
            return False
        if not page.hits and page.scan_complete and page.coverage.complete:
            self._record("stop", "complete coverage with no hits; remaining stages cannot help")
            return False
        return True

    def stage_done(self, stage: str, page: SearchPage) -> None:
        self.stages += 1
        self._record("stage", f"{stage}: hits={len(page.hits)} complete={page.coverage.complete}")
