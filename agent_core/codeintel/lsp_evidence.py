"""Bounded LSP location observations joined to current, permitted source bytes."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import time
from urllib.parse import urlparse

from agent_core.codeintel.budget import BudgetExceeded, QueryBudget
from agent_core.codeintel.config import CodeIntelConfig
from agent_core.codeintel.models import CodeHit, SearchPage, StaleEvidence
from agent_core.codeintel.runtime import ensure_working_set
from agent_core.codeintel.snapshots import contained, read_snapshot
from agent_core.execution import current_execution_scope
from agent_core.lsp import uri_to_path


OPERATIONS = frozenset({"definition", "references", "implementation"})

SYMBOL_KINDS = {1: "file", 2: "module", 3: "namespace", 4: "package", 5: "class", 6: "method",
                7: "property", 8: "field", 9: "constructor", 10: "enum", 11: "interface",
                12: "function", 13: "variable", 14: "constant", 15: "string", 16: "number",
                17: "boolean", 18: "array", 19: "object", 20: "key", 21: "null",
                22: "enum_member", 23: "struct", 24: "event", 25: "operator", 26: "type_parameter"}


def _symbol_kind(value) -> str:
    return SYMBOL_KINDS.get(value, f"kind_{value}")


def _flatten_document_symbols(items, parent="", depth=0):
    """Yield (qualified, kind, points) from hierarchical DocumentSymbol or flat
    SymbolInformation payloads; ranges stay server-declared UTF-16."""
    if depth > 32:
        return
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"]:
            raise ValueError("invalid LSP document symbol")
        name = item["name"][:256]
        kind = _symbol_kind(item.get("kind"))
        if "location" in item:  # SymbolInformation: flat, carries its own URI
            uri, points = location(item["location"])
            container = item.get("containerName")
            qualified = f"{container}.{name}"[:512] if isinstance(container, str) and container else name
            yield uri, qualified, kind, points
            continue
        span = item.get("range")
        if not isinstance(span, dict):
            raise ValueError("invalid LSP document symbol range")
        points = []
        for key_point in ("start", "end"):
            p = span.get(key_point)
            if not isinstance(p, dict) or any(type(p.get(k)) is not int or p[k] < 0 for k in ("line", "character")):
                raise ValueError("invalid LSP position")
            points.append((p["line"], p["character"]))
        if points[1] < points[0]:
            raise ValueError("reversed LSP range")
        qualified = f"{parent}.{name}"[:512] if parent else name
        yield None, qualified, kind, points
        children = item.get("children")
        if children:
            yield from _flatten_document_symbols(children, qualified, depth + 1)


async def query_document_symbols(session, manager, path):
    """Structured DocumentSymbol/SymbolInformation integration with the same
    permission, containment, UTF-16 and current-version checks as location evidence."""
    from agent_core.tools.codeintel import path_allowed
    budget = QueryBudget(session.codeintel_config or CodeIntelConfig(), current_execution_scope())
    root = session.workspace
    source = contained(root, path).relative_to(root).as_posix()
    def allowed(raw):
        return path_allowed(session, raw, "lsp") and path_allowed(session, raw, "code_context")
    if not allowed(source):
        raise PermissionError("LSP source is restricted by read policy")
    async def read(raw):
        if not allowed(raw):
            raise PermissionError("LSP target is restricted by read policy")
        return await asyncio.to_thread(read_snapshot, root, raw, budget)
    data, source_version = await read(source)
    page = SearchPage()
    page.coverage.requested = [source]
    page.coverage.advisories = ["LSP-reported document symbols; server freshness is not certified; validate before editing"]
    response = await asyncio.wait_for(manager.request("document_symbols", path=source, budget=budget, strict_capability=True),
                                      timeout=max(0, budget.deadline - time.monotonic()))
    if isinstance(response, dict) and response.get("unavailable"):
        page.coverage.add_reason(f"lsp_unavailable:document_symbols:{str(response.get('detail'))[:120]}")
        page.coverage.complete = False
        page.usage = budget.usage()
        return page
    items = [] if response is None else response if isinstance(response, list) else [response]
    versions = {source: source_version}
    texts = {source: data.decode("utf-8")}
    try:
        for uri, qualified, kind, points in _flatten_document_symbols(items):
            budget.consume(candidates=1)
            if len(page.hits) >= budget.config.max_results:
                raise BudgetExceeded("max_results")
            try:
                if uri is None:
                    target = source
                else:
                    resolved = uri_to_path(uri)
                    if not resolved.is_absolute():
                        raise ValueError("relative LSP URI")
                    target = contained(root, str(resolved)).relative_to(root).as_posix()
                if target not in versions:
                    data, version = await read(target)
                    versions[target] = version
                    texts[target] = data.decode("utf-8")
                version = versions[target]
                first, last = line_range(texts[target], points)
                end = min(first + 10, last)
                hit = CodeHit(target, first, last, "\n".join(texts[target].splitlines()[first - 1:end]), version,
                    kind="document_symbol", precision="lsp_reported_location", excerpt_start_line=first,
                    excerpt_end_line=end, truncated=end < last, next_start_line=end + 1 if end < last else None,
                    symbol={"qualified_name": qualified, "kind": kind, "resolution": "lsp_document_symbol"})
                budget.output(json.dumps(asdict(hit), ensure_ascii=False))
                page.hits.append(hit)
            except (OSError, ValueError, UnicodeError):
                page.coverage.add_reason("unavailable_or_invalid_lsp_symbol")
        page.scan_complete = True
        for raw, version in versions.items():
            _, current = await read(raw)
            if current.sha256 != version.sha256:
                raise StaleEvidence("source moved during LSP symbol collection")
    except BudgetExceeded as exc:
        page.coverage.add_reason(str(exc))
    page.coverage.checked = ["returned permitted symbols; source hashes observed"]
    page.coverage.complete = False
    page.coverage.add_reason("lsp_server_coverage_unknown")
    page.usage = budget.usage()
    from agent_core.codeintel.runtime import ensure_working_set
    working = ensure_working_set(session)
    for hit in page.hits:
        working.add(hit, reason="lsp_document_symbols")
        working.add_candidate(hit, origin="lsp", evidence_ref="lsp:document_symbols")
    return page


def location(item):
    if not isinstance(item, dict):
        raise ValueError("invalid LSP location")
    uri = item.get("targetUri", item.get("uri"))
    span = item.get("targetSelectionRange", item.get("range"))
    if not isinstance(uri, str) or not isinstance(span, dict):
        raise ValueError("LSP location lacks URI/range")
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"} or parsed.query or parsed.fragment:
        raise ValueError("unsupported LSP location URI")
    points = []
    for key in ("start", "end"):
        p = span.get(key)
        if not isinstance(p, dict) or any(type(p.get(k)) is not int or p[k] < 0 for k in ("line", "character")):
            raise ValueError("invalid LSP position")
        points.append((p["line"], p["character"]))
    if points[1] < points[0]:
        raise ValueError("reversed LSP range")
    return uri, points


def line_range(text, points):
    # LSP positions are zero-based UTF-16 units, with an exclusive end.
    lines = text.split("\n")
    for line, column in points:
        if line >= len(lines):
            raise ValueError("LSP range exceeds source")
        units = lines[line].rstrip("\r").encode("utf-16-le")
        if column * 2 > len(units):
            raise ValueError("LSP column exceeds source")
        units[:column * 2].decode("utf-16-le")  # Reject splitting a surrogate pair.
    start, end = points
    return start[0] + 1, max(start[0] + 1, end[0] + (1 if end[1] else 0))


async def query_evidence(session, manager, operation, *, path, line=0, character=0):
    from agent_core.tools.codeintel import path_allowed
    budget = QueryBudget(session.codeintel_config or CodeIntelConfig(), current_execution_scope())
    root = session.workspace
    source = contained(root, path).relative_to(root).as_posix()
    def allowed(raw):
        return path_allowed(session, raw, "lsp") and path_allowed(session, raw, "code_context")
    if not allowed(source):
        raise PermissionError("LSP source is restricted by read policy")
    async def read(raw):
        if not allowed(raw):
            raise PermissionError("LSP target is restricted by read policy")
        return await asyncio.to_thread(read_snapshot, root, raw, budget)
    _, source_version = await read(source)
    page = SearchPage()
    page.coverage.requested = [source]
    page.coverage.advisories = ["LSP-reported locations; server dependency freshness is not certified; validate candidates before editing"]
    response = await asyncio.wait_for(manager.request(operation, path=source, line=line, character=character, budget=budget, strict_capability=True),
                                      timeout=max(0, budget.deadline - time.monotonic()))
    if isinstance(response, dict) and response.get("unavailable"):
        page.coverage.add_reason(f"lsp_unavailable:{operation}:{str(response.get('detail'))[:120]}")
        page.coverage.complete = False
        page.usage = budget.usage()
        return page
    candidates = [] if response is None else response if isinstance(response, list) else [response]
    versions = {source: source_version}
    try:
        for item in candidates:
            budget.consume(candidates=1)
            if len(page.hits) >= budget.config.max_results:
                raise BudgetExceeded("max_results")
            try:
                uri, points = location(item)
                resolved = uri_to_path(uri)
                if not resolved.is_absolute():
                    raise ValueError("relative LSP URI")
                target = contained(root, str(resolved)).relative_to(root).as_posix()
                data, version = await read(target)
                if target in versions and versions[target].sha256 != version.sha256:
                    raise StaleEvidence("source moved during LSP query")
                versions[target] = version
                text = data.decode("utf-8")
                first, last = line_range(text, points)
                end = min(first + 10, last)
                hit = CodeHit(target, first, last, "\n".join(text.splitlines()[first - 1:end]), version,
                    kind=operation, precision="lsp_reported_location", excerpt_start_line=first,
                    excerpt_end_line=end, truncated=end < last, next_start_line=end + 1 if end < last else None)
                budget.output(json.dumps(asdict(hit), ensure_ascii=False))
                page.hits.append(hit)
            except (OSError, ValueError, UnicodeError):
                # Do not expose restricted/foreign paths or server-supplied bodies.
                page.coverage.add_reason("unavailable_or_invalid_lsp_location")
        page.scan_complete = True
        for raw, version in versions.items():
            _, current = await read(raw)
            if current.sha256 != version.sha256:
                raise StaleEvidence("source moved during LSP evidence collection")
    except BudgetExceeded as exc:
        # Partially read results remain hints, not fresh edit preconditions.
        page.coverage.add_reason(str(exc))
    page.coverage.checked = ["returned permitted locations; source hashes observed"]
    # An LSP response never proves global indexing or reference completeness.
    page.coverage.complete = False
    page.coverage.add_reason("lsp_server_coverage_unknown")
    page.usage = budget.usage()
    working = ensure_working_set(session)
    for hit in page.hits:
        working.add(hit, reason="lsp_location_hint")
        working.add_candidate(hit, origin="lsp", evidence_ref=f"lsp:{operation}")
    return page
