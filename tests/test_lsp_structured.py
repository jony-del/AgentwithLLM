"""Structured LSP document symbols, bounded call hierarchy and budget threading (S5)."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from agent_core.codeintel.budget import BudgetExceeded, QueryBudget
from agent_core.codeintel.config import CodeIntelConfig
from agent_core.codeintel.lsp_evidence import query_document_symbols
from agent_core.codeintel.models import StaleEvidence
from agent_core.codeintel.runtime import ensure_working_set
from agent_core.lsp import LSPManager
from agent_core.tool_config import LSPServerConfig, LSPToolConfig


def session(root):
    return SimpleNamespace(workspace=root, codeintel_config=CodeIntelConfig(), code_permission_rules=None,
                           code_working_set=None, code_read_versions={}, task_run=None)


class SymbolManager:
    def __init__(self, payload):
        self.payload = payload
        self.kwargs = None

    async def request(self, operation, **kwargs):
        self.kwargs = kwargs
        return self.payload


HIERARCHICAL = [
    {"name": "alpha", "kind": 12,
     "range": {"start": {"line": 0, "character": 0}, "end": {"line": 1, "character": 9}},
     "selectionRange": {"start": {"line": 0, "character": 4}, "end": {"line": 0, "character": 9}}},
    {"name": "Beta", "kind": 5,
     "range": {"start": {"line": 4, "character": 0}, "end": {"line": 6, "character": 16}},
     "selectionRange": {"start": {"line": 4, "character": 6}, "end": {"line": 4, "character": 10}},
     "children": [{"name": "method", "kind": 6,
                   "range": {"start": {"line": 5, "character": 4}, "end": {"line": 5, "character": 21}},
                   "selectionRange": {"start": {"line": 5, "character": 8}, "end": {"line": 5, "character": 14}}}]},
]

SOURCE = "def alpha():\n    return 1\n\n\nclass Beta:\n    def method(self):\n        return 2\n"


async def test_document_symbols_hierarchical_flatten_to_candidate_evidence(tmp_path):
    (tmp_path / "a.py").write_text(SOURCE)
    s = session(tmp_path)
    page = await query_document_symbols(s, SymbolManager(HIERARCHICAL), "a.py")
    assert [(h.symbol["qualified_name"], h.symbol["kind"]) for h in page.hits] == [
        ("alpha", "function"), ("Beta", "class"), ("Beta.method", "method")]
    assert all(h.precision == "lsp_reported_location" for h in page.hits)
    assert not page.coverage.complete and "lsp_server_coverage_unknown" in page.coverage.reasons
    working = ensure_working_set(s)
    assert any("lsp:document_symbols" in c["evidence_refs"] for c in working.candidates.values())


async def test_document_symbols_flat_symbol_information_and_foreign_filter(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("def gamma():\n    return 3\n")
    flat = [{"name": "gamma", "kind": 12, "containerName": "mod",
             "location": {"uri": (tmp_path / "b.py").as_uri(),
                          "range": {"start": {"line": 0, "character": 0}, "end": {"line": 1, "character": 9}}}},
            {"name": "foreign", "kind": 12,
             "location": {"uri": (tmp_path.parent / "outside.py").as_uri(),
                          "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}}}}]
    page = await query_document_symbols(session(tmp_path), SymbolManager(flat), "a.py")
    assert [(h.path, h.symbol["qualified_name"]) for h in page.hits] == [("b.py", "mod.gamma")]
    assert "unavailable_or_invalid_lsp_symbol" in page.coverage.reasons


async def test_document_symbols_surrogate_boundary_and_unavailable(tmp_path):
    (tmp_path / "a.py").write_text("😀x = 1\n", encoding="utf-8")
    cutting = [{"name": "bad", "kind": 13,
                "range": {"start": {"line": 0, "character": 1}, "end": {"line": 0, "character": 2}}}]
    page = await query_document_symbols(session(tmp_path), SymbolManager(cutting), "a.py")
    assert not page.hits and "unavailable_or_invalid_lsp_symbol" in page.coverage.reasons
    page = await query_document_symbols(session(tmp_path),
        SymbolManager({"unavailable": True, "detail": "server does not declare documentSymbolProvider"}), "a.py")
    assert not page.hits and not page.coverage.complete
    assert any(r.startswith("lsp_unavailable:document_symbols") for r in page.coverage.reasons)


async def test_document_symbols_reject_source_moving_mid_request(tmp_path):
    source = tmp_path / "a.py"
    source.write_text(SOURCE)
    class Moving:
        async def request(self, operation, **kwargs):
            source.write_text("changed = 1\n")
            return HIERARCHICAL
    with pytest.raises(StaleEvidence):
        await query_document_symbols(session(tmp_path), Moving(), "a.py")


FAKE_SERVER = """
import json, sys
CAPS = {"callHierarchyProvider": True, "documentSymbolProvider": True}
if "--no-symbols" in sys.argv:
    CAPS = {"callHierarchyProvider": True}
def send(value):
    body = json.dumps(value, separators=(',', ':')).encode()
    sys.stdout.buffer.write(f'Content-Length: {len(body)}\\r\\n\\r\\n'.encode() + body)
    sys.stdout.buffer.flush()
while True:
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line: raise SystemExit(0)
        if line in (b'\\r\\n', b'\\n'): break
        k, _, v = line.decode().partition(':'); headers[k.lower().strip()] = v.strip()
    msg = json.loads(sys.stdin.buffer.read(int(headers['content-length'])))
    method = msg.get('method')
    if method == 'initialize': send({'jsonrpc': '2.0', 'id': msg['id'], 'result': {'capabilities': CAPS}})
    elif method == 'textDocument/documentSymbol':
        send({'jsonrpc': '2.0', 'id': msg['id'], 'result': []})
    elif method == 'textDocument/prepareCallHierarchy':
        items = [{'name': 'alpha', 'kind': 12, 'uri': msg['params']['textDocument']['uri'],
                  'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 7}},
                  'selectionRange': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 5}}},
                 {'name': 'beta', 'kind': 12, 'uri': msg['params']['textDocument']['uri'],
                  'range': {'start': {'line': 1, 'character': 0}, 'end': {'line': 1, 'character': 6}},
                  'selectionRange': {'start': {'line': 1, 'character': 0}, 'end': {'line': 1, 'character': 4}}}]
        send({'jsonrpc': '2.0', 'id': msg['id'], 'result': items})
    elif method == 'callHierarchy/incomingCalls':
        send({'jsonrpc': '2.0', 'id': msg['id'],
              'result': [{'from': {'name': 'caller', 'kind': 12}, 'fromRanges': []}]})
    elif method == 'shutdown': send({'jsonrpc': '2.0', 'id': msg['id'], 'result': None})
    elif 'id' in msg: send({'jsonrpc': '2.0', 'id': msg['id'], 'result': []})
    if method == 'exit': raise SystemExit(0)
"""


async def make_manager(tmp_path, *extra_args):
    script = tmp_path / "fake_lsp.py"
    script.write_text(FAKE_SERVER.lstrip(), encoding="utf-8")
    (tmp_path / "demo.py").write_text("alpha()\nbeta()\n# comment\n", encoding="utf-8")
    config = LSPToolConfig(servers=[LSPServerConfig("fake", sys.executable, (str(script), *extra_args),
                                                    {".py": "python"}, timeout=5)])
    return LSPManager(config, tmp_path)


async def test_call_hierarchy_disambiguates_by_position_identifier(tmp_path):
    manager = await make_manager(tmp_path)
    try:
        result = await manager.request("incoming_calls", path="demo.py", line=0, character=1)
        assert result == [{"from": {"name": "caller", "kind": 12}, "fromRanges": []}]
        ambiguous = await manager.request("incoming_calls", path="demo.py", line=2, character=2)
        assert ambiguous["ambiguous"] is True
        assert {c["name"] for c in ambiguous["candidates"]} == {"alpha", "beta"}
    finally:
        await manager.close()


async def test_strict_capability_reports_unavailable_instead_of_empty_success(tmp_path):
    manager = await make_manager(tmp_path, "--no-symbols")
    try:
        page = await query_document_symbols(session(tmp_path), manager, "demo.py")
        assert not page.hits and not page.coverage.complete
        assert any(r.startswith("lsp_unavailable:document_symbols") for r in page.coverage.reasons)
        raw = await manager.request("document_symbols", path="demo.py")
        assert raw == []  # legacy raw path stays pass-through
    finally:
        await manager.close()


async def test_open_document_reads_join_the_caller_budget(tmp_path):
    manager = await make_manager(tmp_path)
    try:
        budget = QueryBudget(CodeIntelConfig())
        await manager.request("document_symbols", path="demo.py", budget=budget)
        assert budget.files >= 1  # the document read was charged to the caller's budget
        cancelled = QueryBudget(CodeIntelConfig())
        cancelled.cancelled.set()
        with pytest.raises(BudgetExceeded):
            await manager.request("document_symbols", path="demo.py", budget=cancelled)
    finally:
        await manager.close()
