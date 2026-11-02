"""Correctness gates for bounded retrieval before large-repository optimization."""
from __future__ import annotations

from dataclasses import replace
import json

import pytest

from agent_core.codeintel.budget import QueryBudget
from agent_core.codeintel.config import CodeIntelConfig
from agent_core.codeintel.models import ChangeSet, SearchRequest
from agent_core.codeintel.runtime import get_service, release_service
from agent_core.codeintel.scanning import walk_files
from agent_core.codeintel.service import CodeIntelligenceService
from agent_core.session import SessionContext
from agent_core.tools.codeintel import CodeSearchTool


async def build(service):
    for _ in range(100):
        status = await service.ensure_index()
        if status["catalog_complete"] and not status["pending"]:
            return
    raise AssertionError(status)


@pytest.fixture
async def service_factory(tmp_path):
    services = []

    async def create(**config):
        service = CodeIntelligenceService(tmp_path / "repo", CodeIntelConfig(watch=False, **config),
                                          database=tmp_path / "index" / "code.sqlite3")
        services.append(service)
        await build(service)
        return service

    yield create
    for service in services:
        await service.close()


async def test_source_names_survive_both_cold_and_indexed_inventory(tmp_path, service_factory):
    root = tmp_path / "repo"
    sources = {"pkg/memory/a.py", "src/build/b.py", "src/dist/c.py", "pkg/env/d.py", "pkg/runs/e.py"}
    excluded = {"memory/runtime.py", "runs/log.py", "build/output.py", "node_modules/x.py", ".git/state.py"}
    for path in sources | excluded:
        file = root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("def needle(): pass\n", encoding="utf-8")
    service = await service_factory()
    cold = {p.relative_to(root).as_posix() for p in walk_files(root, root, QueryBudget(service.config))}
    indexed = {h.path for h in (await service.search(SearchRequest("needle", kind="symbol"))).hits}
    assert cold == indexed == sources
    await service.publish_changes(ChangeSet("excluded-event", tuple(excluded)))
    await service.ensure_index()
    assert (await service.status())["files"] == len(sources)


async def test_configurable_exclusions_are_root_relative_and_isolate_indexes(tmp_path, service_factory):
    root = tmp_path / "repo"
    for path in ("memory/source.py", "src/cache/ignored.py", "pkg/src/cache/keep.py"):
        file = root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("needle = 1\n", encoding="utf-8")
    default = await service_factory()
    custom = await service_factory(exclude_paths=("src/cache",))
    assert custom.store.path != default.store.path
    page = await custom.search(SearchRequest("needle", kind="symbol"))
    assert {h.path for h in page.hits} == {"memory/source.py", "pkg/src/cache/keep.py"}
    cold = {p.relative_to(root).as_posix() for p in walk_files(root, root, QueryBudget(custom.config))}
    assert cold == {h.path for h in page.hits}
    reopened = CodeIntelligenceService(root, custom.config, database=custom.store.path)
    try:
        assert reopened.store.path == custom.store.path
        assert await reopened.catalog_ready()
    finally:
        await reopened.close()


@pytest.mark.parametrize("paths", ["build", ["../escape"], ["/absolute"], ["C:\\absolute"], ["."], [1], [""]])
def test_exclusion_config_rejects_ambiguous_or_escaping_paths(paths):
    with pytest.raises(ValueError, match="exclude_paths"):
        CodeIntelConfig.from_dict({"exclude_paths": paths})


def test_exclusion_config_normalizes_and_can_include_root_source():
    assert CodeIntelConfig.from_dict({"exclude_paths": ["pkg\\generated/", "./pkg/generated"]}).exclude_paths == ("pkg/generated",)
    assert CodeIntelConfig.from_dict({"exclude_paths": []}).exclude_paths == ()


async def test_legacy_inventory_requires_reconciliation_before_claiming_complete(tmp_path, service_factory):
    root = tmp_path / "repo"
    (root / "pkg" / "memory").mkdir(parents=True)
    (root / "pkg" / "memory" / "a.py").write_text("def recovered(): pass\n", encoding="utf-8")
    service = await service_factory()
    with service.store.connect() as db:
        # Simulate an older inventory which never visited nested memory sources.
        db.execute("DELETE FROM text_fts")
        db.execute("DELETE FROM files")
        db.execute("DELETE FROM meta WHERE key='inventory_policy'")
    await service.close()
    reopened = CodeIntelligenceService(root, service.config, database=service.store.path)
    try:
        assert not await reopened.catalog_ready()
        await build(reopened)
        assert (await reopened.search(SearchRequest("recovered", kind="symbol"))).hits
    finally:
        await reopened.close()


@pytest.mark.parametrize("regex", [False, True])
@pytest.mark.parametrize("budget", [{"max_files": 1}, {"max_candidates": 1}])
async def test_empty_partial_pages_advance_to_later_files(tmp_path, service_factory, regex, budget):
    root = tmp_path / "repo"
    root.mkdir()
    for path, text in (("a.txt", "no match\n"), ("b.txt", "still absent\n"), ("c.txt", "Needle\n")):
        (root / path).write_text(text, encoding="utf-8")
    service = await service_factory(**budget)
    request = SearchRequest("needle", kind="text", regex=regex, ignore_case=True)
    cursors = set()
    pages = []
    for _ in range(5):
        page = await service.search(request)
        pages.append(page)
        if page.cursor is None:
            break
        assert page.cursor not in cursors
        cursors.add(page.cursor)
        request = replace(request, cursor=page.cursor)
    assert not pages[0].hits and pages[0].cursor and not pages[0].coverage.complete
    assert [hit.path for page in pages for hit in page.hits] == ["c.txt"]
    assert pages[-1].coverage.complete


async def test_same_line_relations_are_not_lost_at_page_boundaries(tmp_path, service_factory):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def caller(): return left() + right()\n", encoding="utf-8")
    service = await service_factory(max_candidates=1)
    request = SearchRequest("caller", kind="calls", direction="outgoing", limit=1)
    first = await service.search(request)
    second = await service.search(replace(request, cursor=first.cursor))
    assert first.cursor and first.hits[0].start_line == second.hits[0].start_line == 1
    assert [first.hits[0].target_symbol, second.hits[0].target_symbol] == ["left", "right"]
    assert second.coverage.complete


async def test_glob_filtered_candidate_is_a_resumable_empty_page(tmp_path, service_factory):
    root = tmp_path / "repo"
    (root / "src" / "nested").mkdir(parents=True)
    (root / "src" / "nested" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (root / "src" / "z.py").write_text("x = 2\n", encoding="utf-8")
    service = await service_factory(max_candidates=1)
    request = SearchRequest("src/*.py", kind="path")
    first = await service.search(request)
    assert not first.hits and first.cursor
    second = await service.search(replace(request, cursor=first.cursor))
    assert [h.path for h in second.hits] == ["src/z.py"]


async def test_tool_cursor_survives_implicit_working_set_changes(tmp_path):
    for module in ("a", "b"):
        (tmp_path / module).mkdir()
        (tmp_path / module / "a.py").write_text("needle = 1\n", encoding="utf-8")
    session = SessionContext(workspace=tmp_path, codeintel_config=CodeIntelConfig(maintain=False))
    try:
        service = await get_service(session)
        await build(service)
        tool = CodeSearchTool(session)
        first = json.loads((await tool.run({"query": "needle", "kind": "text", "limit": 1})).content)
        assert session.code_working_set.modules() == ("a",)
        second = json.loads((await tool.run({"query": "needle", "kind": "text", "limit": 1, "cursor": first["cursor"]})).content)
        assert second["hits"][0]["path"] == "b/a.py"
        session.code_working_set = None  # Next ReAct task keeps its service alive.
        assert await get_service(session) is service
        assert session.code_working_set is not None
        assert (await tool.run({"query": "needle"})).ok
        session.codeintel_config = replace(session.codeintel_config, max_results=2)
        assert await get_service(session) is not service
        assert not session.code_read_versions
    finally:
        await release_service(session)


async def test_partial_symbol_does_not_fall_back_or_expand_scope(tmp_path, service_factory):
    root = tmp_path / "repo"
    (root / "local").mkdir(parents=True)
    (root / "remote").mkdir()
    (root / "local" / "a.py").write_text("def needle(): pass\n", encoding="utf-8")
    (root / "remote" / "b.py").write_text("needle = 1\n", encoding="utf-8")
    service = await service_factory(max_context_tokens=10)
    page = await service.search(SearchRequest("needle", modules=("local",), expand_scope=True))
    assert not page.hits and not page.cursor
    assert page.usage["files"] == 1
    assert page.coverage.requested == ["module:local under ."]
    assert any("no_scan_progress" in reason for reason in page.coverage.reasons)


async def test_unindexable_files_do_not_disable_auto_text_fallback(tmp_path, service_factory):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.txt").write_text("needle appears in prose\n", encoding="utf-8")
    (root / "binary.dat").write_bytes(b"\x00\x00\x00")
    service = await service_factory()
    page = await service.search(SearchRequest("needle"))
    assert [h.path for h in page.hits] == ["a.txt"]
    assert page.scan_complete and not page.coverage.complete
    assert "excluded_unindexable_files" in page.coverage.reasons


async def test_symbol_extent_and_delivered_excerpt_are_distinct(tmp_path, service_factory):
    root = tmp_path / "repo"
    root.mkdir()
    lines = ["def needle():", *[f"    value_{i} = {i}" for i in range(30)]]
    (root / "a.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    service = await service_factory()
    hit = (await service.search(SearchRequest("needle", kind="symbol"))).hits[0]
    assert (hit.start_line, hit.end_line) == (1, 31)
    assert (hit.excerpt_start_line, hit.excerpt_end_line) == (1, 11)
    assert hit.truncated and hit.next_start_line == 12
    tail = await service.read_region(hit.path, hit.next_start_line, expected_version=hit.version.to_dict())
    assert tail.text == "\n".join(lines[11:])
    assert tail.excerpt_end_line == 31 and not tail.truncated and tail.next_start_line is None


async def test_region_pagination_has_no_fabricated_marker_lines(tmp_path, service_factory):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    service = await service_factory()
    first = await service.read_region("a.txt", 1, 1)
    assert first.text == "one" and first.end_line == 1 and first.next_start_line == 2
    assert first.truncated
    empty = await service.read_region("a.txt", 9)
    assert empty.text == "" and empty.start_line is None and empty.end_line is None
    assert not empty.truncated and empty.next_start_line is None


async def test_regex_spawn_failure_is_partial_and_never_matches_in_process(tmp_path, service_factory, monkeypatch):
    import subprocess
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.txt").write_text("a" * 1000 + "!\n", encoding="utf-8")
    service = await service_factory()

    def denied(*args, **kwargs):
        raise OSError("child creation unavailable")

    monkeypatch.setattr(subprocess, "Popen", denied)
    page = await service.search(SearchRequest("(a+)+$", kind="text", regex=True))
    assert not page.hits and not page.coverage.complete and page.cursor is None
    assert any("regex_engine_failed" in reason for reason in page.coverage.reasons)


async def test_permission_filtered_empty_page_does_not_hide_later_allowed_hits(tmp_path, service_factory):
    root = tmp_path / "repo"
    root.mkdir()
    for name in ("a.py", "b.py"):
        (root / name).write_text("needle = 1\n", encoding="utf-8")
    service = await service_factory(max_candidates=1)
    request = SearchRequest("needle", kind="symbol")
    def allowed(path):
        return path != "a.py"
    first = await service.search(request, allowed=allowed)
    assert not first.hits and first.cursor and "permission_filtered" in first.coverage.reasons
    second = await service.search(replace(request, cursor=first.cursor), allowed=allowed)
    assert [h.path for h in second.hits] == ["b.py"]


async def test_shared_service_release_keeps_other_sessions_working(tmp_path):
    (tmp_path / "a.py").write_text("needle = 1\n", encoding="utf-8")
    config = CodeIntelConfig(maintain=False)
    first = SessionContext(workspace=tmp_path, codeintel_config=config)
    second = SessionContext(workspace=tmp_path, codeintel_config=config)
    try:
        service = await get_service(first)
        await build(service)
        assert await get_service(second) is service
        assert first.code_working_set is not second.code_working_set
        await release_service(first)
        assert not service._closed
        second.code_working_set = None
        assert (await CodeSearchTool(second).run({"query": "needle"})).ok
    finally:
        await release_service(first)
        await release_service(second)


async def test_transaction_projection_uses_the_same_relative_exclusions(tmp_path):
    from agent_core.tools.base import ResourceLock, ToolExecutionContext
    from agent_core.tools.builtin import SearchTextTool
    from agent_core.tools.transaction import JournalStorage, TurnExecutionJournal, WorkspaceTransaction
    root = tmp_path / "repo"
    for relative in ("pkg/memory/a.py", "memory/state.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("needle = 1\n", encoding="utf-8")
    journal = TurnExecutionJournal(JournalStorage.local(tmp_path / "journal", workspace=root))
    txn = WorkspaceTransaction(root, "inventory-policy", journal=journal)
    try:
        txn.declare_resources((ResourceLock("fs", str(root), "read", subtree=True, materialize=False),))
        result = await SearchTextTool(root).run_with_context(
            {"pattern": "needle"}, ToolExecutionContext("inventory-policy", workspace_view=txn))
        assert "pkg/memory/a.py" in result.content
        assert "memory/state.py" not in result.content
        assert not (txn.overlay / "pkg" / "memory" / "a.py").exists()
    finally:
        txn.rollback("test cleanup")
        journal.close()
