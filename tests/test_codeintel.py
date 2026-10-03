from __future__ import annotations

from dataclasses import replace
import asyncio
import hashlib
import os
from pathlib import Path
import time

import pytest

from agent_core.codeintel.budget import BudgetExceeded, QueryBudget
from agent_core.codeintel.config import CodeIntelConfig
from agent_core.codeintel.models import ChangeSet, SearchRequest, StaleEvidence
from agent_core.codeintel.service import CodeIntelligenceService
from agent_core.codeintel.snapshots import read_snapshot
from agent_core.execution import ExecutionScope


async def built(tmp_path: Path, **config) -> CodeIntelligenceService:
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    service = CodeIntelligenceService(root, CodeIntelConfig(watch=False, **config), database=tmp_path / "index" / "code.sqlite3")
    for _ in range(100):
        status = await service.ensure_index()
        if status["catalog_complete"] and not status["pending"]:
            return service
    raise AssertionError(status)


async def test_incremental_facts_delete_move_and_revision(tmp_path):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    source = root / "pkg" / "a.py"
    source.write_text("def target():\n    return 1\n\ndef caller():\n    return target()\n", encoding="utf-8")
    service = await built(tmp_path)
    definitions = await service.search(SearchRequest("target", kind="symbol"))
    assert [h.start_line for h in definitions.hits] == [1]
    old_version = definitions.hits[0].version
    calls = await service.search(SearchRequest("target", kind="calls"))
    assert calls.hits[0].start_line == 5
    assert calls.hits[0].precision == "syntactic"
    source.write_text("def newer():\n    return 2\n", encoding="utf-8")
    await service.publish_changes(ChangeSet("edit1", ("pkg/a.py",)))
    assert not (await service.search(SearchRequest("target", kind="symbol"))).hits
    await service.ensure_index()
    newer = await service.search(SearchRequest("newer", kind="symbol"))
    assert newer.hits[0].version.revision > old_version.revision
    source.rename(root / "pkg" / "b.py")
    await service.publish_changes(ChangeSet("move1", ("pkg/a.py", "pkg/b.py"), "watcher"))
    await service.ensure_index()
    assert [h.path for h in (await service.search(SearchRequest("newer", kind="symbol"))).hits] == ["pkg/b.py"]
    (root / "pkg" / "b.py").unlink()
    await service.publish_changes(ChangeSet("delete1", ("pkg/b.py",)))
    await service.ensure_index()
    assert (await service.status())["files"] == 0
    await service.close()


async def test_query_does_not_enumerate_or_reindex_warm_repository(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("unique_identifier = 42\n", encoding="utf-8")
    service = await built(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("query traversed the repository")
    monkeypatch.setattr(os, "scandir", forbidden)
    monkeypatch.setattr(service.catalog, "advance", forbidden)
    page = await service.search(SearchRequest("unique_identifier"))
    assert page.hits and page.hits[0].version.sha256
    assert page.usage["files"] == 1
    await service.close()


async def test_unreported_change_never_returns_stale_code_and_audit_repairs(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    file = root / "a.py"
    file.write_text("old_name = 1\n")
    service = await built(tmp_path)
    stat = file.stat()
    file.write_text("new_name = 1\n")
    os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    page = await service.search(SearchRequest("old_name"))
    assert not page.hits and not page.coverage.complete
    assert any(r.startswith("stale:") for r in page.coverage.reasons)
    await service.ensure_index(audit=True)
    assert (await service.search(SearchRequest("new_name"))).hits
    await service.close()


async def test_index_queue_survives_restart_and_idempotent_event(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.txt").write_text("one")
    service = await built(tmp_path)
    (root / "a.txt").write_text("two")
    event = ChangeSet("durable", ("a.txt",))
    await service.publish_changes(event)
    generation = (await service.status())["generation"]
    await service.publish_changes(event)
    assert (await service.status())["generation"] == generation
    await service.close()
    reopened = await built(tmp_path)
    assert (await reopened.search(SearchRequest("two"))).hits
    await reopened.close()


async def test_partial_coverage_and_cursor_bound_to_generation(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.txt").write_text("needle\nneedle\nneedle\n")
    service = await built(tmp_path, max_results=1)
    page = await service.search(SearchRequest("needle", limit=1))
    assert len(page.hits) == 1 and page.cursor and not page.coverage.complete
    next_page = await service.search(SearchRequest("needle", limit=1, cursor=page.cursor))
    assert next_page.hits[0].start_line == 2
    await service.publish_changes(ChangeSet("new", ("a.txt",)))
    moved = await service.search(SearchRequest("needle", limit=1, cursor=page.cursor))
    assert not moved.hits and not moved.coverage.complete
    assert any("generation_changed" in reason for reason in moved.coverage.reasons)
    await service.close()


async def test_secrets_redirects_and_permission_filters(tmp_path, directory_redirect):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".env").write_text("sensitive_password")
    (root / "visible.py").write_text("public = 1")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("outside_secret")
    directory_redirect(root / "redirect", outside)
    service = await built(tmp_path)
    assert (await service.status())["files"] == 1
    assert not (await service.search(SearchRequest("public"), allowed=lambda p: False)).hits
    with pytest.raises(ValueError):
        await service.read_region("../outside/secret.py")
    with pytest.raises(ValueError):
        await service.read_region("redirect/secret.py")
    await service.close()


async def test_file_versions_and_worktrees_are_isolated(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    file = root / "a.py"
    file.write_text("x = 1\n")
    service = await built(tmp_path)
    hit = await service.read_region("a.py")
    assert hit.version.sha256 == hashlib.sha256(file.read_bytes()).hexdigest()
    file.write_text("x = 2\n")
    with pytest.raises(StaleEvidence):
        await service.read_region("a.py", expected_version=hit.version.to_dict())
    other = CodeIntelligenceService(tmp_path / "other", database=tmp_path / "other-index" / "code.sqlite3")
    other.workspace.mkdir()
    (other.workspace / "a.py").write_text("x = 1\n")
    with pytest.raises(StaleEvidence):
        await other.read_region("a.py", expected_version=hit.version.to_dict())
    await other.close()
    await service.close()


def test_budget_checks_actual_bytes_and_scope_cancellation(tmp_path):
    (tmp_path / "large.py").write_text("x" * 200)
    budget = QueryBudget(replace(CodeIntelConfig(), max_bytes=100))
    with pytest.raises(BudgetExceeded, match="bytes"):
        read_snapshot(tmp_path, "large.py", budget)
    scope = ExecutionScope.for_workspace(tmp_path, deadline=time.monotonic() - 1)
    with pytest.raises(TimeoutError):
        QueryBudget(CodeIntelConfig(), scope).check()


async def test_regular_edits_reject_stale_versions(tmp_path):
    from agent_core.tools.builtin import ReadTextFileTool, EditFileTool
    file = tmp_path / "a.txt"
    file.write_text("old text\n")
    read = await ReadTextFileTool(tmp_path).run({"path": "a.txt"})
    file.write_text("old text\nexternal change\n")
    with pytest.raises(StaleEvidence):
        await EditFileTool(tmp_path).run({"path": "a.txt", "old_string": "old", "new_string": "new",
            "expected_version": read.metadata["file_version"]})
    assert "external change" in file.read_text()


async def test_transaction_query_reads_overlay_without_copying_subtree(tmp_path):
    from agent_core.tools.transaction import JournalStorage, TurnExecutionJournal, WorkspaceTransaction
    from agent_core.tools.base import ResourceLock, ToolExecutionContext
    from agent_core.tools.builtin import SearchTextTool
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.txt").write_text("original")
    (root / "b.txt").write_text("needle")
    storage = JournalStorage.local(tmp_path / "journal", workspace=root)
    journal = TurnExecutionJournal(storage)
    txn = WorkspaceTransaction(root, "query-turn", journal=journal)
    txn.declare_resources((ResourceLock("fs", str(root / "a.txt"), "write"),))
    (txn.overlay / "a.txt").write_text("needle updated")
    txn.declare_resources((ResourceLock("fs", str(root), "read", subtree=True, materialize=False),))
    tool = SearchTextTool(root)
    result = await tool.run_with_context({"pattern": "needle"}, ToolExecutionContext("query-turn", workspace_view=txn))
    assert "a.txt" in result.content and "b.txt" in result.content
    assert not (txn.overlay / "b.txt").exists()
    (root / "b.txt").write_text("external")
    with pytest.raises(RuntimeError, match="query evidence"):
        txn.commit()
    txn.rollback("test")
    journal.close()


async def test_local_scope_expands_under_one_budget_and_paginates(tmp_path):
    root = tmp_path / "repo"
    (root / "local").mkdir(parents=True)
    (root / "other").mkdir()
    (root / "local" / "a.py").write_text("local_value = 1\n")
    (root / "other" / "b.txt").write_text("needle\nneedle\n")
    service = await built(tmp_path)
    request = SearchRequest("needle", modules=("local",), expand_scope=True, limit=1)
    page = await service.search(request)
    assert page.hits[0].path == "other/b.txt"
    assert page.usage["scope_expansions"] == 1
    assert any("module:local" in p for p in page.coverage.checked)
    second = await service.search(replace(request, cursor=page.cursor))
    assert second.hits[0].start_line == 2
    constrained = await service.search(replace(request, expand_scope=False))
    assert not constrained.hits
    await service.close()


async def test_cancelled_inventory_restarts_uncommitted_directory(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    for i in range(25):
        (root / f"a{i}.py").write_text(f"value_{i} = {i}\n")
    service = CodeIntelligenceService(root, CodeIntelConfig(watch=False), database=tmp_path / "index" / "code.sqlite3")
    scope = ExecutionScope.for_workspace(root)
    bump = service.store.bump
    calls = 0
    def cancel_during_scan(db):
        nonlocal calls
        bump(db)
        calls += 1
        if calls == 5:
            scope.cancellation.cancel()
    monkeypatch.setattr(service.store, "bump", cancel_during_scan)
    with pytest.raises(asyncio.CancelledError):
        await service.ensure_index(scope=scope)
    monkeypatch.setattr(service.store, "bump", bump)
    assert service.catalog._iterator is None
    status = await service.ensure_index()
    assert status["files"] == 25 and status["pending"] == 0
    await scope.close()
    await service.close()


async def test_regex_deadline_returns_partial_and_reaps_process(tmp_path, monkeypatch):
    import subprocess
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.txt").write_text("a" * 100 + "!\n")
    service = await built(tmp_path, query_seconds=0.5)
    original = subprocess.Popen
    children = []
    def spawn(*args, **kwargs):
        process = original(*args, **kwargs)
        children.append(process)
        return process
    monkeypatch.setattr(subprocess, "Popen", spawn)
    start = time.monotonic()
    page = await service.search(SearchRequest("(a+)+$", kind="text", regex=True))
    assert time.monotonic() - start < 5
    assert not page.coverage.complete
    assert children and all(child.poll() is not None for child in children)
    await service.close()


async def test_relation_expansion_and_static_build_dependencies(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def target():\n    pass\ndef middle():\n    target()\ndef entry():\n    middle()\n")
    (root / "pyproject.toml").write_text('[build-system]\nrequires = ["setuptools>=70"]\n')
    service = await built(tmp_path)
    page = await service.expand_relations(["target"], relation="calls", depth=2)
    assert {h.source_symbol for h in page.hits} == {"middle", "entry"}
    dependency = await service.search(SearchRequest("setuptools", kind="build_dependency"))
    assert dependency.hits[0].precision == "manifest"
    await service.close()


async def test_committed_outbox_replays_to_separate_policy_indexes(tmp_path, monkeypatch):
    from agent_core.codeintel.changes import record_commit
    from agent_core.codeintel.store import default_store
    monkeypatch.setenv("POLARIS_HOME", str(tmp_path / "home"))
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "a.py"
    source.write_text("before = 1\n")
    service = CodeIntelligenceService(root, CodeIntelConfig(watch=False), database=default_store(root).with_name("policy.sqlite3"))
    await service.ensure_index()
    source.write_text("after = 1\n")
    record_commit(root, "commit-one", ["a.py"])
    page = await service.search(SearchRequest("before"))
    assert not page.hits and not page.coverage.complete
    await service.ensure_index()
    assert (await service.search(SearchRequest("after"))).hits
    generation = (await service.status())["generation"]
    record_commit(root, "commit-one", ["a.py"])
    await service.search(SearchRequest("after"))
    assert (await service.status())["generation"] == generation
    await service.close()


async def test_context_budget_includes_hit_versions_and_paths(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.txt").write_text("x\n" * 100)
    service = await built(tmp_path, max_context_tokens=1200)
    page = await service.search(SearchRequest("x", kind="text"))
    assert 0 < len(page.hits) < 10
    assert page.usage["output_bytes"] <= 1200
    assert "context_tokens" in page.coverage.reasons
    await service.close()


@pytest.mark.parametrize("path,pattern,expected", [
    ("src/a.py", "*.py", False), ("src/a.py", "src/*.py", True),
    ("src/nested/a.py", "src/*.py", False), ("src/nested/a.py", "**/*.py", True),
    ("a.py", "**/*.py", True), ("other/src/a.py", "src/*.py", False),
])
def test_glob_preserves_directory_component_semantics(path, pattern, expected):
    from agent_core.codeintel.paths import glob_match
    assert glob_match(path, pattern) is expected


def test_code_config_and_cli_build(tmp_path, monkeypatch, capsys):
    from agent_core.cli import main
    from agent_core.config import resolve_codeintel_config
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("cli_value = 1\n")
    config = tmp_path / "settings.toml"
    config.write_text("[codeintel]\nwatch = false\nmax_files = 7\n")
    monkeypatch.setenv("POLARIS_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("AGENT_CODEINTEL", raising=False)
    assert resolve_codeintel_config(config).max_files == 7
    assert main(["code", "build", "--workspace", str(root), "--config", str(config)]) == 0
    assert '"pending": 0' in capsys.readouterr().out
    assert main(["code", "search", "cli_value", "--kind", "text", "--workspace", str(root), "--config", str(config)]) == 0
    assert '"path": "a.py"' in capsys.readouterr().out
    # Symbol queries on a complete catalog also report complete coverage (exit 0);
    # the syntactic-precision caveat is an advisory, not a coverage gap.
    assert main(["code", "search", "cli_value", "--kind", "symbol", "--workspace", str(root), "--config", str(config)]) == 0
    assert "python_syntax_only" in capsys.readouterr().out
    with pytest.raises(ValueError):
        CodeIntelConfig.from_dict({"query_seconds": float("inf")})


async def test_reopened_query_does_not_invalidate_warm_catalog(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("warm = 1\n")
    service = await built(tmp_path)
    before = await service.status()
    database = service.store.path
    await service.close()
    service = CodeIntelligenceService(root, CodeIntelConfig(watch=False), database=database)
    def forbidden(*args, **kwargs):
        raise AssertionError("reopened query started a directory scan")
    monkeypatch.setattr(os, "scandir", forbidden)
    page = await service.search(SearchRequest("warm", kind="text"))
    assert page.hits and page.coverage.complete
    assert page.generation == before["generation"]
    await service.close()


async def test_new_event_during_parse_is_not_overwritten(tmp_path, monkeypatch):
    import agent_core.codeintel.indexer as indexer
    root = tmp_path / "repo"
    root.mkdir()
    file = root / "a.py"
    file.write_text("first = 1\n")
    service = await built(tmp_path)
    file.write_text("second = 2\n")
    await service.publish_changes(ChangeSet("second", ("a.py",)))
    parse = indexer.python_structure
    def concurrent_update(text):
        facts = parse(text)
        file.write_text("third = 3\n")
        service.store.enqueue("third", ("a.py",), "external")
        return facts
    monkeypatch.setattr(indexer, "python_structure", concurrent_update)
    await service.ensure_index()
    assert (await service.status())["pending"] == 1
    monkeypatch.setattr(indexer, "python_structure", parse)
    await service.ensure_index()
    assert (await service.search(SearchRequest("third", kind="symbol"))).hits
    assert not (await service.search(SearchRequest("second", kind="symbol"))).hits
    await service.close()


async def test_malformed_manifest_remains_text_searchable(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "package.json").write_text('{"needle": unfinished')
    service = await built(tmp_path)
    assert (await service.status())["states"] == {"text_only": 1}
    assert (await service.search(SearchRequest("needle", kind="text"))).hits
    await service.close()


async def test_missing_watcher_reconciles_external_edits(tmp_path, monkeypatch):
    import sys
    from agent_core.codeintel.watcher import maintain
    monkeypatch.setitem(sys.modules, "watchdog.events", None)
    root = tmp_path / "repo"
    root.mkdir()
    file = root / "a.py"
    file.write_text("original = 1\n")
    service = await built(tmp_path, reconcile_seconds=0.1)
    service._background = asyncio.create_task(maintain(service))
    try:
        file.write_text("external = 1\n")
        for _ in range(50):
            if (await service.search(SearchRequest("external", kind="symbol"))).hits:
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError("periodic reconciliation did not observe external edit")
        assert service.watch_state == "periodic_reconciliation_only"
    finally:
        await service.close()


async def test_search_during_build_never_fails_with_locked_database(tmp_path, monkeypatch):
    """Regression: outbox replay during a chunked build must not raise database-is-locked."""
    from agent_core.codeintel.changes import record_commit
    from agent_core.codeintel.store import default_store
    monkeypatch.setenv("POLARIS_HOME", str(tmp_path / "home"))
    root = tmp_path / "repo"
    root.mkdir()
    for i in range(1200):
        (root / f"f{i:05d}.py").write_text(f"def fn_{i}():\n    return {i}\n", encoding="utf-8")
    service = CodeIntelligenceService(root, CodeIntelConfig(watch=False, query_seconds=60, max_files=10000),
                                      database=default_store(root))
    errors: list[str] = []
    state = {"building": True}

    async def builder() -> None:
        while state["building"]:
            status = await service.ensure_index()
            if status["catalog_complete"] and status["pending"] == 0:
                return

    async def searcher() -> None:
        n = 0
        while state["building"] and n < 150:
            record_commit(root, f"ev-{n}", ["f00001.py"])
            try:
                await service.search(SearchRequest("fn_1", kind="text"))
            except Exception as exc:  # noqa: BLE001 - any escape here is the regression
                errors.append(f"{type(exc).__name__}: {exc}")
            n += 1
            await asyncio.sleep(0.01)

    build = asyncio.create_task(builder())
    await asyncio.sleep(0.2)
    probe = asyncio.create_task(searcher())
    await probe
    state["building"] = False
    await build
    assert not errors
    assert (await service.search(SearchRequest("fn_1", kind="text"))).hits
    await service.close()


async def test_disabled_layer_issues_no_versions(tmp_path):
    from agent_core.tools.builtin import ReadTextFileTool
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    tool = ReadTextFileTool(tmp_path)

    class Stub:
        codeintel_config = CodeIntelConfig(enabled=False)
    tool._code_session = Stub()
    result = await tool.run({"path": "a.txt"})
    assert result.ok and "file_version" not in result.metadata
    tool._code_session = None  # absent session config means defaults (enabled)
    result = await tool.run({"path": "a.txt"})
    assert result.metadata["file_version"]["sha256"]


def _agent(tmp_path, **config_overrides):
    from agent_core.providers.fake import FakeProvider
    from agent_core.react import ReActAgent, ReActConfig
    from agent_core.storage import JSONLRunLogger
    (tmp_path / "runs").mkdir(parents=True, exist_ok=True)
    config = ReActConfig(run_dir=str(tmp_path / "runs"), **config_overrides)
    return ReActAgent(FakeProvider(), config, logger=JSONLRunLogger(str(tmp_path / "runs")), workspace=tmp_path)


def test_disabled_layer_unregisters_code_tools(tmp_path):
    from agent_core.codeintel.config import CodeIntelConfig as _Cfg
    agent = _agent(tmp_path, codeintel=_Cfg(enabled=False))
    names = {tool.name for tool in agent.registry.list()} | {item.name for item in agent.registry.deferred()}
    assert not {"code_search", "code_relations", "code_context"} & names
    agent_enabled = _agent(tmp_path / "second")
    enabled_names = {item.name for item in agent_enabled.registry.deferred()}
    assert {"code_search", "code_relations", "code_context"} <= enabled_names


def test_post_compact_attachments_merge_references_and_content(tmp_path):
    from agent_core.codeintel.config import CodeIntelConfig as _Cfg
    from agent_core.codeintel.models import CodeHit, FileVersion
    from agent_core.codeintel.working_set import TaskWorkingSet
    (tmp_path / "a.py").write_text("value = 1\n", encoding="utf-8")
    agent = _agent(tmp_path)
    key = str((tmp_path / "a.py").resolve())
    agent.session.record_read(key, "value = 1\n")
    working = TaskWorkingSet()
    working.add(CodeHit("a.py", 1, 1, "", FileVersion("wt", "a.py", "0" * 64), module="."), reason="read_text_file")
    agent.session.code_working_set = working
    attachments = agent._build_read_attachments()
    assert len(attachments) == 2
    assert "a.py:1-1" in attachments[0].content
    assert "value = 1" in attachments[1].content

    disabled = _agent(tmp_path / "second", codeintel=_Cfg(enabled=False))
    disabled.session.record_read(str((tmp_path / "a.py").resolve()), "value = 1\n")
    disabled.session.code_working_set = working
    attachments = disabled._build_read_attachments()
    assert len(attachments) == 1 and "value = 1" in attachments[0].content


async def test_disabled_layer_skips_version_recording(tmp_path):
    from agent_core.models import ToolCall, ToolResult
    from agent_core.codeintel.config import CodeIntelConfig as _Cfg
    (tmp_path / "a.py").write_text("value = 1\n", encoding="utf-8")
    agent = _agent(tmp_path, codeintel=_Cfg(enabled=False))
    agent._record_read_result(
        ToolCall("read_text_file", {"path": "a.py"}),
        ToolResult("read_text_file", "value = 1\n", metadata={
            "file_version": {"worktree_id": "wt", "path": "a.py", "sha256": "0" * 64, "revision": 0},
            "start_line": 1, "end_line": 1}))
    assert agent.session.code_read_versions == {}
    assert agent.session.code_working_set is None
    # Explicit caller-supplied preconditions remain enforceable even when disabled.
    from agent_core.tools.builtin import EditFileTool
    with pytest.raises(StaleEvidence):
        await EditFileTool(tmp_path).run({"path": "a.py", "old_string": "value", "new_string": "x",
            "expected_version": {"path": "a.py", "sha256": "1" * 64, "worktree_id": "wt"}})


async def test_edit_precondition_uses_session_bound_versions(tmp_path):
    """The executor-bound read_versions path (no explicit expected_version) rejects stale edits."""
    from agent_core.tools.base import ToolExecutionContext
    from agent_core.tools.builtin import EditFileTool, ReadTextFileTool
    file = tmp_path / "a.txt"
    file.write_text("old text\n", encoding="utf-8")
    read = await ReadTextFileTool(tmp_path).run({"path": "a.txt"})
    versions = {str((tmp_path / "a.txt").resolve()): read.metadata["file_version"]}
    file.write_text("old text\nexternal change\n", encoding="utf-8")
    context = ToolExecutionContext("turn", logical_workspace=tmp_path, read_versions=versions)
    with pytest.raises(StaleEvidence):
        await EditFileTool(tmp_path).run_with_context(
            {"path": "a.txt", "old_string": "old", "new_string": "new"}, context)


async def test_cold_index_search_hints_at_build(tmp_path):
    from agent_core.codeintel.runtime import release_service
    from agent_core.session import SessionContext
    from agent_core.tools.builtin import SearchTextTool
    (tmp_path / "a.txt").write_text("needle here\n", encoding="utf-8")
    session = SessionContext(workspace=tmp_path, codeintel_config=CodeIntelConfig(maintain=False, watch=False))
    tool = SearchTextTool(tmp_path)
    tool._code_session = session
    try:
        result = await tool.run({"pattern": "needle"})
        assert "needle here" in result.content
        assert "code index not built" in result.content
    finally:
        await release_service(session)


async def test_symbol_search_reports_complete_with_precision_advisory(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def target():\n    return 1\n", encoding="utf-8")
    service = await built(tmp_path)
    page = await service.search(SearchRequest("target", kind="symbol"))
    assert page.coverage.complete
    assert any("python_syntax_only" in a for a in page.coverage.advisories)
    assert not page.coverage.reasons
    await service.close()


async def test_path_search_uses_component_glob_semantics(tmp_path):
    root = tmp_path / "repo"
    (root / "src" / "nested").mkdir(parents=True)
    (root / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (root / "src" / "nested" / "b.py").write_text("y = 2\n", encoding="utf-8")
    service = await built(tmp_path)
    page = await service.search(SearchRequest("src/*.py", kind="path"))
    assert [h.path for h in page.hits] == ["src/a.py"]
    await service.close()


async def test_read_region_truncates_at_output_budget(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "big.txt").write_text("x" * 100 + "\n" + ("y" * 100 + "\n") * 50, encoding="utf-8")
    service = await built(tmp_path, max_context_tokens=1000)
    hit = await service.read_region("big.txt", 1, 200)
    assert "truncated: context output budget" in hit.text
    assert (hit.end_line or 0) < 51
    await service.close()


async def test_batched_regex_matches_across_files_with_one_child(tmp_path, monkeypatch):
    import subprocess
    root = tmp_path / "repo"
    root.mkdir()
    for i in range(30):
        (root / f"f{i:02d}.py").write_text(f"def fn_{i}():\n    return {i}\n", encoding="utf-8")
    service = await built(tmp_path)
    original = subprocess.Popen
    children = []
    def spawn(*args, **kwargs):
        process = original(*args, **kwargs)
        children.append(process)
        return process
    monkeypatch.setattr(subprocess, "Popen", spawn)
    page = await service.search(SearchRequest(r"fn_1\d", kind="text", regex=True))
    paths = {h.path for h in page.hits}
    assert "f10.py" in paths and "f19.py" in paths
    assert len(children) == 1
    await service.close()


async def test_index_works_when_polaris_home_is_redirected(tmp_path, monkeypatch, directory_redirect):
    """Regression: a symlinked/junctioned POLARIS_HOME must not trip the redirect guard."""
    from agent_core.codeintel.store import default_store
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    directory_redirect(tmp_path / "link-home", real_home)
    monkeypatch.setenv("POLARIS_HOME", str(tmp_path / "link-home"))
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def target():\n    return 1\n", encoding="utf-8")
    service = CodeIntelligenceService(root, CodeIntelConfig(watch=False), database=default_store(root))
    for _ in range(50):
        status = await service.ensure_index()
        if status["catalog_complete"] and status["pending"] == 0:
            break
    assert status["catalog_complete"]
    assert (await service.search(SearchRequest("target", kind="symbol"))).hits
    await service.close()


async def test_catalog_scan_survives_directory_deleted_mid_iteration(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def kept():\n    return 1\n", encoding="utf-8")
    service = await built(tmp_path)

    class Vanished:
        def __next__(self):
            raise FileNotFoundError("directory vanished")
        def close(self):
            pass

    with service.store.connect() as db:
        db.execute("INSERT INTO scan_dirs VALUES('gone')")
        db.execute("UPDATE meta SET value='0' WHERE key='catalog_complete'")
    service.catalog._iterator = Vanished()
    service.catalog._directory = "gone"
    status = await service.ensure_index()
    assert status["catalog_complete"]
    assert (await service.search(SearchRequest("kept", kind="symbol"))).hits
    await service.close()


async def test_walk_tolerates_directory_deleted_mid_scan(tmp_path, monkeypatch):
    from agent_core.tools.builtin import SearchTextTool
    root = tmp_path / "repo"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "a.py").write_text("needle = 1\n", encoding="utf-8")
    (root / "gone").mkdir()
    (root / "gone" / "b.py").write_text("needle = 2\n", encoding="utf-8")
    real_scandir = os.scandir

    def fake_scandir(path):
        if str(path).endswith("gone"):
            raise FileNotFoundError("vanished")
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", fake_scandir)
    result = await SearchTextTool(root).run({"pattern": "needle", "path": "."})
    assert result.ok
    assert "sub/a.py" in result.content


async def test_transaction_query_dir_membership_guard(tmp_path):
    """A membership change in a scanned directory invalidates listing evidence at commit."""
    from agent_core.tools.transaction import JournalStorage, TurnExecutionJournal, WorkspaceTransaction
    from agent_core.tools.base import ResourceLock, ToolExecutionContext
    from agent_core.tools.builtin import SearchTextTool
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.txt").write_text("needle", encoding="utf-8")
    storage = JournalStorage.local(tmp_path / "journal", workspace=root)
    journal = TurnExecutionJournal(storage)
    txn = WorkspaceTransaction(root, "query-turn-membership", journal=journal)
    txn.declare_resources((ResourceLock("fs", str(root), "read", subtree=True, materialize=False),))
    tool = SearchTextTool(root)
    result = await tool.run_with_context({"pattern": "needle"}, ToolExecutionContext("query-turn-membership", workspace_view=txn))
    assert "a.txt" in result.content
    (root / "new.txt").write_text("appeared mid-turn", encoding="utf-8")
    with pytest.raises(RuntimeError, match="membership"):
        txn.commit()
    txn.rollback("test")
    journal.close()


async def test_index_disk_quota_stops_slice_cleanly(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n", encoding="utf-8")
    service = CodeIntelligenceService(root, CodeIntelConfig(watch=False), database=tmp_path / "index" / "code.sqlite3")
    status = await service.ensure_index()
    assert status["stop_reason"] is None
    service.config = replace(service.config, max_index_bytes=1)
    status = await service.ensure_index()
    assert status["stop_reason"] == "index_disk_quota"
    await service.close()
