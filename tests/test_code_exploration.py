from __future__ import annotations

from dataclasses import replace
import json

import pytest

from agent_core.codeintel.config import CodeIntelConfig
from agent_core.codeintel.models import CodeHit, FileVersion, SearchCoverage, SearchPage, SearchRequest
from agent_core.codeintel.rendering import compact_code_result
from agent_core.codeintel.runtime import ensure_working_set, get_service, notify_committed, release_service
from agent_core.codeintel.snapshots import worktree_id
from agent_core.codeintel.strategy import ExplorationState
from agent_core.codeintel.working_set import TaskWorkingSet
from agent_core.session import SessionContext
from agent_core.task_runtime import TaskContract, TaskRun, TaskStore, capture_revision
from agent_core.tools.codeintel import CodeSearchTool
from agent_core.tools.task_control import TaskStateTool
from agent_core.tools.transaction import JournalStorage


def test_repeated_query_advice_never_turns_partial_coverage_into_absence():
    state = ExplorationState()
    request = SearchRequest("needle", path="src")
    partial = SearchPage(generation=1, coverage=SearchCoverage(reasons=["pending_index"], complete=False))
    first = state.record(request, partial)
    second = state.record(request, partial)
    assert first["no_gain_repeats"] == 0 and second["no_gain_repeats"] == 1
    assert second["next_action"] == "refresh_or_narrow_scope" and not second["coverage_complete"]
    assert "needle" not in json.dumps(list(state.queries.values()))
    complete = SearchPage(generation=1, coverage=SearchCoverage(complete=True))
    assert state.record(request, complete)["next_action"] == "change_query_or_expand_scope"
    assert state.record(request, complete)["no_gain_repeats"] == 1
    complete.generation = 2
    assert state.record(request, complete)["no_gain_repeats"] == 0
    state.invalidate()
    assert state.record(request, complete)["no_gain_repeats"] == 0


def test_cursor_progress_is_distinct_and_history_is_bounded():
    state = ExplorationState(max_entries=4)
    page = SearchPage(cursor="next", generation=3)
    request = SearchRequest("needle")
    first = state.record(request, page)
    continuation = state.record(replace(request, cursor="page-one"), page)
    assert first["query_id"] != continuation["query_id"] and continuation["no_gain_repeats"] == 0
    assert continuation["next_action"] == "continue_cursor"
    for index in range(12):
        state.record(SearchRequest(str(index)), page)
    assert len(state.queries) == 4


def test_restored_evidence_remains_historical_and_permission_filtered(tmp_path):
    working = TaskWorkingSet()
    identity = worktree_id(tmp_path)
    working.binding = (identity, "policy")
    for path in ("a.py", "denied.py"):
        working.add(CodeHit(path, 1, 1, "SOURCE_BODY_MUST_NOT_PERSIST", FileVersion(identity, path, "a" * 64)))
    working.exploration.record(SearchRequest("needle"), SearchPage())
    saved = working.to_state()
    assert "SOURCE_BODY" not in json.dumps(saved)
    restored = TaskWorkingSet()
    restored.binding = working.binding
    restored.restore(saved, allowed=lambda path: path != "denied.py")
    assert [entry["path"] for entry in restored.entries.values()] == ["a.py"]
    assert next(iter(restored.entries.values()))["freshness"] == "needs_revalidation"
    assert next(iter(restored.exploration.queries.values()))["freshness"] == "needs_revalidation"
    restored.invalidate(("a.py",))
    assert next(iter(restored.entries.values()))["freshness"] == "stale"
    assert len(restored.render_references(700).encode()) <= 700
    assert "next_action" in restored.render_references(1500)
    for binding in (("different-workspace", "policy"), (identity, "different-policy")):
        other = TaskWorkingSet()
        other.binding = binding
        other.restore(saved, allowed=lambda _: True)
        assert not other.entries and not other.exploration.queries


@pytest.mark.parametrize("snapshot", [False, True])
async def test_search_history_survives_task_save_without_restoring_edit_versions(tmp_path, snapshot):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def needle(): return 1\n", encoding="utf-8")
    storage = JournalStorage.local(tmp_path / "state", workspace=root)
    task_store = TaskStore(storage)
    from agent_core.revisions import RevisionTracker
    baseline = await RevisionTracker(root, snapshots=task_store.snapshots).capture() if snapshot else capture_revision(root)
    session = SessionContext(workspace=root, task_run=TaskRun(TaskContract("find needle"), baseline),
                             task_store=task_store, codeintel_config=CodeIntelConfig(maintain=False, watch=False))
    service = await get_service(session)
    try:
        for _ in range(20):
            status = await service.ensure_index()
            if status["catalog_complete"] and not status["pending"]:
                break
        tool = CodeSearchTool(session)
        args = {"query": "needle", "kind": "symbol", "path": "."}
        first = json.loads((await tool.run(args)).content)
        second = json.loads((await tool.run(args)).content)
        assert first["exploration"]["new_evidence"] == 1
        assert second["exploration"]["no_gain_repeats"] == 1
        assert second["hits"]  # Repeated requests still execute and return real evidence.
        await session.persist_task_async()
        assert json.loads(task_store.path.read_text())["v"] == 5
        restored = SessionContext(workspace=root, task_run=task_store.load(), task_store=task_store)
        assert restored.task_run.baseline == baseline
        assert bool(restored.task_run.baseline.snapshot_id) is snapshot
        working = ensure_working_set(restored)
        assert working.entries and restored.code_read_versions == {}
        assert all(entry["freshness"] == "needs_revalidation" for entry in working.entries.values())
        queries = json.loads(TaskStateTool(restored)._invoke({"section": "code_queries"}).content)
        assert queries["total"] == 1 and queries["page"][0]["no_gain_repeats"] == 1
        assert queries["page"][0]["freshness"] == "needs_revalidation"
        (root / "a.py").write_text("def needle(): return 2\n", encoding="utf-8")
        await notify_committed(session, ("a.py",), "change")
        assert next(iter(session.code_working_set.entries.values()))["freshness"] == "stale"
        assert not session.code_read_versions
    finally:
        await release_service(session)


def test_invalid_history_is_discarded_observably(tmp_path, caplog):
    task = TaskRun(TaskContract("restore"), capture_revision(tmp_path), code_exploration={"v": 999})
    session = SessionContext(workspace=tmp_path, task_run=task)
    working = ensure_working_set(session)
    assert not working.entries and not working.exploration.queries
    assert "discarding invalid historical code exploration" in caplog.text


async def test_react_records_exploration_in_logs_and_transcript(tmp_path, monkeypatch):
    from agent_core.models import LLMResult, ToolCall
    from agent_core.react import ReActAgent, ReActConfig
    from agent_core.storage import read_events
    from agent_core.tools.transaction import _secure_mkdir

    home = tmp_path.parent / "exploration-home"
    _secure_mkdir(home)
    monkeypatch.setenv("POLARIS_HOME", str(home))

    class Provider:
        calls = 0

        async def complete(self, messages, tools, config, stream=None, should_cancel=None):
            self.calls += 1
            if self.calls <= 2:
                call = ToolCall("code_search", {"query": "needle", "kind": "symbol", "path": "."}, id=f"q{self.calls}")
            elif self.calls == 3:
                call = ToolCall("task_state", {"section": "code_queries"}, id="history")
            else:
                return LLMResult("located needle", stop_reason="end")
            return LLMResult("locating", tool_calls=[call], stop_reason="tool_use")

    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def needle(): return 1\n", encoding="utf-8")
    agent = ReActAgent(Provider(), ReActConfig(run_dir=str(tmp_path / "runs"),
        session_dir=str(tmp_path.parent / "transcripts"), project_instructions=False, git_context=False,
        codeintel=CodeIntelConfig(maintain=False, watch=False)), workspace=root)
    agent.registry.activate("code_search")
    service = await get_service(agent.session)
    try:
        await service.ensure_index()
        result = await agent.run("Locate needle; do not edit files.")
        searches = [m for m in result.messages if m.role == "tool" and m.name == "code_search"]
        assert len(searches) == 2
        assert searches[-1].metadata["codeintel"]["exploration"]["no_gain_repeats"] == 1
        assert agent.session.task_run.code_exploration["queries"][0]["attempts"] == 2
        assert agent.transcript is not None
        transcript = agent.transcript.path.read_text(encoding="utf-8")
        assert '"no_gain_repeats": 1' in transcript
        events = list(read_events(agent.logger.path))
        assert any(e["event"] == "tool_result" and "no_gain_repeats" in json.dumps(e) for e in events)
        assert result.answer == "located needle"
    finally:
        await release_service(agent.session)
        agent.logger.close()


def test_changed_policy_drops_history_and_edit_authority(tmp_path):
    session = SessionContext(workspace=tmp_path, task_run=TaskRun(TaskContract("policy"), capture_revision(tmp_path)))
    working = ensure_working_set(session)
    hit = CodeHit("a.py", 1, 1, "", FileVersion(worktree_id(tmp_path), "a.py", "a" * 64))
    working.add(hit)
    session.task_run.code_exploration = working.to_state()
    session.code_read_versions["a.py"] = hit.version.to_dict()
    class Restricted:
        deny = ["a.py"]
        ask = []
        def deny_match(self, name, args):
            return str(args["path"]).endswith("a.py")
        def ask_match(self, name, args):
            return False
    rules = Restricted()
    session.code_permission_rules = lambda: rules
    assert not ensure_working_set(session).entries
    assert not session.code_read_versions


def test_exploration_projection_survives_structured_compaction():
    page = SearchPage(hits=[CodeHit("a.py", 1, 1, "x" * 3000, FileVersion("wt", "a.py", "a" * 64))]).to_dict()
    page["exploration"] = {"query_id": "b" * 64, "no_gain_repeats": 2, "next_action": "read_candidate_context"}
    compact = compact_code_result(json.dumps(page), "code_search", 1500)
    result = json.loads(compact)
    assert result["exploration"]["no_gain_repeats"] == 2
    assert result["exploration"]["freshness"].startswith("historical")
    assert result["exploration"]["next_action"] == "recover_original_page"
    assert result["coverage"]["complete"] is False


def test_history_has_a_serialized_byte_budget(tmp_path):
    working = TaskWorkingSet()
    working.binding = (worktree_id(tmp_path), "")
    for index in range(128):
        path = f"src/{index}.py"
        working.add(CodeHit(path, 1, 1, "", FileVersion(working.binding[0], path, "a" * 64)), reason="注释" * 1000)
    saved = working.to_state()
    assert len(json.dumps(saved, ensure_ascii=False).encode()) <= 128 * 1024
    restored = TaskWorkingSet()
    restored.binding = working.binding
    restored.restore(saved, allowed=lambda _: True)
    assert restored.entries


@pytest.mark.parametrize("path", ["../outside", "C:/outside", "a\\b", ".env"])
def test_restored_evidence_rejects_unsafe_names(tmp_path, path):
    working = TaskWorkingSet()
    working.binding = (worktree_id(tmp_path), "")
    working.add(CodeHit(path, 1, 1, "", FileVersion(working.binding[0], path, "a" * 64)))
    restored = TaskWorkingSet()
    restored.binding = working.binding
    with pytest.raises(ValueError, match="identity"):
        restored.restore(working.to_state(), allowed=lambda _: True)
