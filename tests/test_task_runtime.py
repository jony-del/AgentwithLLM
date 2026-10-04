from dataclasses import replace
from pathlib import Path
import sys

import pytest

from agent_core.task_runtime import (
    PlanStep, TaskContract, TaskRun, TaskStore, VerificationCheck, VerificationEvidence,
    capture_revision, verify_completion,
)
from agent_core.tools.transaction import JournalStorage


def test_final_revision_invalidates_previous_evidence(tmp_path):
    from hashlib import sha256
    import json
    source = tmp_path / "f.py"
    source.write_text("x=1\n")
    baseline = capture_revision(tmp_path)
    check = VerificationCheck("test", ("python", "-m", "pytest"))
    task = TaskRun(TaskContract("fix", checks=(check,)), baseline)
    source.write_text("x=2\n")
    tested = capture_revision(tmp_path)
    task.evidence.append(VerificationEvidence("test", tested.digest, tested.workspace,
        sha256(json.dumps(check.argv).encode()).hexdigest(), 0, "completed"))
    assert verify_completion(task, tested).status == "completed"
    source.write_text("x=3\n")
    assert verify_completion(task, capture_revision(tmp_path)).status == "unverified"


def test_changes_require_checks_and_scope_is_enforced(tmp_path):
    initial = capture_revision(tmp_path)
    task = TaskRun(TaskContract("fix", allowed_paths=("src",)), initial)
    (tmp_path / "unrelated.py").write_text("x=1\n")
    report = verify_completion(task, capture_revision(tmp_path))
    assert report.status == "unverified"
    assert len(report.issues) == 2


@pytest.mark.parametrize("steps", [
    [PlanStep("a", "a", ("b",)), PlanStep("b", "b", ("a",))],
    [PlanStep("a", "a", ("missing",))],
    [PlanStep("a", "a"), PlanStep("a", "duplicate")],
    [PlanStep("a", "a"), PlanStep("b", "b", ("a",), "completed")],
])
def test_plan_rejects_cycles_unknown_dependencies_and_early_completion(tmp_path, steps):
    task = TaskRun(TaskContract("fix"), capture_revision(tmp_path))
    with pytest.raises(ValueError):
        task.replace_plan(steps)
    assert task.plan_revision == 0


def test_failure_budget_requires_replan_or_code_change(tmp_path):
    task = TaskRun(TaskContract("fix"), capture_revision(tmp_path))
    for _ in range(3):
        task.record_failure("test", {"target": "f"}, "ToolFailed", "unchanged")
    assert task.stalled()
    assert verify_completion(task, task.baseline).status == "blocked"
    task.replace_plan([PlanStep("a", "different strategy", status="completed")])
    task.record_failure("test", {"target": "f"}, "ToolFailed", "unchanged")
    assert not task.stalled()


def test_task_state_roundtrip_does_not_persist_permissions(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    task = TaskRun(TaskContract("fix", constraints=("keep API",)), capture_revision(repo))
    task.replace_plan([PlanStep("a", "inspect", status="completed"), PlanStep("b", "fix", ("a",))])
    store = TaskStore(JournalStorage.local(tmp_path / "state", workspace=repo))
    store.save(task)
    restored = store.load()
    assert restored is not None and restored.id == task.id
    assert restored.plan[1].depends_on == ("a",)
    assert "permissions" not in store.path.read_text()


def test_read_only_completion_and_protocol_truncation(tmp_path):
    task = TaskRun(TaskContract("explain"), capture_revision(tmp_path))
    assert verify_completion(task, task.baseline).status == "completed"
    assert verify_completion(task, task.baseline, truncated=True).status == "unverified"
    assert verify_completion(task, task.baseline, running_processes=True).status == "unverified"
    assert verify_completion(task, task.baseline, termination_proven=False).status == "unverified"


async def test_verification_executes_and_detects_command_mutation(tmp_path):
    from agent_core.process_supervisor import ProcessSupervisor
    from agent_core.session import SessionContext
    from agent_core.tool_config import ShellToolConfig
    from agent_core.tools.task_control import RunVerificationTool
    check = VerificationCheck("pass", (sys.executable, "-c", "print('ok')"))
    task = TaskRun(TaskContract("fix", checks=(check,)), capture_revision(tmp_path))
    supervisor = ProcessSupervisor(ShellToolConfig(), tmp_path.parent / "process-logs")
    session = SessionContext(workspace=tmp_path, task_run=task, process_supervisor=supervisor)
    tool = RunVerificationTool(session)
    try:
        result = await tool.run({"check_id": "pass", "argv": list(check.argv)})
        assert result.ok, result.content
        assert verify_completion(task, capture_revision(tmp_path)).status == "completed"
        wrong = await tool.run({"check_id": "pass", "argv": [sys.executable, "-c", "print('different')"]})
        assert not wrong.ok
        check = replace(check, argv=(sys.executable, "-c", "from pathlib import Path; Path('f.py').write_text('x=1')"))
        task.contract = replace(task.contract, checks=(check,))
        changed = await tool.run({"check_id": "pass", "argv": list(check.argv)})
        assert not changed.ok
        assert task.evidence[-1].state == "revision_changed"
    finally:
        await supervisor.shutdown()


async def test_caller_checks_cannot_be_replaced_by_model(tmp_path):
    from agent_core.session import SessionContext
    from agent_core.tools.task_control import UpdateTaskPlanTool
    check = VerificationCheck("required", ("python", "-m", "pytest"))
    task = TaskRun(TaskContract("fix", checks=(check,)), capture_revision(tmp_path), checks_locked=True)
    tool = UpdateTaskPlanTool(SessionContext(task_run=task))
    result = await tool.run({"steps": [], "checks": []})
    assert not result.ok
    assert task.contract.checks == (check,)


async def test_run_tests_cancels_supervised_process(tmp_path):
    import asyncio
    from agent_core.process_supervisor import ProcessSupervisor, safe_process_environment
    from agent_core.tool_config import ShellToolConfig
    supervisor = ProcessSupervisor(ShellToolConfig(), tmp_path / "logs")
    task = asyncio.create_task(supervisor.run_argv([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path, timeout=60))
    while not supervisor.running():
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not supervisor.running()
    assert "OPENAI_API_KEY" not in safe_process_environment()
    await supervisor.shutdown()


def _scripted_agent(tmp_path, responses, *, project_instructions=False, session_dir=""):
    from agent_core.models import LLMResult
    from agent_core.providers.fake import FakeProvider
    from agent_core.react import ReActAgent, ReActConfig
    from agent_core.codeintel.config import CodeIntelConfig

    class Scripted(FakeProvider):
        def _compute(self, messages):
            response = responses.pop(0) if responses else LLMResult("done", stop_reason="end")
            if isinstance(response, Exception):
                raise response
            return response

    return ReActAgent(Scripted(), ReActConfig(
        run_dir=str(tmp_path / "runs"), session_dir=session_dir, project_instructions=project_instructions,
        permission="bypass", codeintel=CodeIntelConfig(enabled=False),
    ), workspace=tmp_path)


async def test_react_edit_verification_completion_and_durable_state(tmp_path):
    from agent_core.models import LLMResult, ToolCall
    check = VerificationCheck("test", (sys.executable, "-c", "assert open('f.py').read() == 'x=2\\n'"))
    (tmp_path / "f.py").write_text("x=1\n")
    agent = _scripted_agent(tmp_path, [
        LLMResult("", tool_calls=[ToolCall("edit_file", {"path": "f.py", "old_string": "x=1", "new_string": "x=2"})]),
        LLMResult("", tool_calls=[ToolCall("run_verification", {"check_id": "test", "argv": list(check.argv)})]),
        LLMResult("fixed", stop_reason="end"),
    ], session_dir=str(tmp_path.parent / "task-transcripts"))
    try:
        result = await agent.run("fix f.py", task_contract=TaskContract("fix f.py", checks=(check,), allowed_paths=("f.py",)))
        assert result.status == "completed", result.answer
        assert result.verification is not None and result.verification.changed_paths == ("f.py",)
        saved = agent.session.task_store.load()
        assert saved.status == "completed" and saved.checks_locked
    finally:
        await agent.fire_session_end("test")
        await agent.runtime.close()


async def test_react_cannot_claim_completion_without_verification(tmp_path):
    from agent_core.models import LLMResult, ToolCall
    agent = _scripted_agent(tmp_path, [LLMResult("", tool_calls=[ToolCall("write_text_file", {"path": "f.py", "content": "x=1\n"})])])
    try:
        result = await agent.run("create f.py")
        assert result.status == "unverified"
        assert "Verification status: unverified" in result.answer
        assert agent.session.task_store.load().status == "unverified"
    finally:
        await agent.fire_session_end("test")
        await agent.runtime.close()


async def test_worktree_switch_edits_through_executor_and_preserves_original(tmp_path):
    import subprocess
    from agent_core.models import ToolCall
    for args in (["init", "-q"], ["config", "user.email", "test@example.invalid"], ["config", "user.name", "Test"]):
        subprocess.run(["git", *args], cwd=tmp_path, check=True)
    (tmp_path / "f.py").write_text("x=1\n")
    subprocess.run(["git", "add", "f.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    agent = _scripted_agent(tmp_path, [])
    try:
        state = await agent.session.worktree_manager.create_and_enter("editing")
        assert agent.executor.journal_storage.workspace == state.path
        result = (await agent.executor.execute_many([ToolCall("edit_file", {
            "path": "f.py", "old_string": "x=1", "new_string": "x=2",
        })]))[0]
        assert result.ok, result.content
        assert (state.path / "f.py").read_text() == "x=2\n"
        assert (tmp_path / "f.py").read_text() == "x=1\n"
        await agent.session.worktree_manager.exit("remove", discard_changes=True)
        assert agent.executor.journal_storage.workspace == tmp_path.resolve()
    finally:
        await agent.fire_session_end("test")
        await agent.runtime.close()


async def test_identical_tool_failures_stop_without_spinning(tmp_path):
    from agent_core.models import LLMResult, ToolCall
    agent = _scripted_agent(tmp_path, [LLMResult("", tool_calls=[ToolCall("missing", {})]) for _ in range(10)])
    try:
        result = await agent.run("do work")
        assert result.status == "blocked"
        assert result.steps <= 5
        assert "repeated failures" in result.answer
    finally:
        await agent.fire_session_end("test")
        await agent.runtime.close()


def test_source_packages_named_memory_are_part_of_revision(tmp_path):
    package = tmp_path / "agent_core" / "memory"
    package.mkdir(parents=True)
    source = package / "store.py"
    source.write_text("old")
    before = capture_revision(tmp_path)
    source.write_text("new")
    assert capture_revision(tmp_path).digest != before.digest


async def test_resume_task_restores_goal_and_rejects_missing_state(tmp_path):
    agent = _scripted_agent(tmp_path, [])
    try:
        with pytest.raises(ValueError, match="unfinished task"):
            await agent.run("continue", resume_task=True)
        # An incomplete plan keeps this read-only task unfinished.
        baseline = capture_revision(tmp_path)
        task = TaskRun(TaskContract("original goal"), baseline, execution_workspace=str(tmp_path.resolve()))
        task.replace_plan([PlanStep("a", "unfinished")])
        agent.session.task_store.save(task)
        result = await agent.run("continue", resume_task=True)
        assert result.status == "unverified"
        assert agent.session.task_run.id == task.id
        assert agent.session.task_run.contract.goal == "original goal"
        task.execution_workspace = str(tmp_path / "other")
        agent.session.task_store.save(task)
        with pytest.raises(ValueError, match="recorded execution workspace"):
            await agent.run("continue", resume_task=True)
    finally:
        await agent.fire_session_end("test")
        await agent.runtime.close()


async def test_nested_rule_context_follows_complete_tool_result_batch(tmp_path):
    from agent_core.models import LLMResult, ToolCall
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "AGENTS.md").write_text("Use local conventions")
    (package / "a.py").write_text("a=1")
    (package / "b.py").write_text("b=2")
    agent = _scripted_agent(tmp_path, [LLMResult("", tool_calls=[
        ToolCall("read_text_file", {"path": "pkg/a.py"}),
        ToolCall("read_text_file", {"path": "pkg/b.py"}),
    ])], project_instructions=True)
    try:
        result = await agent.run("explain the package")
        results = [i for i, message in enumerate(result.messages) if message.role == "tool"]
        rules = [i for i, message in enumerate(result.messages) if message.metadata.get("instruction_scope")]
        assert len(results) == 2 and results[1] == results[0] + 1
        assert len(rules) == 1 and rules[0] > results[-1]
        assert "Use local conventions" in result.messages[rules[0]].content
    finally:
        await agent.fire_session_end("test")
        await agent.runtime.close()


async def test_child_task_stores_do_not_overwrite_parent_or_sibling(tmp_path):
    import asyncio
    agent = _scripted_agent(tmp_path, [])
    children = [agent._make_subagent_child("read_only") for _ in range(2)]
    try:
        original = TaskRun(TaskContract("parent goal"), capture_revision(tmp_path))
        agent.session.task_store.save(original)
        paths = {agent.session.task_store.path, *(child.session.task_store.path for child in children)}
        assert len(paths) == 3
        await asyncio.gather(*(child.run(f"child {i}") for i, child in enumerate(children)))
        assert agent.session.task_store.load().id == original.id
        assert agent.session.task_store.load().contract.goal == "parent goal"
        assert [child.session.task_store.load().contract.goal for child in children] == ["child 0", "child 1"]
    finally:
        for child in children:
            await child.fire_session_end("test")
            await child.runtime.close()
        await agent.fire_session_end("test")
        await agent.runtime.close()


async def test_supervised_argv_bounds_output_and_timeout_and_drops_secrets(tmp_path, monkeypatch):
    from agent_core.process_supervisor import ProcessSupervisor
    from agent_core.tool_config import ShellToolConfig
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-credential")
    supervisor = ProcessSupervisor(ShellToolConfig(preview_bytes=256, log_bytes=512), tmp_path / "logs")
    try:
        output = await supervisor.run_argv([sys.executable, "-c", "print('x'*100000)"], tmp_path, timeout=10)
        assert len(output["output"].encode()) <= 256
        assert Path(output["output_path"]).stat().st_size <= 512
        secret = await supervisor.run_argv([sys.executable, "-c", "import os; print('OPENAI_API_KEY' in os.environ)"], tmp_path, timeout=10)
        assert secret["output"].strip() == "False"
        timed = await supervisor.run_argv([sys.executable, "-c", "import time; time.sleep(10)"], tmp_path, timeout=0.25)
        assert timed["state"] == "timed_out"
        assert not supervisor.running()
    finally:
        await supervisor.shutdown()


async def test_provider_error_persists_failed_task_status(tmp_path):
    agent = _scripted_agent(tmp_path, [RuntimeError("provider failed")])
    try:
        with pytest.raises(RuntimeError, match="provider failed"):
            await agent.run("do work")
        assert agent.session.task_store.load().status == "failed"
    finally:
        await agent.fire_session_end("test")
        await agent.runtime.close()


async def test_stop_hook_mutation_invalidates_previous_verification(tmp_path):
    from agent_core.hooks import HookOutcome
    from agent_core.models import LLMResult, ToolCall
    source = tmp_path / "f.py"
    source.write_text("original")
    check = VerificationCheck("read", (sys.executable, "-c", "assert open('f.py').read() == 'original'"))
    agent = _scripted_agent(tmp_path, [LLMResult("", tool_calls=[
        ToolCall("run_verification", {"check_id": "read", "argv": list(check.argv)})]),
        LLMResult("done", stop_reason="end"),
    ])

    class MutatingHook:
        async def on_stop(self, context):
            source.write_text("changed by hook")
            return HookOutcome()

    agent.hooks.stop_hooks.append(MutatingHook())
    try:
        result = await agent.run("verify", task_contract=TaskContract("verify", checks=(check,)))
        assert result.status == "unverified"
        assert not result.verification.evidence
        assert source.read_text() == "changed by hook"
    finally:
        await agent.fire_session_end("test")
        await agent.runtime.close()
