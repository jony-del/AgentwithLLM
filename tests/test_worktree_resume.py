import subprocess

import pytest

from agent_core.task_runtime import TaskContract, TaskRun, capture_revision
from tests.test_task_runtime import _scripted_agent


async def initialized(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, timeout=10)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True, timeout=10)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, timeout=10)
    (tmp_path / "f.py").write_text("x=1\n")
    subprocess.run(["git", "add", "f.py"], cwd=tmp_path, check=True, timeout=10)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True, timeout=10)
    agent = _scripted_agent(tmp_path, [])
    agent.session.task_run = TaskRun(TaskContract("change f"), capture_revision(tmp_path))
    state = await agent.session.worktree_manager.create_and_enter("owned")
    return agent, state


async def test_resume_restores_recorded_owned_worktree_and_task_store(tmp_path):
    from agent_core.worktree import WorktreeManager
    agent, state = await initialized(tmp_path)
    try:
        record = agent.session.task_run.workspace_binding
        assert record["path"] == str(state.path)
        await agent._bind_execution_workspace(tmp_path)
        # Simulate a fresh process in the project with the old private task record.
        agent.session.task_run.execution_workspace = str(state.path)
        agent.session.persist_task()
        agent.session.worktree_manager = WorktreeManager(agent.session, agent.registry, agent.sandbox, agent.config.tools.worktree)
        result = await agent.run("continue", resume_task=True)
        assert result.status == "completed", result.answer
        assert agent.session.workspace == state.path
        assert agent.executor.journal_storage.workspace == state.path
        assert agent.session.worktree_manager.active.path == state.path
        assert agent.session.task_store.load().workspace_binding == record
    finally:
        await agent.session.worktree_manager.exit("remove", discard_changes=True)
        await agent.fire_session_end("test")
        await agent.runtime.close()


@pytest.mark.parametrize("field,value", [("branch", "main"), ("original_workspace", "C:/foreign"), ("slug", "../escape"), ("base_sha", "bad")])
async def test_foreign_or_corrupt_binding_cannot_publish(tmp_path, field, value):
    agent, state = await initialized(tmp_path)
    record = dict(agent.session.task_run.workspace_binding)
    try:
        await agent._bind_execution_workspace(tmp_path)
        record[field] = value
        with pytest.raises((ValueError, RuntimeError)):
            await agent.session.worktree_manager.restore_owned(record)
        assert agent.session.workspace == tmp_path.resolve()
        assert agent.executor.journal_storage.workspace == tmp_path.resolve()
    finally:
        await agent._bind_execution_workspace(state.path)
        await agent.session.worktree_manager.exit("remove", discard_changes=True)
        await agent.fire_session_end("test")
        await agent.runtime.close()


async def test_branch_change_after_checkpoint_is_refused(tmp_path):
    agent, state = await initialized(tmp_path)
    record = dict(agent.session.task_run.workspace_binding)
    try:
        subprocess.run(["git", "checkout", "-qb", "foreign"], cwd=state.path, check=True, timeout=10)
        await agent._bind_execution_workspace(tmp_path)
        with pytest.raises(ValueError, match="registered"):
            await agent.session.worktree_manager.restore_owned(record)
        assert agent.session.workspace == tmp_path.resolve()
    finally:
        subprocess.run(["git", "checkout", "-q", state.branch], cwd=state.path, check=True, timeout=10)
        await agent._bind_execution_workspace(state.path)
        await agent.session.worktree_manager.exit("remove", discard_changes=True)
        await agent.fire_session_end("test")
        await agent.runtime.close()
