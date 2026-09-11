"""Session-isolation contract tests (docs/core-contracts.md §2, audit §9.3).

``ReActAgent.resume_loaded_session`` replaces the whole ``SessionRuntime`` in one
step; every piece of session-scoped state — permission grants, todos, plan state,
approved workflow digests, read-file state, scheduler jobs, counters — belongs to
the old runtime and must not survive the swap. Each test drives a real
``await agent.resume_loaded_session(loaded)`` against an on-disk transcript.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_core.builtin_hooks import StopCompletionHook
from agent_core.chat_commands import dispatch
from agent_core.hooks import HookContext, HookEvent
from agent_core.memory import MemoryConfig
from agent_core.models import Message
from agent_core.providers.fake import FakeProvider
from agent_core.react import ReActAgent, ReActConfig
from agent_core.transcript import (
    TranscriptStore,
    load_transcript,
    project_dir,
)
from agent_core.ui import NullUI


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setenv("POLARIS_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(ws)
    return ws


def _config(tmp_path: Path, session_dir) -> ReActConfig:
    return ReActConfig(
        run_dir=str(tmp_path / "runs"),
        session_dir=str(session_dir),
        memory=MemoryConfig(enabled=False),
        project_instructions=False,
        git_context=False,
    )


def _make_agent(workspace: Path, tmp_path: Path, session_dir, session_id: str = "current-session") -> ReActAgent:
    return ReActAgent(
        FakeProvider(),
        _config(tmp_path, session_dir),
        workspace=workspace,
        session_id=session_id,
    )


async def _resume_target(session_dir, workspace: Path, session_id: str = "target-session"):
    store = TranscriptStore(session_dir, workspace, session_id)
    try:
        await store.append_message(Message("user", "older turn"))
    finally:
        store.close()
    return load_transcript(store.path)


# --- a. session permission grants -------------------------------------------


async def test_session_permission_grants_do_not_survive_resume(workspace, tmp_path):
    session_root = tmp_path / "sessions"
    loaded = await _resume_target(session_root, workspace)
    agent = _make_agent(workspace, tmp_path, session_root)
    try:
        old_policy = agent.permissions
        old_policy.add_session_rule("shell(git status)")
        old_policy._session_allow.add("read_file")
        old_policy._session_allow_commands.add("git status")
        assert not old_policy._session_rules.is_empty

        await agent.resume_loaded_session(loaded)

        new_policy = agent.permissions
        assert new_policy is not old_policy
        assert new_policy._session_rules.is_empty
        assert new_policy._session_allow == set()
        assert new_policy._session_allow_commands == set()
    finally:
        agent.logger.close()


# --- b. todos / plan / workflow approvals / read state -----------------------


async def test_todo_plan_workflow_and_read_state_do_not_survive_resume(workspace, tmp_path):
    session_root = tmp_path / "sessions"
    loaded = await _resume_target(session_root, workspace)
    agent = _make_agent(workspace, tmp_path, session_root)
    try:
        old_session = agent.session
        old_session.todos.replace([{"content": "unfinished task", "status": "in_progress"}])
        old_session.plan_state.enter("default", workspace / "plan.md")
        old_session.approved_workflow_digests.add("workflow-digest")
        old_session.record_read("/a", "snapshot")

        await agent.resume_loaded_session(loaded)

        assert agent.session is not old_session
        assert agent.session.todos.items() == []
        assert agent.session.plan_state.active is False
        assert agent.session.plan_state.artifact_path is None
        assert agent.session.approved_workflow_digests == set()
        assert agent.session.read_file_state == {}
    finally:
        agent.logger.close()


# --- c. owned resources, counters, per-session flags -------------------------


async def test_resources_counters_and_flags_reset_on_resume(workspace, tmp_path):
    session_root = tmp_path / "sessions"
    loaded = await _resume_target(session_root, workspace)
    agent = _make_agent(workspace, tmp_path, session_root)
    try:
        agent._session_input_tokens = 120
        agent._session_output_tokens = 34
        agent._last_message_uuid = "old-head"
        agent.fast_mode = True
        agent.session_title = "old title"
        agent._unsandboxed_permission_ack = True
        old_runtime = agent.runtime
        old_supervisor = agent.process_supervisor
        old_transcript = agent.transcript
        assert agent._session_input_tokens == 120

        await agent.resume_loaded_session(loaded)

        assert agent.runtime is not old_runtime
        assert old_runtime.closed is True
        assert agent.process_supervisor is not old_supervisor
        assert agent.session_id == "target-session"
        assert agent.session_id != old_runtime.descriptor.session_id
        assert agent.transcript is not old_transcript
        assert agent.transcript is not None
        assert agent.transcript.session_id == "target-session"
        # Session counters and their property aliases restart at zero, and the
        # chain head restarts at the new runtime's empty durable head.
        assert agent.runtime.counters == {"input_tokens": 0, "output_tokens": 0}
        assert agent._session_input_tokens == 0
        assert agent._session_output_tokens == 0
        assert agent._last_message_uuid is None
        assert agent.fast_mode is False
        assert agent.session_title is None  # the target transcript carries no title
        assert agent._unsandboxed_permission_ack is False
    finally:
        agent.logger.close()


# --- d. scheduler jobs --------------------------------------------------------


async def test_resume_deletes_old_sessions_scheduler_jobs(workspace, tmp_path):
    class RecordingSchedulerStore:
        def __init__(self) -> None:
            self.deleted: list[tuple[str, str]] = []

        # SessionRuntime.close() routes this through asyncio.to_thread, so the
        # mock must be synchronous (an async def would leak an unawaited coroutine).
        def delete_session_jobs(self, session_id, agent_id):
            self.deleted.append((session_id, agent_id))

    session_root = tmp_path / "sessions"
    loaded = await _resume_target(session_root, workspace)
    agent = _make_agent(workspace, tmp_path, session_root)
    try:
        store = RecordingSchedulerStore()
        agent.runtime.scheduler_store = store
        old_session_id = agent.session_id
        old_agent_id = agent.session.agent_id

        await agent.resume_loaded_session(loaded)

        assert store.deleted == [(old_session_id, old_agent_id)]
        assert agent.runtime.scheduler_store is None
    finally:
        agent.logger.close()


# --- e. StopCompletionHook follows the live runtime ---------------------------


def _stop_context() -> HookContext:
    return HookContext(event=HookEvent.STOP, messages=[], stop_hook_active=False)


async def test_stop_completion_hook_follows_the_new_runtime(workspace, tmp_path):
    session_root = tmp_path / "sessions"
    loaded = await _resume_target(session_root, workspace)
    agent = _make_agent(workspace, tmp_path, session_root)
    try:
        hook = next(h for h in agent.hooks.stop_hooks if isinstance(h, StopCompletionHook))
        # The getter closes over the agent, so pre-resume it sees the old context.
        agent.session.todos.replace([{"content": "old open todo", "status": "in_progress"}])
        assert (await hook.on_stop(_stop_context())).block is True

        await agent.resume_loaded_session(loaded)

        # Post-resume the same hook instance observes the NEW context: the old
        # open todo is gone with the old runtime, so a stop is allowed again...
        assert (await hook.on_stop(_stop_context())).block is False
        # ...and an open todo recorded on the new context blocks it once more.
        agent.session.todos.replace([{"content": "new open todo", "status": "pending"}])
        blocked = await hook.on_stop(_stop_context())
        assert blocked.block is True
        assert "new open todo" in (blocked.additional_context or "")
    finally:
        agent.logger.close()


# --- f. chat /resume cross-project rejection + /branch fork -------------------


async def test_chat_resume_rejects_cross_project_target_and_branch_forks_it(
    tmp_path, monkeypatch, capsys
):
    session_root = tmp_path / "s"
    project_a = tmp_path / "a"
    project_b = tmp_path / "b"
    project_a.mkdir()
    project_b.mkdir()
    monkeypatch.setenv("POLARIS_HOME", str(tmp_path / "home"))
    source_id = "c" * 32
    store = TranscriptStore(session_root, project_a, source_id)
    first = Message("user", "hello")
    second = Message("assistant", "hi")
    second.parent_uuid = first.uuid
    try:
        assert await store.append_message(first)
        assert await store.append_message(second)
    finally:
        store.close()

    # The chat session runs in project B; the saved session belongs to project A.
    monkeypatch.chdir(project_b)
    agent = ReActAgent(FakeProvider(), _config(tmp_path, session_root), workspace=project_b)
    original_session_id = agent.session_id
    try:
        rejected = await dispatch(f"/resume {source_id}", agent, NullUI(), [])
        out = capsys.readouterr().out
        assert rejected.history is None
        assert "belongs to" in out and "/branch" in out
        assert agent.session_id == original_session_id

        branched = await dispatch(f"/branch {source_id}", agent, NullUI(), [])
        out = capsys.readouterr().out
        assert "Branched into session" in out
        history = branched.history
        assert history is not None
        assert [message.content for message in history] == ["hello", "hi"]
        # The fork remaps every uuid and re-links parents inside the clone.
        assert {message.uuid for message in history}.isdisjoint({first.uuid, second.uuid})
        assert history[0].parent_uuid is None
        assert history[1].parent_uuid == history[0].uuid
        # The branch is a new session owned by the CURRENT project.
        assert agent.session_id != source_id
        assert agent.session_id != original_session_id
        assert agent.transcript is not None
        assert agent.transcript.path.parent == project_dir(session_root, project_b)
    finally:
        agent.logger.close()
