from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from agent_core.agents.team import FileLock, TeamPermissionError, TeamStore
from agent_core.models import LLMResult, ToolCall, ToolRisk
from agent_core.permissions import PermissionMode, PermissionPolicy
from agent_core.providers import FakeProvider
from agent_core.providers.base import LLMProvider, StreamHandler
from agent_core.react import ReActAgent, ReActConfig
from agent_core.session import SessionContext
from agent_core.tools.catalog import default_tools
from agent_core.tools.executor import ToolExecutor
from agent_core.tools.registry import ToolRegistry
from agent_core.tools.team import (
    TaskCreateTool,
    TaskUpdateTool,
    TeamCreateTool,
    TeamInboxReadTool,
    TeamMessageSendTool,
    TeammateSpawnTool,
)


async def test_team_create_builds_config_tasks_and_leader_inbox(tmp_path: Path) -> None:
    store = TeamStore(tmp_path)
    team = await store.create_team("alpha", "ship the feature", "lead")

    team_dir = tmp_path / team["id"]
    assert (team_dir / "team.json").exists()
    assert (team_dir / "tasks.json").exists()
    assert (team_dir / "inbox" / "lead.jsonl").exists()
    assert await store.list_tasks(team["id"]) == []
    assert (await store.get_team(team["id"]))["leader"] == "lead"


async def test_inbox_writes_are_file_locked_under_concurrency(tmp_path: Path) -> None:
    store = TeamStore(tmp_path)
    team = await store.create_team("alpha", "coordinate")
    team_id = team["id"]
    await store.add_member(team_id, "worker", "researcher")

    # 50 concurrent sends: each offloads to a worker thread inside the store, so
    # the sidecar FileLock is what keeps appends atomic.
    await asyncio.gather(
        *(store.send_message(team_id, "leader", "worker", f"message {index}") for index in range(50))
    )

    messages = await store.read_inbox(team_id, "worker", unread_only=False)
    assert len(messages) == 50
    assert len({message["id"] for message in messages}) == 50
    assert all(message["to"] == "worker" for message in messages)


async def test_event_reads_are_file_locked_against_writers(tmp_path: Path) -> None:
    store = TeamStore(tmp_path)
    team = await store.create_team("alpha", "coordinate")
    team_id = team["id"]
    events = store._events_file(team_id)

    # Exercises the blocking internal directly: the FileLock held here must make a
    # concurrent reader (on its own thread) wait until the lock is released.
    with ThreadPoolExecutor(max_workers=1) as pool:
        with FileLock(store._lock_file(events)):
            future = pool.submit(store._read_events_sync, team_id)
            time.sleep(0.05)
            assert not future.done()
        records = future.result(timeout=1)

    assert records[-1]["event"] == "team_created"


async def test_task_update_enforces_owner_permissions_and_claiming(tmp_path: Path) -> None:
    store = TeamStore(tmp_path)
    team = await store.create_team("alpha", "coordinate")
    team_id = team["id"]
    await store.add_member(team_id, "alice", "researcher")
    await store.add_member(team_id, "bob", "reviewer")
    task = await store.create_task(team_id, "inspect", "inspect code")

    updated, assigned_to = await store.update_task(team_id, task["id"], "leader", owner="alice")
    assert assigned_to == "alice"
    assert updated["owner"] == "alice"
    assert updated["status"] == "assigned"

    with pytest.raises(TeamPermissionError):
        await store.update_task(team_id, task["id"], "bob", status="completed")

    updated, _ = await store.update_task(team_id, task["id"], "alice", status="completed", result="done")
    assert updated["status"] == "completed"
    assert updated["result"] == "done"

    unowned = await store.create_task(team_id, "claim me", "unowned task")
    claimed, _ = await store.update_task(team_id, unowned["id"], "bob", status="in_progress")
    assert claimed["owner"] == "bob"
    assert claimed["status"] == "in_progress"


async def test_inbox_unread_cursor_advances(tmp_path: Path) -> None:
    store = TeamStore(tmp_path)
    team = await store.create_team("alpha", "coordinate")
    team_id = team["id"]
    await store.add_member(team_id, "worker", "researcher")
    await store.send_message(team_id, "leader", "worker", "first")
    await store.send_message(team_id, "leader", "worker", "second")

    assert len(await store.read_inbox(team_id, "worker")) == 2
    assert await store.read_inbox(team_id, "worker") == []
    assert len(await store.read_inbox(team_id, "worker", unread_only=False)) == 2


async def test_team_tools_create_tasks_assign_and_message(tmp_path: Path) -> None:
    session = SessionContext(team_store=TeamStore(tmp_path))
    created = await TeamCreateTool(session).run({"name": "alpha", "goal": "coordinate", "leader_name": "lead"})
    assert created.ok
    team_id = created.metadata["team_id"]
    assert session.team_id == team_id
    assert session.agent_name == "lead"

    await session.team_store.add_member(team_id, "worker", "researcher")
    task = await TaskCreateTool(session).run(
        {"team_id": team_id, "title": "inspect", "description": "inspect code"}
    )
    assert task.ok
    task_id = task.metadata["task_id"]

    assigned = await TaskUpdateTool(session).run({"team_id": team_id, "task_id": task_id, "owner": "worker"})
    assert assigned.ok
    messages = await session.team_store.read_inbox(team_id, "worker")
    assert messages[0]["kind"] == "assignment"
    assert "Task assigned" in messages[0]["content"]


async def test_teammate_spawn_tool_uses_session_factory() -> None:
    calls: list[tuple[str, str, str, str | None, str, str | None]] = []

    async def factory(
        team_id: str, name: str, role: str, task_id: str | None, preset: str, model: str | None = None
    ) -> str:
        calls.append((team_id, name, role, task_id, preset, model))
        return "spawned"

    session = SessionContext(teammate_factory=factory)
    result = await TeammateSpawnTool(session).run(
        {"team_id": "team_abc", "name": "worker", "role": "researcher", "task_id": "task_1", "tool_preset": "full"}
    )
    assert result.ok
    assert result.content == "spawned"
    # No model given → None forwarded (inherit the parent's model).
    assert calls == [("team_abc", "worker", "researcher", "task_1", "full", None)]


async def test_teammate_spawn_forwards_model_override() -> None:
    seen: list = []

    async def factory(
        team_id: str, name: str, role: str, task_id: str | None, preset: str, model: str | None = None
    ) -> str:
        seen.append(model)
        return "spawned"

    result = await TeammateSpawnTool(SessionContext(teammate_factory=factory)).run(
        {"team_id": "t", "name": "w", "role": "r", "model": "claude-sonnet-4-6"}
    )
    assert result.metadata["model"] == "claude-sonnet-4-6"
    assert seen == ["claude-sonnet-4-6"]


async def test_teammate_spawn_different_tasks_can_run_concurrently(tmp_path: Path) -> None:
    async def factory(
        team_id: str, name: str, role: str, task_id: str | None, preset: str, model: str | None = None
    ) -> str:
        await asyncio.sleep(0.15)
        return f"{name}:{task_id}"

    registry = ToolRegistry()
    registry.register(TeammateSpawnTool(SessionContext(workspace=tmp_path, teammate_factory=factory)))
    executor = ToolExecutor(registry, PermissionPolicy(PermissionMode.AUTO), max_workers=2)

    start = time.perf_counter()
    results = await executor.execute_many(
        [
            ToolCall("teammate_spawn", {"team_id": "team_a", "name": "alice", "role": "r", "task_id": "task_1"}),
            ToolCall("teammate_spawn", {"team_id": "team_a", "name": "bob", "role": "r", "task_id": "task_2"}),
        ]
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 0.28
    assert [result.content for result in results] == ["alice:task_1", "bob:task_2"]


async def test_teammate_spawn_same_task_is_serial(tmp_path: Path) -> None:
    async def factory(
        team_id: str, name: str, role: str, task_id: str | None, preset: str, model: str | None = None
    ) -> str:
        await asyncio.sleep(0.15)
        return f"{name}:{task_id}"

    registry = ToolRegistry()
    registry.register(TeammateSpawnTool(SessionContext(workspace=tmp_path, teammate_factory=factory)))
    executor = ToolExecutor(registry, PermissionPolicy(PermissionMode.AUTO), max_workers=2)

    start = time.perf_counter()
    await executor.execute_many(
        [
            ToolCall("teammate_spawn", {"team_id": "team_a", "name": "alice", "role": "r", "task_id": "task_1"}),
            ToolCall("teammate_spawn", {"team_id": "team_a", "name": "bob", "role": "r", "task_id": "task_1"}),
        ]
    )
    elapsed = time.perf_counter() - start

    assert elapsed >= 0.28


def test_teammate_only_tools_are_not_in_default_tool_set() -> None:
    names = {tool.name for tool in default_tools()}
    assert {"team_create", "task_create", "teammate_spawn", "task_update", "team_status"} <= names
    assert "team_inbox_read" not in names
    assert "team_message_send" not in names
    assert TeamInboxReadTool().risk is ToolRisk.READ
    assert TeamMessageSendTool().risk is ToolRisk.WRITE


class _CompletingTeamProvider(LLMProvider):
    def __init__(self, team_id: str, task_id: str) -> None:
        self.team_id = team_id
        self.task_id = task_id
        self.calls = 0

    async def complete(
        self,
        messages,
        tools: list[dict[str, Any]],
        config: dict[str, Any],
        stream: StreamHandler | None = None,
        should_cancel=None,
    ) -> LLMResult:
        self.calls += 1
        if self.calls == 1:
            return LLMResult(
                content="working",
                tool_calls=[
                    ToolCall("team_inbox_read", {"team_id": self.team_id}),
                    ToolCall(
                        "task_update",
                        {
                            "team_id": self.team_id,
                            "task_id": self.task_id,
                            "status": "completed",
                            "result": "done",
                        },
                    ),
                    ToolCall(
                        "team_message_send",
                        {
                            "team_id": self.team_id,
                            "to": "leader",
                            "content": "done",
                            "task_id": self.task_id,
                            "kind": "completion",
                        },
                    ),
                ],
                stop_reason="tool_use",
            )
        return LLMResult(content="worker final", stop_reason="end")


async def test_react_spawn_teammate_can_update_task_and_message_leader(tmp_path: Path) -> None:
    store = TeamStore(tmp_path / "teams")
    team = await store.create_team("alpha", "coordinate")
    team_id = team["id"]
    await store.add_member(team_id, "worker", "researcher")
    task = await store.create_task(team_id, "inspect", "inspect code", owner="worker")
    provider = _CompletingTeamProvider(team_id, task["id"])
    agent = ReActAgent(
        provider,
        ReActConfig(run_dir=str(tmp_path / "runs"), permission="auto"),
        team_store=store,
    )

    answer = await agent._spawn_teammate(team_id, "worker", "researcher", task["id"])

    assert answer == "worker final"
    assert (await store.get_task(team_id, task["id"]))["status"] == "completed"
    leader_messages = await store.read_inbox(team_id, "leader")
    assert leader_messages[-1]["kind"] == "completion"
    assert leader_messages[-1]["content"] == "done"


async def test_teammate_no_longer_runs_blanket_auto(tmp_path: Path) -> None:
    # Teammates used to be forced permission="auto" (a child-side escalation). Now they
    # run the preset-mapped mode with explicit allow rules for the coordination tools.
    store = TeamStore(tmp_path / "teams")
    team = await store.create_team("alpha", "coordinate")
    agent = ReActAgent(
        FakeProvider(),
        ReActConfig(run_dir=str(tmp_path / "runs"), permission="auto"),
        team_store=store,
    )

    built = await agent._make_teammate_child(team["id"], "worker", "researcher", None, "read_only")
    assert not isinstance(built, str)
    child, _prompt = built
    assert child.config.permission == PermissionMode.DEFAULT
    # Coordination tools are allow-ruled so a read_only teammate can still report back.
    assert child.config.permission_rules.allow_matches("task_update", {})
    assert child.config.permission_rules.allow_matches("team_message_send", {})


# --- per-teammate model override (heterogeneous team) ------------------------


async def test_teammate_child_uses_model_override(tmp_path: Path) -> None:
    store = TeamStore(tmp_path / "teams")
    team = await store.create_team("alpha", "coordinate")
    agent = ReActAgent(
        FakeProvider(),
        ReActConfig(run_dir=str(tmp_path / "runs"), permission="auto", model="claude-opus-4-8"),
        team_store=store,
    )

    built = await agent._make_teammate_child(
        team["id"], "worker", "researcher", None, "read_only", "claude-haiku-4-5-20251001"
    )
    assert not isinstance(built, str)
    child, _prompt = built
    assert child.config.model == "claude-haiku-4-5-20251001"  # teammate override
    assert agent.config.model == "claude-opus-4-8"  # leader unchanged


async def test_teammate_child_inherits_parent_model_by_default(tmp_path: Path) -> None:
    store = TeamStore(tmp_path / "teams")
    team = await store.create_team("alpha", "coordinate")
    agent = ReActAgent(
        FakeProvider(),
        ReActConfig(run_dir=str(tmp_path / "runs"), permission="auto", model="claude-opus-4-8"),
        team_store=store,
    )

    built = await agent._make_teammate_child(team["id"], "worker", "researcher", None, "read_only")
    assert not isinstance(built, str)
    child, _prompt = built
    assert child.config.model == "claude-opus-4-8"  # back-compat: no override


async def test_teammate_unknown_model_refused(tmp_path: Path) -> None:
    store = TeamStore(tmp_path / "teams")
    team = await store.create_team("alpha", "coordinate")
    agent = ReActAgent(
        FakeProvider(),
        ReActConfig(run_dir=str(tmp_path / "runs"), permission="auto"),
        team_store=store,
    )

    answer = await agent._spawn_teammate(team["id"], "worker", "researcher", None, "read_only", "gpt-4")
    assert "unsupported model" in answer
