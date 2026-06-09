from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from agent_core.agents.team import FileLock, TeamPermissionError, TeamStore
from agent_core.models import LLMResult, ToolCall, ToolRisk
from agent_core.permissions import PermissionMode, PermissionPolicy
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


def test_team_create_builds_config_tasks_and_leader_inbox(tmp_path: Path) -> None:
    store = TeamStore(tmp_path)
    team = store.create_team("alpha", "ship the feature", "lead")

    team_dir = tmp_path / team["id"]
    assert (team_dir / "team.json").exists()
    assert (team_dir / "tasks.json").exists()
    assert (team_dir / "inbox" / "lead.jsonl").exists()
    assert store.list_tasks(team["id"]) == []
    assert store.get_team(team["id"])["leader"] == "lead"


def test_inbox_writes_are_file_locked_under_concurrency(tmp_path: Path) -> None:
    store = TeamStore(tmp_path)
    team = store.create_team("alpha", "coordinate")
    team_id = team["id"]
    store.add_member(team_id, "worker", "researcher")

    def send(index: int) -> None:
        store.send_message(team_id, "leader", "worker", f"message {index}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(send, range(50)))

    messages = store.read_inbox(team_id, "worker", unread_only=False)
    assert len(messages) == 50
    assert len({message["id"] for message in messages}) == 50
    assert all(message["to"] == "worker" for message in messages)


def test_event_reads_are_file_locked_against_writers(tmp_path: Path) -> None:
    store = TeamStore(tmp_path)
    team = store.create_team("alpha", "coordinate")
    team_id = team["id"]
    events = store._events_file(team_id)

    with ThreadPoolExecutor(max_workers=1) as pool:
        with FileLock(store._lock_file(events)):
            future = pool.submit(store.read_events, team_id)
            time.sleep(0.05)
            assert not future.done()
        records = future.result(timeout=1)

    assert records[-1]["event"] == "team_created"


def test_task_update_enforces_owner_permissions_and_claiming(tmp_path: Path) -> None:
    store = TeamStore(tmp_path)
    team = store.create_team("alpha", "coordinate")
    team_id = team["id"]
    store.add_member(team_id, "alice", "researcher")
    store.add_member(team_id, "bob", "reviewer")
    task = store.create_task(team_id, "inspect", "inspect code")

    updated, assigned_to = store.update_task(team_id, task["id"], "leader", owner="alice")
    assert assigned_to == "alice"
    assert updated["owner"] == "alice"
    assert updated["status"] == "assigned"

    with pytest.raises(TeamPermissionError):
        store.update_task(team_id, task["id"], "bob", status="completed")

    updated, _ = store.update_task(team_id, task["id"], "alice", status="completed", result="done")
    assert updated["status"] == "completed"
    assert updated["result"] == "done"

    unowned = store.create_task(team_id, "claim me", "unowned task")
    claimed, _ = store.update_task(team_id, unowned["id"], "bob", status="in_progress")
    assert claimed["owner"] == "bob"
    assert claimed["status"] == "in_progress"


def test_inbox_unread_cursor_advances(tmp_path: Path) -> None:
    store = TeamStore(tmp_path)
    team = store.create_team("alpha", "coordinate")
    team_id = team["id"]
    store.add_member(team_id, "worker", "researcher")
    store.send_message(team_id, "leader", "worker", "first")
    store.send_message(team_id, "leader", "worker", "second")

    assert len(store.read_inbox(team_id, "worker")) == 2
    assert store.read_inbox(team_id, "worker") == []
    assert len(store.read_inbox(team_id, "worker", unread_only=False)) == 2


def test_team_tools_create_tasks_assign_and_message(tmp_path: Path) -> None:
    session = SessionContext(team_store=TeamStore(tmp_path))
    created = TeamCreateTool(session).run({"name": "alpha", "goal": "coordinate", "leader_name": "lead"})
    assert created.ok
    team_id = created.metadata["team_id"]
    assert session.team_id == team_id
    assert session.agent_name == "lead"

    session.team_store.add_member(team_id, "worker", "researcher")
    task = TaskCreateTool(session).run(
        {"team_id": team_id, "title": "inspect", "description": "inspect code"}
    )
    assert task.ok
    task_id = task.metadata["task_id"]

    assigned = TaskUpdateTool(session).run({"team_id": team_id, "task_id": task_id, "owner": "worker"})
    assert assigned.ok
    messages = session.team_store.read_inbox(team_id, "worker")
    assert messages[0]["kind"] == "assignment"
    assert "Task assigned" in messages[0]["content"]


def test_teammate_spawn_tool_uses_session_factory() -> None:
    calls: list[tuple[str, str, str, str | None, str]] = []

    def factory(team_id: str, name: str, role: str, task_id: str | None, preset: str) -> str:
        calls.append((team_id, name, role, task_id, preset))
        return "spawned"

    session = SessionContext(teammate_factory=factory)
    result = TeammateSpawnTool(session).run(
        {"team_id": "team_abc", "name": "worker", "role": "researcher", "task_id": "task_1", "tool_preset": "full"}
    )
    assert result.ok
    assert result.content == "spawned"
    assert calls == [("team_abc", "worker", "researcher", "task_1", "full")]


def test_teammate_spawn_different_tasks_can_run_concurrently(tmp_path: Path) -> None:
    def factory(team_id: str, name: str, role: str, task_id: str | None, preset: str) -> str:
        time.sleep(0.15)
        return f"{name}:{task_id}"

    registry = ToolRegistry()
    registry.register(TeammateSpawnTool(SessionContext(workspace=tmp_path, teammate_factory=factory)))
    executor = ToolExecutor(registry, PermissionPolicy(PermissionMode.AUTO), max_workers=2)

    start = time.perf_counter()
    results = executor.execute_many(
        [
            ToolCall("teammate_spawn", {"team_id": "team_a", "name": "alice", "role": "r", "task_id": "task_1"}),
            ToolCall("teammate_spawn", {"team_id": "team_a", "name": "bob", "role": "r", "task_id": "task_2"}),
        ]
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 0.28
    assert [result.content for result in results] == ["alice:task_1", "bob:task_2"]


def test_teammate_spawn_same_task_is_serial(tmp_path: Path) -> None:
    def factory(team_id: str, name: str, role: str, task_id: str | None, preset: str) -> str:
        time.sleep(0.15)
        return f"{name}:{task_id}"

    registry = ToolRegistry()
    registry.register(TeammateSpawnTool(SessionContext(workspace=tmp_path, teammate_factory=factory)))
    executor = ToolExecutor(registry, PermissionPolicy(PermissionMode.AUTO), max_workers=2)

    start = time.perf_counter()
    executor.execute_many(
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

    def complete(
        self,
        messages,
        tools: list[dict[str, Any]],
        config: dict[str, Any],
        stream: StreamHandler | None = None,
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


def test_react_spawn_teammate_can_update_task_and_message_leader(tmp_path: Path) -> None:
    store = TeamStore(tmp_path / "teams")
    team = store.create_team("alpha", "coordinate")
    team_id = team["id"]
    store.add_member(team_id, "worker", "researcher")
    task = store.create_task(team_id, "inspect", "inspect code", owner="worker")
    provider = _CompletingTeamProvider(team_id, task["id"])
    agent = ReActAgent(
        provider,
        ReActConfig(run_dir=str(tmp_path / "runs"), permission="auto"),
        team_store=store,
    )

    answer = agent._spawn_teammate(team_id, "worker", "researcher", task["id"])

    assert answer == "worker final"
    assert store.get_task(team_id, task["id"])["status"] == "completed"
    leader_messages = store.read_inbox(team_id, "leader")
    assert leader_messages[-1]["kind"] == "completion"
    assert leader_messages[-1]["content"] == "done"
