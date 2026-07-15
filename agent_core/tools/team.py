from __future__ import annotations

import json
from typing import Any

from agent_core.agents.team import TeamError, TeamPermissionError, TeamStore
from agent_core.models import ToolRisk, ToolResult
from agent_core.permission_types import (
    DecisionSource,
    PermissionContext,
    PermissionMode,
    PermissionResult,
)
from agent_core.session import SessionAwareMixin
from agent_core.tools.base import ConcurrencySpec, ResourceLock, Tool
from agent_core.tools.catalog import builtin_tool

_PRESETS = {"read_only", "full"}


def _internal_state_permission(reason: str) -> PermissionResult:
    return PermissionResult.allow(reason, decision_source=DecisionSource.TOOL)


def _team_side_effect_permission(
    tool_name: str,
    arguments: dict[str, Any],
    context: PermissionContext,
    reason: str,
) -> PermissionResult:
    allow_rule = context.rules.allow_match(tool_name, arguments) if context.rules is not None else None
    if allow_rule is not None:
        return PermissionResult.allow(
            "team action allowed by rule",
            decision_source=DecisionSource.RULE,
            matched_rule=allow_rule,
        )
    if tool_name in context.session_authorizations.tool_names:
        return PermissionResult.allow("team action allowed for this session", decision_source=DecisionSource.RULE)
    if context.mode is PermissionMode.BYPASS:
        return PermissionResult.passthrough(reason)
    return PermissionResult.ask(
        reason,
        classifier_approvable=context.mode is not PermissionMode.PLAN,
    )


def _render(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _team_id(arguments: dict[str, object], fallback: str | None) -> str:
    return str(arguments.get("team_id") or fallback or "").strip()


def _resource_id(*parts: object) -> str:
    return ":".join(str(part or "_").strip() or "_" for part in parts)


def _store(tool_name: str, maybe_store: object | None) -> TeamStore | ToolResult:
    if maybe_store is None:
        return ToolResult(
            tool_name,
            "Team collaboration is not available in this context.",
            ok=False,
            metadata={"error_type": "Unavailable"},
        )
    return maybe_store  # type: ignore[return-value]


def _team_error(tool_name: str, exc: Exception) -> ToolResult:
    if isinstance(exc, TeamPermissionError):
        return ToolResult(tool_name, str(exc), ok=False, metadata={"error_type": "PermissionDenied"})
    if isinstance(exc, TeamError):
        return ToolResult(tool_name, str(exc), ok=False, metadata={"error_type": "BadArgs"})
    return ToolResult(tool_name, f"Team tool error: {type(exc).__name__}: {exc}", ok=False)


@builtin_tool
class TeamCreateTool(SessionAwareMixin, Tool):
    name = "team_create"
    description = (
        "Create a file-backed agent team for explicit multi-agent collaboration. This creates "
        "shared team config, a shared task list, and the leader inbox under runs/teams."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Short team name."},
            "goal": {"type": "string", "description": "Overall team objective."},
            "leader_name": {
                "type": "string",
                "description": "Leader agent name; defaults to leader. Use letters, digits, _, ., or -.",
            },
        },
        "required": ["name", "goal"],
    }
    risk = ToolRisk.WRITE

    async def check_permissions(
        self, arguments: dict[str, Any], context: PermissionContext
    ) -> PermissionResult:
        return _team_side_effect_permission(
            self.name, arguments, context, "creating a team changes shared orchestration state"
        )

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        return ConcurrencySpec(
            (
                ResourceLock("session", "team_context", "write"),
                ResourceLock("team_create", "global", "write"),
            )
        )

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        store = _store(self.name, self.session.team_store)
        if isinstance(store, ToolResult):
            return store
        try:
            team = await store.create_team(
                str(arguments.get("name", "")),
                str(arguments.get("goal", "")),
                str(arguments.get("leader_name", "leader") or "leader"),
            )
        except Exception as exc:  # noqa: BLE001 - tools should return observations
            return _team_error(self.name, exc)
        self.session.agent_name = team["leader"]
        self.session.team_id = team["id"]
        return ToolResult(self.name, _render(team), metadata={"team_id": team["id"]})


@builtin_tool
class TaskCreateTool(SessionAwareMixin, Tool):
    name = "task_create"
    description = "Create a task in a team's shared task list. The task starts pending unless an owner is provided."
    input_schema = {
        "type": "object",
        "properties": {
            "team_id": {"type": "string", "description": "Team id returned by team_create."},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "owner": {"type": "string", "description": "Optional existing teammate/leader owner."},
            "priority": {"type": "string", "description": "Optional priority label; defaults to normal."},
        },
        "required": ["team_id", "title", "description"],
    }
    risk = ToolRisk.WRITE

    async def check_permissions(
        self, arguments: dict[str, Any], context: PermissionContext
    ) -> PermissionResult:
        return _internal_state_permission("Task creation is internal coordination state")

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        team_id = _team_id(arguments, self.session.team_id)
        return ConcurrencySpec(
            (
                ResourceLock("session", "team_context", "write"),
                ResourceLock("team_tasks", team_id or "_", "write"),
            )
        )

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        store = _store(self.name, self.session.team_store)
        if isinstance(store, ToolResult):
            return store
        team_id = _team_id(arguments, self.session.team_id)
        try:
            task = await store.create_task(
                team_id,
                str(arguments.get("title", "")),
                str(arguments.get("description", "")),
                str(arguments["owner"]) if arguments.get("owner") else None,
                str(arguments["priority"]) if arguments.get("priority") else None,
            )
        except Exception as exc:  # noqa: BLE001
            return _team_error(self.name, exc)
        self.session.team_id = team_id
        return ToolResult(self.name, _render(task), metadata={"team_id": team_id, "task_id": task["id"]})


@builtin_tool
class TeammateSpawnTool(SessionAwareMixin, Tool):
    name = "teammate_spawn"
    description = (
        "Create or reuse a teammate in a team and run one teammate work turn. Assign work first "
        "with task_update(owner=...) or pass task_id to focus the teammate on a task."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "team_id": {"type": "string", "description": "Team id returned by team_create."},
            "name": {"type": "string", "description": "Teammate name; letters, digits, _, ., or -."},
            "role": {"type": "string", "description": "Teammate role/instructions."},
            "task_id": {"type": "string", "description": "Optional task to focus on."},
            "tool_preset": {
                "type": "string",
                "enum": ["read_only", "full"],
                "description": "read_only gives read/search plus team tools; full also allows file writes.",
            },
            "model": {
                "type": "string",
                "description": (
                    "Optional model for the teammate (e.g. claude-haiku-4-5-..., "
                    "claude-sonnet-4-6, claude-opus-4-8). Omit to inherit the parent's "
                    "model, so a team can mix models by role."
                ),
            },
        },
        "required": ["team_id", "name", "role"],
    }
    risk = ToolRisk.WRITE

    async def check_permissions(
        self, arguments: dict[str, Any], context: PermissionContext
    ) -> PermissionResult:
        if context.mode is PermissionMode.PLAN and str(arguments.get("tool_preset", "read_only")) == "full":
            return PermissionResult.deny(
                "plan mode cannot spawn a writable teammate",
                decision_source=DecisionSource.CENTRAL_SAFETY,
            )
        return _team_side_effect_permission(
            self.name, arguments, context, "spawning a teammate consumes external model capacity"
        )

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        team_id = _team_id(arguments, self.session.team_id)
        name = str(arguments.get("name", "")).strip()
        task_id = str(arguments["task_id"]).strip() if arguments.get("task_id") else None
        preset = str(arguments.get("tool_preset", "read_only"))
        if preset not in _PRESETS:
            preset = "read_only"
        fs_mode = "write" if preset == "full" else "read"
        locks = [
            ResourceLock("member", _resource_id(team_id, name), "write"),
            ResourceLock("fs", str(self.session.workspace.resolve()), fs_mode, subtree=True),
        ]
        if task_id:
            locks.append(ResourceLock("task", _resource_id(team_id, task_id), "write"))
        return ConcurrencySpec(tuple(locks))

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        """Run one teammate turn on the current event loop so teammates overlap."""
        factory = self.session.teammate_factory
        if factory is None:
            return ToolResult(
                self.name,
                "Team teammates are not available in this context.",
                ok=False,
                metadata={"error_type": "Unavailable"},
            )
        team_id = _team_id(arguments, self.session.team_id)
        name = str(arguments.get("name", "")).strip()
        role = str(arguments.get("role", "")).strip()
        task_id = str(arguments["task_id"]).strip() if arguments.get("task_id") else None
        preset = str(arguments.get("tool_preset", "read_only"))
        if preset not in _PRESETS:
            preset = "read_only"
        model = str(arguments.get("model", "")).strip() or None
        try:
            answer = await factory(team_id, name, role, task_id, preset, model)
        except Exception as exc:  # noqa: BLE001
            return _team_error(self.name, exc)
        return ToolResult(
            self.name,
            answer,
            metadata={
                "team_id": team_id,
                "teammate": name,
                "task_id": task_id,
                "preset": preset,
                "model": model,
            },
        )


@builtin_tool
class TaskUpdateTool(SessionAwareMixin, Tool):
    name = "task_update"
    description = (
        "Update a team task. Leaders can assign/reassign owners; teammates can claim unowned "
        "tasks or update tasks they own. Statuses: pending, assigned, in_progress, blocked, completed."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "team_id": {"type": "string", "description": "Team id returned by team_create."},
            "task_id": {"type": "string"},
            "status": {
                "type": "string",
                "enum": ["pending", "assigned", "in_progress", "blocked", "completed"],
            },
            "owner": {"type": "string", "description": "Assign or claim the task owner."},
            "note": {"type": "string", "description": "Progress note to append."},
            "result": {"type": "string", "description": "Final or partial result."},
        },
        "required": ["team_id", "task_id"],
    }
    risk = ToolRisk.WRITE

    async def check_permissions(
        self, arguments: dict[str, Any], context: PermissionContext
    ) -> PermissionResult:
        return _internal_state_permission("Task updates are internal coordination state")

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        team_id = _team_id(arguments, self.session.team_id)
        task_id = str(arguments.get("task_id", "")).strip()
        locks = [ResourceLock("task", _resource_id(team_id, task_id), "write")]
        if arguments.get("owner"):
            locks.append(ResourceLock("member", _resource_id(team_id, arguments["owner"]), "write"))
        return ConcurrencySpec(tuple(locks))

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        store = _store(self.name, self.session.team_store)
        if isinstance(store, ToolResult):
            return store
        team_id = _team_id(arguments, self.session.team_id)
        actor = self.session.agent_name
        try:
            task, assigned_to = await store.update_task(
                team_id,
                str(arguments.get("task_id", "")),
                actor,
                status=str(arguments["status"]) if arguments.get("status") else None,
                owner=str(arguments["owner"]) if arguments.get("owner") else None,
                note=str(arguments["note"]) if arguments.get("note") else None,
                result=str(arguments["result"]) if arguments.get("result") is not None else None,
            )
            if assigned_to and assigned_to != actor:
                await store.send_message(
                    team_id,
                    actor,
                    assigned_to,
                    f"Task assigned: {task['id']} - {task['title']}",
                    task_id=str(task["id"]),
                    kind="assignment",
                )
        except Exception as exc:  # noqa: BLE001
            return _team_error(self.name, exc)
        return ToolResult(self.name, _render(task), metadata={"team_id": team_id, "task_id": task["id"]})


@builtin_tool
class TeamStatusTool(SessionAwareMixin, Tool):
    name = "team_status"
    description = "Read a team's shared config, task list, members, and recent team events."
    input_schema = {
        "type": "object",
        "properties": {
            "team_id": {"type": "string", "description": "Team id returned by team_create."},
            "recent_events": {"type": "integer", "description": "Number of recent events to include; default 20."},
        },
        "required": ["team_id"],
    }
    risk = ToolRisk.READ

    async def check_permissions(
        self, arguments: dict[str, Any], context: PermissionContext
    ) -> PermissionResult:
        return _internal_state_permission("Team status is internal read-only state")

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        team_id = _team_id(arguments, self.session.team_id)
        return ConcurrencySpec((ResourceLock("team_status", team_id or "_", "read"),))

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        store = _store(self.name, self.session.team_store)
        if isinstance(store, ToolResult):
            return store
        team_id = _team_id(arguments, self.session.team_id)
        try:
            status = await store.status(team_id, int(arguments.get("recent_events", 20)))
        except Exception as exc:  # noqa: BLE001
            return _team_error(self.name, exc)
        return ToolResult(self.name, _render(status), metadata={"team_id": team_id})


class TeamInboxReadTool(SessionAwareMixin, Tool):
    name = "team_inbox_read"
    description = "Read this teammate's own team inbox. By default returns unread messages and advances the cursor."
    input_schema = {
        "type": "object",
        "properties": {
            "team_id": {"type": "string", "description": "Team id; defaults to this teammate's current team."},
            "unread_only": {"type": "boolean", "description": "Defaults to true."},
        },
        "required": [],
    }
    risk = ToolRisk.READ

    async def check_permissions(
        self, arguments: dict[str, Any], context: PermissionContext
    ) -> PermissionResult:
        return _internal_state_permission("Team inbox is internal coordination state")

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        team_id = _team_id(arguments, self.session.team_id)
        mode = "write" if bool(arguments.get("unread_only", True)) else "read"
        return ConcurrencySpec((ResourceLock("member", _resource_id(team_id, self.session.agent_name), mode),))

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        store = _store(self.name, self.session.team_store)
        if isinstance(store, ToolResult):
            return store
        team_id = _team_id(arguments, self.session.team_id)
        try:
            messages = await store.read_inbox(
                team_id,
                self.session.agent_name,
                unread_only=bool(arguments.get("unread_only", True)),
            )
        except Exception as exc:  # noqa: BLE001
            return _team_error(self.name, exc)
        return ToolResult(self.name, _render(messages), metadata={"team_id": team_id, "count": len(messages)})


class TeamMessageSendTool(SessionAwareMixin, Tool):
    name = "team_message_send"
    description = "Send a file-backed message to another teammate's inbox using a file lock."
    input_schema = {
        "type": "object",
        "properties": {
            "team_id": {"type": "string", "description": "Team id; defaults to this teammate's current team."},
            "to": {"type": "string", "description": "Recipient teammate/leader name."},
            "content": {"type": "string"},
            "task_id": {"type": "string", "description": "Optional related task id."},
            "kind": {"type": "string", "description": "Message kind; defaults to message."},
        },
        "required": ["to", "content"],
    }
    risk = ToolRisk.WRITE

    async def check_permissions(
        self, arguments: dict[str, Any], context: PermissionContext
    ) -> PermissionResult:
        return _team_side_effect_permission(
            self.name, arguments, context, "sending a teammate message changes shared coordination state"
        )

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        team_id = _team_id(arguments, self.session.team_id)
        locks = [ResourceLock("member", _resource_id(team_id, arguments.get("to", "")), "write")]
        if arguments.get("task_id"):
            locks.append(ResourceLock("task", _resource_id(team_id, arguments["task_id"]), "read"))
        return ConcurrencySpec(tuple(locks))

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        store = _store(self.name, self.session.team_store)
        if isinstance(store, ToolResult):
            return store
        team_id = _team_id(arguments, self.session.team_id)
        try:
            message = await store.send_message(
                team_id,
                self.session.agent_name,
                str(arguments.get("to", "")),
                str(arguments.get("content", "")),
                task_id=str(arguments["task_id"]) if arguments.get("task_id") else None,
                kind=str(arguments["kind"]) if arguments.get("kind") else None,
            )
        except Exception as exc:  # noqa: BLE001
            return _team_error(self.name, exc)
        return ToolResult(self.name, _render(message), metadata={"team_id": team_id, "message_id": message["id"]})
