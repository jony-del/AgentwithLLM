"""Local control-plane tools that reuse existing session/team/MCP state."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from agent_core.agents.team import FileLock
from agent_core.config import load_agent_toml, user_settings_path
from agent_core.models import ToolRisk, ToolResult
from agent_core.permission_types import DecisionSource, PermissionContext, PermissionMode, PermissionResult
from agent_core.session import SessionAwareMixin
from agent_core.tools.base import Tool
from agent_core.tools.catalog import builtin_tool
from agent_core.tools.registry import RegistryAwareMixin
from agent_core.tools.team import _render, _store, _team_error


@builtin_tool
class ToolSearchTool(RegistryAwareMixin, Tool):
    name = "tool_search"
    description = "Search available tools and activate matching deferred capabilities for this agent."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "required": ["query"],
    }
    risk = ToolRisk.READ

    def _invoke(self, arguments: dict[str, object]) -> ToolResult:
        if self.registry is None:
            return ToolResult(self.name, "Tool registry is unavailable.", ok=False)
        try:
            results = self.registry.search(
                str(arguments.get("query", "")),
                max_results=int(str(arguments.get("max_results", 8))),
            )
        except (RuntimeError, ValueError) as exc:
            return ToolResult(self.name, f"Tool search failed: {exc}", ok=False)
        return ToolResult(self.name, json.dumps(results, indent=2, ensure_ascii=False), metadata={"count": len(results)})


@builtin_tool
class AskUserQuestionTool(SessionAwareMixin, Tool):
    name = "ask_user_question"
    description = "Ask the live user up to three structured questions and return their answers."
    input_schema = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array", "minItems": 1, "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "question": {"type": "string"},
                        "options": {
                            "type": "array", "minItems": 2, "maxItems": 3,
                            "items": {
                                "type": "object",
                                "properties": {"label": {"type": "string"}, "description": {"type": "string"}},
                                "required": ["label", "description"],
                            },
                        },
                    },
                    "required": ["id", "question", "options"],
                },
            }
        },
        "required": ["questions"],
    }
    risk = ToolRisk.READ
    requires_user_interaction = True

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        raw = arguments.get("questions")
        if not isinstance(raw, list) or not 1 <= len(raw) <= 3 or self.session.ask_user is None:
            return ToolResult(self.name, "A live question callback is unavailable.", ok=False)
        questions = [dict(item) for item in raw if isinstance(item, dict)]
        answers = await self.session.ask_user(questions)
        return ToolResult(self.name, json.dumps(answers, ensure_ascii=False, indent=2), metadata={"count": len(answers)})


@builtin_tool
class EnterPlanTool(SessionAwareMixin, Tool):
    name = "enter_plan"
    description = "Enter the existing plan-mode workflow and allocate the session-owned plan artifact."
    input_schema = {"type": "object", "properties": {}, "required": []}
    risk = ToolRisk.WRITE

    async def check_permissions(self, arguments: dict[str, Any], context: PermissionContext) -> PermissionResult:
        if context.mode is PermissionMode.PLAN:
            return PermissionResult.allow("plan mode is already active")
        return PermissionResult.allow("entering plan mode only narrows permissions", decision_source=DecisionSource.TOOL)

    def _invoke(self, arguments: dict[str, object]) -> ToolResult:
        if self.session.plan_state.active:
            return ToolResult(self.name, "Plan mode is already active.")
        setter = self.session.permission_mode_setter
        if setter is None:
            return ToolResult(self.name, "Permission mode switching is unavailable.", ok=False)
        setter(PermissionMode.PLAN.value, source="enter_plan")
        path = self.session.plan_store.path_for(self.session.session_id, self.session.agent_id)
        return ToolResult(self.name, f"Entered plan mode. Plan artifact: {path}", metadata={"path": str(path)})


class _TeamReadTool(SessionAwareMixin, Tool):
    risk = ToolRisk.READ

    async def check_permissions(self, arguments: dict[str, Any], context: PermissionContext) -> PermissionResult:
        return PermissionResult.allow("internal team state read", decision_source=DecisionSource.TOOL)


@builtin_tool
class TaskGetTool(_TeamReadTool):
    name = "task_get"
    description = "Read one task from the current TeamStore."
    input_schema = {
        "type": "object", "properties": {"team_id": {"type": "string"}, "task_id": {"type": "string"}},
        "required": ["team_id", "task_id"],
    }

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        store = _store(self.name, self.session.team_store)
        if isinstance(store, ToolResult):
            return store
        try:
            task = await store.get_task(str(arguments.get("team_id", "")), str(arguments.get("task_id", "")))
        except Exception as exc:
            return _team_error(self.name, exc)
        return ToolResult(self.name, _render(task), metadata={"task_id": task["id"]})


@builtin_tool
class TaskListTool(_TeamReadTool):
    name = "task_list"
    description = "List tasks from a team, optionally filtered by owner or status."
    input_schema = {
        "type": "object",
        "properties": {"team_id": {"type": "string"}, "owner": {"type": "string"}, "status": {"type": "string"}},
        "required": ["team_id"],
    }

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        store = _store(self.name, self.session.team_store)
        if isinstance(store, ToolResult):
            return store
        try:
            tasks = await store.list_tasks(str(arguments.get("team_id", "")))
        except Exception as exc:
            return _team_error(self.name, exc)
        for key in ("owner", "status"):
            if arguments.get(key):
                tasks = [task for task in tasks if task.get(key) == str(arguments[key])]
        return ToolResult(self.name, _render(tasks), metadata={"count": len(tasks)})


@builtin_tool
class TeamDeleteTool(SessionAwareMixin, Tool):
    name = "team_delete"
    description = "Permanently delete one TeamStore team and its tasks/inboxes."
    input_schema = {"type": "object", "properties": {"team_id": {"type": "string"}}, "required": ["team_id"]}
    risk = ToolRisk.DANGEROUS
    requires_user_interaction = True

    async def check_permissions(self, arguments: dict[str, Any], context: PermissionContext) -> PermissionResult:
        return PermissionResult.ask("deleting team state is permanent", bypass_immune=True, classifier_approvable=False)

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        store = _store(self.name, self.session.team_store)
        if isinstance(store, ToolResult):
            return store
        team_id = str(arguments.get("team_id", ""))
        try:
            await store.delete_team(team_id)
        except Exception as exc:
            return _team_error(self.name, exc)
        if self.session.team_id == team_id:
            self.session.team_id = None
        return ToolResult(self.name, f"Deleted team {team_id}")


_CONFIG_SETTINGS = {
    "model", "provider", "effort", "permission",
    "tools.shell.timeout", "tools.shell.auto_background_seconds",
    "tools.shell.max_tasks", "tools.lsp.autodetect",
}


def _nested_get(document: Any, setting: str) -> Any:
    current = document
    for part in setting.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _write_setting(path: Path, setting: str, value: object) -> None:
    try:
        import tomlkit
    except ModuleNotFoundError as exc:
        raise RuntimeError("config writes require the core tomlkit dependency") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path.with_suffix(path.suffix + ".lock")):
        document = tomlkit.parse(path.read_text(encoding="utf-8")) if path.exists() else tomlkit.document()
        current = document
        parts = setting.split(".")
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = tomlkit.table()
            current = current[part]
        current[parts[-1]] = value
        encoded = tomlkit.dumps(document).encode("utf-8")
        fd, name = tempfile.mkstemp(prefix=".settings.", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


@builtin_tool
class ConfigTool(Tool):
    name = "config"
    description = "Read effective configuration or write one allowlisted user-level setting."
    deferred = True
    input_schema = {
        "type": "object",
        "properties": {"setting": {"type": "string"}, "value": {}},
        "required": ["setting"],
    }
    risk = ToolRisk.WRITE

    async def check_permissions(self, arguments: dict[str, Any], context: PermissionContext) -> PermissionResult:
        if "value" not in arguments:
            return PermissionResult.allow("configuration reads are local and read-only")
        if str(arguments.get("setting", "")) not in _CONFIG_SETTINGS:
            return PermissionResult.deny("setting is not in the user-writable allowlist")
        return PermissionResult.ask(
            "writing user-level configuration requires confirmation",
            bypass_immune=True, classifier_approvable=False,
        )

    def _invoke(self, arguments: dict[str, object]) -> ToolResult:
        setting = str(arguments.get("setting", ""))
        if "value" not in arguments:
            value = _nested_get(load_agent_toml(), setting)
            return ToolResult(self.name, json.dumps({"setting": setting, "value": value}, default=str))
        if setting not in _CONFIG_SETTINGS:
            return ToolResult(self.name, "Setting is not writable.", ok=False)
        try:
            _write_setting(user_settings_path(), setting, arguments.get("value"))
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            return ToolResult(self.name, f"Config write failed: {exc}", ok=False)
        return ToolResult(self.name, f"Updated user setting {setting}", metadata={"path": str(user_settings_path())})


def _resource_payload(value: Any) -> object:
    if is_dataclass(value):
        return asdict(value)  # type: ignore[arg-type]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    return str(value)


@builtin_tool
class ListMCPResourcesTool(SessionAwareMixin, Tool):
    name = "list_mcp_resources"
    description = "List resources through the session's existing MCP connections."
    deferred = True
    input_schema = {"type": "object", "properties": {"server": {"type": "string"}}, "required": []}
    risk = ToolRisk.READ

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        manager = self.session.mcp_manager
        if manager is None:
            return ToolResult(self.name, "No MCP manager is connected.", ok=False)
        try:
            resources = await asyncio.to_thread(manager.list_resources, arguments.get("server"))
        except Exception as exc:
            return ToolResult(self.name, f"MCP resource listing failed: {exc}", ok=False)
        return ToolResult(self.name, json.dumps(resources, ensure_ascii=False, indent=2), metadata={"count": len(resources)})


@builtin_tool
class ReadMCPResourceTool(SessionAwareMixin, Tool):
    name = "read_mcp_resource"
    description = "Read one resource through an existing MCP server connection."
    deferred = True
    input_schema = {
        "type": "object", "properties": {"server": {"type": "string"}, "uri": {"type": "string"}},
        "required": ["server", "uri"],
    }
    risk = ToolRisk.READ

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        manager = self.session.mcp_manager
        if manager is None:
            return ToolResult(self.name, "No MCP manager is connected.", ok=False)
        try:
            response = await asyncio.to_thread(
                manager.read_resource, str(arguments.get("server", "")), str(arguments.get("uri", ""))
            )
        except Exception as exc:
            return ToolResult(self.name, f"MCP resource read failed: {exc}", ok=False)
        content = json.dumps(_resource_payload(response), ensure_ascii=False, indent=2, default=str)
        if len(content) > 100_000:
            content = content[:100_000] + "\n[MCP resource truncated]"
        return ToolResult(self.name, content)
