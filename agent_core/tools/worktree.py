from __future__ import annotations

import json
from typing import Any

from agent_core.models import ToolRisk, ToolResult
from agent_core.permission_types import PermissionContext, PermissionResult
from agent_core.session import SessionAwareMixin
from agent_core.tools.base import ConcurrencySpec, ResourceLock, Tool
from agent_core.tools.catalog import builtin_tool


@builtin_tool
class EnterWorktreeTool(SessionAwareMixin, Tool):
    name = "enter_worktree"
    description = "Create a session-owned Git worktree and atomically switch this session into it."
    deferred = True
    input_schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": []}
    risk = ToolRisk.WRITE

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        return ConcurrencySpec((ResourceLock("session", "workspace", "write"),), exclusive=True)

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        manager = self.session.worktree_manager
        if manager is None:
            return ToolResult(self.name, "Worktree manager was not bound to the session.", ok=False)
        try:
            state = await manager.create_and_enter(str(arguments["name"]) if arguments.get("name") else None)
            details = await manager.summary(state)
        except (OSError, ValueError, RuntimeError) as exc:
            return ToolResult(self.name, f"Worktree creation refused: {exc}", ok=False)
        return ToolResult(self.name, json.dumps(details, indent=2), metadata=details)


@builtin_tool
class ExitWorktreeTool(SessionAwareMixin, Tool):
    name = "exit_worktree"
    description = "Leave the active session-owned worktree, keeping it or safely removing it."
    deferred = True
    input_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["keep", "remove"]},
            "discard_changes": {"type": "boolean"},
        },
        "required": [],
    }
    risk = ToolRisk.DANGEROUS

    async def check_permissions(self, arguments: dict[str, Any], context: PermissionContext) -> PermissionResult:
        if str(arguments.get("action", "keep")) == "remove" or arguments.get("discard_changes"):
            return PermissionResult.ask(
                "removing a worktree or discarding changes requires interactive confirmation",
                bypass_immune=True, classifier_approvable=False,
            )
        return PermissionResult.allow("keeping a worktree preserves all work")

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        return ConcurrencySpec((ResourceLock("session", "workspace", "write"),), exclusive=True)

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        manager = self.session.worktree_manager
        if manager is None:
            return ToolResult(self.name, "No session-owned worktree is active.", ok=False)
        try:
            details = await manager.exit(
                str(arguments.get("action", "keep")), discard_changes=bool(arguments.get("discard_changes", False))
            )
        except (OSError, ValueError, RuntimeError) as exc:
            return ToolResult(self.name, f"Worktree exit refused: {exc}", ok=False)
        return ToolResult(self.name, json.dumps(details, indent=2), metadata=details)
