"""Dedicated plan artifact and explicit plan-exit workflow tools."""

from __future__ import annotations

from typing import Any

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


@builtin_tool
class WritePlanTool(SessionAwareMixin, Tool):
    name = "write_plan"
    description = (
        "Write the complete implementation plan to the agent-owned plan artifact. "
        "This tool is available only in plan mode and does not accept a file path."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "maxLength": 262144},
        },
        "required": ["content"],
    }
    risk = ToolRisk.WRITE

    async def check_permissions(
        self, arguments: dict[str, Any], context: PermissionContext
    ) -> PermissionResult:
        if context.mode is not PermissionMode.PLAN or not context.plan_state.active:
            return PermissionResult.deny(
                "write_plan is available only during an active planning workflow",
                decision_source=DecisionSource.CENTRAL_SAFETY,
            )
        return PermissionResult.allow("dedicated plan artifact write")

    def concurrency_spec(self, arguments: dict[str, Any]) -> ConcurrencySpec:
        path = self.session.plan_state.artifact_path
        return ConcurrencySpec((ResourceLock("plan", str(path or "unbound"), "write"),), exclusive=True)

    def _invoke(self, arguments: dict[str, Any]) -> ToolResult:
        path = self.session.plan_state.artifact_path
        if not self.session.plan_state.active or path is None:
            return ToolResult(self.name, "No active plan workflow.", ok=False, metadata={"error_type": "Unavailable"})
        content = str(arguments.get("content", ""))
        try:
            self.session.plan_store.write(path, content)
        except (OSError, ValueError) as exc:
            return ToolResult(self.name, f"Plan write failed: {exc}", ok=False, metadata={"error_type": "PlanWrite"})
        return ToolResult(
            self.name,
            f"Plan artifact updated: {path}",
            metadata={"path": str(path), "chars": len(content)},
        )


@builtin_tool
class ExitPlanTool(SessionAwareMixin, Tool):
    name = "exit_plan"
    description = "Present the saved plan for approval and restore the mode active before plan mode."
    input_schema = {"type": "object", "properties": {}, "required": []}
    risk = ToolRisk.WRITE
    requires_user_interaction = True

    async def check_permissions(
        self, arguments: dict[str, Any], context: PermissionContext
    ) -> PermissionResult:
        if context.mode is not PermissionMode.PLAN or not context.plan_state.active:
            return PermissionResult.deny(
                "exit_plan can only be used from an active plan workflow",
                decision_source=DecisionSource.CENTRAL_SAFETY,
            )
        plan = self.session.plan_store.read(self.session.plan_state.artifact_path)
        if plan is None or not plan.strip():
            return PermissionResult.deny(
                "a non-empty plan artifact is required before exiting plan mode",
                decision_source=DecisionSource.TOOL,
            )
        return PermissionResult.ask(
            "approve the plan and exit plan mode",
            decision_source=DecisionSource.TOOL,
            metadata={"plan_chars": len(plan)},
            bypass_immune=True,
        )

    def concurrency_spec(self, arguments: dict[str, Any]) -> ConcurrencySpec:
        return ConcurrencySpec((ResourceLock("session", "permission_mode", "write"),), exclusive=True)

    def _invoke(self, arguments: dict[str, Any]) -> ToolResult:
        state = self.session.plan_state
        plan = self.session.plan_store.read(state.artifact_path)
        if plan is None or not plan.strip():
            return ToolResult(self.name, "No non-empty plan artifact exists.", ok=False, metadata={"error_type": "NoPlan"})
        previous = state.previous_mode or PermissionMode.DEFAULT.value
        setter = self.session.permission_mode_setter
        if setter is None:
            return ToolResult(self.name, "Permission mode switching is unavailable.", ok=False)
        path = state.artifact_path
        setter(previous, source="exit_plan")
        return ToolResult(
            self.name,
            f"Plan approved; restored permission mode {previous}.\n\n## Approved Plan\n{plan}",
            metadata={"path": str(path) if path else "", "restored_mode": previous},
        )
