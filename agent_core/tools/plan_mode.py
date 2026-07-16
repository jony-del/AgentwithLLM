"""Dedicated plan artifact and explicit plan-exit workflow tools."""

from __future__ import annotations

from typing import Any

from agent_core.command_security import analyze_command
from agent_core.models import ToolRisk, ToolResult
from agent_core.permission_rules import parse_rule
from agent_core.permission_types import (
    DecisionSource,
    PermissionBehavior,
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
    input_schema = {
        "type": "object",
        "properties": {
            "requested_permissions": {
                "type": "array",
                "maxItems": 32,
                "items": {
                    "type": "object",
                    "properties": {
                        "rule": {"type": "string", "minLength": 1, "maxLength": 1024},
                        "reason": {"type": "string", "minLength": 1, "maxLength": 500},
                    },
                    "required": ["rule", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": [],
        "additionalProperties": False,
    }
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
        try:
            requested = self._validated_requested_permissions(arguments)
        except ValueError as exc:
            return PermissionResult.deny(
                f"invalid requested permission bundle: {exc}",
                decision_source=DecisionSource.CENTRAL_SAFETY,
            )
        return PermissionResult.ask(
            (
                "approve the plan and exit plan mode"
                + (f" with {len(requested)} scoped session grant(s)" if requested else "")
            ),
            decision_source=DecisionSource.TOOL,
            metadata={"plan_chars": len(plan), "requested_permission_count": len(requested)},
            bypass_immune=True,
        )

    def _validated_requested_permissions(self, arguments: dict[str, Any]) -> tuple[str, ...]:
        raw_items = arguments.get("requested_permissions", [])
        if not isinstance(raw_items, list) or len(raw_items) > 32:
            raise ValueError("requested_permissions must contain at most 32 items")
        registered = self.session.registered_tool_names
        accepted: list[str] = []
        for item in raw_items:
            if not isinstance(item, dict):
                raise ValueError("each request must contain rule and reason")
            rule_text = str(item.get("rule", "")).strip()
            reason = str(item.get("reason", "")).strip()
            if not reason:
                raise ValueError("each requested rule needs a reason")
            parsed = parse_rule(rule_text)
            if parsed is None or parsed.content is None:
                raise ValueError(f"blanket or malformed rule is not allowed: {rule_text!r}")
            if parsed.tool_name not in registered:
                raise ValueError(f"tool is not registered: {parsed.tool_name}")
            content = parsed.content
            if "*" in content or "?" in content or "$(" in content or "`" in content:
                raise ValueError(f"wildcard/dynamic scope is not allowed: {rule_text!r}")
            if parsed.tool_name == "run_command":
                analysis = analyze_command(content)
                if analysis.behavior is PermissionBehavior.DENY or analysis.category in {
                    "destructive", "dynamic", "environment", "persistence", "protected", "secret"
                }:
                    raise ValueError(f"unsafe shell grant ({analysis.category}): {rule_text!r}")
            elif parsed.tool_name.startswith("web_") and not content.startswith("domain:"):
                raise ValueError("web grants must use domain:<host> scope")
            elif any(token in content.casefold() for token in (".git", ".env", ".ssh", "agent.toml")):
                raise ValueError(f"protected path scope is not allowed: {rule_text!r}")
            accepted.append(rule_text)
        return tuple(dict.fromkeys(accepted))

    def concurrency_spec(self, arguments: dict[str, Any]) -> ConcurrencySpec:
        return ConcurrencySpec((ResourceLock("session", "permission_mode", "write"),), exclusive=True)

    def _invoke(self, arguments: dict[str, Any]) -> ToolResult:
        state = self.session.plan_state
        plan = self.session.plan_store.read(state.artifact_path)
        if plan is None or not plan.strip():
            return ToolResult(self.name, "No non-empty plan artifact exists.", ok=False, metadata={"error_type": "NoPlan"})
        previous = state.previous_mode or PermissionMode.DEFAULT.value
        try:
            requested = self._validated_requested_permissions(arguments)
        except ValueError as exc:
            return ToolResult(self.name, f"Permission bundle rejected: {exc}", ok=False)
        grant = self.session.permission_grant_setter
        if requested and grant is None:
            return ToolResult(self.name, "Session permission grants are unavailable.", ok=False)
        try:
            for rule in requested:
                assert grant is not None
                grant(rule)
        except (OSError, ValueError) as exc:
            return ToolResult(self.name, f"Session permission grant failed: {exc}", ok=False)
        setter = self.session.permission_mode_setter
        if setter is None:
            return ToolResult(self.name, "Permission mode switching is unavailable.", ok=False)
        path = state.artifact_path
        setter(previous, source="exit_plan")
        return ToolResult(
            self.name,
            f"Plan approved; restored permission mode {previous}.\n\n## Approved Plan\n{plan}",
            metadata={
                "path": str(path) if path else "",
                "restored_mode": previous,
                "approved_permission_count": len(requested),
            },
        )
