from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from agent_core.models import ToolRisk

if TYPE_CHECKING:
    from agent_core.models import ToolCall
    from agent_core.tools.base import Tool

# Asks the user about a pending tool. Args: tool name, risk value, arguments.
# Returns "once" (run now), "always" (allow for the rest of the session), or "deny".
Prompter = Callable[[str, str, dict[str, Any]], str]


class PermissionMode(str, Enum):
    DEFAULT = "default"
    ACCEPTEDITS = "acceptedits"
    PLAN = "plan"
    AUTO = "auto"
    DONTASK = "dontask"


@dataclass(slots=True)
class PermissionDecision:
    allowed: bool
    dry_run: bool = False
    ask_user: bool = False
    reason: str = ""


class PermissionPolicy:
    def __init__(
        self,
        mode: PermissionMode | str = PermissionMode.DEFAULT,
        prompter: Prompter | None = None,
    ) -> None:
        self.mode = PermissionMode(mode)
        # We can only ask the user when a prompter is wired (an interactive UI).
        # Without one, an "ask" collapses into a denial (matches old non-interactive).
        self.prompter = prompter
        self.interactive = prompter is not None
        # Tools the user chose to "always allow" for the lifetime of this session.
        self._session_allow: set[str] = set()

    def decide(self, tool: "Tool") -> PermissionDecision:
        risk = tool.risk
        if tool.name in self._session_allow:
            return PermissionDecision(True, reason="allowed for this session")
        if self.mode == PermissionMode.PLAN:
            if risk in {ToolRisk.WRITE, ToolRisk.DANGEROUS}:
                return PermissionDecision(True, dry_run=True, reason="plan mode dry-run")
            return PermissionDecision(True, reason="plan mode read allowed")
        if self.mode == PermissionMode.ACCEPTEDITS:
            if risk in {ToolRisk.READ, ToolRisk.WRITE}:
                return PermissionDecision(True, reason="acceptedits allows read/write")
            return self._maybe_ask("acceptedits requires confirmation for dangerous tools")
        if self.mode == PermissionMode.AUTO:
            if risk == ToolRisk.DANGEROUS:
                return PermissionDecision(False, reason="auto denies dangerous tools")
            return PermissionDecision(True, reason="auto allows read/write")
        if self.mode == PermissionMode.DONTASK:
            if risk == ToolRisk.DANGEROUS:
                return PermissionDecision(False, reason="dontask denies dangerous tools")
            return PermissionDecision(True, reason="dontask allows safe tools")
        if risk == ToolRisk.READ:
            return PermissionDecision(True, reason="default allows read tools")
        return self._maybe_ask("default requires confirmation")

    def confirm(self, decision: PermissionDecision, tool: "Tool", tool_call: "ToolCall") -> PermissionDecision:
        if not decision.ask_user or self.prompter is None:
            return decision
        choice = self.prompter(tool.name, tool.risk.value, tool_call.arguments)
        if choice == "always":
            self._session_allow.add(tool.name)
            return PermissionDecision(True, reason="user allowed for this session")
        if choice == "once":
            return PermissionDecision(True, reason="user confirmed")
        return PermissionDecision(False, reason="user rejected")

    def _maybe_ask(self, reason: str) -> PermissionDecision:
        if self.interactive:
            return PermissionDecision(False, ask_user=True, reason=reason)
        return PermissionDecision(False, reason=f"{reason}; non-interactive")
