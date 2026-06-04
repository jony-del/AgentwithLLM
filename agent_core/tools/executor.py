from __future__ import annotations

from dataclasses import asdict

from agent_core.hooks import HookPipeline
from agent_core.models import ToolCall, ToolResult
from agent_core.permissions import PermissionPolicy
from agent_core.storage import JSONLRunLogger
from agent_core.tools.registry import ToolRegistry
from agent_core.ui import AgentUI, NullUI


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        permissions: PermissionPolicy,
        hooks: HookPipeline | None = None,
        logger: JSONLRunLogger | None = None,
        ui: AgentUI | None = None,
    ) -> None:
        self.registry = registry
        self.permissions = permissions
        self.hooks = hooks or HookPipeline()
        self.logger = logger
        self.ui = ui or NullUI()

    def execute(self, tool_call: ToolCall) -> ToolResult:
        rewritten_call, pre_results = self.hooks.run_pre(tool_call)
        if self.logger:
            self.logger.write(
                "tool_pre",
                {
                    "tool_call": asdict(rewritten_call),
                    "pre_results": [asdict(result) for result in pre_results],
                },
            )
        if any(not result.allowed for result in pre_results):
            result = ToolResult(rewritten_call.name, "Tool rejected by pre hook", ok=False)
            return self._finish(rewritten_call, result, None)

        try:
            tool = self.registry.get(rewritten_call.name)
        except KeyError:
            result = ToolResult(
                rewritten_call.name,
                f"Unknown tool: {rewritten_call.name}",
                ok=False,
                metadata={"error_type": "UnknownTool"},
            )
            return self._finish(rewritten_call, result, "unknown tool")

        self.ui.on_tool_call(tool.name, tool.risk.value, rewritten_call.arguments)
        decision = self.permissions.decide(tool)
        decision = self.permissions.confirm(decision, tool, rewritten_call)
        if self.logger:
            self.logger.write("permission", {"tool": tool.name, "decision": asdict(decision)})
        if not decision.allowed:
            result = ToolResult(tool.name, f"Tool denied: {decision.reason}", ok=False)
            return self._finish(rewritten_call, result, decision.reason)
        if decision.dry_run:
            result = ToolResult(tool.name, f"Dry-run: would execute {tool.name} with {rewritten_call.arguments}")
            return self._finish(rewritten_call, result, decision.reason)

        try:
            result = tool.run(rewritten_call.arguments)
        except Exception as exc:
            result = ToolResult(tool.name, f"Tool error: {exc}", ok=False, metadata={"error_type": type(exc).__name__})
        result = self.hooks.run_post(rewritten_call, result)
        return self._finish(rewritten_call, result, decision.reason)

    def _finish(self, tool_call: ToolCall, result: ToolResult, reason: str | None) -> ToolResult:
        """Log, surface the observation to the UI, and return — one exit for every path."""
        self._log_result(tool_call, result, reason)
        self.ui.on_tool_result(result)
        return result

    def _log_result(self, tool_call: ToolCall, result: ToolResult, reason: str | None) -> None:
        if self.logger:
            self.logger.write(
                "tool_result",
                {"tool_call": asdict(tool_call), "result": asdict(result), "reason": reason},
            )
