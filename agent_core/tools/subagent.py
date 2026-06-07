"""Sub-agent dispatch — the Task/Explore-style fan-out tool.

``dispatch_agent`` hands a self-contained sub-task to a fresh ``ReActAgent`` with a clean
context and a narrowed tool set, returning only its final answer. This keeps the main
agent's context lean while still letting it delegate open-ended investigations.

The actual child is built by ``SessionContext.subagent_factory`` (injected by
``ReActAgent`` as a closure), so this tool never imports the agent class — no import
cycle. Recursion is prevented two ways: the child never receives this tool, and a depth
ceiling on the session is enforced by the factory.
"""

from __future__ import annotations

from agent_core.models import ToolRisk, ToolResult
from agent_core.session import SessionAwareMixin
from agent_core.tools.base import Tool
from agent_core.tools.catalog import builtin_tool

_PRESETS = {"read_only", "full"}


@builtin_tool
class DispatchAgentTool(SessionAwareMixin, Tool):
    name = "dispatch_agent"
    description = (
        "Delegate a self-contained sub-task to a fresh sub-agent that has its own clean "
        "context and returns a concise result. Ideal for open-ended investigation (e.g. "
        "'find everywhere X is configured and summarise') so the main context stays focused. "
        "tool_preset 'read_only' (default) gives the child read/search tools only; 'full' also "
        "allows file writes. The child cannot itself dispatch sub-agents."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "A complete, standalone description of the sub-task."},
            "tool_preset": {
                "type": "string",
                "enum": ["read_only", "full"],
                "description": "Capability level for the child (default read_only).",
            },
        },
        "required": ["task"],
    }
    risk = ToolRisk.WRITE

    def run(self, arguments: dict[str, object]) -> ToolResult:
        task = str(arguments.get("task", "")).strip()
        if not task:
            return ToolResult(self.name, "task must not be empty", ok=False, metadata={"error_type": "BadArgs"})
        preset = str(arguments.get("tool_preset", "read_only"))
        if preset not in _PRESETS:
            preset = "read_only"

        factory = self.session.subagent_factory
        if factory is None:
            return ToolResult(
                self.name,
                "Sub-agents are not available in this context.",
                ok=False,
                metadata={"error_type": "Unavailable"},
            )
        try:
            answer = factory(task, preset)
        except Exception as exc:  # noqa: BLE001 - a child failure must not crash the parent run
            return ToolResult(self.name, f"Sub-agent error: {type(exc).__name__}: {exc}", ok=False)
        return ToolResult(self.name, answer, metadata={"preset": preset})
