"""Task-planning tool — a Claude-Code-style ``TodoWrite``.

``update_todos`` lets the model maintain a structured, multi-step plan that persists
across tool turns (in ``SessionContext.todos``) and is surfaced to the live UI. It is
session-aware (``SessionAwareMixin``) rather than workspace-scoped: nothing touches disk.
"""

from __future__ import annotations

from agent_core.models import ToolRisk, ToolResult
from agent_core.session import SessionAwareMixin
from agent_core.tools.base import ConcurrencySpec, ResourceLock, Tool
from agent_core.tools.catalog import builtin_tool


@builtin_tool
class UpdateTodosTool(SessionAwareMixin, Tool):
    name = "update_todos"
    description = (
        "Record or update a structured to-do list for the current task. Pass the COMPLETE "
        "list each time (it replaces the previous one). Use this to plan multi-step work and "
        "to keep exactly one item 'in_progress' as you go. Statuses: pending, in_progress, "
        "completed."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "The full to-do list, in order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "What the step is."},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                            "description": "Defaults to pending if omitted.",
                        },
                    },
                    "required": ["content"],
                },
            }
        },
        "required": ["todos"],
    }
    risk = ToolRisk.READ

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        # This tool notifies the UI from its body, so keep it on the serial path.
        return ConcurrencySpec((ResourceLock("session", "todos", "write"),), exclusive=True)

    def run(self, arguments: dict[str, object]) -> ToolResult:
        todos = arguments.get("todos")
        if not isinstance(todos, list):
            return ToolResult(self.name, "todos must be a list", ok=False, metadata={"error_type": "BadArgs"})
        items = self.session.todos.replace(todos)
        self.session.notify_todos()
        return ToolResult(
            self.name,
            self.session.todos.render(),
            metadata={"count": len(items)},
        )
