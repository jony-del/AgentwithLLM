"""Read-only inspection of the unified background registry."""

from __future__ import annotations

import json

from agent_core.models import ToolResult, ToolRisk
from agent_core.session import SessionAwareMixin
from agent_core.tools.base import ConcurrencySpec, Tool
from agent_core.tools.catalog import builtin_tool


@builtin_tool
class BackgroundTasksTool(SessionAwareMixin, Tool):
    name = "background_tasks"
    description = "List background shell/agent/teammate handles, states and result metadata."
    input_schema = {"type": "object", "properties": {}}
    risk = ToolRisk.READ

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        return ConcurrencySpec(())

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        manager = self.session.background_tasks
        records = manager.snapshots() if manager is not None else []
        for record in records:
            record.pop("result", None)
        return ToolResult(self.name, json.dumps(records, ensure_ascii=False))
