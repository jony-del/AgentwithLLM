from __future__ import annotations

import json
from typing import Any

from agent_core.models import ToolResult, ToolRisk
from agent_core.session import SessionAwareMixin
from agent_core.tools.base import ConcurrencySpec, ResourceLock, Tool
from agent_core.tools.catalog import builtin_tool


@builtin_tool
class PlanCodeTaskTool(SessionAwareMixin, Tool):
    name = "plan_code_task"
    description = "Create a dependency plan from bounded multilingual symbol and relation evidence. Queries should name relevant symbols. Coverage limits are reported."
    input_schema = {"type": "object", "properties": {"queries": {"type": "array", "minItems": 1, "maxItems": 4,
                    "items": {"type": "string", "minLength": 1, "maxLength": 256}}}, "required": ["queries"]}
    risk = ToolRisk.READ
    safely_cancellable = True
    execution_timeout = 600.0

    def concurrency_spec(self, arguments: dict[str, Any]) -> ConcurrencySpec:
        return ConcurrencySpec((ResourceLock("fs", str(self.session.workspace), "read", subtree=True, materialize=False),
                                ResourceLock("task", self.session.agent_id, "write")))

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        if self.session.plan_code_task is None:
            return ToolResult(self.name, "Code planner unavailable", ok=False)
        record = await self.session.plan_code_task(arguments["queries"])
        return ToolResult(self.name, json.dumps(record, ensure_ascii=False), metadata={"plan": record})
