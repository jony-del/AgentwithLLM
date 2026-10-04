"""Read-only independent review entry point."""
from __future__ import annotations

import json
from typing import Any

from agent_core.models import ToolResult, ToolRisk
from agent_core.session import SessionAwareMixin
from agent_core.tools.base import ConcurrencySpec, ResourceLock, Tool
from agent_core.tools.catalog import builtin_tool


@builtin_tool
class RunReviewTool(SessionAwareMixin, Tool):
    name = "run_review"
    description = "Run a fresh-context, tool-free review against the task source baseline. Findings are version-bound; fix blockers and rerun checks."
    input_schema = {"type": "object", "properties": {}}
    risk = ToolRisk.READ
    safely_cancellable = True
    execution_timeout = 600.0

    def concurrency_spec(self, arguments: dict[str, Any]) -> ConcurrencySpec:
        return ConcurrencySpec((ResourceLock("fs", str(self.session.workspace), "read", subtree=True, materialize=False),
                                ResourceLock("task", self.session.agent_id, "write")))

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        if self.session.task_run is None or self.session.review_task is None:
            return ToolResult(self.name, "Review unavailable", ok=False)
        record = await self.session.review_task(await self.session.capture_revision())
        return ToolResult(self.name, json.dumps(record, ensure_ascii=False), ok=record["status"] == "passed", metadata={"review": record})
