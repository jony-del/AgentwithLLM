from __future__ import annotations

import hashlib
import json
from typing import Any, cast

from agent_core.models import ToolRisk, ToolResult
from agent_core.permission_types import DecisionSource, PermissionContext, PermissionResult
from agent_core.session import SessionAwareMixin
from agent_core.sandbox import SandboxAwareMixin
from agent_core.tools.base import Tool
from agent_core.tools.catalog import builtin_tool
from agent_core.workflow_runtime import WorkflowError, WorkflowRuntime


@builtin_tool
class WorkflowRunTool(SessionAwareMixin, SandboxAwareMixin, Tool):
    name = "workflow_run"
    description = "Run an installed plugin workflow in an isolated Node orchestration runtime."
    deferred = True
    input_schema = {
        "type": "object",
        "properties": {
            "workflow": {"type": "string"},
            "args": {"type": "object"},
        },
        "required": ["workflow"],
        "additionalProperties": False,
    }
    risk = ToolRisk.DANGEROUS

    async def check_permissions(
        self, arguments: dict[str, Any], context: PermissionContext
    ) -> PermissionResult:
        workflows = self.session.plugin_workflows
        name = str(arguments.get("workflow") or "")
        source = workflows.get(name)
        if source is None:
            return PermissionResult.deny("unknown plugin workflow", decision_source=DecisionSource.TOOL)
        if not context.sandbox.enabled:
            return PermissionResult.deny(
                "plugin workflows require an enforcing sandbox backend",
                decision_source=DecisionSource.TOOL,
            )
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if digest in self.session.approved_workflow_digests:
            return PermissionResult.allow(
                "this exact plugin workflow digest was confirmed earlier in the session",
                decision_source=DecisionSource.TOOL,
            )
        return PermissionResult.ask(
            f"run plugin workflow {name} (digest {digest[:12]})",
            decision_source=DecisionSource.TOOL,
            bypass_immune=True,
            metadata={"workflow": name, "digest": digest},
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        source = self.session.plugin_workflows.get(str(arguments.get("workflow") or ""))
        factory = self.session.subagent_factory
        if source is None or factory is None:
            return ToolResult(self.name, "Workflow or sub-agent runtime is unavailable.", ok=False)
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        self.session.approved_workflow_digests.add(digest)
        try:
            result = await WorkflowRuntime().run(
                source,
                dict(arguments.get("args") or {}),
                cast(Any, factory),
                sandbox=self.sandbox,
                workspace=self.session.workspace,
                require_sandbox=True,
            )
        except (WorkflowError, OSError, TimeoutError) as exc:
            return ToolResult(self.name, f"Workflow failed: {exc}", ok=False)
        return ToolResult(self.name, json.dumps(result, ensure_ascii=False, default=str))
