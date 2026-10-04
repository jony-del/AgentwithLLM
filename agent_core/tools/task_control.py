"""Task planning and verification tools; evidence is produced by execution, never supplied by the model."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from agent_core.command_security import analyze_command
from agent_core.models import ToolResult, ToolRisk
from agent_core.permission_types import PermissionBehavior, PermissionContext, PermissionMode, PermissionResult
from agent_core.sandbox import SandboxAwareMixin, SandboxInvocation
from agent_core.session import SessionAwareMixin
from agent_core.task_runtime import PlanStep, VerificationCheck, VerificationEvidence
from agent_core.tools.base import ConcurrencySpec, ExecutionScope, ResourceLock, Tool
from agent_core.tools.catalog import builtin_tool


@builtin_tool
class UpdateTaskPlanTool(SessionAwareMixin, Tool):
    name = "update_task_plan"
    description = (
        "Set a dependency-checked task plan and, if not supplied by the caller, verification checks. "
        "Each check has id, kind and explicit argv. Update the plan after a failed attempt. "
        "Completed steps require completed dependencies; completion still requires executed verification."
    )
    input_schema = {"type": "object", "properties": {
        "steps": {"type": "array", "maxItems": 128, "items": {"type": "object", "properties": {
            "id": {"type": "string", "minLength": 1}, "description": {"type": "string", "minLength": 1},
            "depends_on": {"type": "array", "items": {"type": "string"}},
            "status": {"enum": ["pending", "in_progress", "completed", "blocked"]},
            "paths": {"type": "array", "maxItems": 64, "items": {"type": "string"}},
        }, "required": ["id", "description"]}},
        "checks": {"type": "array", "maxItems": 32, "items": {"type": "object", "properties": {
            "id": {"type": "string", "minLength": 1}, "argv": {"type": "array", "minItems": 1,
                "maxItems": 256, "items": {"type": "string"}},
            "kind": {"enum": ["test", "build", "lint", "typecheck", "check"]},
        }, "required": ["id", "argv"]}},
    }, "required": ["steps"]}
    risk = ToolRisk.READ

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        return ConcurrencySpec((ResourceLock("task", self.session.session_id, "write"),), exclusive=True)

    def _invoke(self, arguments: dict[str, Any]) -> ToolResult:
        task = self.session.task_run
        if task is None:
            return ToolResult(self.name, "No active task", ok=False)
        try:
            steps = [PlanStep(x["id"], x["description"], tuple(x.get("depends_on", ())), x.get("status", "pending"), tuple(x.get("paths", ())))
                     for x in arguments["steps"]]
            contract = task.contract
            if "checks" in arguments:
                checks = tuple(VerificationCheck(x["id"], tuple(x["argv"]), x.get("kind", "test"))
                               for x in arguments["checks"])
                if task.checks_locked and checks != contract.checks:
                    raise ValueError("caller-provided verification checks cannot be changed by the model")
                contract = replace(contract, checks=checks)
            task.replace_plan(steps)
            task.contract = contract
            self.session.persist_task()
        except (KeyError, TypeError, ValueError) as exc:
            return ToolResult(self.name, str(exc), ok=False, metadata={"error_type": "InvalidTaskPlan"})
        return ToolResult(self.name, task.context())


@builtin_tool
class RunVerificationTool(SessionAwareMixin, SandboxAwareMixin, Tool):
    name = "run_verification"
    description = "Execute one registered verification check with explicit argv. Records exit status against the workspace revision."
    input_schema = {"type": "object", "properties": {
        "check_id": {"type": "string"},
        "argv": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "string"}},
        "timeout": {"type": "integer", "minimum": 1, "maximum": 600},
    }, "required": ["check_id", "argv"]}
    risk = ToolRisk.DANGEROUS
    safely_cancellable = True
    execution_timeout = 600.0

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        return ConcurrencySpec((ResourceLock("fs", str(self.session.workspace), "write", subtree=True),
                                ResourceLock("task", self.session.session_id, "write")))

    async def check_permissions(self, arguments: dict[str, Any], context: PermissionContext) -> PermissionResult:
        import shlex
        argv = arguments.get("argv", [])
        analysis = analyze_command(shlex.join(str(x) for x in argv))
        if analysis.behavior is PermissionBehavior.DENY:
            return PermissionResult.deny(analysis.reason)
        if not analysis.bypass_immune and (context.mode is PermissionMode.BYPASS or
                (context.rules is not None and context.rules.allow_match(self.name, arguments) is not None)):
            return PermissionResult.passthrough("explicit verification invocation may be allowed by policy")
        return PermissionResult.ask("verification executes project code", classifier_approvable=True,
                                    bypass_immune=analysis.bypass_immune)

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        task = self.session.task_run
        supervisor = self.session.process_supervisor
        if task is None or supervisor is None:
            return ToolResult(self.name, "Task or process supervisor unavailable", ok=False)
        argv = tuple(str(x) for x in arguments["argv"])
        check = next((c for c in task.contract.checks if c.id == arguments["check_id"]), None)
        if check is None or check.argv != argv:
            return ToolResult(self.name, "Check argv does not match the task contract", ok=False,
                              metadata={"error_type": "VerificationContractMismatch"})
        before = await self.session.capture_revision()
        host = list(argv)
        guest = list(argv)
        capability = Path(argv[0]).stem.casefold()
        if capability in {"python", "python3", "python3.11", "python3.12"}:
            capability = "python"
            if not Path(argv[0]).is_absolute():
                host[0] = sys.executable
        required: tuple[str, ...] = ()
        if capability in {"python", "bash", "git", "node", "pwsh", "rg"}:
            guest[0] = "@" + capability
            required = (capability,)
        invocation = SandboxInvocation.create(host, guest_argv=guest, required_guest_capabilities=required,
                                              scope=ExecutionScope.for_workspace(self.session.workspace, network="deny"))
        spec, shell = self.sandbox.wrap_invocation(invocation)
        if shell or not isinstance(spec, (list, tuple)):
            return ToolResult(self.name, "verification requires explicit argv", ok=False)
        output = await supervisor.run_argv(list(spec), self.session.workspace, timeout=int(arguments.get("timeout", 300)))
        after = await self.session.capture_revision()
        state = str(output["state"]) if before.digest == after.digest else "revision_changed"
        code = output["exit_code"]
        evidence = VerificationEvidence(check.id, after.digest, after.workspace,
            hashlib.sha256(json.dumps(argv).encode()).hexdigest(), code if isinstance(code, int) else None,
            state, str(output["output_path"]))
        task.evidence.append(evidence)
        task.evidence = task.evidence[-100:]
        if state == "completed" and code == 0:
            task.checkpoints.append({"revision": after.digest, "workspace": after.workspace,
                                     "plan_revision": task.plan_revision, "check_id": check.id})
            task.checkpoints = task.checkpoints[-32:]
        await self.session.persist_task_async()
        return ToolResult(self.name, str(output.pop("output")), ok=state == "completed" and code == 0,
                          metadata={**output, "verification": {"check_id": check.id, "revision": after.digest, "state": state}})


@builtin_tool
class TaskStateTool(SessionAwareMixin, Tool):
    name = "task_state"
    description = "Read the durable task contract, checks, plan, attempts, evidence, review findings or checkpoints in bounded pages."
    input_schema = {"type": "object", "properties": {
        "section": {"enum": ["goal", "checks", "plan", "failures", "evidence", "reviews", "checkpoints"]},
        "offset": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 4096},
    }, "required": ["section"]}
    risk = ToolRisk.READ

    def _invoke(self, arguments: dict[str, Any]) -> ToolResult:
        task = self.session.task_run
        if task is None:
            return ToolResult(self.name, "No active task", ok=False)
        section = str(arguments["section"])
        offset = max(0, int(arguments.get("offset", 0)))
        if section == "goal":
            source = task.contract.goal
            limit = min(4096, int(arguments.get("limit", 4096)))
        else:
            source = ([asdict(x) for x in task.contract.checks] if section == "checks" else
                      [asdict(x) for x in task.plan] if section == "plan" else
                      [asdict(x) for x in task.evidence] if section == "evidence" else getattr(task, section))
            limit = min(8, int(arguments.get("limit", 8)))
        page = source[offset:offset + limit]
        return ToolResult(self.name, json.dumps({"section": section, "page": page,
            "total": len(source), "next_offset": offset + limit if offset + limit < len(source) else None}, ensure_ascii=False))
