"""Behavioral verification entry point and restricted internal probe executor."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
import sys
from typing import Any
import uuid

from agent_core.execution import ExecutionScope
from agent_core.models import ToolCall, ToolResult, ToolRisk
from agent_core.mcp.adapter import MCPTool
from agent_core.permission_types import PermissionContext, PermissionResult
from agent_core.sandbox import SandboxInvocation
from agent_core.session import SessionAwareMixin
from agent_core.task_runtime import WorkspaceRevision
from agent_core.tools.base import ConcurrencySpec, ResourceLock, Tool
from agent_core.tools.builtin import ReadTextFileTool
from agent_core.tools.catalog import builtin_tool
from agent_core.tools.codeintel import path_allowed
from agent_core.tools.executor import ToolExecutor
from agent_core.tools.registry import ToolRegistry
from agent_core.tools.task_control import RunVerificationTool
from agent_core.verifier import bounded_thread, source_unchanged


@builtin_tool
class RunVerifierTool(SessionAwareMixin, Tool):
    name = "run_verifier"
    description = ("Run independent functional/adversarial probes in a sandboxed source copy. "
                   "Use after a completed stage or before completion; inspect findings and fix blockers. "
                   "A real sandbox is required. The verifier cannot edit the project.")
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}
    risk = ToolRisk.READ
    safely_cancellable = True
    execution_timeout = 600.0

    def concurrency_spec(self, arguments: dict[str, Any]) -> ConcurrencySpec:
        return ConcurrencySpec((ResourceLock("fs", str(self.session.workspace), "read", subtree=True),
                                ResourceLock("task", self.session.session_id, "write")), exclusive=True)

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        if self.session.task_run is None or self.session.run_verifier is None:
            return ToolResult(self.name, "Verifier unavailable", ok=False)
        record = await self.session.run_verifier(await self.session.capture_revision())
        return ToolResult(self.name, json.dumps(record, ensure_ascii=False), ok=record["status"] == "passed",
                          metadata={"verifier": record})


class _VerifierRead(ReadTextFileTool):
    def __init__(self, workspace: Path, session: Any):
        super().__init__(workspace)
        self.origin_session = session

    async def check_permissions(self, arguments: dict[str, Any], context: PermissionContext) -> PermissionResult:
        relative = str(arguments.get("path", ""))
        if not path_allowed(self.origin_session, relative, "run_verifier"):
            return PermissionResult.deny("verifier read restricted by original policy")
        return await super().check_permissions(arguments, context)


class VerifierProbeTool(RunVerificationTool):
    # Only exposed in the verifier's private registry, never in the main registry.
    name = "verifier_probe"
    description = "Execute an assertion-bearing CLI/API/test probe. Record expected vs actual and a framework evidence ID."
    input_schema = {"type": "object", "properties": {
        "argv": {"type": "array", "minItems": 1, "maxItems": 256,
                 "items": {"type": "string", "maxLength": 16384}},
        "kind": {"enum": ["functional", "adversarial"]},
        "criterion": {"type": "string", "minLength": 1, "maxLength": 2000},
        "expected_exit_code": {"type": "integer", "minimum": 0, "maximum": 255},
        "expected_output": {"type": "string", "minLength": 1, "maxLength": 4000},
        "timeout": {"type": "integer", "minimum": 1, "maximum": 600},
    }, "required": ["argv", "kind", "criterion", "expected_exit_code", "expected_output"],
       "additionalProperties": False}

    def __init__(self, session: Any, sandbox: Any, copy: Path, revision: WorkspaceRevision,
                 probes: list[dict[str, Any]]):
        self.bind_session(session)
        self.bind_sandbox(sandbox)
        self.copy = copy
        self.revision = revision
        self.probes = probes

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        if not self.sandbox.is_enabled():
            return ToolResult(self.name, "Behavioral probes require a prepared real sandbox", ok=False)
        argv = list(arguments["argv"])
        if sum(map(len, argv)) > 16_384:
            return ToolResult(self.name, "Probe argv budget exceeded", ok=False)
        before = await self.session.capture_revision()
        if before.digest != self.revision.digest or before.workspace != self.revision.workspace:
            return ToolResult(self.name, "Original workspace changed before probe", ok=False)
        if not await bounded_thread(source_unchanged, self.revision, self.copy):
            return ToolResult(self.name, "Verifier copy source changed", ok=False)
        host, guest = list(argv), list(argv)
        capability = Path(argv[0]).stem.casefold()
        if capability in {"python", "python3", "python3.11", "python3.12"}:
            capability = "python"
            if not Path(argv[0]).is_absolute():
                host[0] = sys.executable
        required: tuple[str, ...] = ()
        if capability in {"python", "bash", "git", "node", "pwsh", "rg"}:
            guest[0] = "@" + capability
            required = (capability,)
        scope = ExecutionScope.for_workspace(self.copy, read_only_roots=(self.session.workspace,),
                                            private_temp=self.copy.parent, network="deny")
        invocation = SandboxInvocation.create(host, guest_argv=guest,
                    required_guest_capabilities=required, scope=scope)
        command, shell = self.sandbox.wrap_invocation(invocation)
        if shell or not isinstance(command, (list, tuple)):
            return ToolResult(self.name, "Probe requires explicit argv", ok=False)
        supervisor = self.session.process_supervisor
        if supervisor is None:
            return ToolResult(self.name, "Process supervisor unavailable", ok=False)
        output = await supervisor.run_argv(list(command), self.copy,
                                                               timeout=int(arguments.get("timeout", 60)))
        actual = str(output["output"])
        stable = await bounded_thread(source_unchanged, self.revision, self.copy)
        final = await self.session.capture_revision()
        stable = stable and final.digest == self.revision.digest and final.workspace == self.revision.workspace
        ok = (stable and output["state"] == "completed" and
              output["exit_code"] == arguments["expected_exit_code"] and arguments["expected_output"] in actual)
        record = {"id": uuid.uuid4().hex, "kind": arguments["kind"], "criterion": arguments["criterion"],
                  "argv": argv, "expected_exit_code": arguments["expected_exit_code"],
                  "expected_output": arguments["expected_output"], "exit_code": output["exit_code"],
                  "state": str(output["state"]) if stable else "revision_changed", "ok": ok,
                  "output": actual[-4000:], "output_truncated": len(actual) > 4000 or bool(output.get("truncated")),
                  "output_path": str(output["output_path"]),
                  "output_hash": hashlib.sha256(actual.encode()).hexdigest(),
                  "revision": self.revision.digest, "workspace": self.revision.workspace}
        self.probes.append(record)
        if self.session.logger is not None:
            await self.session.logger.write("verifier_probe", record)
        return ToolResult(self.name, json.dumps(record, ensure_ascii=False), ok=ok,
                          metadata={"probe_executed": True})


class VerifierBrowserProbeTool(Tool):
    """Delegate only existing browser MCP tools through their original permission gate."""
    name = "verifier_browser_probe"
    description = "Invoke an already-connected browser tool and record observed output against an explicit expectation."
    risk = ToolRisk.READ
    safely_cancellable = True
    execution_timeout = 600.0

    def __init__(self, agent: Any, tools: Sequence[Tool], copy: Path, revision: WorkspaceRevision,
                 probes: list[dict[str, Any]]):
        self.agent, self.copy, self.revision, self.probes = agent, copy, revision, probes
        registry = ToolRegistry()
        for tool in tools:
            registry.register(tool)
        self.executor = ToolExecutor(registry, agent.permissions, hooks=agent.hooks, logger=agent.logger,
            ui=agent.ui, permission_classifier=agent.permission_classifier, parallel_tools=False,
            journal_storage=agent.executor.journal_storage)
        self.input_schema = {"type": "object", "properties": {
            "tool": {"enum": [tool.name for tool in tools]}, "arguments": {"type": "object"},
            "kind": {"enum": ["functional", "adversarial"]},
            "criterion": {"type": "string", "minLength": 1, "maxLength": 2000},
            "expected_output": {"type": "string", "minLength": 1, "maxLength": 4000},
        }, "required": ["tool", "arguments", "kind", "criterion", "expected_output"], "additionalProperties": False}

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        from agent_core.execution import current_execution_scope
        if arguments["tool"] not in {tool.name for tool in self.executor.registry.list()}:
            return ToolResult(self.name, "Unconnected browser capability", ok=False)
        before = await self.agent.session.capture_revision()
        if (before.digest != self.revision.digest or before.workspace != self.revision.workspace or
                not await bounded_thread(source_unchanged, self.revision, self.copy)):
            return ToolResult(self.name, "Workspace changed before browser probe", ok=False)
        outcomes = await self.executor.execute_many([ToolCall(arguments["tool"], arguments["arguments"])],
                                                   execution_scope=current_execution_scope())
        actual = outcomes[0]
        final = await self.agent.session.capture_revision()
        stable = (final.digest == self.revision.digest and final.workspace == self.revision.workspace and
                  await bounded_thread(source_unchanged, self.revision, self.copy))
        ok = actual.ok and stable and arguments["expected_output"] in actual.content
        record = {"id": uuid.uuid4().hex, "kind": arguments["kind"], "criterion": arguments["criterion"],
            "browser_tool": arguments["tool"], "expected_output": arguments["expected_output"],
            "output": actual.content[-4000:], "output_truncated": len(actual.content) > 4000,
            "output_hash": hashlib.sha256(actual.content.encode()).hexdigest(), "ok": ok,
            "state": "completed" if stable else "revision_changed", "revision": self.revision.digest,
            "workspace": self.revision.workspace}
        self.probes.append(record)
        await self.agent.logger.write("verifier_browser_probe", record)
        return ToolResult(self.name, json.dumps(record, ensure_ascii=False), ok=ok,
                          metadata={"probe_executed": True})


def probe_executor(agent: Any, copy: Path, revision: WorkspaceRevision,
                   probes: list[dict[str, Any]]) -> ToolExecutor:
    if not agent.sandbox.is_enabled():
        raise RuntimeError("Behavioral verification requires a prepared real sandbox; configure [sandbox] first")
    registry = ToolRegistry()
    registry.register(_VerifierRead(copy, agent.session))
    registry.register(VerifierProbeTool(agent.session, agent.sandbox, copy, revision, probes))
    browser_tools = [tool for tool in agent.registry.list() if
        isinstance(tool, MCPTool) and any(marker in tool._server.casefold()
        for marker in ("playwright", "claude-in-chrome", "chrome_devtools", "chrome-devtools"))]
    if browser_tools:
        registry.register(VerifierBrowserProbeTool(agent, browser_tools, copy, revision, probes))
    registry.rebind_workspace(str(copy))
    # A separate scheduler avoids nested leases while retaining the parent's exact
    # policy, classifier and tool hooks. No implicit grants and no recursive agents.
    return ToolExecutor(registry, agent.permissions, hooks=agent.hooks, logger=agent.logger,
                        ui=agent.ui, permission_classifier=agent.permission_classifier, parallel_tools=False,
                        journal_storage=agent.executor.journal_storage)
