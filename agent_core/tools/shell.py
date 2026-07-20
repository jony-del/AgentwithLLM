"""Dialect-specific shell tools backed by one process supervisor."""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

from agent_core.command_security import analyze_command
from agent_core.models import ToolRisk, ToolResult
from agent_core.permission_types import PermissionBehavior, PermissionContext, PermissionMode, PermissionResult
from agent_core.process_supervisor import (
    ProcessSupervisor,
    encoded_powershell,
    powershell_utf8_command,
    resolve_bash_executable,
    resolve_powershell_executable,
)
from agent_core.sandbox import SandboxAwareMixin
from agent_core.session import SessionAwareMixin
from agent_core.tools.base import ConcurrencySpec, ExecutionScope, ResourceLock, Tool
from agent_core.tools.catalog import builtin_tool


def _supervisor(tool: str, session: Any) -> ProcessSupervisor | ToolResult:
    value = session.process_supervisor
    if not isinstance(value, ProcessSupervisor):
        return ToolResult(tool, "Process supervisor is unavailable in this session.", ok=False)
    return value


class _ShellTool(SessionAwareMixin, SandboxAwareMixin, Tool):
    risk = ToolRisk.DANGEROUS
    dialect: str

    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer", "minimum": 1},
            "description": {"type": "string", "maxLength": 200},
            "run_in_background": {"type": "boolean"},
            "dangerously_disable_sandbox": {"type": "boolean"},
        },
        "required": ["command"],
    }

    async def check_permissions(self, arguments: dict[str, Any], context: PermissionContext) -> PermissionResult:
        command = str(arguments.get("command", ""))
        if arguments.get("dangerously_disable_sandbox"):
            if not self.sandbox.config.allow_unsandboxed_commands:
                return PermissionResult.deny(
                    "managed sandbox configuration forbids unsandboxed commands",
                    metadata={"dialect": self.dialect, "sandbox_override": True},
                )
            return PermissionResult.ask(
                "disabling command isolation always requires explicit user approval",
                classifier_approvable=False,
                bypass_immune=True,
                metadata={"dialect": self.dialect, "sandbox_override": True},
            )
        analysis = analyze_command(command)
        metadata = {"category": analysis.category, "segments": len(analysis.segments), "dialect": self.dialect}
        if analysis.behavior is PermissionBehavior.DENY:
            return PermissionResult.deny(analysis.reason, metadata=metadata)
        if analysis.behavior is PermissionBehavior.ALLOW:
            return PermissionResult.allow(analysis.reason, metadata=metadata)
        if context.mode is PermissionMode.BYPASS and not analysis.bypass_immune:
            return PermissionResult.passthrough(analysis.reason, metadata=metadata)
        if analysis.category in {"development", "file_mutation"}:
            return PermissionResult.passthrough(analysis.reason, metadata=metadata)
        return PermissionResult.ask(
            analysis.reason,
            metadata=metadata,
            classifier_approvable=analysis.classifier_approvable,
            bypass_immune=analysis.bypass_immune,
        )

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        return ConcurrencySpec((ResourceLock("fs", str(self.session.workspace.resolve()), "write", subtree=True),))

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        command = str(arguments.get("command", ""))
        if not command.strip():
            return ToolResult(self.name, "command must not be empty", ok=False)
        supervisor = _supervisor(self.name, self.session)
        if isinstance(supervisor, ToolResult):
            return supervisor
        config = supervisor.config
        timeout = max(
            1, min(int(str(arguments.get("timeout", config.timeout))), config.max_timeout)
        )
        try:
            if self.dialect == "bash":
                executable = resolve_bash_executable(config.bash.executable or os.getenv("POLARIS_BASH_PATH"))
                argv: object = [executable, "-lc", command]
            else:
                executable = resolve_powershell_executable(config.powershell.executable)
                argv = [
                    executable, "-NoLogo", "-NoProfile", "-NonInteractive",
                    "-OutputFormat", "Text", "-EncodedCommand",
                    encoded_powershell(powershell_utf8_command(command)),
                ]
            if not bool(arguments.get("dangerously_disable_sandbox", False)):
                argv, shell = self.sandbox.wrap(
                    argv, False, command=command,
                    scope=ExecutionScope.for_workspace(self.session.workspace, network="deny"),
                )
                if shell or isinstance(argv, str):
                    raise RuntimeError("sandbox returned a shell-string command; explicit argv is required")
            if not isinstance(argv, (list, tuple)):
                raise RuntimeError("sandbox returned invalid argv")
            task = await supervisor.start(
                self.dialect, command, self.session.workspace,
                argv=[str(item) for item in argv], timeout=timeout,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            return ToolResult(self.name, f"{self.dialect} failed to start: {exc}", ok=False)
        if bool(arguments.get("run_in_background", False)):
            return ToolResult(
                self.name,
                f"Task {task.id} started in background. Use task_output or task_stop.",
                metadata={"task_id": task.id, "state": task.state, "output_path": str(task.log_path)},
            )
        foreground: float = float(timeout)
        if config.auto_background_seconds > 0:
            foreground = min(float(timeout), config.auto_background_seconds)
        try:
            deadline = time.monotonic() + foreground
            while task.state == "running":
                callback = self.session.should_background
                if callback is not None and callback():
                    return ToolResult(
                        self.name,
                        f"Task {task.id} was moved to background by Ctrl+B.",
                        metadata={"task_id": task.id, "state": "running", "interactive_backgrounded": True},
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                try:
                    await supervisor.wait(task.id, min(0.1, remaining))
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            await supervisor.stop(task.id)
            raise
        except asyncio.TimeoutError:
            if foreground >= timeout:
                await supervisor.stop(task.id)
            else:
                return ToolResult(
                    self.name,
                    f"Task {task.id} is still running and was moved to background after {foreground:g}s.",
                    metadata={"task_id": task.id, "state": "running", "auto_backgrounded": True},
                )
        output = await supervisor.output(task.id, block=False, timeout=0, tail_lines=None)
        content = str(output.pop("output"))
        state = str(output["state"])
        return ToolResult(
            self.name,
            f"{content}\n[exit code: {output['exit_code']}; state: {state}]".strip(),
            ok=state == "completed",
            metadata=output,
        )


@builtin_tool
class BashTool(_ShellTool):
    name = "bash"
    dialect = "bash"
    description = "Run a Bash command with syntax analysis, sandboxing, timeout, and background supervision."


@builtin_tool
class PowerShellTool(_ShellTool):
    name = "powershell"
    dialect = "powershell"
    description = "Run a PowerShell command via UTF-16LE EncodedCommand with AST syntax validation."


@builtin_tool
class TaskOutputTool(SessionAwareMixin, Tool):
    name = "task_output"
    description = "Read bounded output and state for a supervised shell task."
    input_schema = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "block": {"type": "boolean"},
            "timeout": {"type": "number", "minimum": 0, "maximum": 60},
            "tail_lines": {"type": "integer", "minimum": 0, "maximum": 10000},
        },
        "required": ["task_id"],
    }
    risk = ToolRisk.READ

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        supervisor = _supervisor(self.name, self.session)
        if isinstance(supervisor, ToolResult):
            return supervisor
        try:
            output = await supervisor.output(
                str(arguments.get("task_id", "")),
                block=bool(arguments.get("block", True)),
                timeout=float(str(arguments.get("timeout", 30))),
                tail_lines=int(str(arguments["tail_lines"])) if arguments.get("tail_lines") is not None else None,
            )
        except (KeyError, ValueError) as exc:
            return ToolResult(self.name, str(exc), ok=False)
        content = str(output.pop("output"))
        return ToolResult(self.name, content or "(no output)", metadata=output)


@builtin_tool
class TaskStopTool(SessionAwareMixin, Tool):
    name = "task_stop"
    description = "Stop a supervised shell task and its entire process tree."
    input_schema = {
        "type": "object",
        "properties": {"task_id": {"type": "string"}},
        "required": ["task_id"],
    }
    risk = ToolRisk.WRITE

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        supervisor = _supervisor(self.name, self.session)
        if isinstance(supervisor, ToolResult):
            return supervisor
        try:
            task = await supervisor.stop(str(arguments.get("task_id", "")))
        except KeyError as exc:
            return ToolResult(self.name, str(exc), ok=False)
        return ToolResult(self.name, f"Task {task.id}: {task.state}", metadata={"task_id": task.id, "state": task.state})


@builtin_tool
class SleepTool(Tool):
    name = "sleep"
    description = "Pause cooperatively for at most 60 seconds; cancellation stops immediately."
    input_schema = {
        "type": "object",
        "properties": {"seconds": {"type": "number", "minimum": 0, "maximum": 60}},
        "required": ["seconds"],
    }
    risk = ToolRisk.READ

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        seconds = float(str(arguments.get("seconds", 0)))
        if not 0 <= seconds <= 60:
            return ToolResult(self.name, "seconds must be between 0 and 60", ok=False)
        await asyncio.sleep(seconds)
        return ToolResult(self.name, json.dumps({"slept": seconds}))
