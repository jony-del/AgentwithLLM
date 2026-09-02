"""Agent tools whose commands execute inside a SWE-bench solver container."""

from __future__ import annotations

from typing import Any

from agent_core.models import ToolRisk, ToolResult
from agent_core.permission_types import DecisionSource, PermissionContext, PermissionResult
from agent_core.tools.base import ConcurrencySpec, ExecutionSafety, ResourceLock, Tool

from .runtime import InstanceRuntime


def _timeout(value: object, default: float) -> float:
    try:
        if not isinstance(value, (int, float, str)):
            return default
        return min(1800.0, max(1.0, float(value)))
    except (TypeError, ValueError):
        return default


class _RuntimeTool(Tool):
    risk = ToolRisk.DANGEROUS
    execution_safety = ExecutionSafety.FINAL_ONLY

    def __init__(self, runtime: InstanceRuntime) -> None:
        self.runtime = runtime

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        return ConcurrencySpec((ResourceLock("solver", "container", "write", subtree=True),))

    async def check_permissions(
        self, arguments: dict[str, Any], context: PermissionContext
    ) -> PermissionResult:
        command = str(arguments.get("command", "")).strip()
        if not command:
            return PermissionResult.deny("command must not be empty", decision_source=DecisionSource.TOOL)
        # The only command path exposed to the model is the disposable, network-disabled
        # SWE-bench runtime.  Central permission preflight still applies deny rules and
        # schema validation before this explicit capability is reached.
        return PermissionResult.allow(
            "command runs inside the isolated SWE-bench solver runtime",
            decision_source=DecisionSource.SANDBOX,
        )


class SWEbenchShellTool(_RuntimeTool):
    name = "swebench_shell"
    description = (
        "Run a shell command in the isolated SWE-bench repository container. The working "
        "directory is /testbed and network access is disabled. Use this for dependency-free "
        "code inspection, git commands, formatting, and focused checks."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "minLength": 1},
            "timeout": {"type": "number", "minimum": 1, "maximum": 1800},
        },
        "required": ["command"],
    }

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        command = str(arguments.get("command", ""))
        timeout = _timeout(arguments.get("timeout", 300), 300.0)
        result = await self.runtime.exec(command, timeout=timeout)
        return ToolResult(
            self.name,
            result.render(),
            ok=result.ok,
            metadata={
                "returncode": result.returncode,
                "duration": result.duration,
                "timed_out": result.timed_out,
                "runtime": "swebench",
            },
        )


class SWEbenchTestTool(_RuntimeTool):
    name = "swebench_test"
    description = (
        "Run tests in the isolated SWE-bench repository container. Supply a focused command "
        "such as `pytest tests/foo.py -q`; the default is `pytest -q`. Test failures are "
        "returned as observations so the Agent can iterate."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "minLength": 1, "description": "Test command; defaults to pytest -q."},
            "timeout": {"type": "number", "minimum": 1, "maximum": 1800},
        },
        "required": [],
    }

    def __init__(self, runtime: InstanceRuntime, default_command: str = "pytest -q") -> None:
        super().__init__(runtime)
        self.default_command = default_command

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        command = str(arguments.get("command") or self.default_command).strip()
        timeout = _timeout(arguments.get("timeout", 600), 600.0)
        result = await self.runtime.exec(command, timeout=timeout)
        return ToolResult(
            self.name,
            result.render(),
            ok=result.ok,
            metadata={
                "returncode": result.returncode,
                "duration": result.duration,
                "timed_out": result.timed_out,
                "runtime": "swebench",
                "test_command": command,
            },
        )


def build_swebench_registry(runtime: InstanceRuntime, workspace: str, *, test_command: str | None = None):
    """Build a least-privilege registry for one benchmark instance."""
    from agent_core.react import ReActAgent

    registry = ReActAgent.default_registry()
    allowed = {"list_dir", "read_text_file", "search_text", "glob", "edit_file", "multi_edit", "apply_patch", "write_text_file", "git_diff", "echo"}
    for tool in registry.list():
        if tool.name not in allowed:
            registry.unregister(tool.name)
    for item in registry.deferred():
        registry.unregister(item.name)
    registry.rebind_workspace(workspace)
    registry.register(SWEbenchShellTool(runtime))
    registry.register(SWEbenchTestTool(runtime, test_command or "pytest -q"))
    return registry
