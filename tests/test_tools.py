import time
from pathlib import Path

import pytest

from agent_core.hooks import HookPipeline, HookResult
from agent_core.models import ToolCall, ToolResult, ToolRisk
from agent_core.permissions import PermissionMode, PermissionPolicy
from agent_core.tools.base import ConcurrencySpec, ResourceLock, Tool
from agent_core.tools.builtin import EchoTool, ReadTextFileTool, WriteTextFileTool
from agent_core.tools.executor import ToolExecutor
from agent_core.tools.registry import ToolRegistry


class RewriteToEchoHook:
    def before_tool(self, tool_call: ToolCall) -> HookResult:
        return HookResult(tool_call=ToolCall("echo", {"text": "rewritten"}))


class RewritePathHook:
    def before_tool(self, tool_call: ToolCall) -> HookResult:
        if tool_call.name == "path_sleep":
            return HookResult(tool_call=ToolCall("path_sleep", {**tool_call.arguments, "path": "same.txt"}))
        return HookResult()


class PathSleepTool(Tool):
    name = "path_sleep"
    description = "Sleep while holding a declared resource lock."
    input_schema = {"type": "object", "properties": {}}
    risk = ToolRisk.READ

    def concurrency_spec(self, arguments: dict) -> ConcurrencySpec:
        return ConcurrencySpec(
            (
                ResourceLock(
                    "fs",
                    str(arguments["path"]),
                    str(arguments.get("mode", "read")),  # type: ignore[arg-type]
                    subtree=bool(arguments.get("subtree", False)),
                ),
            )
        )

    def run(self, arguments: dict) -> ToolResult:
        time.sleep(float(arguments.get("delay", 0.1)))
        return ToolResult(self.name, str(arguments.get("label", arguments["path"])))


class DangerousCountingTool(Tool):
    name = "dangerous_count"
    description = "Dangerous tool that records if it ran."
    input_schema = {"type": "object", "properties": {}}
    risk = ToolRisk.DANGEROUS

    def __init__(self) -> None:
        self.calls = 0

    def concurrency_spec(self, arguments: dict) -> ConcurrencySpec:
        return ConcurrencySpec()

    def run(self, arguments: dict) -> ToolResult:
        self.calls += 1
        return ToolResult(self.name, "ran")


class WriteCountingTool(Tool):
    name = "write_count"
    description = "Write tool that records if it ran."
    input_schema = {"type": "object", "properties": {}}
    risk = ToolRisk.WRITE

    def __init__(self) -> None:
        self.calls = 0

    def concurrency_spec(self, arguments: dict) -> ConcurrencySpec:
        return ConcurrencySpec()

    def run(self, arguments: dict) -> ToolResult:
        self.calls += 1
        return ToolResult(self.name, "ran")


class StateAppendTool(Tool):
    name = "state_append"
    description = "Append to shared state and record state snapshots during preparation."
    input_schema = {"type": "object", "properties": {}}
    risk = ToolRisk.READ

    def __init__(self) -> None:
        self.state: list[str] = []
        self.spec_snapshots: list[tuple[str, ...]] = []

    def concurrency_spec(self, arguments: dict) -> ConcurrencySpec:
        self.spec_snapshots.append(tuple(self.state))
        return ConcurrencySpec()

    def run(self, arguments: dict) -> ToolResult:
        value = str(arguments["value"])
        self.state.append(value)
        return ToolResult(self.name, value)


def test_executor_returns_failed_result_for_unknown_tool() -> None:
    registry = ToolRegistry()
    executor = ToolExecutor(registry, PermissionPolicy(PermissionMode.AUTO))

    result = executor.execute(ToolCall("missing_tool"))

    assert not result.ok
    assert result.name == "missing_tool"
    assert result.metadata["error_type"] == "UnknownTool"


def test_executor_uses_tool_call_rewritten_by_pre_hook() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    executor = ToolExecutor(
        registry,
        PermissionPolicy(PermissionMode.AUTO),
        HookPipeline(pre_hooks=[RewriteToEchoHook()]),
    )

    result = executor.execute(ToolCall("missing_tool"))

    assert result.ok
    assert result.name == "echo"
    assert result.content == "rewritten"


def test_execute_many_runs_independent_read_tools_concurrently() -> None:
    registry = ToolRegistry()
    registry.register(PathSleepTool())
    executor = ToolExecutor(registry, PermissionPolicy(PermissionMode.AUTO), max_workers=2)

    start = time.perf_counter()
    results = executor.execute_many(
        [
            ToolCall("path_sleep", {"path": "a.txt", "mode": "read", "delay": 0.2, "label": "a"}),
            ToolCall("path_sleep", {"path": "a.txt", "mode": "read", "delay": 0.2, "label": "b"}),
        ]
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 0.35
    assert [result.content for result in results] == ["a", "b"]


def test_execute_many_preserves_input_order_not_completion_order() -> None:
    registry = ToolRegistry()
    registry.register(PathSleepTool())
    executor = ToolExecutor(registry, PermissionPolicy(PermissionMode.AUTO), max_workers=2)

    results = executor.execute_many(
        [
            ToolCall("path_sleep", {"path": "a.txt", "mode": "read", "delay": 0.15, "label": "slow"}),
            ToolCall("path_sleep", {"path": "b.txt", "mode": "read", "delay": 0.01, "label": "fast"}),
        ]
    )

    assert [result.content for result in results] == ["slow", "fast"]


def test_execute_many_serializes_conflicting_file_locks() -> None:
    registry = ToolRegistry()
    registry.register(PathSleepTool())
    executor = ToolExecutor(registry, PermissionPolicy(PermissionMode.AUTO), max_workers=2)

    start = time.perf_counter()
    executor.execute_many(
        [
            ToolCall("path_sleep", {"path": "same.txt", "mode": "read", "delay": 0.15}),
            ToolCall("path_sleep", {"path": "same.txt", "mode": "write", "delay": 0.15}),
        ]
    )
    same_file_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    executor.execute_many(
        [
            ToolCall("path_sleep", {"path": "a.txt", "mode": "write", "delay": 0.15}),
            ToolCall("path_sleep", {"path": "b.txt", "mode": "write", "delay": 0.15}),
        ]
    )
    different_file_elapsed = time.perf_counter() - start

    assert same_file_elapsed >= 0.28
    assert different_file_elapsed < 0.28


def test_execute_many_uses_rewritten_call_for_resource_locks() -> None:
    registry = ToolRegistry()
    registry.register(PathSleepTool())
    executor = ToolExecutor(
        registry,
        PermissionPolicy(PermissionMode.AUTO),
        HookPipeline(pre_hooks=[RewritePathHook()]),
        max_workers=2,
    )

    start = time.perf_counter()
    executor.execute_many(
        [
            ToolCall("path_sleep", {"path": "a.txt", "mode": "write", "delay": 0.15}),
            ToolCall("path_sleep", {"path": "b.txt", "mode": "write", "delay": 0.15}),
        ]
    )
    elapsed = time.perf_counter() - start

    assert elapsed >= 0.28


def test_execute_many_handles_denied_dry_run_and_unknown_without_running() -> None:
    registry = ToolRegistry()
    dangerous = DangerousCountingTool()
    write = WriteCountingTool()
    registry.register(dangerous)
    registry.register(write)

    denied_executor = ToolExecutor(registry, PermissionPolicy(PermissionMode.AUTO))
    denied = denied_executor.execute_many(
        [ToolCall("missing"), ToolCall("dangerous_count"), ToolCall("write_count")]
    )
    assert [result.name for result in denied] == ["missing", "dangerous_count", "write_count"]
    assert not denied[0].ok and denied[0].metadata["error_type"] == "UnknownTool"
    assert not denied[1].ok
    assert denied[2].ok
    assert dangerous.calls == 0
    assert write.calls == 1

    dry_run_executor = ToolExecutor(registry, PermissionPolicy(PermissionMode.PLAN))
    dry_run = dry_run_executor.execute_many([ToolCall("write_count")])
    assert dry_run[0].content.startswith("Dry-run:")
    assert write.calls == 1


def test_parallel_disabled_prepares_each_call_after_previous_run() -> None:
    registry = ToolRegistry()
    tool = StateAppendTool()
    registry.register(tool)
    executor = ToolExecutor(
        registry,
        PermissionPolicy(PermissionMode.AUTO),
        parallel_tools=False,
        max_workers=2,
    )

    results = executor.execute_many(
        [
            ToolCall("state_append", {"value": "first"}),
            ToolCall("state_append", {"value": "second"}),
        ]
    )

    assert [result.content for result in results] == ["first", "second"]
    assert tool.state == ["first", "second"]
    assert tool.spec_snapshots == [(), ("first",)]


def test_execute_many_cancelled_returns_one_result_per_input() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    executor = ToolExecutor(registry, PermissionPolicy(PermissionMode.AUTO), max_workers=2)

    results = executor.execute_many(
        [ToolCall("echo", {"text": "a"}), ToolCall("missing")],
        should_cancel=lambda: True,
    )

    assert len(results) == 2
    assert [result.metadata["error_type"] for result in results] == ["Cancelled", "Cancelled"]


def test_file_tools_resolve_paths_inside_workspace(tmp_path: Path) -> None:
    writer = WriteTextFileTool(tmp_path)
    reader = ReadTextFileTool(tmp_path)

    writer.run({"path": "nested/example.txt", "content": "hello"})

    assert reader.run({"path": "nested/example.txt"}).content == "hello"


def test_file_tools_reject_paths_outside_workspace(tmp_path: Path) -> None:
    writer = WriteTextFileTool(tmp_path)

    with pytest.raises(ValueError, match="escapes workspace"):
        writer.run({"path": "../outside.txt", "content": "nope"})
