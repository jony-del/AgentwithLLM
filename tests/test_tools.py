from pathlib import Path

import pytest

from agent_core.hooks import HookPipeline, HookResult
from agent_core.models import ToolCall
from agent_core.permissions import PermissionMode, PermissionPolicy
from agent_core.tools.demo import EchoTool, ReadTextFileTool, WriteTextFileTool
from agent_core.tools.executor import ToolExecutor
from agent_core.tools.registry import ToolRegistry


class RewriteToEchoHook:
    def before_tool(self, tool_call: ToolCall) -> HookResult:
        return HookResult(tool_call=ToolCall("echo", {"text": "rewritten"}))


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


def test_file_tools_resolve_paths_inside_workspace(tmp_path: Path) -> None:
    writer = WriteTextFileTool(tmp_path)
    reader = ReadTextFileTool(tmp_path)

    writer.run({"path": "nested/example.txt", "content": "hello"})

    assert reader.run({"path": "nested/example.txt"}).content == "hello"


def test_file_tools_reject_paths_outside_workspace(tmp_path: Path) -> None:
    writer = WriteTextFileTool(tmp_path)

    with pytest.raises(ValueError, match="escapes workspace"):
        writer.run({"path": "../outside.txt", "content": "nope"})
