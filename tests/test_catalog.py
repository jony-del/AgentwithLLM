from pathlib import Path

from agent_core.models import ToolResult, ToolRisk
from agent_core.react import ReActAgent
from agent_core.tools import catalog
from agent_core.tools.base import Tool
from agent_core.tools.catalog import builtin_tool, builtin_tool_classes, default_tools

# Every tool the project ships today; discovery must find all of them.
_EXPECTED = {
    "echo",
    "read_text_file",
    "list_dir",
    "search_text",
    "git_diff",
    "write_text_file",
    "edit_file",
    "bash",
    "run_tests",
}


def test_discovery_finds_all_builtins() -> None:
    names = {tool.name for tool in default_tools()}
    assert _EXPECTED <= names


def test_default_tools_have_unique_names() -> None:
    names = [tool.name for tool in default_tools()]
    assert len(names) == len(set(names))  # no duplicate registration


def test_default_registry_registers_every_builtin() -> None:
    registry = ReActAgent.default_registry()
    for name in _EXPECTED:
        assert registry.get(name).name == name


def test_default_tools_inject_workspace(tmp_path: Path) -> None:
    tools = {t.name: t for t in default_tools(tmp_path)}
    # Workspace-scoped tool gets the workspace; workspace-agnostic tool still builds.
    assert tools["read_text_file"].workspace == tmp_path.resolve()
    assert tools["echo"].name == "echo"


def test_builtin_tool_decorator_registers_and_returns() -> None:
    class _TmpTool(Tool):
        name = "_tmp_tool"
        description = "temp"
        input_schema = {"type": "object", "properties": {}}
        risk = ToolRisk.READ

        def _invoke(self, arguments: dict) -> ToolResult:
            return ToolResult(self.name, "ok")

    before = len(catalog._BUILTIN)
    returned = builtin_tool(_TmpTool)
    try:
        assert returned is _TmpTool  # decorator returns the class unchanged
        assert _TmpTool in builtin_tool_classes()
    finally:
        catalog._BUILTIN.remove(_TmpTool)  # don't leak into other tests
    assert len(catalog._BUILTIN) == before
