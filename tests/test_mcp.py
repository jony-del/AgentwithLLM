import importlib.util
import sys
from pathlib import Path

import pytest

# The unit tests below run without the optional SDK (agent_core.mcp imports it lazily);
# only the integration test at the end needs it installed.
_HAS_MCP = importlib.util.find_spec("mcp") is not None

from agent_core.config import resolve_mcp_config
from agent_core.mcp.adapter import MCPAdapter, MCPTool, _flatten_content
from agent_core.mcp.config import MCPConfig, MCPServerConfig
from agent_core.models import ToolRisk


# --- config parsing (no SDK, no servers needed) --------------------------------


def test_config_from_dict_parses_both_transports() -> None:
    config = MCPConfig.from_dict(
        {
            "servers": {
                "fs": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["-m", "server", "."],
                    "env": {"TOKEN": "x"},
                    "risk": "read",
                },
                "weather": {
                    "transport": "streamable-http",
                    "url": "http://localhost:8000/mcp",
                    "headers": {"Authorization": "Bearer t"},
                    "risk": "write",
                },
            }
        }
    )
    by_name = {server.name: server for server in config.servers}
    assert set(by_name) == {"fs", "weather"}

    fs = by_name["fs"]
    assert fs.transport == "stdio"
    assert fs.command == "python"
    assert fs.args == ["-m", "server", "."]  # toml array passes through untouched
    assert fs.env == {"TOKEN": "x"}  # toml table passes through untouched
    assert fs.risk == "read"

    weather = by_name["weather"]
    assert weather.url == "http://localhost:8000/mcp"
    assert weather.headers == {"Authorization": "Bearer t"}
    assert weather.risk == "write"


def test_config_defaults_risk_to_dangerous_and_enabled() -> None:
    config = MCPConfig.from_dict({"servers": {"x": {"transport": "stdio", "command": "echo"}}})
    server = config.servers[0]
    assert server.risk == "dangerous"  # safe default
    assert server.enabled is True


def test_config_from_dict_empty_or_malformed_yields_no_servers() -> None:
    assert MCPConfig.from_dict(None).servers == []
    assert MCPConfig.from_dict({}).servers == []
    assert MCPConfig.from_dict({"servers": "not-a-table"}).servers == []


def test_resolve_mcp_config_reads_toml_table(tmp_path: Path) -> None:
    toml = tmp_path / "agent.toml"
    toml.write_text(
        """
[mcp.servers.fs]
transport = "stdio"
command = "python"
args = ["-m", "server"]
risk = "read"
""",
        encoding="utf-8",
    )
    config = resolve_mcp_config(toml)
    assert len(config.servers) == 1
    assert config.servers[0].name == "fs"
    assert config.servers[0].risk == "read"


def test_resolve_mcp_config_absent_table_is_empty(tmp_path: Path) -> None:
    toml = tmp_path / "agent.toml"
    toml.write_text('model = "claude-sonnet-4-6"\n', encoding="utf-8")
    assert resolve_mcp_config(toml).servers == []


# --- adapter / tool wrapping (fake manager, no SDK call) -----------------------


class _Descriptor:
    def __init__(self, name, description="", input_schema=None):
        self.name = name
        self.description = description
        self.inputSchema = input_schema


class _Block:
    def __init__(self, text=None, type="text"):
        self.text = text
        self.type = type


class _Result:
    def __init__(self, content, isError=False):
        self.content = content
        self.isError = isError


class _FakeManager:
    """Stand-in for MCPClientManager: no event loop, no SDK."""

    def __init__(self, tools, result=None, error=None):
        self._tools = tools
        self._result = result
        self._error = error
        self.calls = []

    def tools(self):
        return list(self._tools)

    def call_tool(self, server, tool, arguments, timeout=None):
        self.calls.append((server, tool, arguments))
        if self._error is not None:
            raise self._error
        return self._result


def test_adapter_prefixes_names_and_maps_risk() -> None:
    fs = MCPServerConfig(name="fs", risk="read")
    db = MCPServerConfig(name="db")  # default risk = dangerous
    manager = _FakeManager(
        [(fs, _Descriptor("read_file")), (db, _Descriptor("query"))]
    )
    tools = {t.name: t for t in MCPAdapter(manager).list_tools()}
    assert set(tools) == {"fs__read_file", "db__query"}
    assert tools["fs__read_file"].risk is ToolRisk.READ
    assert tools["db__query"].risk is ToolRisk.DANGEROUS  # default


def test_tool_run_flattens_text_and_marks_ok() -> None:
    server = MCPServerConfig(name="s", risk="read")
    result = _Result([_Block(text="line1"), _Block(text="line2")], isError=False)
    manager = _FakeManager([], result=result)
    tool = MCPTool(manager, server, _Descriptor("t"))

    out = tool.run({"a": 1})
    assert out.ok is True
    assert out.content == "line1\nline2"
    assert out.metadata["mcp_server"] == "s"
    assert manager.calls == [("s", "t", {"a": 1})]  # remote name, not the prefixed name


def test_tool_run_maps_is_error_to_not_ok() -> None:
    manager = _FakeManager([], result=_Result([_Block(text="boom")], isError=True))
    tool = MCPTool(manager, MCPServerConfig(name="s"), _Descriptor("t"))
    assert tool.run({}).ok is False


def test_tool_run_exception_becomes_failed_result() -> None:
    manager = _FakeManager([], error=TimeoutError("slow"))
    tool = MCPTool(manager, MCPServerConfig(name="s"), _Descriptor("t"))
    out = tool.run({})
    assert out.ok is False
    assert "MCP tool error" in out.content


def test_tool_uses_default_schema_when_descriptor_has_none() -> None:
    tool = MCPTool(_FakeManager([]), MCPServerConfig(name="s"), _Descriptor("t", input_schema=None))
    assert tool.input_schema == {"type": "object", "properties": {}}


def test_flatten_content_notes_non_text_blocks() -> None:
    assert _flatten_content([_Block(text="hi"), _Block(text=None, type="image")]) == "hi\n[image block]"
    assert _flatten_content(None) == ""


# --- error reporting -----------------------------------------------------------


def test_describe_mcp_error_unwraps_nested_exception_groups() -> None:
    from agent_core.cli import _describe_mcp_error

    # anyio nests the real cause inside (possibly several) ExceptionGroups.
    leaf = RuntimeError("Connection closed")
    grouped = ExceptionGroup("outer", [ExceptionGroup("inner", [leaf])])
    assert _describe_mcp_error(grouped) == "RuntimeError: Connection closed"

    # A plain exception is described as-is.
    assert _describe_mcp_error(ValueError("nope")) == "ValueError: nope"


# --- integration (requires the optional `mcp` SDK and a real subprocess) --------

_SERVER = '''
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("testsrv")


@mcp.tool()
def echo(text: str) -> str:
    return text


if __name__ == "__main__":
    mcp.run(transport="stdio")
'''


@pytest.mark.skipif(not _HAS_MCP, reason="the optional `mcp` SDK is not installed")
def test_stdio_roundtrip_through_manager(tmp_path: Path) -> None:
    from agent_core.mcp.client import MCPClientManager

    server_file = tmp_path / "srv.py"
    server_file.write_text(_SERVER, encoding="utf-8")
    config = MCPConfig(
        servers=[
            MCPServerConfig(
                name="t",
                transport="stdio",
                command=sys.executable,
                args=[str(server_file)],
                risk="read",
            )
        ]
    )
    manager = MCPClientManager(config, connect_timeout=30)
    manager.start()
    try:
        discovered = [descriptor.name for _, descriptor in manager.tools()]
        assert "echo" in discovered

        tools = {t.name: t for t in MCPAdapter(manager).list_tools()}
        assert "t__echo" in tools
        assert tools["t__echo"].risk is ToolRisk.READ

        out = tools["t__echo"].run({"text": "hello mcp"})
        assert out.ok is True
        assert out.content == "hello mcp"
    finally:
        manager.close()
