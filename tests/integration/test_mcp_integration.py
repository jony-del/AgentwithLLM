"""Offline MCP integration over a real local HTTP transport (audit phase 5).

The stdio roundtrip already lives in tests/test_mcp.py; this chain adds the
streamable-http transport against a real FastMCP server subprocess: discovery,
a tool roundtrip, the cleanup path after a call timeout (suspect marking and a
bounded close while the server is still up), and connection recovery afterwards.
Everything is local-loopback — no external endpoints, no credentials.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agent_core.mcp.adapter import MCPAdapter
from agent_core.mcp.client import MCPClientManager
from agent_core.mcp.config import MCPConfig, MCPServerConfig

pytestmark = pytest.mark.integration

_SERVER = '''
import sys
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("http-itg")


@mcp.tool()
def echo(text: str) -> str:
    return "echo:" + text


@mcp.tool()
def slow(text: str) -> str:
    import time
    time.sleep(20)
    return text


mcp.run(transport="streamable-http", host="127.0.0.1", port=int(sys.argv[1]))
'''


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_for_port(port: int, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.2)
    raise AssertionError(f"MCP HTTP server never listened on 127.0.0.1:{port}")


@pytest.fixture
def http_server(tmp_path: Path):
    port = _free_port()
    server_file = tmp_path / "http_server.py"
    server_file.write_text(_SERVER, encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(server_file), str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_port(port)
        yield port
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def _config(port: int) -> MCPConfig:
    return MCPConfig(
        servers=[
            MCPServerConfig(
                name="h",
                transport="streamable-http",
                url=f"http://127.0.0.1:{port}/mcp",
                risk="read",
            )
        ]
    )


async def test_http_roundtrip_timeout_cleanup_and_recovery(http_server) -> None:
    port = http_server
    manager = MCPClientManager(_config(port), connect_timeout=30, call_timeout=3)
    manager.start()
    try:
        discovered = [descriptor.name for _, descriptor in manager.tools()]
        assert {"echo", "slow"} <= set(discovered)

        tools = {tool.name: tool for tool in MCPAdapter(manager).list_tools()}
        assert "h__echo" in tools
        result = await tools["h__echo"].run({"text": "hello"})
        assert result.ok is True
        assert result.content == "echo:hello"

        # The slow tool outruns an explicit 3s call budget: the call must fail
        # as a timeout (never hang) and mark the server suspect for the next
        # call's health check. (The adapter path swallows transport errors into
        # a failed ToolResult, so the cleanup contract is pinned at the manager.)
        with pytest.raises(TimeoutError):
            manager.call_tool("h", "slow", {"text": "stuck"}, timeout=3)

        # Recovery on the SAME manager: the next call health-checks, reconnects
        # once and completes over a fresh session.
        recovered = manager.call_tool("h", "echo", {"text": "recovered"})
        assert "echo:recovered" in str(getattr(recovered, "content", ""))
    finally:
        # close() must be bounded even though the server (and its stuck slow
        # call) is still alive.
        manager.close()

    # A fresh manager connects to the still-running server without residue from
    # the timed-out call.
    second = MCPClientManager(_config(port), connect_timeout=30, call_timeout=10)
    second.start()
    try:
        tools = {tool.name: tool for tool in MCPAdapter(second).list_tools()}
        result = await tools["h__echo"].run({"text": "again"})
        assert result.ok is True
        assert result.content == "echo:again"
    finally:
        second.close()
