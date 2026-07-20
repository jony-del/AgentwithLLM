from __future__ import annotations

import asyncio
import os
import threading
from contextlib import AsyncExitStack
from typing import Any

from agent_core.mcp.config import MCPConfig, MCPServerConfig


class MCPClientManager:
    """Bridge the synchronous agent to the asynchronous ``mcp`` SDK.

    The SDK is fully async (``ClientSession``/``stdio_client``/``streamablehttp_client``
    are async context managers built on anyio task groups), while the agent calls
    ``Tool.run`` synchronously. So this manager owns a single background thread running
    one asyncio event loop; every MCP session is opened and lives inside that loop, and
    ``call_tool`` submits work to it and blocks for the result.

    The anyio catch: an ``async with`` must be exited in the *same task* that entered it.
    So a single long-lived ``_serve`` coroutine opens every server inside one
    ``AsyncExitStack``, signals readiness, then awaits a stop ``Event``; ``close()`` sets
    that event so the same task unwinds the stack. Per-call ``call_tool`` is submitted as
    a separate task — safe, because it only reads/writes the session's anyio streams
    within the same loop and never exits an ``async with``.
    """

    def __init__(
        self,
        config: MCPConfig,
        *,
        connect_timeout: float = 30.0,
        call_timeout: float = 60.0,
    ) -> None:
        self._config = config
        self._connect_timeout = connect_timeout
        self._call_timeout = call_timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._serve_future: Any = None  # concurrent.futures.Future
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._sessions: dict[str, Any] = {}
        self._tools: list[tuple[MCPServerConfig, Any]] = []
        self._closed = False

    # -- lifecycle -----------------------------------------------------------------

    def start(self) -> "MCPClientManager":
        """Spin up the loop thread and connect every enabled server. Blocking.

        Raises whatever connecting raised (e.g. ``ModuleNotFoundError`` if the ``mcp``
        extra isn't installed, or a transport error) after tearing the thread down.
        """
        if self._loop is not None:
            return self
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-loop", daemon=True)
        self._thread.start()
        self._serve_future = asyncio.run_coroutine_threadsafe(self._serve(), self._loop)
        if not self._ready.wait(self._connect_timeout):
            self.close()
            raise TimeoutError(f"MCP servers did not connect within {self._connect_timeout}s")
        if self._error is not None:
            error = self._error
            self.close()
            raise error
        return self

    async def _serve(self) -> None:
        self._stop = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:
                for server in self._config.servers:
                    if server.enabled:
                        await self._connect(stack, server)
                # Ready only once every server connected; then hold the stack open until
                # close() sets the stop event, so it unwinds here, in this same task.
                self._ready.set()
                await self._stop.wait()
        except BaseException as exc:  # noqa: BLE001 - surfaced to start() on the caller thread
            self._error = exc
            self._ready.set()

    async def _connect(self, stack: AsyncExitStack, server: MCPServerConfig) -> None:
        from mcp import ClientSession

        read, write = await self._open_transport(stack, server)
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._sessions[server.name] = session
        listed = await session.list_tools()
        for descriptor in listed.tools:
            self._tools.append((server, descriptor))

    async def _open_transport(self, stack: AsyncExitStack, server: MCPServerConfig):
        transport = (server.transport or "stdio").lower()
        if transport == "stdio":
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=server.command,
                args=list(server.args),
                # Always pass a full env so the child inherits PATH etc. (env=None gives
                # the SDK's minimal default), with per-server overrides layered on top.
                env={**os.environ, **server.env},
                cwd=server.cwd or None,
            )
            streams = await stack.enter_async_context(stdio_client(params))
        elif transport in ("streamable-http", "streamable_http", "http"):
            from mcp.client.streamable_http import streamablehttp_client

            streams = await stack.enter_async_context(
                streamablehttp_client(server.url, headers=server.headers or None)
            )
        else:
            raise ValueError(f"Unknown MCP transport '{server.transport}' for server '{server.name}'")
        # streamable-http yields (read, write, get_session_id); stdio yields (read, write).
        # Take the first two so both shapes work across SDK versions.
        return streams[0], streams[1]

    def close(self) -> None:
        """Tear down sessions, the loop, and the thread. Idempotent and exception-safe."""
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        if loop is None:
            return
        # Ask the _serve task to unwind its AsyncExitStack in its own task, then wait.
        if self._stop is not None:
            loop.call_soon_threadsafe(self._stop.set)
        if self._serve_future is not None:
            try:
                self._serve_future.result(timeout=10)
            except Exception:
                pass
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        try:
            loop.close()
        except Exception:
            pass

    # -- query / call --------------------------------------------------------------

    def tools(self) -> list[tuple[MCPServerConfig, Any]]:
        """The ``(server_config, tool_descriptor)`` pairs discovered across all servers."""
        return list(self._tools)

    def call_tool(self, server: str, tool: str, arguments: dict[str, Any] | None, timeout: float | None = None):
        """Invoke a tool on a connected server, blocking for the ``CallToolResult``."""
        if self._loop is None or self._closed:
            raise RuntimeError("MCP manager is not running")
        session = self._sessions.get(server)
        if session is None:
            raise KeyError(f"Unknown MCP server: {server}")
        future = asyncio.run_coroutine_threadsafe(
            session.call_tool(tool, arguments or {}), self._loop
        )
        return future.result(timeout if timeout is not None else self._call_timeout)

    def list_resources(self, server: str | None = None, timeout: float | None = None) -> list[dict[str, Any]]:
        if self._loop is None or self._closed:
            raise RuntimeError("MCP manager is not running")

        async def collect() -> list[dict[str, Any]]:
            records: list[dict[str, Any]] = []
            for name, session in self._sessions.items():
                if server is not None and name != server:
                    continue
                response = await session.list_resources()
                for item in response.resources:
                    records.append({
                        "server": name,
                        "uri": str(item.uri),
                        "name": getattr(item, "name", None),
                        "description": getattr(item, "description", None),
                        "mime_type": getattr(item, "mimeType", None),
                    })
            return records

        future = asyncio.run_coroutine_threadsafe(collect(), self._loop)
        return future.result(timeout if timeout is not None else self._call_timeout)

    def read_resource(self, server: str, uri: str, timeout: float | None = None) -> Any:
        if self._loop is None or self._closed:
            raise RuntimeError("MCP manager is not running")
        session = self._sessions.get(server)
        if session is None:
            raise KeyError(f"Unknown MCP server: {server}")
        future = asyncio.run_coroutine_threadsafe(session.read_resource(uri), self._loop)
        return future.result(timeout if timeout is not None else self._call_timeout)
