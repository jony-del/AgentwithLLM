from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import ipaddress
import logging
import os
import re
import socket
import subprocess
import threading
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Callable, cast
from urllib.parse import urlsplit

from pydantic import FileUrl

from agent_core.env_security import expand_host_env
from agent_core.mcp.config import MCPConfig, MCPServerConfig
from agent_core.process_tree import terminate_pid_tree


logger = logging.getLogger(__name__)

# How often a scope-aware wait wakes to check the cancellation token.
_CANCEL_POLL_INTERVAL = 0.05


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
        health_timeout: float = 5.0,
        channel_servers: set[str] | None = None,
        notification_sink: Callable[[str, str, dict[str, str]], None] | None = None,
    ) -> None:
        self._config = config
        self._connect_timeout = connect_timeout
        self._call_timeout = call_timeout
        self._health_timeout = health_timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._serve_future: Any = None  # concurrent.futures.Future
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._sessions: dict[str, Any] = {}
        self._tools: list[tuple[MCPServerConfig, Any]] = []
        self._closed = False
        self._channel_servers = set(channel_servers or ())
        self._notification_sink = notification_sink
        # Sessions left possibly desynced by a timed-out/cancelled in-flight call;
        # the next call on one health-checks it (and reconnects once) first.
        self._suspect: set[str] = set()
        self._health_lock = threading.Lock()
        # Exit stacks owned by reconnects (the original sessions live in _serve's).
        self._extra_stacks: dict[str, AsyncExitStack] = {}

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
        session = await self._open_session(stack, server)
        initialized = await session.initialize()
        if server.name in self._channel_servers:
            experimental = getattr(initialized.capabilities, "experimental", None) or {}
            if "claude/channel" not in experimental:
                raise RuntimeError(
                    f"configured channel MCP server {server.name} did not declare claude/channel"
                )
        self._sessions[server.name] = session
        listed = await session.list_tools()
        for descriptor in listed.tools:
            self._tools.append((server, descriptor))

    async def _open_session(self, stack: AsyncExitStack, server: MCPServerConfig):
        """Open the transport and enter a ``ClientSession`` onto ``stack``."""
        from mcp import ClientSession, types

        read, write = await self._open_transport(stack, server)
        async def handle_message(message: Any) -> None:
            if server.name not in self._channel_servers or self._notification_sink is None:
                return
            try:
                raw = message.model_dump(by_alias=True)
                root = raw.get("root", raw) if isinstance(raw, dict) else {}
                if not isinstance(root, dict) or root.get("method") != "notifications/claude/channel":
                    return
                params = root.get("params")
                if not isinstance(params, dict) or not isinstance(params.get("content"), str):
                    return
                raw_meta = params.get("meta")
                meta = {
                    str(key): str(value)[:1000]
                    for key, value in raw_meta.items()
                    if re.fullmatch(r"[A-Za-z0-9_]+", str(key))
                } if isinstance(raw_meta, dict) else {}
                self._notification_sink(server.name, params["content"][:16_384], meta)
            except Exception:
                return

        async def list_roots(_context: Any) -> Any:
            roots = []
            for raw in server.roots:
                expanded = raw
                uri = expanded if "://" in expanded else Path(expanded).expanduser().resolve().as_uri()
                roots.append(types.Root(uri=FileUrl(uri), name=Path(expanded).name or None))
            return types.ListRootsResult(roots=roots)

        return await stack.enter_async_context(ClientSession(
            read,
            write,
            message_handler=handle_message,
            list_roots_callback=cast(Any, list_roots) if server.roots else None,
        ))

    async def _open_transport(self, stack: AsyncExitStack, server: MCPServerConfig):
        transport = (server.transport or "stdio").lower()
        public_addresses: tuple[str, ...] = ()
        if server.network_policy == "public-only":
            if transport not in {"streamable-http", "streamable_http", "http"}:
                raise ValueError("untrusted remote MCP requires streamable HTTPS")
            if server.headers or server.headers_helper or server.env or server.oauth:
                raise ValueError("untrusted remote MCP must not receive credentials or host environment")
            public_addresses = _validate_public_remote(server.url)
        allow_sensitive = (not server.discovered) or server.allow_sensitive_env
        headers = {
            key: _expand_env(value, allow_sensitive=allow_sensitive)
            for key, value in server.headers.items()
        }
        if server.headers_helper:
            headers.update(_run_headers_helper(server))
        if transport == "stdio":
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=server.command,
                args=list(server.args),
                # Never expose the full Polaris environment to an MCP child.  A small
                # runtime baseline keeps executable lookup/platform startup working;
                # everything else must be explicitly configured for this server.
                env=_minimal_stdio_env(server),
                cwd=server.cwd or None,
            )
            streams = await stack.enter_async_context(stdio_client(params))
        elif transport in ("streamable-http", "streamable_http", "http"):
            import httpx2
            from mcp.client.streamable_http import streamable_http_client

            timeout = httpx2.Timeout(server.timeout, read=max(server.timeout, 300.0))
            if public_addresses:
                client = _public_http_client(
                    server.url,
                    public_addresses,
                    timeout=timeout,
                )
            else:
                client = httpx2.AsyncClient(
                    headers=headers or None,
                    timeout=timeout,
                    follow_redirects=False,
                    trust_env=False,
                )
            http_client = await stack.enter_async_context(client)
            streams = await stack.enter_async_context(
                streamable_http_client(server.url, http_client=http_client)
            )
        elif transport == "sse":
            from mcp.client.sse import sse_client

            streams = await stack.enter_async_context(
                sse_client(server.url, headers=headers or None)
            )
        elif transport in {"ws", "wss", "websocket"}:
            from mcp.client.websocket import websocket_client

            streams = await stack.enter_async_context(websocket_client(server.url))
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
                # A wedged transport exit (e.g. an unresponsive stdio child) must
                # not hang shutdown: cancel the serve task so the stack unwinds
                # through CancelledError instead.
                logger.warning("MCP close: serve task did not unwind within 10s; cancelling it")
                self._serve_future.cancel()
        # Close reconnect-owned stacks and cancel anything still pending (a hung
        # call_tool coroutine, a wedged unwind) so loop.stop() is actually reached.
        try:
            asyncio.run_coroutine_threadsafe(self._shutdown_tasks(), loop).result(timeout=5)
        except Exception:
            logger.warning("MCP close: event loop did not drain pending tasks within 5s")
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logger.warning("MCP close: loop thread is still alive after 5s")
        try:
            loop.close()
        except Exception:
            pass

    async def _shutdown_tasks(self) -> None:
        """Runs on the MCP loop: close reconnect stacks, then cancel stragglers."""
        stacks, self._extra_stacks = list(self._extra_stacks.values()), {}
        for stack in stacks:
            with contextlib.suppress(Exception):
                await stack.aclose()
        current = asyncio.current_task()
        pending = [
            task for task in asyncio.all_tasks()
            if task is not current and not task.done()
        ]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    # -- query / call --------------------------------------------------------------

    def tools(self) -> list[tuple[MCPServerConfig, Any]]:
        """The ``(server_config, tool_descriptor)`` pairs discovered across all servers."""
        return list(self._tools)

    def call_tool(
        self,
        server: str,
        tool: str,
        arguments: dict[str, Any] | None,
        timeout: float | None = None,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ):
        """Invoke a tool on a connected server, blocking for the ``CallToolResult``.

        ``should_cancel`` (e.g. an execution scope's ``cancelled``) is polled at a
        short interval while waiting; when it reports cancellation the in-flight
        future is cancelled and ``asyncio.CancelledError`` is raised instead of
        parking this thread until the timeout.
        """
        if self._loop is None or self._closed:
            raise RuntimeError("MCP manager is not running")
        if server not in self._sessions:
            raise KeyError(f"Unknown MCP server: {server}")
        self._ensure_healthy(server)
        session = self._sessions[server]
        future = asyncio.run_coroutine_threadsafe(
            session.call_tool(tool, arguments or {}), self._loop
        )
        bounded = timeout if timeout is not None else server_config.timeout if (server_config := next((item for item in self._config.servers if item.name == server), None)) is not None else self._call_timeout
        return self._bounded_result(
            future,
            bounded,
            suspect=server,
            should_cancel=should_cancel,
            message=f"MCP tool call timed out after {bounded:.3f}s",
        )

    def list_resources(self, server: str | None = None, timeout: float | None = None) -> list[dict[str, Any]]:
        if self._loop is None or self._closed:
            raise RuntimeError("MCP manager is not running")
        if server is not None:
            self._ensure_healthy(server)

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
        bounded = timeout if timeout is not None else self._call_timeout
        # Only a single-server listing can attribute a timeout to that session;
        # a fan-out listing leaves the suspect set untouched.
        return self._bounded_result(
            future,
            bounded,
            suspect=server,
            message=f"MCP resource listing timed out after {bounded:.3f}s",
        )

    def read_resource(self, server: str, uri: str, timeout: float | None = None) -> Any:
        if self._loop is None or self._closed:
            raise RuntimeError("MCP manager is not running")
        if server not in self._sessions:
            raise KeyError(f"Unknown MCP server: {server}")
        self._ensure_healthy(server)
        session = self._sessions[server]
        future = asyncio.run_coroutine_threadsafe(session.read_resource(uri), self._loop)
        bounded = timeout if timeout is not None else self._call_timeout
        return self._bounded_result(
            future,
            bounded,
            suspect=server,
            message=f"MCP resource read timed out after {bounded:.3f}s",
        )

    # -- timeout recovery ----------------------------------------------------------

    def _bounded_result(
        self,
        future: concurrent.futures.Future[Any],
        bounded: float,
        *,
        suspect: str | None = None,
        should_cancel: Callable[[], bool] | None = None,
        message: str,
    ) -> Any:
        """Block the calling thread on an MCP-loop future, bounded by ``bounded``.

        On timeout the future is cancelled and ``suspect`` (a server name, if the
        wait was attributable to one session) is marked for a health check before
        its next use. With ``should_cancel`` the wait wakes at a short interval so
        a cancelled execution scope interrupts it promptly; the thread never parks
        longer than ``bounded`` either way.
        """
        if should_cancel is None:
            try:
                return future.result(bounded)
            except concurrent.futures.TimeoutError:
                future.cancel()
                if suspect is not None:
                    self._suspect.add(suspect)
                raise TimeoutError(message) from None
        deadline = time.monotonic() + bounded
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                future.cancel()
                if suspect is not None:
                    self._suspect.add(suspect)
                raise TimeoutError(message) from None
            try:
                return future.result(min(remaining, _CANCEL_POLL_INTERVAL))
            except concurrent.futures.TimeoutError:
                if should_cancel():
                    future.cancel()
                    if suspect is not None:
                        self._suspect.add(suspect)
                    raise asyncio.CancelledError("cancelled") from None

    def _ensure_healthy(self, server: str) -> None:
        """Health-check a suspect session, dropping and reconnecting it once on failure."""
        if self._closed or self._loop is None or server not in self._suspect:
            return
        with self._health_lock:
            if server not in self._suspect:
                return  # another worker thread already recovered it
            session = self._sessions.get(server)
            probe = None
            if session is not None:
                probe = getattr(session, "send_ping", None) or getattr(session, "list_tools", None)
            if probe is not None:
                try:
                    asyncio.run_coroutine_threadsafe(probe(), self._loop).result(self._health_timeout)
                    self._suspect.discard(server)
                    return
                except Exception:
                    pass
            server_config = next(
                (item for item in self._config.servers if item.name == server), None
            )
            if server_config is None:
                raise KeyError(f"Unknown MCP server: {server}")
            asyncio.run_coroutine_threadsafe(
                self._reconnect_session(server_config), self._loop
            ).result(self._connect_timeout)
            self._suspect.discard(server)

    async def _reconnect_session(self, server: MCPServerConfig) -> None:
        """Runs on the MCP loop: replace one server's session with a fresh one.

        The old session's context stays on its original exit stack (anyio requires
        exiting in the entering task); the new stack is owned here and unwound by
        ``close()``.
        """
        stack = AsyncExitStack()
        try:
            session = await self._open_session(stack, server)
            await session.initialize()
        except BaseException:
            with contextlib.suppress(Exception):
                await stack.aclose()
            raise
        previous = self._extra_stacks.get(server.name)
        self._extra_stacks[server.name] = stack
        self._sessions[server.name] = session
        if previous is not None:
            with contextlib.suppress(Exception):
                await previous.aclose()


def _minimal_stdio_env(server: MCPServerConfig) -> dict[str, str]:
    baseline_names = {
        "PATH", "PATHEXT", "SystemRoot", "COMSPEC", "WINDIR",
        "TMP", "TEMP", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE",
    }
    result = {
        name: value
        for name in baseline_names
        if (value := os.environ.get(name)) is not None
    }
    allow_sensitive = (not server.discovered) or server.allow_sensitive_env
    result.update(
        {key: _expand_env(value, allow_sensitive=allow_sensitive) for key, value in server.env.items()}
    )
    return result


def _validate_public_remote(url: str) -> tuple[str, ...]:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("discovered remote MCP URL must be credential-free HTTPS")
    try:
        addresses = {
            str(item[4][0])
            for item in socket.getaddrinfo(
                parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM
            )
        }
    except OSError as exc:
        raise ValueError(f"could not resolve remote MCP host: {parsed.hostname}") from exc
    if not addresses:
        raise ValueError(f"remote MCP host has no addresses: {parsed.hostname}")
    for raw in addresses:
        if not ipaddress.ip_address(raw).is_global:
            raise ValueError(f"remote MCP host resolves to a non-public address: {raw}")
    return tuple(sorted(addresses))


def _public_http_client(
    url: str,
    addresses: tuple[str, ...],
    *,
    timeout: Any,
) -> Any:
    """Build a proxy-free, redirect-free client pinned to prevalidated DNS answers."""

    import httpcore2
    import httpx2
    from httpcore2._backends.auto import AutoBackend

    expected_host = (urlsplit(url).hostname or "").casefold()

    class _PinnedBackend(httpcore2.AsyncNetworkBackend):
        def __init__(self) -> None:
            self._delegate = AutoBackend()

        async def connect_tcp(
            self,
            host: str,
            port: int,
            timeout: float | None = None,
            local_address: str | None = None,
            socket_options: Any = None,
        ) -> Any:
            if host.casefold().rstrip(".") != expected_host.rstrip("."):
                raise OSError("remote MCP attempted to connect to an unpinned host")
            last_error: BaseException | None = None
            for address in addresses:
                try:
                    return await self._delegate.connect_tcp(
                        address,
                        port,
                        timeout=timeout,
                        local_address=local_address,
                        socket_options=socket_options,
                    )
                except Exception as exc:  # try every address from the validated set
                    last_error = exc
            assert last_error is not None
            raise last_error

        async def connect_unix_socket(
            self, path: str, timeout: float | None = None, socket_options: Any = None
        ) -> Any:
            raise OSError("remote MCP cannot use Unix sockets")

        async def sleep(self, seconds: float) -> None:
            await self._delegate.sleep(seconds)

    transport = httpx2.AsyncHTTPTransport(trust_env=False, retries=0)
    # httpx2 does not yet expose resolver injection, while httpcore2 does. Replacing
    # the freshly-created pool's backend keeps TLS SNI/certificate validation bound to
    # the original hostname while the TCP connection uses only the approved addresses.
    transport._pool._network_backend = _PinnedBackend()
    return httpx2.AsyncClient(
        transport=transport,
        timeout=timeout,
        follow_redirects=False,
        trust_env=False,
    )


def _expand_env(value: str, *, allow_sensitive: bool = True) -> str:
    return expand_host_env(value, allow_sensitive=allow_sensitive)


def _run_headers_helper(server: MCPServerConfig) -> dict[str, str]:
    env = {
        **os.environ,
        "CLAUDE_CODE_MCP_SERVER_NAME": server.name,
        "CLAUDE_CODE_MCP_SERVER_URL": server.url,
    }
    try:
        # Popen (not subprocess.run) so the timeout path can kill the whole tree;
        # run() would only kill the shell and leak the helper's children.
        proc = subprocess.Popen(
            server.headers_helper,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            # A dedicated session makes the shell a group leader, so the timeout
            # path can signal the helper's whole process group (POSIX).
            start_new_session=os.name != "nt",
        )
    except OSError as exc:
        raise RuntimeError(f"MCP headersHelper failed: {type(exc).__name__}") from exc
    try:
        stdout, _stderr = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired as exc:
        terminate_pid_tree(proc.pid)
        with contextlib.suppress(Exception):
            proc.wait(timeout=1)
        raise RuntimeError(f"MCP headersHelper failed: {type(exc).__name__}") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"MCP headersHelper failed: {type(exc).__name__}") from exc
    if proc.returncode != 0:
        raise RuntimeError("MCP headersHelper returned a non-zero exit status")
    try:
        value = json.loads(stdout)
    except ValueError as exc:
        raise RuntimeError("MCP headersHelper did not return JSON") from exc
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise RuntimeError("MCP headersHelper must return an object of string headers")
    return value
