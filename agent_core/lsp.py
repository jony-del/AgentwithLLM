"""Minimal stdlib JSON-RPC/LSP client with lazy server lifecycle."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from urllib.parse import unquote, urlparse

from agent_core.tool_config import LSPServerConfig, LSPToolConfig
from agent_core.tools.base import ExecutionScope


def path_to_uri(path: Path) -> str:
    return path.resolve().as_uri()


def uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"unsupported LSP URI: {uri}")
    path = unquote(parsed.path)
    if os.name == "nt" and path.startswith("/"):
        path = path[1:]
    return Path(path)


@dataclass(slots=True)
class _Server:
    config: LSPServerConfig
    workspace: Path
    process: asyncio.subprocess.Process | None = None
    reader_task: asyncio.Task[None] | None = None
    stderr_task: asyncio.Task[None] | None = None
    pending: dict[int, asyncio.Future[Any]] = field(default_factory=dict)
    next_id: int = 1
    restarts: int = 0
    initialized: bool = False
    documents: dict[str, int] = field(default_factory=dict)
    diagnostics: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stderr_tail: bytearray = field(default_factory=bytearray)
    sandbox: Any = None
    private_temp: Path | None = None

    async def start(self) -> None:
        executable = shutil.which(self.config.command)
        if executable is None and Path(self.config.command).is_file():
            executable = str(Path(self.config.command).resolve())
        if executable is None:
            raise RuntimeError(f"LSP server executable not found: {self.config.command}")
        env = {**os.environ, **self.config.env}
        for key in list(env):
            if key.casefold() in {"http_proxy", "https_proxy", "all_proxy"}:
                env.pop(key, None)
        env["NO_PROXY"] = "*"
        argv: object = [executable, *self.config.args]
        if self.sandbox is not None:
            if self.private_temp is None:
                self.private_temp = Path(tempfile.mkdtemp(prefix=f"polaris-lsp-{self.config.name}-"))
            argv, shell = self.sandbox.wrap(
                argv, False,
                scope=ExecutionScope.for_workspace(
                    self.workspace, private_temp=self.private_temp, network="deny",
                    workspace_writable=False,
                ),
            )
            if shell or isinstance(argv, str):
                raise RuntimeError("LSP sandbox returned a shell command; explicit argv is required")
        if not isinstance(argv, (list, tuple)):
            raise RuntimeError("LSP command provider returned invalid argv")
        self.process = await asyncio.create_subprocess_exec(
            *(str(item) for item in argv),
            cwd=str(self.workspace),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.reader_task = asyncio.create_task(self._read_loop(), name=f"lsp-{self.config.name}-reader")
        self.stderr_task = asyncio.create_task(self._stderr_loop(), name=f"lsp-{self.config.name}-stderr")
        root_uri = path_to_uri(self.workspace)
        await self.request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": root_uri,
                "workspaceFolders": [{"uri": root_uri, "name": self.workspace.name}],
                "capabilities": {"textDocument": {"publishDiagnostics": {"relatedInformation": True}}},
                "initializationOptions": self.config.initialization_options,
            },
            allow_restart=False,
        )
        await self.notify("initialized", {})
        self.initialized = True

    async def _read_loop(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        reader = self.process.stdout
        try:
            while True:
                headers: dict[str, str] = {}
                while True:
                    line = await reader.readline()
                    if not line:
                        raise EOFError("LSP server closed stdout")
                    if line in {b"\r\n", b"\n"}:
                        break
                    key, _, value = line.decode("ascii", errors="replace").partition(":")
                    headers[key.casefold().strip()] = value.strip()
                length = int(headers.get("content-length", "0"))
                if length <= 0 or length > 16 * 1024 * 1024:
                    raise ValueError(f"invalid LSP Content-Length: {length}")
                message = json.loads((await reader.readexactly(length)).decode("utf-8"))
                self._route(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(RuntimeError(f"LSP server {self.config.name} crashed: {exc}"))
            self.pending.clear()
            self.initialized = False

    async def _stderr_loop(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        while True:
            chunk = await self.process.stderr.read(4096)
            if not chunk:
                return
            self.stderr_tail.extend(chunk)
            if len(self.stderr_tail) > 32_768:
                del self.stderr_tail[:-32_768]

    def _route(self, message: Any) -> None:
        if not isinstance(message, dict):
            return
        if "id" in message:
            future = self.pending.pop(message["id"], None)
            if future is None or future.done():
                return
            if "error" in message:
                future.set_exception(RuntimeError(f"LSP error: {message['error']}"))
            else:
                future.set_result(message.get("result"))
            return
        if message.get("method") == "textDocument/publishDiagnostics":
            params = message.get("params", {})
            if isinstance(params, dict):
                self.diagnostics[str(params.get("uri", ""))] = list(params.get("diagnostics", []))

    async def _write(self, message: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None or self.process.returncode is not None:
            raise RuntimeError(f"LSP server {self.config.name} is not running")
        body = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        framed = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
        async with self.write_lock:
            self.process.stdin.write(framed)
            await self.process.stdin.drain()

    async def request(self, method: str, params: object, *, allow_restart: bool = True) -> Any:
        if self.process is None or self.process.returncode is not None:
            if not allow_restart:
                raise RuntimeError(f"LSP server {self.config.name} is not running")
            raise RuntimeError(f"LSP server {self.config.name} needs restart")
        request_id = self.next_id
        self.next_id += 1
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        await self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        try:
            return await asyncio.wait_for(future, self.config.timeout)
        except asyncio.TimeoutError:
            self.pending.pop(request_id, None)
            raise TimeoutError(f"LSP request timed out: {method}") from None

    async def notify(self, method: str, params: object) -> None:
        await self._write({"jsonrpc": "2.0", "method": method, "params": params})

    async def close(self) -> None:
        if self.process is None:
            if self.private_temp is not None:
                await asyncio.to_thread(shutil.rmtree, self.private_temp, True)
            return
        try:
            if self.process.returncode is None and self.initialized:
                await self.request("shutdown", {}, allow_restart=False)
                await self.notify("exit", {})
        except Exception:
            pass
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 2)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        for task in (self.reader_task, self.stderr_task):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(task for task in (self.reader_task, self.stderr_task) if task is not None),
            return_exceptions=True,
        )
        if self.private_temp is not None:
            await asyncio.to_thread(shutil.rmtree, self.private_temp, True)


class LSPManager:
    def __init__(
        self, config: LSPToolConfig, workspace: str | Path, *, event_sink: Any = None,
        sandbox: Any = None,
    ) -> None:
        self.config = config
        self.workspace = Path(workspace).resolve()
        self.event_sink = event_sink
        self.sandbox = sandbox
        self._servers: dict[str, _Server] = {}

    def rebind_workspace(self, workspace: str | Path) -> None:
        if self._servers:
            raise RuntimeError("cannot switch workspace while LSP servers are running")
        self.workspace = Path(workspace).resolve()

    def _config_for(self, path: Path) -> LSPServerConfig:
        suffix = path.suffix.lower()
        for config in self.config.servers:
            if suffix in config.extensions or suffix.lstrip(".") in config.extensions:
                return config
        raise ValueError(f"no LSP server configured for extension {suffix!r}")

    async def _server_for(self, path: Path) -> _Server:
        config = self._config_for(path)
        server = self._servers.get(config.name)
        if server is not None and server.process is not None and server.process.returncode is None and server.initialized:
            return server
        restarts = server.restarts + 1 if server is not None else 0
        if restarts > self.config.max_restarts:
            raise RuntimeError(f"LSP server {config.name} exceeded restart limit")
        if server is not None:
            await server.close()
        server = _Server(config, self.workspace, restarts=restarts, sandbox=self.sandbox)
        self._servers[config.name] = server
        try:
            await server.start()
        except Exception as exc:
            await server.close()
            await self._event("lsp_server_state", {"server": config.name, "state": "failed", "error": str(exc)})
            raise
        await self._event("lsp_server_state", {"server": config.name, "state": "running", "restart": restarts})
        return server

    async def _event(self, kind: str, payload: dict[str, object]) -> None:
        if self.event_sink is not None:
            try:
                await self.event_sink(kind, payload)
            except Exception:
                pass

    def _resolve(self, raw: str | Path) -> Path:
        path = (self.workspace / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
        if path != self.workspace and self.workspace not in path.parents:
            raise ValueError("LSP path escapes workspace")
        return path

    async def open_document(self, path: str | Path) -> tuple[_Server, Path]:
        resolved = self._resolve(path)
        server = await self._server_for(resolved)
        uri = path_to_uri(resolved)
        text = resolved.read_text(encoding="utf-8")
        if uri not in server.documents:
            language = server.config.extensions.get(resolved.suffix.lower(), resolved.suffix.lstrip("."))
            server.documents[uri] = 1
            await server.notify("textDocument/didOpen", {
                "textDocument": {"uri": uri, "languageId": language, "version": 1, "text": text}
            })
        return server, resolved

    async def notify_saved(self, path: str | Path) -> None:
        resolved = self._resolve(path)
        try:
            server = await self._server_for(resolved)
        except ValueError:
            return
        uri = path_to_uri(resolved)
        if uri not in server.documents:
            await self.open_document(resolved)
            return
        version = server.documents[uri] + 1
        server.documents[uri] = version
        text = resolved.read_text(encoding="utf-8")
        await server.notify("textDocument/didChange", {
            "textDocument": {"uri": uri, "version": version}, "contentChanges": [{"text": text}]
        })
        await server.notify("textDocument/didSave", {"textDocument": {"uri": uri}, "text": text})

    async def request(self, operation: str, *, path: str | None = None, line: int = 0, character: int = 0, query: str = "") -> Any:
        if operation == "workspace_symbols":
            if not self.config.servers:
                raise ValueError("no LSP servers configured")
            config = self.config.servers[0]
            server = self._servers.get(config.name)
            if server is None:
                server = _Server(config, self.workspace)
                self._servers[config.name] = server
                await server.start()
            return await server.request("workspace/symbol", {"query": query})
        if path is None:
            raise ValueError(f"path is required for {operation}")
        server, resolved = await self.open_document(path)
        uri = path_to_uri(resolved)
        text_document = {"uri": uri}
        position = {"line": max(0, line), "character": max(0, character)}
        methods = {
            "definition": "textDocument/definition",
            "references": "textDocument/references",
            "hover": "textDocument/hover",
            "document_symbols": "textDocument/documentSymbol",
            "implementation": "textDocument/implementation",
            "prepare_call_hierarchy": "textDocument/prepareCallHierarchy",
            "incoming_calls": "callHierarchy/incomingCalls",
            "outgoing_calls": "callHierarchy/outgoingCalls",
        }
        if operation == "diagnostics":
            return server.diagnostics.get(uri, [])
        method = methods.get(operation)
        if method is None:
            raise ValueError(f"unsupported LSP operation: {operation}")
        if operation == "document_symbols":
            params: dict[str, Any] = {"textDocument": text_document}
        elif operation == "references":
            params = {"textDocument": text_document, "position": position, "context": {"includeDeclaration": True}}
        elif operation in {"incoming_calls", "outgoing_calls"}:
            prepared = await server.request("textDocument/prepareCallHierarchy", {"textDocument": text_document, "position": position})
            item = prepared[0] if isinstance(prepared, list) and prepared else None
            if item is None:
                return []
            params = {"item": item}
        else:
            params = {"textDocument": text_document, "position": position}
        return await server.request(method, params)

    async def close(self) -> None:
        await asyncio.gather(*(server.close() for server in self._servers.values()), return_exceptions=True)
        self._servers.clear()
