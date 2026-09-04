"""Restricted JavaScript workflow orchestration bridge.

The JavaScript context receives only ``meta``, ``args``, ``agent`` and ``pipeline``.
Actual agents remain Python-owned, so every child keeps the normal Polaris sandbox,
permission, accounting and cancellation behavior.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Awaitable, Callable

from jsonschema import Draft202012Validator
from agent_core.tools.base import ExecutionScope
from agent_core.sandbox import SandboxInvocation


class WorkflowError(RuntimeError):
    pass


_MAX_WORKFLOW_FRAME_BYTES = 1024 * 1024


_RUNNER = r"""
const vm = require('node:vm');
const MAX_FRAME = 1024 * 1024;
let nextId = 1, total = 0, initialized = false;
const waiting = new Map();
function send(v) {
  let body = Buffer.from(JSON.stringify(v), 'utf8');
  if (body.length > MAX_FRAME) body = Buffer.from(JSON.stringify({type:'error', error:'result_too_large'}));
  const header = Buffer.allocUnsafe(4); header.writeUInt32BE(body.length, 0);
  process.stdout.write(Buffer.concat([header, body]));
}
function agent(prompt, options = {}) {
  if (++total > 1000) return Promise.reject(new Error('workflow agent limit exceeded'));
  const id = nextId++;
  send({type:'agent', id, prompt:String(prompt).slice(0,100000), options});
  return new Promise((resolve, reject) => waiting.set(id, {resolve, reject}));
}
async function pipeline(items, fn) {
  if (!Array.isArray(items)) throw new TypeError('pipeline items must be an array');
  const out = new Array(items.length); let cursor = 0;
  async function worker() { while (true) { const i = cursor++; if (i >= items.length) return; out[i] = await fn(items[i], i); } }
  await Promise.all(Array.from({length: Math.min(16, items.length)}, worker));
  return out;
}
async function receive(msg) {
  if (!initialized && msg.type === 'init') {
    initialized = true;
    const source = String(msg.source).replace(/^\s*export\s+const\s+meta\s*=/m, 'const meta =');
    if (/\bimport\s*(?:\(|[^('])/.test(source)) { send({type:'error', error:'module loading is disabled'}); return; }
    const context = vm.createContext(Object.freeze({args:Object.freeze(msg.args || {}), agent, pipeline}), {
      codeGeneration: {strings:false, wasm:false}
    });
    try {
      const script = new vm.Script(`(async () => { ${source}\n })()`, {filename:'plugin-workflow.js'});
      const result = await script.runInContext(context, {timeout:1000});
      send({type:'result', result});
    } catch (e) { send({type:'error', error:String(e && e.message || e)}); }
    return;
  }
  if (msg.type === 'agent_result') {
    const pending = waiting.get(msg.id); if (!pending) return; waiting.delete(msg.id);
    if (msg.ok) pending.resolve(msg.result); else pending.resolve(null);
  }
}
let input = Buffer.alloc(0);
process.stdin.on('data', chunk => {
  input = Buffer.concat([input, chunk]);
  while (input.length >= 4) {
    const size = input.readUInt32BE(0);
    if (size > MAX_FRAME) { send({type:'error', error:'frame_too_large'}); process.exitCode = 2; return; }
    if (input.length < 4 + size) return;
    const body = input.subarray(4, 4 + size); input = input.subarray(4 + size);
    let msg; try { msg = JSON.parse(body.toString('utf8')); }
    catch (_) { send({type:'error', error:'invalid_json'}); continue; }
    if (!msg || typeof msg !== 'object' || Array.isArray(msg)) {
      send({type:'error', error:'invalid_frame_type'}); continue;
    }
    void receive(msg);
  }
});
process.stdin.on('end', () => {
  if (input.length) send({type:'error', error:'truncated_frame'});
});
"""


def _encode_frame(message: dict[str, Any]) -> bytes:
    body = json.dumps(message, ensure_ascii=False, default=str).encode("utf-8")
    if len(body) > _MAX_WORKFLOW_FRAME_BYTES:
        raise WorkflowError("workflow frame exceeded size limit")
    return struct.pack(">I", len(body)) + body


async def _read_frame(stream: asyncio.StreamReader) -> dict[str, Any]:
    try:
        header = await stream.readexactly(4)
    except asyncio.IncompleteReadError as exc:
        code = "truncated frame header" if exc.partial else "runtime closed"
        raise WorkflowError(f"workflow {code}") from exc
    size = struct.unpack(">I", header)[0]
    if size > _MAX_WORKFLOW_FRAME_BYTES:
        raise WorkflowError("workflow frame exceeded size limit")
    try:
        body = await stream.readexactly(size)
    except asyncio.IncompleteReadError as exc:
        raise WorkflowError("workflow truncated frame body") from exc
    try:
        message = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise WorkflowError("workflow runtime returned invalid JSON") from exc
    if not isinstance(message, dict):
        raise WorkflowError("workflow runtime returned a non-object frame")
    return message


async def _terminate_workflow_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        if os.name == "nt":
            from agent_core.hook_adapters import (
                _windows_descendant_pids,
                _windows_terminate_pid,
            )

            def terminate_windows() -> None:
                descendants = _windows_descendant_pids(process.pid)
                for pid in [*reversed(descendants), process.pid]:
                    if _windows_terminate_pid(pid):
                        continue
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                        timeout=3,
                    )

            await asyncio.to_thread(terminate_windows)
        else:
            try:
                getattr(os, "killpg")(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(process.wait(), timeout=1)
        except asyncio.TimeoutError:
            if os.name != "nt":
                try:
                    getattr(os, "killpg")(process.pid, getattr(signal, "SIGKILL", 9))
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            await process.wait()


class WorkflowRuntime:
    def __init__(self, *, max_concurrency: int = 16, max_agents: int = 1000) -> None:
        self.max_concurrency = max(1, min(16, max_concurrency))
        self.max_agents = max(1, min(1000, max_agents))

    async def run(
        self,
        source: str,
        args: dict[str, Any],
        agent_factory: Callable[..., Awaitable[str]],
        *,
        timeout: float = 3600.0,
        sandbox: Any = None,
        workspace: str | Path | None = None,
        require_sandbox: bool = False,
    ) -> Any:
        if require_sandbox and (sandbox is None or not sandbox.is_enabled()):
            raise WorkflowError("plugin workflows require an enforcing sandbox backend")
        guest_mode = bool(sandbox is not None and sandbox.uses_guest)
        node = "node" if guest_mode else shutil.which("node")
        if node is None:
            raise WorkflowError("Node 24 or newer is required for plugin workflows")
        if not guest_mode:
            version = await asyncio.create_subprocess_exec(
                node, "--version", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await version.communicate()
            match = re.match(rb"v(\d+)", stdout.strip())
            if match is None or int(match.group(1)) < 24:
                raise WorkflowError("Node 24 or newer is required for plugin workflows")
        private_temp = Path(tempfile.mkdtemp(prefix="polaris-workflow-"))
        argv: list[str] = [node, "--permission", "-e", _RUNNER]
        if sandbox is not None and sandbox.is_enabled():
            invocation = SandboxInvocation.create(
                argv,
                guest_argv=["@node", "--permission", "-e", _RUNNER],
                required_guest_capabilities=("node",),
                scope=ExecutionScope.for_workspace(
                    Path(workspace or Path.cwd()).resolve(),
                    private_temp=private_temp,
                    network="deny",
                    workspace_writable=False,
                ),
            )
            wrapped, shell = sandbox.wrap_invocation(invocation)
            if shell or not isinstance(wrapped, list) or not wrapped:
                shutil.rmtree(private_temp, ignore_errors=True)
                raise WorkflowError("workflow sandbox did not return explicit argv")
            argv = [str(item) for item in wrapped]
        minimal_env = {
            key: value
            for key in (
                "PATH", "PATHEXT", "SystemRoot", "COMSPEC", "WINDIR", "TMP", "TEMP",
                # OCI client configuration only; none of these are forwarded into
                # the Linux guest because the wrapped argv has no --env flags.
                "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "XDG_CONFIG_HOME",
                "XDG_RUNTIME_DIR", "CONTAINER_HOST", "CONTAINERS_CONF",
                "CONTAINERS_STORAGE_CONF", "DOCKER_HOST", "DOCKER_CONTEXT",
            )
            if (value := os.environ.get(key)) is not None
        }
        try:
            process_options: dict[str, Any] = {}
            if os.name == "nt":
                process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                process_options["start_new_session"] = True
            process = await asyncio.create_subprocess_exec(
                *(str(item) for item in argv),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=minimal_env,
                **process_options,
            )
        except Exception:
            shutil.rmtree(private_temp, ignore_errors=True)
            raise
        assert process.stdin is not None and process.stdout is not None
        process_stdin = process.stdin
        process_stdout = process.stdout
        write_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks: set[asyncio.Task[None]] = set()
        count = 0

        async def send(message: dict[str, Any]) -> None:
            try:
                encoded = _encode_frame(message)
            except WorkflowError:
                if message.get("type") != "agent_result":
                    raise
                encoded = _encode_frame(
                    {"type": "agent_result", "id": message.get("id"), "ok": False,
                     "error": "result_too_large"}
                )
            async with write_lock:
                process_stdin.write(encoded)
                await process_stdin.drain()

        async def handle(message: dict[str, Any]) -> None:
            nonlocal count
            count += 1
            request_id = message.get("id")
            if count > self.max_agents:
                await send({"type": "agent_result", "id": request_id, "ok": False})
                return
            raw_options = message.get("options")
            options: dict[str, Any] = raw_options if isinstance(raw_options, dict) else {}
            model = str(options.get("model") or "") or None
            schema = options.get("schema") if isinstance(options.get("schema"), dict) else None
            prompt = str(message.get("prompt") or "")
            if schema is not None:
                prompt += (
                    "\n\nReturn only a JSON value matching this JSON Schema; do not use a "
                    "Markdown fence:\n" + json.dumps(schema, ensure_ascii=False)
                )
            try:
                async with semaphore:
                    result: Any = await agent_factory(prompt, "read_only", model)
                if schema is not None:
                    result = json.loads(result)
                    Draft202012Validator(schema).validate(result)
                await send({"type": "agent_result", "id": request_id, "ok": True, "result": result})
            except Exception:
                await send({"type": "agent_result", "id": request_id, "ok": False})

        try:
            await send({"type": "init", "source": source, "args": args})
            async with asyncio.timeout(timeout):
                while True:
                    try:
                        message = await _read_frame(process_stdout)
                    except WorkflowError as exc:
                        if str(exc) != "workflow runtime closed":
                            raise
                        stderr = await process.stderr.read(501) if process.stderr else b""
                        raise WorkflowError(
                            "workflow runtime exited unexpectedly: "
                            + stderr.decode("utf-8", "replace")[:500]
                        ) from exc
                    if message.get("type") == "agent":
                        task = asyncio.create_task(handle(message))
                        tasks.add(task)
                        task.add_done_callback(tasks.discard)
                    elif message.get("type") == "result":
                        await asyncio.gather(*tasks, return_exceptions=True)
                        return message.get("result")
                    elif message.get("type") == "error":
                        raise WorkflowError(str(message.get("error") or "workflow failed"))
                    else:
                        raise WorkflowError("workflow runtime returned an invalid frame type")
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await _terminate_workflow_process(process)
            shutil.rmtree(private_temp, ignore_errors=True)
