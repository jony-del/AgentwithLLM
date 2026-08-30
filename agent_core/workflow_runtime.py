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
import tempfile
from pathlib import Path
from typing import Any, Awaitable, Callable

from jsonschema import Draft202012Validator
from agent_core.tools.base import ExecutionScope
from agent_core.sandbox import SandboxInvocation


class WorkflowError(RuntimeError):
    pass


_RUNNER = r"""
const vm = require('node:vm');
const readline = require('node:readline');
const rl = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
let nextId = 1, total = 0, initialized = false;
const waiting = new Map();
function send(v) { process.stdout.write(JSON.stringify(v) + '\n'); }
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
rl.on('line', async line => {
  let msg; try { msg = JSON.parse(line); } catch (_) { return; }
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
});
"""


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
            process = await asyncio.create_subprocess_exec(
                *(str(item) for item in argv),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=minimal_env,
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
            encoded = json.dumps(message, ensure_ascii=False, default=str).encode("utf-8") + b"\n"
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

        await send({"type": "init", "source": source, "args": args})
        try:
            async with asyncio.timeout(timeout):
                while True:
                    line = await process_stdout.readline()
                    if not line:
                        stderr = await process.stderr.read() if process.stderr else b""
                        raise WorkflowError(
                            "workflow runtime exited unexpectedly: "
                            + stderr.decode("utf-8", "replace")[:500]
                        )
                    if len(line) > 1_000_000:
                        raise WorkflowError("workflow runtime message exceeded size limit")
                    try:
                        message = json.loads(line)
                    except ValueError as exc:
                        raise WorkflowError("workflow runtime returned invalid JSON") from exc
                    if message.get("type") == "agent":
                        task = asyncio.create_task(handle(message))
                        tasks.add(task)
                        task.add_done_callback(tasks.discard)
                    elif message.get("type") == "result":
                        await asyncio.gather(*tasks, return_exceptions=True)
                        return message.get("result")
                    elif message.get("type") == "error":
                        raise WorkflowError(str(message.get("error") or "workflow failed"))
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if process.returncode is None:
                process.kill()
            await process.wait()
            shutil.rmtree(private_temp, ignore_errors=True)
