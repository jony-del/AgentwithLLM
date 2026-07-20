"""Cancellation-safe process supervision shared by Bash and PowerShell tools."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
import uuid

from agent_core.tool_config import ShellToolConfig

EventSink = Callable[[str, dict[str, object]], Awaitable[None]]


class ShellUnavailableError(RuntimeError):
    pass


def _safe_command_preview(dialect: str, command: str) -> str:
    """Return a non-secret-bearing structural preview; arguments never enter audit logs."""
    match = re.match(r"\s*([A-Za-z][A-Za-z0-9_.-]{0,40})\b", command)
    return f"{match.group(1)} …" if match else f"<{dialect} command>"


def resolve_bash_executable(configured: str | None = None) -> str:
    """Return trusted Bash; WindowsApps/WSL shims are deliberately rejected."""
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return str(path.resolve())
        raise ShellUnavailableError(f"configured Bash does not exist: {path}")
    if os.name == "nt":
        candidates = [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "bin" / "bash.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "usr" / "bin" / "bash.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Git" / "bin" / "bash.exe",
        ]
        for path in candidates:
            if path.is_file():
                return str(path.resolve())
        raise ShellUnavailableError(
            "Git for Windows Bash is required for agent commands. Install Git for Windows "
            "or set POLARIS_BASH_PATH / [tools.shell.bash].executable. WindowsApps bash.exe "
            "and WSL are not accepted."
        )
    executable = shutil.which("bash")
    if executable:
        return executable
    raise ShellUnavailableError("Bash is required; install bash or configure [tools.shell.bash].executable")


def resolve_powershell_executable(configured: str | None = None) -> str:
    if configured:
        path = Path(configured).expanduser()
        resolved = shutil.which(str(path)) or (str(path.resolve()) if path.is_file() else None)
        if resolved:
            return resolved
        raise ShellUnavailableError(f"configured PowerShell does not exist: {path}")
    executable = shutil.which("pwsh")
    if executable:
        return executable
    if os.name == "nt":
        executable = shutil.which("powershell")
        if executable:
            return executable
    raise ShellUnavailableError("PowerShell is unavailable; install pwsh or configure its executable")


def encoded_powershell(command: str) -> str:
    return base64.b64encode(command.encode("utf-16le")).decode("ascii")


def powershell_utf8_command(command: str) -> str:
    return (
        "$ProgressPreference='SilentlyContinue';"
        "$OutputEncoding=[Text.UTF8Encoding]::new();"
        "&{" + command + "}2>&1|ForEach-Object{"
        "$s=($_|Out-String);$b=[Text.Encoding]::UTF8.GetBytes($s);"
        "$o=[Console]::OpenStandardOutput();$o.Write($b,0,$b.Length);$o.Flush()};"
        "if($null-ne $LASTEXITCODE){exit $LASTEXITCODE}"
    )


@dataclass(slots=True)
class ProcessTask:
    id: str
    dialect: str
    command_digest: str
    command_preview: str
    cwd: Path
    log_path: Path
    process: asyncio.subprocess.Process
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    returncode: int | None = None
    state: str = "running"
    preview: bytearray = field(default_factory=bytearray)
    output_bytes: int = 0
    logged_bytes: int = 0
    truncated: bool = False
    drain_task: asyncio.Task[None] | None = None
    timeout_task: asyncio.Task[None] | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def duration(self) -> float:
        return (self.finished_at or time.monotonic()) - self.started_at


class ProcessSupervisor:
    def __init__(
        self,
        config: ShellToolConfig,
        root: str | Path,
        *,
        event_sink: EventSink | None = None,
    ) -> None:
        self.config = config
        self.root = Path(root).resolve()
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.root = (
                Path(tempfile.gettempdir()) / "polaris-processes" / self.root.name
            ).resolve()
            self.root.mkdir(parents=True, exist_ok=True)
        self._event_sink = event_sink
        self._tasks: dict[str, ProcessTask] = {}
        self._history: dict[str, dict[str, object]] = {}
        self._closed = False
        for path in self.root.glob("*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(value, dict) and isinstance(value.get("task_id"), str):
                self._history[str(value["task_id"])] = value

    def tasks(self) -> list[ProcessTask]:
        return list(self._tasks.values())

    def running(self) -> list[ProcessTask]:
        return [item for item in self._tasks.values() if item.process.returncode is None]

    async def _event(self, kind: str, task: ProcessTask, **extra: object) -> None:
        if self._event_sink is None:
            return
        payload: dict[str, object] = {
            "task_id": task.id,
            "dialect": task.dialect,
            "command_digest": task.command_digest,
            "command_preview": task.command_preview,
            "cwd": str(task.cwd),
            "output_pointer": str(task.log_path),
            **extra,
        }
        try:
            await self._event_sink(kind, payload)
        except Exception:
            pass

    def _persist(self, task: ProcessTask) -> None:
        payload: dict[str, object] = {
            "task_id": task.id, "dialect": task.dialect,
            "command_digest": task.command_digest, "command_preview": task.command_preview,
            "cwd": str(task.cwd), "output_path": str(task.log_path), "state": task.state,
            "exit_code": task.returncode, "duration": round(task.duration, 3),
            "output_bytes": task.output_bytes, "truncated": task.truncated,
        }
        temporary = self.root / f".{task.id}.{uuid.uuid4().hex}.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, self.root / f"{task.id}.json")
        self._history[task.id] = payload

    async def syntax_check(
        self, dialect: str, command: str, executable: str, *, timeout: float = 10
    ) -> None:
        if dialect == "bash":
            argv = [executable, "-n", "-c", command]
        else:
            target = base64.b64encode(command.encode("utf-8")).decode("ascii")
            parser = (
                "$s=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('" + target + "'));"
                "$t=$null;$e=$null;[System.Management.Automation.Language.Parser]::ParseInput($s,[ref]$t,[ref]$e)|Out-Null;"
                "if($e.Count){$e|ForEach-Object{$_.Message};exit 2}"
            )
            argv = [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded_powershell(parser)]
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            output, _ = await asyncio.wait_for(
                process.communicate(), timeout=max(0.1, min(timeout, 10))
            )
        except asyncio.TimeoutError:
            await self._kill_process_tree(process)
            raise ValueError(f"{dialect} syntax check timed out") from None
        if process.returncode:
            detail = output.decode("utf-8", errors="replace").strip()[:4000]
            raise ValueError(f"{dialect} syntax error: {detail or 'parser rejected command'}")

    async def start(
        self,
        dialect: str,
        command: str,
        cwd: str | Path,
        *,
        argv: list[str] | None = None,
        timeout: float | None = None,
    ) -> ProcessTask:
        if self._closed:
            raise RuntimeError("process supervisor is closed")
        if len(self.running()) >= self.config.max_tasks:
            raise RuntimeError(f"background task limit reached ({self.config.max_tasks})")
        deadline = time.monotonic() + timeout if timeout is not None else None
        executable = (
            resolve_bash_executable(self.config.bash.executable or os.getenv("POLARIS_BASH_PATH"))
            if dialect == "bash"
            else resolve_powershell_executable(self.config.powershell.executable)
        )
        await self.syntax_check(
            dialect, command, executable,
            timeout=(max(0.1, deadline - time.monotonic()) if deadline is not None else 10),
        )
        remaining_timeout = deadline - time.monotonic() if deadline is not None else None
        if remaining_timeout is not None and remaining_timeout <= 0:
            raise TimeoutError(f"{dialect} command deadline expired during syntax analysis")
        if argv is None:
            if dialect == "bash":
                argv = [executable, "-lc", command]
            else:
                argv = [
                    executable,
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-OutputFormat",
                    "Text",
                    "-EncodedCommand",
                    encoded_powershell(powershell_utf8_command(command)),
                ]
        resolved_cwd = Path(cwd).resolve()
        task_id = uuid.uuid4().hex[:12]
        log_path = self.root / f"{task_id}.log"
        if os.name == "nt":
            process = await asyncio.create_subprocess_exec(
                *argv, cwd=str(resolved_cwd), stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            process = await asyncio.create_subprocess_exec(
                *argv, cwd=str(resolved_cwd), stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT, start_new_session=True,
            )
        digest = hashlib.sha256(command.encode("utf-8", errors="replace")).hexdigest()
        task = ProcessTask(
            task_id,
            dialect,
            digest,
            _safe_command_preview(dialect, command),
            resolved_cwd,
            log_path,
            process,
        )
        self._tasks[task_id] = task
        self._persist(task)
        task.drain_task = asyncio.create_task(self._drain(task), name=f"polaris-task-{task_id}")
        if remaining_timeout is not None:
            task.timeout_task = asyncio.create_task(
                self._enforce_timeout(task, remaining_timeout), name=f"polaris-timeout-{task_id}"
            )
        await self._event("task_started", task, pid=process.pid)
        return task

    async def _enforce_timeout(self, task: ProcessTask, timeout: float) -> None:
        try:
            await asyncio.wait_for(task.done.wait(), timeout)
        except asyncio.TimeoutError:
            if task.state == "running":
                task.state = "timed_out"
                await self._kill_process_tree(task.process)

    async def _drain(self, task: ProcessTask) -> None:
        stream = task.process.stdout
        assert stream is not None
        with task.log_path.open("wb") as log:
            while True:
                chunk = await stream.read(65536)
                if not chunk:
                    break
                task.output_bytes += len(chunk)
                task.preview.extend(chunk)
                overflow = len(task.preview) - self.config.preview_bytes
                if overflow > 0:
                    del task.preview[:overflow]
                    task.truncated = True
                remaining = self.config.log_bytes - task.logged_bytes
                if remaining > 0:
                    written = chunk[:remaining]
                    log.write(written)
                    log.flush()
                    task.logged_bytes += len(written)
                if len(chunk) > remaining:
                    task.truncated = True
            try:
                os.fsync(log.fileno())
            except OSError:
                pass
        task.returncode = await task.process.wait()
        task.finished_at = time.monotonic()
        if task.state == "running":
            task.state = "completed" if task.returncode == 0 else "failed"
        task.done.set()
        self._persist(task)
        if task.timeout_task is not None and task.timeout_task is not asyncio.current_task():
            task.timeout_task.cancel()
        await self._event(
            "task_finished",
            task,
            exit_code=task.returncode,
            duration=task.duration,
            state=task.state,
            output_bytes=task.output_bytes,
            truncated=task.truncated,
        )

    async def wait(self, task_id: str, timeout: float | None = None) -> ProcessTask:
        task = self.get(task_id)
        if timeout is None:
            await task.done.wait()
        else:
            await asyncio.wait_for(task.done.wait(), timeout=max(0.0, timeout))
        return task

    def get(self, task_id: str) -> ProcessTask:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise KeyError(f"Unknown task: {task_id}") from exc

    async def output(self, task_id: str, *, block: bool, timeout: float, tail_lines: int | None) -> dict[str, object]:
        task = self._tasks.get(task_id)
        if task is None:
            try:
                historical = dict(self._history[task_id])
            except KeyError as exc:
                raise KeyError(f"Unknown task: {task_id}") from exc
            path = Path(str(historical.get("output_path", "")))
            try:
                with path.open("rb") as handle:
                    handle.seek(max(0, path.stat().st_size - self.config.preview_bytes))
                    content = handle.read(self.config.preview_bytes).decode("utf-8", errors="replace")
            except OSError:
                content = ""
            if tail_lines is not None:
                content = "\n".join(content.splitlines()[-max(0, min(int(tail_lines), 10_000)):])
            historical["output"] = content
            historical["historical"] = True
            return historical
        if block and task.state == "running":
            try:
                await self.wait(task_id, timeout=min(max(timeout, 0.0), 60.0))
            except asyncio.TimeoutError:
                pass
        content = bytes(task.preview).decode("utf-8", errors="replace")
        if tail_lines is not None:
            content = "\n".join(content.splitlines()[-max(0, min(int(tail_lines), 10_000)):])
        return {
            "task_id": task.id,
            "state": task.state,
            "exit_code": task.returncode,
            "duration": round(task.duration, 3),
            "output": content,
            "output_path": str(task.log_path),
            "truncated": task.truncated,
            "output_bytes": task.output_bytes,
        }

    async def stop(self, task_id: str) -> ProcessTask:
        task = self.get(task_id)
        if task.process.returncode is not None:
            return task
        if task.state == "running":
            task.state = "stopped"
        await self._kill_process_tree(task.process)
        if task.drain_task is not None:
            try:
                await asyncio.wait_for(task.drain_task, self.config.shutdown_grace_seconds + 2)
            except asyncio.TimeoutError:
                task.drain_task.cancel()
        task.finished_at = task.finished_at or time.monotonic()
        task.done.set()
        self._persist(task)
        await self._event("task_stopped", task, duration=task.duration)
        return task

    async def _kill_process_tree(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        if os.name == "nt":
            killer = await asyncio.create_subprocess_exec(
                "taskkill", "/PID", str(process.pid), "/T", "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
        else:
            try:
                getattr(os, "killpg")(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                await asyncio.wait_for(process.wait(), self.config.shutdown_grace_seconds)
                return
            except asyncio.TimeoutError:
                try:
                    getattr(os, "killpg")(process.pid, getattr(signal, "SIGKILL", 9))
                except ProcessLookupError:
                    pass
        try:
            await asyncio.wait_for(process.wait(), self.config.shutdown_grace_seconds)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    async def shutdown(self) -> None:
        self._closed = True
        await asyncio.gather(*(self.stop(item.id) for item in self.running()), return_exceptions=True)
