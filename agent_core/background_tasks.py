"""Run-owned background operations over existing shell and agent backends.

The session owns the registry/outbox; execution scopes still own all live work.
No task is detached from its originating run's cancellation or deadline.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import json
import time
from typing import Any
import uuid

from agent_core.background_store import BackgroundTaskStore
from agent_core.execution import ExecutionScope, current_execution_scope
from agent_core.tool_config import BackgroundTaskConfig

_ACTIVE = {"pending", "running"}
_TERMINAL = {"completed", "failed", "stopped", "lost", "timed_out", "deadline", "interrupted",
             "max_steps", "no_progress", "unverified", "blocked", "cancelled"}


def _bounded(text: str, limit: int) -> str:
    data = text.encode("utf-8", errors="replace")
    if len(data) <= limit:
        return text
    return data[:max(0, limit - 32)].decode("utf-8", errors="ignore") + "\n[... truncated ...]"


@dataclass(slots=True)
class AgentTaskOutcome:
    answer: str
    status: str = "completed"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BackgroundTask:
    id: str
    kind: str
    owner_run: str
    description: str
    state: str = "running"
    backgrounded: bool = False
    required: bool = True
    result: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    notified: bool = False
    started_at: float = field(default_factory=time.monotonic)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    background_signal: asyncio.Event = field(default_factory=asyncio.Event)
    worker: asyncio.Task[Any] | None = None
    process_task: Any = None
    stop_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def snapshot(self) -> dict[str, Any]:
        return {"task_id": self.id, "task_type": self.kind, "owner_run": self.owner_run,
                "description": self.description, "state": self.state,
                "backgrounded": self.backgrounded, "required": self.required,
                "result": self.result, "metadata": dict(self.metadata), "notified": self.notified}


class BackgroundTaskManager:
    def __init__(self, config: BackgroundTaskConfig, supervisor: Any, *,
                 store: BackgroundTaskStore | None = None, agent_id: str = "leader",
                 notify_ui: Callable[[list[dict[str, Any]]], None] | None = None) -> None:
        self.config, self.supervisor, self.store = config, supervisor, store
        self.agent_id, self.notify_ui = agent_id, notify_ui
        self.active_run = "standalone"
        self.records: dict[str, BackgroundTask] = {}
        self.events: list[dict[str, Any]] = []
        self.changed = asyncio.Event()
        self._save_lock = asyncio.Lock()
        self.persistence_error: str | None = None
        if store is not None:
            saved = store.load()
            for raw in saved.get("records", []):
                task = BackgroundTask(str(raw["task_id"]), str(raw["task_type"]),
                                      str(raw["owner_run"]), str(raw["description"]),
                                      state=str(raw["state"]), backgrounded=bool(raw["backgrounded"]),
                                      required=bool(raw["required"]), result=str(raw.get("result", "")),
                                      metadata=dict(raw.get("metadata", {})), notified=bool(raw.get("notified")))
                if task.state in _ACTIVE:
                    task.state, task.notified = "lost", False
                task.done.set()
                self.records[task.id] = task
            self.events = list(saved.get("events", []))
            for task in self.records.values():
                if task.state == "lost":
                    self._enqueue(task, "finished")

    def begin_run(self, run_id: str | None = None) -> str:
        self.active_run = run_id or uuid.uuid4().hex
        # Starting a different user task archives old notices; records remain queryable.
        self.events = [event for event in self.events if event["owner_run"] == self.active_run]
        return self.active_run

    def get(self, task_id: str) -> BackgroundTask:
        try:
            return self.records[task_id]
        except KeyError as exc:
            raise KeyError(f"Unknown background task: {task_id}") from exc

    def running(self, *, current_run: bool = True) -> list[BackgroundTask]:
        return [task for task in self.records.values() if task.state in _ACTIVE
                and (not current_run or task.owner_run == self.active_run)]

    def snapshots(self) -> list[dict[str, Any]]:
        return [task.snapshot() for task in self.records.values()]

    def _touch(self) -> None:
        self.changed.set()
        if self.notify_ui is not None:
            try:
                self.notify_ui(self.snapshots())
            except Exception:
                pass  # Observational UI failures must not strand owned work.

    def ensure_capacity(self, *, agent: bool = False) -> None:
        pending_ids = {event["task_id"] for event in self.events}
        for task in list(self.records.values()):
            if len(self.records) < self.config.max_records:
                break
            if (task.done.is_set() and task.id not in pending_ids and
                    (task.owner_run != self.active_run or not task.required or task.state == "completed")):
                del self.records[task.id]
        if (len(self.records) >= self.config.max_records or
                len(self.events) + 2 * len(self.running(current_run=False)) + 2 > self.config.max_notifications):
            raise RuntimeError("background record/notification limit reached; consume pending results first")
        if agent and sum(task.kind != "shell" for task in self.running(current_run=False)) >= self.config.max_agents:
            raise RuntimeError("background agent limit reached")

    async def save(self) -> None:
        if self.store is None:
            return
        async with self._save_lock:
            # Snapshot under the serial write lock, never persist stale pre-await state.
            records, events = self.snapshots(), [dict(event) for event in self.events]
            pending = asyncio.create_task(asyncio.to_thread(self.store.save, records, events))
            try:
                await asyncio.shield(pending)
                self.persistence_error = None
            except asyncio.CancelledError:
                await pending
                raise
            except Exception as exc:
                self.persistence_error = type(exc).__name__
                raise

    def _enqueue(self, task: BackgroundTask, event_type: str, tail: str = "") -> None:
        event_id = f"{task.id}:{event_type}"
        if task.notified and event_type == "finished":
            return
        if not any(event["event_id"] == event_id for event in self.events):
            self.events.append({"event_id": event_id, "task_id": task.id, "owner_agent": self.agent_id,
                                "owner_run": task.owner_run, "event": event_type, "task_type": task.kind,
                                "state": task.state, "description": task.description,
                                "output_path": task.metadata.get("output_path"),
                                "summary": _bounded(task.result, 2048), "tail": _bounded(tail, 2048)})
        if event_type == "finished":
            task.notified = True
        self._touch()

    async def register_shell(self, process: Any, description: str) -> BackgroundTask:
        task = BackgroundTask(process.id, "shell", self.active_run, description[:200],
                              state=process.state, process_task=process, done=process.done,
                              metadata={"output_path": str(process.log_path), "dialect": process.dialect})
        self.records[task.id] = task
        try:
            await self.save()
        except BaseException:
            await self.supervisor.stop(process.id)
            self.records.pop(task.id, None)
            raise
        self._touch()
        return task

    def background(self, task_id: str) -> bool:
        task = self.get(task_id)
        if task.state not in _ACTIVE or task.backgrounded:
            return False
        task.backgrounded = True
        task.background_signal.set()
        self._touch()
        return True

    def background_all(self) -> None:
        for task in self.running():
            self.background(task.id)

    def consume_background(self, callback: Callable[[], bool] | None) -> None:
        if self.config.enabled and callback is not None and callback():
            self.background_all()

    async def publish_shell(self, kind: str, payload: dict[str, object]) -> None:
        task = self.records.get(str(payload.get("task_id", "")))
        if task is None:
            return
        if kind == "task_stalled" and task.backgrounded:
            self._enqueue(task, "interactive_input", str(payload.get("tail", "")))
        elif kind == "task_finished":
            task.state = str(payload["state"])
            task.metadata["exit_code"] = payload.get("exit_code")
            if payload.get("persistence_error"):
                task.metadata["persistence_error"] = payload["persistence_error"]
            if task.backgrounded:
                self._enqueue(task, "finished")
            self._touch()
        else:
            return
        try:
            await self.save()
        except (OSError, ValueError):
            # Retain the in-memory outbox; completion verification reports degraded persistence.
            pass

    async def mark_background(self, task: BackgroundTask) -> None:
        self.background(task.id)
        task.backgrounded = True
        if task.process_task is not None and task.process_task.stalled_tail:
            self._enqueue(task, "interactive_input", task.process_task.stalled_tail)
        if task.done.is_set():
            if task.process_task is not None:
                task.state = task.process_task.state
            self._enqueue(task, "finished")
        await self.save()

    async def start_agent(self, kind: str, description: str,
                          operation: Callable[[], Awaitable[AgentTaskOutcome | str]], *,
                          background: bool, metadata: dict[str, Any] | None = None) -> BackgroundTask:
        self.ensure_capacity(agent=True)
        scope = current_execution_scope()
        if scope is None:
            raise RuntimeError("background agents require an active execution scope")
        task = BackgroundTask(uuid.uuid4().hex[:12], kind, self.active_run, description[:200],
                              backgrounded=background, metadata=metadata or {})
        self.records[task.id] = task
        try:
            await self.save()
        except BaseException:
            del self.records[task.id]
            raise
        if task.done.is_set():
            # A CLI stop may race the initial durable registration before launch.
            return task
        try:
            scope.raise_if_cancelled()
        except BaseException:
            task.state = "stopped"
            task.done.set()
            raise

        async def run() -> None:
            try:
                outcome = await operation()
                if isinstance(outcome, str):
                    outcome = AgentTaskOutcome(outcome)
                if outcome.status not in _TERMINAL:
                    raise ValueError(f"agent operation returned an invalid terminal status: {outcome.status}")
                task.result = _bounded(outcome.answer, self.config.result_max_bytes)
                task.state = outcome.status
                task.metadata.update(outcome.metadata)
            except asyncio.CancelledError:
                task.state = "stopped"
                raise
            except Exception as exc:
                task.state, task.result = "failed", _bounded(str(exc), 2048)
            finally:
                task.done.set()
                if task.backgrounded:
                    self._enqueue(task, "finished")
                self._touch()
                try:
                    await self.save()
                except Exception:
                    pass

        operation_coro = run()
        try:
            task.worker = scope.create_task(operation_coro, name=f"background-{kind}-{task.id}")
        except BaseException:
            operation_coro.close()
            task.state = "stopped"
            task.done.set()
            raise
        scope.tasks.add_cleanup(lambda: self.stop(task.id, explicit=False))
        await asyncio.sleep(0)
        self._touch()
        return task

    async def await_agent(self, task: BackgroundTask, callback: Callable[[], bool] | None) -> bool:
        """Return True when released to background, False when the operation finishes."""
        deadline = (time.monotonic() + self.config.agent_auto_background_seconds
                    if self.config.agent_auto_background_seconds > 0 else None)
        while not task.done.is_set():
            self.consume_background(callback)
            if task.backgrounded or (deadline is not None and time.monotonic() >= deadline):
                await self.mark_background(task)
                return True
            try:
                await asyncio.wait_for(task.background_signal.wait(), 0.05)
            except TimeoutError:
                pass
        return False

    async def stop(self, task_id: str, *, explicit: bool = True) -> BackgroundTask:
        task = self.get(task_id)
        async with task.stop_lock:
            if explicit:
                task.required = False
            if not task.done.is_set():
                if task.kind == "shell":
                    await self.supervisor.stop(task.id)
                    task.state = task.process_task.state
                elif task.worker is not None:
                    task.worker.cancel()
                    await asyncio.gather(task.worker, return_exceptions=True)
                    if not task.done.is_set():
                        task.state = "stopped"
                        task.done.set()
                else:
                    task.state = "stopped"
                    task.done.set()
            await self.save()
            self._touch()
        return task

    async def close_run(self, run_id: str) -> None:
        await asyncio.gather(*(self.stop(task.id, explicit=False) for task in self.records.values()
                               if task.owner_run == run_id and not task.done.is_set()))

    async def close(self) -> None:
        await asyncio.gather(*(self.stop(task.id, explicit=False) for task in self.running(current_run=False)))

    async def output(self, task_id: str, *, block: bool, timeout: float,
                     tail_lines: int | None) -> dict[str, Any]:
        task = self.get(task_id)
        if task.kind == "shell":
            return await self.supervisor.output(task_id, block=block, timeout=timeout, tail_lines=tail_lines)
        if block and not task.done.is_set():
            try:
                await asyncio.wait_for(task.done.wait(), max(0, min(timeout, 60)))
            except TimeoutError:
                pass
        metadata = task.snapshot()
        metadata.pop("result")
        return {**metadata, "output": task.result, "background_task_id": task.id}

    def pending_events(self, seen: set[str]) -> list[dict[str, Any]]:
        # Old terminal results remain readable, but cannot steer a new user task.
        return [event for event in self.events if event["event_id"] not in seen
                and event["owner_run"] == self.active_run and event["owner_agent"] == self.agent_id]

    async def acknowledge(self, event_ids: set[str]) -> None:
        removed = [event for event in self.events if event["event_id"] in event_ids]
        self.events = [event for event in self.events if event["event_id"] not in event_ids]
        try:
            await self.save()
        except BaseException:
            present = {event["event_id"] for event in self.events}
            self.events = [event for event in removed if event["event_id"] not in present] + self.events
            raise

    async def wait_for_change(self, scope: ExecutionScope) -> None:
        scope.raise_if_cancelled()
        try:
            await scope.run_awaitable(self.changed.wait(), timeout=0.1)
        except TimeoutError:
            scope.raise_if_cancelled()

    def completion_issues(self) -> tuple[str, ...]:
        issues = [f"background task {task.id}: {task.state}" for task in self.records.values()
                  if task.owner_run == self.active_run and task.required and task.state != "completed"]
        issues.extend(f"background task {task.id}: process state persistence failed"
            for task in self.records.values() if task.owner_run == self.active_run
            and task.required and task.metadata.get("persistence_error"))
        if self.persistence_error:
            issues.append("background notification persistence failed: " + self.persistence_error)
        return tuple(issues)

    def format_events(self, events: list[dict[str, Any]]) -> str:
        # The caller emits/acks exactly one event per message, never a truncated batch.
        value = dict(events[0])
        limit = self.config.notification_max_bytes
        fields = ("summary", "tail", "description", "output_path")
        for key in fields:
            if isinstance(value.get(key), str):
                value[key] = _bounded(value[key], max(64, (limit - 512) // 4))
        serialized = json.dumps([value], ensure_ascii=False)
        while len(serialized.encode("utf-8")) > limit:
            candidates = [key for key in fields if isinstance(value.get(key), str) and value[key]]
            if not candidates:
                raise ValueError("background notification identity exceeds message budget")
            largest = max(candidates, key=lambda key: len(json.dumps(value[key], ensure_ascii=False).encode("utf-8")))
            raw_bytes = len(value[largest].encode("utf-8"))
            value[largest] = _bounded(value[largest], raw_bytes // 2) if raw_bytes > 64 else ""
            serialized = json.dumps([value], ensure_ascii=False)
        return serialized
