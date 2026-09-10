"""Frozen contract (schema v1): the unified execution scope for one run tree.

``ExecutionScope`` is the single authority for deadline, cancellation, and owned
resources of an execution tree (run loop, provider attempts, tool batches, hooks,
MCP calls, sub-agents). The invariants below are contractual — pinned by
``tests/test_contracts.py`` — and must not change without a schema bump:

1. ``child()`` only ever tightens: deadline takes the min, ``network`` can never
   widen from ``deny`` to ``allow``, ``workspace_writable`` can never widen from
   False to True.
2. One execution tree shares exactly one ``CancellationToken`` and one
   ``ExecutionTaskRegistry``; ``child()`` carries both over unchanged.
3. Cancellation has three layers with a fixed relationship: ``cancel_probe``
   (external async signal, e.g. Esc) folds into the token on first observation;
   the token provides cooperative cancellation (``raise_if_cancelled`` raises
   ``asyncio.CancelledError``); ``asyncio.Task.cancel`` is structural and reserved
   for the executor's ``safely_cancellable`` tasks. The run loop converts only
   its own scope's token cancellation into an interrupted result.
4. ``remaining_budget(requested)`` returns ``min(requested, wall-clock left)``;
   ``run_awaitable`` enforces that bound and raises ``TimeoutError`` past it.
5. Whoever creates a scope closes it: ``close()`` cancels the token and reaps
   every registered task and cleanup callback.

Known bypasses that do NOT yet draw from the scope budget (phase-3 integration
targets, not part of this contract): the httpx ``ProviderConfig.timeout``, the
MCP client thread-side future, prompt/agent hook ``wait_for`` calls, and
``ProcessSupervisor._enforce_timeout``.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Mapping


class CancellationToken:
    """Frozen contract (schema v1): thread-safe cooperative cancellation shared by one execution tree."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason = "cancelled"

    def cancel(self, reason: str = "cancelled") -> None:
        self._reason = reason or "cancelled"
        self._event.set()

    @property
    def reason(self) -> str:
        return self._reason

    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled():
            raise asyncio.CancelledError(self._reason)


class ExecutionTaskRegistry:
    """Frozen contract (schema v1): own async tasks and cleanup callbacks created by an execution tree."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()
        self._cleanups: list[Callable[[], object]] = []

    def create_task(self, awaitable: Awaitable[Any], *, name: str | None = None) -> asyncio.Task[Any]:
        async def run() -> Any:
            return await awaitable

        task: asyncio.Task[Any] = asyncio.create_task(run(), name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def add_cleanup(self, callback: Callable[[], object]) -> None:
        self._cleanups.append(callback)

    async def close(self, timeout: float = 5.0) -> None:
        async def reap() -> None:
            tasks = [task for task in self._tasks if not task.done()]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            callbacks, self._cleanups = list(reversed(self._cleanups)), []
            for callback in callbacks:
                with contextlib.suppress(Exception):
                    value = callback()
                    if inspect.isawaitable(value):
                        await value

        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(reap(), timeout=max(0.01, timeout))


@dataclass(frozen=True, slots=True)
class ExecutionScope:
    """Frozen contract (schema v1): provider-neutral authority and lifecycle budget for one execution tree."""

    workspace: Path
    git_common_dir: Path | None = None
    read_only_roots: tuple[Path, ...] = ()
    writable_roots: tuple[Path, ...] = ()
    private_temp: Path | None = None
    network: Literal["deny", "allow"] = "deny"
    workspace_writable: bool = True
    deadline: float | None = None
    cancellation: CancellationToken = field(default_factory=CancellationToken, compare=False)
    cancel_probe: Callable[[], bool] | None = field(default=None, compare=False, repr=False)
    tasks: ExecutionTaskRegistry = field(default_factory=ExecutionTaskRegistry, compare=False)
    provider_gate: Any | None = field(default=None, compare=False, repr=False)
    recovery_context: Mapping[str, str] = field(default_factory=dict, compare=False)
    cleanup_timeout: float = 5.0

    @classmethod
    def for_workspace(
        cls,
        workspace: str | Path,
        *,
        git_common_dir: str | Path | None = None,
        read_only_roots: tuple[str | Path, ...] = (),
        writable_roots: tuple[str | Path, ...] = (),
        private_temp: str | Path | None = None,
        network: Literal["deny", "allow"] = "deny",
        workspace_writable: bool = True,
        deadline: float | None = None,
        cancellation: CancellationToken | None = None,
        cancel_probe: Callable[[], bool] | None = None,
        tasks: ExecutionTaskRegistry | None = None,
        provider_gate: Any | None = None,
        recovery_context: Mapping[str, str] | None = None,
        cleanup_timeout: float = 5.0,
    ) -> "ExecutionScope":
        return cls(
            Path(workspace).resolve(),
            Path(git_common_dir).resolve() if git_common_dir is not None else None,
            tuple(Path(item).resolve() for item in read_only_roots),
            tuple(Path(item).resolve() for item in writable_roots),
            Path(private_temp).resolve() if private_temp is not None else None,
            network,
            workspace_writable,
            deadline,
            cancellation or CancellationToken(),
            cancel_probe,
            tasks or ExecutionTaskRegistry(),
            provider_gate,
            dict(recovery_context or {}),
            max(0.01, float(cleanup_timeout)),
        )

    def cancelled(self) -> bool:
        return self.cancellation.cancelled() or bool(
            self.cancel_probe is not None and self.cancel_probe()
        )

    def raise_if_cancelled(self) -> None:
        if self.cancelled():
            if not self.cancellation.cancelled():
                self.cancellation.cancel("cancel probe requested cancellation")
            self.cancellation.raise_if_cancelled()
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("execution deadline exceeded")

    def remaining_budget(self, requested: float | None = None) -> float | None:
        remaining = None if self.deadline is None else max(0.0, self.deadline - time.monotonic())
        if requested is None:
            return remaining
        requested = max(0.0, float(requested))
        return requested if remaining is None else min(requested, remaining)

    def child(
        self,
        *,
        deadline: float | None = None,
        workspace: str | Path | None = None,
        network: Literal["deny", "allow"] | None = None,
        workspace_writable: bool | None = None,
        recovery_context: Mapping[str, str] | None = None,
    ) -> "ExecutionScope":
        child_deadline = deadline
        if self.deadline is not None:
            child_deadline = self.deadline if deadline is None else min(self.deadline, deadline)
        child_network = self.network if network is None else network
        if self.network == "deny" and child_network == "allow":
            child_network = "deny"
        child_writable = self.workspace_writable if workspace_writable is None else workspace_writable
        if not self.workspace_writable:
            child_writable = False
        context = dict(self.recovery_context)
        context.update(recovery_context or {})
        return replace(
            self,
            workspace=(Path(workspace).resolve() if workspace is not None else self.workspace),
            network=child_network,
            workspace_writable=child_writable,
            deadline=child_deadline,
            recovery_context=context,
        )

    def with_provider_gate(self, gate: object) -> "ExecutionScope":
        return replace(self, provider_gate=gate)

    async def run_awaitable(
        self,
        awaitable: Awaitable[Any],
        *,
        timeout: float | None = None,
    ) -> Any:
        self.raise_if_cancelled()
        bounded = self.remaining_budget(timeout)
        if bounded is not None:
            if bounded <= 0:
                if inspect.iscoroutine(awaitable):
                    awaitable.close()
                raise TimeoutError("execution deadline exceeded")
            return await asyncio.wait_for(awaitable, timeout=bounded)
        return await awaitable

    async def sleep(self, delay: float) -> None:
        await self.run_awaitable(asyncio.sleep(max(0.0, delay)))

    def create_task(self, awaitable: Awaitable[Any], *, name: str | None = None) -> asyncio.Task[Any]:
        self.raise_if_cancelled()
        return self.tasks.create_task(awaitable, name=name)

    async def close(self) -> None:
        self.cancellation.cancel("execution scope closed")
        await self.tasks.close(self.cleanup_timeout)


_ACTIVE_EXECUTION_SCOPE: ContextVar[ExecutionScope | None] = ContextVar(
    "polaris_execution_scope", default=None
)


def current_execution_scope() -> ExecutionScope | None:
    return _ACTIVE_EXECUTION_SCOPE.get()


@contextlib.contextmanager
def execution_scope_context(scope: ExecutionScope):
    token = _ACTIVE_EXECUTION_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _ACTIVE_EXECUTION_SCOPE.reset(token)
