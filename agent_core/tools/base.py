from __future__ import annotations

import asyncio
from abc import ABC
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol

from agent_core.models import ToolRisk, ToolResult
from agent_core.permission_types import PermissionContext, PermissionResult

class ToolDisplayProvider:
    """Optional UI-rendering hints a tool may supply for a nicer live trace.

    Both hooks return plain strings (or ``None`` to decline) so tools stay free
    of any UI/Rich dependency — the renderer owns presentation. ``ConsoleUI``
    consults these via ``ToolExecutor``; a ``None`` lets the renderer fall back
    to its generic formatting.
    """

    def render_args(self, arguments: dict[str, Any]) -> str | None:
        """A compact one-line argument summary for the ``● tool(...)`` header."""
        return None

    def render_result(self, arguments: dict[str, Any], result: ToolResult) -> str | None:
        """A unified-diff string to show under the result branch (e.g. edits)."""
        return None

LockMode = Literal["read", "write"]


def coerce_int(value: object) -> int:
    """Convert a JSON-compatible scalar without accepting arbitrary objects."""
    if isinstance(value, (int, float, str, bytes, bytearray)):
        return int(value)
    raise TypeError(f"expected an integer-compatible value, got {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class ResourceLock:
    """A logical resource a tool call reads or writes.

    ``namespace`` separates unrelated resource kinds. ``key`` identifies the resource
    within that namespace. When ``subtree`` is true, the key also covers children
    beneath it; this is mostly used for workspace directory locks.
    """

    namespace: str
    key: str
    mode: LockMode
    subtree: bool = False
    # A consumer can make an ordering edge a hard success dependency.  This is
    # deliberately carried by the concrete resource rather than inferred from
    # ToolRisk: a READ label alone does not prove what a tool needs to succeed.
    requires_success: bool = False


@dataclass(frozen=True, slots=True)
class ConcurrencySpec:
    """How a tool call may be scheduled relative to other calls in the same turn."""

    locks: tuple[ResourceLock, ...] = ()
    exclusive: bool = False


class ExecutionSafety(str, Enum):
    """When a tool is allowed to perform real work within a model turn."""

    SPECULATIVE_SAFE = "speculative_safe"
    TRANSACTIONAL = "transactional"
    FINAL_ONLY = "final_only"


@dataclass(frozen=True, slots=True)
class ToolExecutionPolicy:
    """Trusted execution and concurrency policy for one concrete invocation.

    The defaults fail closed.  Risk and MCP annotations are permission/display
    signals and never implicitly upgrade this policy.
    """

    safety: ExecutionSafety = ExecutionSafety.FINAL_ONLY
    concurrency: ConcurrencySpec = ConcurrencySpec(exclusive=True)
    transaction_backend: str | None = None
    idempotency_key: str | None = None
    safely_cancellable: bool = False
    # Required upper bound for provisional work that cannot be cancelled safely.
    execution_timeout: float = 30.0


class WorkspaceView(Protocol):
    """Minimal turn-scoped workspace projection exposed to tools."""

    workspace: Path

    def execution_root(self) -> Path: ...


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """Context supplied by the scheduler without changing third-party signatures."""

    turn_id: str
    workspace_view: WorkspaceView | None = None
    provisional: bool = False


_EXECUTION_CONTEXT: ContextVar[ToolExecutionContext | None] = ContextVar(
    "polaris_tool_execution_context", default=None
)


def current_execution_context() -> ToolExecutionContext | None:
    return _EXECUTION_CONTEXT.get()


def execution_workspace(default: str | Path) -> Path:
    """Resolve the workspace visible to the current call.

    ``asyncio.to_thread`` copies context variables, so synchronous built-ins see
    the same overlay as async-native tools without mutating shared tool objects.
    """

    context = current_execution_context()
    if context is not None and context.workspace_view is not None:
        return context.workspace_view.execution_root()
    return Path(default).resolve()


@dataclass(frozen=True, slots=True)
class ExecutionScope:
    """Per-call filesystem/network boundaries passed to execution providers.

    A scope is immutable so concurrent tools cannot accidentally mutate global
    sandbox state while another command is being prepared.  Backends that do not
    yet understand the extra roots still receive the active workspace through the
    existing wrap seam and therefore fail no less safely than before.
    """

    workspace: Path
    git_common_dir: Path | None = None
    read_only_roots: tuple[Path, ...] = ()
    writable_roots: tuple[Path, ...] = ()
    private_temp: Path | None = None
    network: Literal["deny", "allow"] = "deny"
    workspace_writable: bool = True

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
    ) -> "ExecutionScope":
        return cls(
            Path(workspace).resolve(),
            Path(git_common_dir).resolve() if git_common_dir is not None else None,
            tuple(Path(item).resolve() for item in read_only_roots),
            tuple(Path(item).resolve() for item in writable_roots),
            Path(private_temp).resolve() if private_temp is not None else None,
            network,
            workspace_writable,
        )


class WorkspacePathMixin:
    """Confine file/command access to a workspace root.

    ``resolve_workspace_path`` rejects any path that escapes the workspace (via
    ``..`` or an absolute path), so tools can't read or write outside the project
    directory. Tools that only need the root (command runners) read ``self.workspace``.
    """

    def __init__(self, workspace: str | Path | None = None) -> None:
        self._workspace = Path(workspace or Path.cwd()).resolve()

    @property
    def workspace(self) -> Path:
        return execution_workspace(self._workspace)

    @workspace.setter
    def workspace(self, value: str | Path) -> None:
        self._workspace = Path(value).resolve()

    def bind_workspace(self, workspace: str | Path) -> None:
        """Atomically repoint a workspace-scoped tool at a new project root."""
        self._workspace = Path(workspace).resolve()

    def resolve_workspace_path(self, raw_path: object) -> Path:
        path = Path(str(raw_path))
        resolved = (self.workspace / path).resolve() if not path.is_absolute() else path.resolve()
        if resolved != self.workspace and self.workspace not in resolved.parents:
            raise ValueError(f"Path escapes workspace: {path}")
        return resolved

    def workspace_lock(
        self,
        raw_path: object,
        mode: LockMode,
        *,
        subtree: bool = False,
        requires_success: bool = False,
    ) -> ResourceLock:
        return ResourceLock(
            "fs",
            str(self.resolve_workspace_path(raw_path)),
            mode,
            subtree=subtree,
            requires_success=requires_success,
        )


class Tool(ABC, ToolDisplayProvider):
    name: str
    description: str
    input_schema: dict[str, Any]
    risk: ToolRisk = ToolRisk.READ
    # ``acceptedits`` is a capability mode, not a blanket grant for every tool
    # labelled WRITE.  Only workspace-native file editors opt in; coordination,
    # sub-agent, skill, MCP, and other stateful tools keep the fail-safe default.
    accept_edits_safe: bool = False
    # These tools must remain an ASK even in bypass/auto; a live user or an
    # explicitly configured PermissionRequest approval channel must resolve it.
    requires_user_interaction: bool = False
    # Unknown and existing third-party tools are authoritative-response-only.
    # Audited built-ins opt in explicitly on their concrete classes.
    execution_safety: ExecutionSafety = ExecutionSafety.FINAL_ONLY
    transaction_backend: str | None = None
    safely_cancellable: bool = False
    execution_timeout: float = 30.0

    def schema_for_llm(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def concurrency_spec(self, arguments: dict[str, Any]) -> ConcurrencySpec:
        """Return the resources touched by this concrete call.

        The default is intentionally conservative: tools that do not declare their
        resources run one at a time in original order.
        """
        return ConcurrencySpec(exclusive=True)

    def execution_policy(self, arguments: dict[str, Any]) -> ToolExecutionPolicy:
        """Return the trusted policy declared by Python code for this call."""

        return ToolExecutionPolicy(
            safety=self.execution_safety,
            concurrency=self.concurrency_spec(arguments),
            transaction_backend=self.transaction_backend,
            safely_cancellable=self.safely_cancellable,
            execution_timeout=max(0.1, float(self.execution_timeout)),
        )

    async def check_permissions(
        self,
        arguments: dict[str, Any],
        context: PermissionContext,
    ) -> PermissionResult:
        """Return this tool's argument-aware permission recommendation.

        PASSTHROUGH is deliberately the base contract: an un-migrated or third-party
        tool must still pass through the central policy and conservative ToolRisk
        fallback instead of accidentally becoming trusted merely by subclassing Tool.
        """
        return PermissionResult.passthrough()

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool — the single (async) execution entry point.

        The default offloads the blocking :meth:`_invoke` to a worker thread so
        ordinary tools stay simple synchronous code without ever blocking the event
        loop. Async-native tools (those that spawn child agents or use an async
        transport) override ``run`` directly; the executor detects the override and
        awaits them on the loop so their work can overlap.
        """
        return await asyncio.to_thread(self._invoke, arguments)

    async def run_with_context(
        self,
        arguments: dict[str, Any],
        execution_context: ToolExecutionContext,
    ) -> ToolResult:
        """Execute with a turn context while preserving the legacy ``run`` API."""

        token = _EXECUTION_CONTEXT.set(execution_context)
        try:
            return await self.run(arguments)
        finally:
            _EXECUTION_CONTEXT.reset(token)

    def _invoke(self, arguments: dict[str, Any]) -> ToolResult:
        """Blocking implementation hook for ordinary tools (internal detail).

        Runs on a worker thread via the default :meth:`run`, bounded by the
        executor's ``max_workers`` semaphore. Tools implement either this or an
        async ``run`` override — never both.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _invoke() or override run()")
