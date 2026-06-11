from __future__ import annotations

import asyncio
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agent_core.models import ToolRisk, ToolResult

LockMode = Literal["read", "write"]


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


@dataclass(frozen=True, slots=True)
class ConcurrencySpec:
    """How a tool call may be scheduled relative to other calls in the same turn."""

    locks: tuple[ResourceLock, ...] = ()
    exclusive: bool = False


class WorkspacePathMixin:
    """Confine file/command access to a workspace root.

    ``resolve_workspace_path`` rejects any path that escapes the workspace (via
    ``..`` or an absolute path), so tools can't read or write outside the project
    directory. Tools that only need the root (command runners) read ``self.workspace``.
    """

    def __init__(self, workspace: str | Path | None = None) -> None:
        self.workspace = Path(workspace or Path.cwd()).resolve()

    def resolve_workspace_path(self, raw_path: object) -> Path:
        path = Path(str(raw_path))
        resolved = (self.workspace / path).resolve() if not path.is_absolute() else path.resolve()
        if resolved != self.workspace and self.workspace not in resolved.parents:
            raise ValueError(f"Path escapes workspace: {path}")
        return resolved

    def workspace_lock(self, raw_path: object, mode: LockMode, *, subtree: bool = False) -> ResourceLock:
        return ResourceLock("fs", str(self.resolve_workspace_path(raw_path)), mode, subtree=subtree)


class Tool(ABC):
    name: str
    description: str
    input_schema: dict[str, Any]
    risk: ToolRisk = ToolRisk.READ

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

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool — the single (async) execution entry point.

        The default offloads the blocking :meth:`_invoke` to a worker thread so
        ordinary tools stay simple synchronous code without ever blocking the event
        loop. Async-native tools (those that spawn child agents or use an async
        transport) override ``run`` directly; the executor detects the override and
        awaits them on the loop so their work can overlap.
        """
        return await asyncio.to_thread(self._invoke, arguments)

    def _invoke(self, arguments: dict[str, Any]) -> ToolResult:
        """Blocking implementation hook for ordinary tools (internal detail).

        Runs on a worker thread via the default :meth:`run`, bounded by the
        executor's ``max_workers`` semaphore. Tools implement either this or an
        async ``run`` override — never both.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _invoke() or override run()")
