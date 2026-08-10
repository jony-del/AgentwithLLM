from __future__ import annotations

import builtins
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_core.tools.adapters import ToolAdapter
from agent_core.tools.base import (
    ConcurrencySpec,
    ExecutionSafety,
    ResourceLock,
    Tool,
    ToolExecutionPolicy,
    WorkspacePathMixin,
)

if TYPE_CHECKING:
    from agent_core.tool_config import ToolPolicyConfig


@dataclass(slots=True)
class DeferredTool:
    name: str
    description: str
    factory: Callable[[], Tool]
    available: Callable[[], tuple[bool, str | None]] | None = None
    metadata: dict[str, Any] | None = None


class RegistryAwareMixin:
    """Mixin for tools such as tool_search that need their owning registry."""

    registry: "ToolRegistry | None" = None

    def bind_registry(self, registry: "ToolRegistry") -> None:
        self.registry = registry


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._deferred: dict[str, DeferredTool] = {}
        self._runtime: dict[str, Any] = {}
        self._workspace: str | None = None
        self._policy_overrides: dict[str, ToolPolicyConfig] = {}
        self._policy_overrides_trusted = False
        self._lock = threading.RLock()

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        if isinstance(tool, RegistryAwareMixin):
            tool.bind_registry(self)
        self._bind_tool(tool)

    def bind_runtime(
        self,
        *,
        session: Any = None,
        sandbox: Any = None,
        web_policy: Any = None,
        unattended: bool = False,
    ) -> None:
        self._runtime = {
            "session": session,
            "sandbox": sandbox,
            "web_policy": web_policy,
            "unattended": unattended,
        }
        for tool in self._tools.values():
            self._bind_tool(tool)

    def _bind_tool(self, tool: Tool) -> None:
        if self._workspace is not None and isinstance(tool, WorkspacePathMixin):
            tool.bind_workspace(self._workspace)
        if not self._runtime:
            return
        from agent_core.sandbox import SandboxAwareMixin
        from agent_core.session import SessionAwareMixin
        from agent_core.tools.web import WebPolicyAwareMixin

        if isinstance(tool, SessionAwareMixin) and self._runtime.get("session") is not None:
            tool.bind_session(self._runtime["session"])
        if isinstance(tool, SandboxAwareMixin) and self._runtime.get("sandbox") is not None:
            tool.bind_sandbox(self._runtime["sandbox"])
        if isinstance(tool, WebPolicyAwareMixin) and self._runtime.get("web_policy") is not None:
            tool.bind_web_policy(self._runtime["web_policy"], unattended=bool(self._runtime["unattended"]))
        session = self._runtime.get("session")
        if session is not None:
            session.registered_tool_names = frozenset(set(session.registered_tool_names) | {tool.name})

    def register_deferred(
        self,
        name: str,
        description: str,
        factory: Callable[[], Tool],
        *,
        available: Callable[[], tuple[bool, str | None]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if name in self._tools or name in self._deferred:
            raise ValueError(f"Tool already registered: {name}")
        self._deferred[name] = DeferredTool(name, description, factory, available, metadata)

    def activate(self, name: str) -> Tool:
        existing = self._tools.get(name)
        if existing is not None:
            return existing
        try:
            deferred = self._deferred[name]
        except KeyError as exc:
            raise KeyError(f"Unknown deferred tool: {name}") from exc
        if deferred.available is not None:
            ok, reason = deferred.available()
            if not ok:
                raise RuntimeError(reason or f"Tool dependency unavailable: {name}")
        tool = deferred.factory()
        if tool.name != name:
            raise ValueError(f"Deferred factory for {name!r} returned {tool.name!r}")
        del self._deferred[name]
        self.register(tool)
        return tool

    def register_adapter(self, adapter: ToolAdapter, *, deferred: bool = False) -> None:
        for tool in adapter.list_tools():
            if deferred:
                metadata: dict[str, Any] | None = None
                server = getattr(tool, "_server", None)
                remote = getattr(tool, "_remote", None)
                if server and remote:
                    metadata = {"kind": "mcp", "server": str(server), "remote": str(remote)}

                def factory(bound_tool: Tool = tool) -> Tool:
                    return bound_tool

                self.register_deferred(
                    tool.name,
                    tool.description,
                    factory,
                    metadata=metadata,
                )
            else:
                self.register(tool)

    def unregister(self, name: str) -> None:
        """Drop a tool if present (idempotent). Used to hide conditionally-disabled tools."""
        with self._lock:
            self._tools.pop(name, None)
            self._deferred.pop(name, None)
            self._sync_session_names()

    def replace_group(
        self,
        remove_names: set[str],
        active: list[Tool],
        deferred: list[DeferredTool],
    ) -> None:
        """Validate and atomically replace one runtime-owned group of tools."""

        new_names = [tool.name for tool in active] + [item.name for item in deferred]
        duplicates = sorted({name for name in new_names if new_names.count(name) > 1})
        if duplicates:
            raise ValueError("Duplicate tools in candidate group: " + ", ".join(duplicates))
        with self._lock:
            occupied = (set(self._tools) | set(self._deferred)) - set(remove_names)
            collisions = sorted(occupied & set(new_names))
            if collisions:
                raise ValueError("Tool already registered: " + ", ".join(collisions))
            tools = {key: value for key, value in self._tools.items() if key not in remove_names}
            lazy = {key: value for key, value in self._deferred.items() if key not in remove_names}
            for tool in active:
                tools[tool.name] = tool
            for item in deferred:
                lazy[item.name] = item
            previous_tools, previous_deferred = self._tools, self._deferred
            self._tools, self._deferred = tools, lazy
            try:
                for tool in active:
                    if isinstance(tool, RegistryAwareMixin):
                        tool.bind_registry(self)
                    self._bind_tool(tool)
            except Exception:
                self._tools, self._deferred = previous_tools, previous_deferred
                self._sync_session_names()
                raise
            self._sync_session_names()

    def _sync_session_names(self) -> None:
        session = self._runtime.get("session")
        if session is not None:
            session.registered_tool_names = frozenset(self._tools)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas_for_llm(self) -> builtins.list[dict[str, object]]:
        return [tool.schema_for_llm() for tool in self.list()]

    def set_policy_overrides(
        self,
        policies: dict[str, ToolPolicyConfig],
        *,
        trusted: bool = True,
    ) -> None:
        """Install local execution policies.

        Untrusted configuration may still make a policy more conservative, but it
        cannot upgrade a tool into speculative or transactional execution.
        """

        self._policy_overrides = dict(policies)
        self._policy_overrides_trusted = bool(trusted)

    @staticmethod
    def _argument_value(arguments: dict[str, Any], dotted: str) -> object:
        value: object = arguments
        for part in dotted.split("."):
            if not isinstance(value, dict) or part not in value:
                raise KeyError(dotted)
            value = value[part]
        if isinstance(value, (dict, list)) or value is None:
            raise ValueError(f"resource argument {dotted!r} must be a scalar")
        return value

    def execution_policy(self, tool: Tool, arguments: dict[str, Any]) -> ToolExecutionPolicy:
        declared = tool.execution_policy(arguments)
        override = self._policy_overrides.get(tool.name)
        if override is None:
            return declared
        if not self._policy_overrides_trusted:
            rank = {
                ExecutionSafety.SPECULATIVE_SAFE: 0,
                ExecutionSafety.TRANSACTIONAL: 1,
                ExecutionSafety.FINAL_ONLY: 2,
            }
            if rank[override.safety] <= rank[declared.safety]:
                return declared
            return ToolExecutionPolicy(safety=override.safety)
        try:
            locks: list[ResourceLock] = []
            for template in override.resources:
                raw_key = (
                    template.key
                    if template.key is not None
                    else self._argument_value(arguments, str(template.argument))
                )
                key = str(raw_key)
                if template.namespace == "fs" and self._workspace is not None:
                    from pathlib import Path

                    path = Path(key)
                    key = str((Path(self._workspace) / path).resolve() if not path.is_absolute() else path.resolve())
                locks.append(
                    ResourceLock(
                        template.namespace,
                        key,
                        template.mode,
                        subtree=template.subtree,
                        requires_success=template.requires_success,
                    )
                )
            idempotency_key = None
            if override.idempotency_argument:
                idempotency_key = str(
                    self._argument_value(arguments, override.idempotency_argument)
                )
            if override.safety is ExecutionSafety.TRANSACTIONAL and not override.transaction_backend:
                raise ValueError("transactional policy has no backend")
            return ToolExecutionPolicy(
                safety=override.safety,
                concurrency=ConcurrencySpec(tuple(locks), exclusive=override.exclusive),
                transaction_backend=override.transaction_backend,
                idempotency_key=idempotency_key,
                safely_cancellable=override.safely_cancellable,
                execution_timeout=override.execution_timeout,
            )
        except (KeyError, TypeError, ValueError):
            # A malformed/missing template must never make a call less constrained.
            return ToolExecutionPolicy()

    def deferred(self) -> builtins.list[DeferredTool]:
        return list(self._deferred.values())

    def search(
        self, query: str, *, max_results: int = 8
    ) -> builtins.list[dict[str, Any]]:
        """Search permitted active/deferred tools and activate deferred matches."""
        terms = [item.casefold() for item in query.split() if item.strip()]
        if not terms:
            return []
        candidates: list[tuple[int, str, str, bool]] = []
        for tool in self._tools.values():
            haystack = f"{tool.name} {tool.description}".casefold()
            score = sum(3 if term in tool.name.casefold() else 1 for term in terms if term in haystack)
            if score:
                candidates.append((score, tool.name, tool.description, False))
        for item in self._deferred.values():
            if item.available is not None and not item.available()[0]:
                continue
            haystack = f"{item.name} {item.description}".casefold()
            score = sum(3 if term in item.name.casefold() else 1 for term in terms if term in haystack)
            if score:
                candidates.append((score, item.name, item.description, True))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        results: list[dict[str, Any]] = []
        for _score, name, description, was_deferred in candidates[: max(1, min(max_results, 20))]:
            if was_deferred:
                self.activate(name)
            results.append({"name": name, "description": description, "activated": was_deferred})
        return results

    def rebind_workspace(self, workspace: str) -> None:
        self._workspace = workspace
        for tool in self._tools.values():
            if isinstance(tool, WorkspacePathMixin):
                tool.bind_workspace(workspace)

    @property
    def workspace(self) -> str | None:
        return self._workspace
