from __future__ import annotations

import builtins
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent_core.tools.adapters import ToolAdapter
from agent_core.tools.base import Tool, WorkspacePathMixin


@dataclass(slots=True)
class DeferredTool:
    name: str
    description: str
    factory: Callable[[], Tool]
    available: Callable[[], tuple[bool, str | None]] | None = None


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
    ) -> None:
        if name in self._tools or name in self._deferred:
            raise ValueError(f"Tool already registered: {name}")
        self._deferred[name] = DeferredTool(name, description, factory, available)

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

    def register_adapter(self, adapter: ToolAdapter) -> None:
        for tool in adapter.list_tools():
            self.register(tool)

    def unregister(self, name: str) -> None:
        """Drop a tool if present (idempotent). Used to hide conditionally-disabled tools."""
        self._tools.pop(name, None)
        self._deferred.pop(name, None)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas_for_llm(self) -> builtins.list[dict[str, object]]:
        return [tool.schema_for_llm() for tool in self.list()]

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
