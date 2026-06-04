from __future__ import annotations

from agent_core.tools.adapters import ToolAdapter
from agent_core.tools.base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def register_adapter(self, adapter: ToolAdapter) -> None:
        for tool in adapter.list_tools():
            self.register(tool)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas_for_llm(self) -> list[dict[str, object]]:
        return [tool.schema_for_llm() for tool in self.list()]

