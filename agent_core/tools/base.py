from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agent_core.models import ToolRisk, ToolResult


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

    @abstractmethod
    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool."""

