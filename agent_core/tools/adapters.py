from __future__ import annotations

from abc import ABC, abstractmethod

from agent_core.tools.base import Tool


class ToolAdapter(ABC):
    @abstractmethod
    def list_tools(self) -> list[Tool]:
        """Return tools exposed by this adapter."""


class MCPAdapter(ToolAdapter):
    """V1 placeholder for Model Context Protocol tool integration."""

    def list_tools(self) -> list[Tool]:
        return []


class LCPAdapter(ToolAdapter):
    """V1 placeholder for local/custom context protocol integration."""

    def list_tools(self) -> list[Tool]:
        return []

