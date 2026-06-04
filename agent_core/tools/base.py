from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from agent_core.models import ToolRisk, ToolResult


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

