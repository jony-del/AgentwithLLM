from __future__ import annotations

from pathlib import Path

from agent_core.models import ToolRisk, ToolResult
from agent_core.tools.base import Tool


class WorkspacePathMixin:
    def __init__(self, workspace: str | Path | None = None) -> None:
        self.workspace = Path(workspace or Path.cwd()).resolve()

    def resolve_workspace_path(self, raw_path: object) -> Path:
        path = Path(str(raw_path))
        resolved = (self.workspace / path).resolve() if not path.is_absolute() else path.resolve()
        if resolved != self.workspace and self.workspace not in resolved.parents:
            raise ValueError(f"Path escapes workspace: {path}")
        return resolved


class EchoTool(Tool):
    name = "echo"
    description = "Echo text back to the agent."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    risk = ToolRisk.READ

    def run(self, arguments: dict[str, object]) -> ToolResult:
        return ToolResult(name=self.name, content=str(arguments.get("text", "")))


class ReadTextFileTool(WorkspacePathMixin, Tool):
    name = "read_text_file"
    description = "Read a UTF-8 text file from the current workspace."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    risk = ToolRisk.READ

    def run(self, arguments: dict[str, object]) -> ToolResult:
        path = self.resolve_workspace_path(arguments["path"])
        return ToolResult(name=self.name, content=path.read_text(encoding="utf-8"))


class WriteTextFileTool(WorkspacePathMixin, Tool):
    name = "write_text_file"
    description = "Write UTF-8 text to a file inside the current workspace."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }
    risk = ToolRisk.WRITE

    def run(self, arguments: dict[str, object]) -> ToolResult:
        path = self.resolve_workspace_path(arguments["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(arguments.get("content", "")), encoding="utf-8")
        return ToolResult(name=self.name, content=f"Wrote {path}")
