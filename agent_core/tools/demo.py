from __future__ import annotations

from agent_core.models import ToolRisk, ToolResult
from agent_core.tools.base import Tool, WorkspacePathMixin

# Re-exported for backwards compatibility: WorkspacePathMixin now lives in base.py.
__all__ = ["EchoTool", "ReadTextFileTool", "WorkspacePathMixin", "WriteTextFileTool"]


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
    description = (
        "Read a UTF-8 text file from the current workspace. Optionally pass `offset` "
        "(1-based start line) and/or `limit` (line count) to page through a large document."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "1-based first line to read (optional)."},
            "limit": {"type": "integer", "description": "Maximum number of lines to read (optional)."},
        },
        "required": ["path"],
    }
    risk = ToolRisk.READ

    def run(self, arguments: dict[str, object]) -> ToolResult:
        path = self.resolve_workspace_path(arguments["path"])
        text = path.read_text(encoding="utf-8")
        offset = arguments.get("offset")
        limit = arguments.get("limit")
        # No paging requested: return the whole file verbatim (original behavior).
        if offset is None and limit is None:
            return ToolResult(name=self.name, content=text)
        lines = text.splitlines()
        start = max(int(offset) - 1, 0) if offset is not None else 0
        end = start + int(limit) if limit is not None else len(lines)
        return ToolResult(name=self.name, content="\n".join(lines[start:end]))


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
