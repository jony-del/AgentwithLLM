from __future__ import annotations

from typing import Any

from agent_core.models import ToolRisk, ToolResult
from agent_core.notebook import edit_notebook
from agent_core.permission_types import PermissionContext, PermissionResult
from agent_core.permission_safety import ordinary_write_permission
from agent_core.session import SessionAwareMixin
from agent_core.tools.base import (
    ConcurrencySpec,
    ExecutionSafety,
    ResourceLock,
    Tool,
    execution_workspace,
)
from agent_core.tools.catalog import builtin_tool


@builtin_tool
class NotebookEditTool(SessionAwareMixin, Tool):
    name = "notebook_edit"
    description = "Replace, insert, or delete one Jupyter notebook cell after a verified read."
    deferred = True
    input_schema = {
        "type": "object",
        "properties": {
            "notebook_path": {"type": "string"},
            "cell_id": {"type": "string"},
            "new_source": {"type": "string"},
            "cell_type": {"type": "string", "enum": ["code", "markdown", "raw"]},
            "edit_mode": {"type": "string", "enum": ["replace", "insert", "delete"]},
        },
        "required": ["notebook_path", "new_source", "edit_mode"],
    }
    risk = ToolRisk.WRITE
    accept_edits_safe = True
    execution_safety = ExecutionSafety.TRANSACTIONAL
    transaction_backend = "workspace"

    async def check_permissions(self, arguments: dict[str, Any], context: PermissionContext) -> PermissionResult:
        projected = dict(arguments)
        projected["path"] = projected.get("notebook_path")
        return ordinary_write_permission(self.name, projected, context)

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        path = (execution_workspace(self.session.workspace) / str(arguments.get("notebook_path", ""))).resolve()
        return ConcurrencySpec((ResourceLock("fs", str(path), "write"),))

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        workspace = execution_workspace(self.session.workspace)
        path = (workspace / str(arguments.get("notebook_path", ""))).resolve()
        if path != workspace and workspace not in path.parents:
            return ToolResult(self.name, "Path escapes workspace", ok=False)
        logical_path = (
            self.session.workspace / str(arguments.get("notebook_path", ""))
        ).resolve()
        expected = getattr(self.session, "notebook_reads", {}).get(str(logical_path))
        if expected is None:
            return ToolResult(self.name, "Notebook must be read with read_text_file before editing.", ok=False)
        config = self.session.tool_suite.notebook if self.session.tool_suite is not None else None
        try:
            result = edit_notebook(
                path,
                expected=expected,
                cell_id=str(arguments["cell_id"]) if arguments.get("cell_id") else None,
                new_source=str(arguments.get("new_source", "")),
                cell_type=str(arguments["cell_type"]) if arguments.get("cell_type") else None,
                edit_mode=str(arguments.get("edit_mode", "")),
                max_bytes=config.max_bytes if config is not None else 16 * 1024 * 1024,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            return ToolResult(self.name, f"Notebook edit refused: {exc}", ok=False)
        self.session.notebook_reads[str(logical_path)] = {
            "sha256": result["sha256"], "mtime_ns": result["mtime_ns"], "size": result["size"]
        }
        return ToolResult(self.name, f"Notebook cell edit complete: {result['edit_mode']} {result.get('cell_id')}", metadata=result)
