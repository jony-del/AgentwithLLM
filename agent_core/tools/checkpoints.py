"""Preview and transactionally restore explicit source paths from task checkpoints."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent_core.models import ToolResult, ToolRisk
from agent_core.permission_safety import inspect_paths
from agent_core.permission_types import PermissionContext, PermissionResult
from agent_core.session import SessionAwareMixin, SessionContext
from agent_core.tools.base import ConcurrencySpec, ExecutionSafety, ResourceLock, Tool, WorkspacePathMixin
from agent_core.tools.catalog import builtin_tool
from agent_core.tools.codeintel import path_allowed


@builtin_tool
class CreateCheckpointTool(SessionAwareMixin, Tool):
    name = "create_checkpoint"
    description = "Save an owned source checkpoint in private storage before an experiment; returns a checkpoint id."
    input_schema = {"type": "object", "properties": {"label": {"type": "string", "maxLength": 256}}}
    risk = ToolRisk.READ
    # Source/blob copying uses a worker thread, which must be drained on cancellation.
    safely_cancellable = False
    execution_timeout = 600.0

    def concurrency_spec(self, arguments: dict[str, Any]) -> ConcurrencySpec:
        return ConcurrencySpec((ResourceLock("fs", str(self.session.workspace), "read", subtree=True),
                                ResourceLock("task", self.session.agent_id, "write")))

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        task, store = self.session.task_run, self.session.checkpoint_store
        if task is None or store is None:
            return ToolResult(self.name, "Task checkpoint store unavailable", ok=False)
        revision = await self.session.capture_revision()
        checkpoint = await asyncio.to_thread(store.capture, task, revision, str(arguments.get("label", "")),
                                              allowed=lambda path: path_allowed(self.session, path, self.name))
        if not task.baseline_checkpoint and revision.digest == task.baseline.digest and revision.workspace == task.baseline.workspace:
            task.baseline_checkpoint = checkpoint.id
        await self.session.persist_task_async()
        return ToolResult(self.name, f"Checkpoint {checkpoint.id}: {len(checkpoint.files)} files",
                          metadata={"checkpoint_id": checkpoint.id, "revision": checkpoint.revision})


@builtin_tool
class RollbackCheckpointTool(WorkspacePathMixin, SessionAwareMixin, Tool):
    name = "rollback_checkpoint"
    description = "Preview selected paths from an owned checkpoint, then apply using expected_revision and explicit approval. Invalidates verification."
    input_schema = {"type": "object", "properties": {
        "checkpoint_id": {"type": "string", "pattern": "^[0-9a-f]{32}$"},
        "paths": {"type": "array", "minItems": 1, "maxItems": 128, "uniqueItems": True,
                  "items": {"type": "string", "minLength": 1}},
        "preview": {"type": "boolean"}, "expected_revision": {"type": "string"},
    }, "required": ["checkpoint_id", "paths"]}
    risk = ToolRisk.WRITE
    execution_safety = ExecutionSafety.TRANSACTIONAL
    transaction_backend = "workspace"

    def __init__(self, workspace: str | Path | None = None) -> None:
        WorkspacePathMixin.__init__(self, workspace)
        SessionAwareMixin.__init__(self, SessionContext(workspace=self._workspace))

    def concurrency_spec(self, arguments: dict[str, Any]) -> ConcurrencySpec:
        # The expected revision covers the whole workspace, not just selected paths.
        return ConcurrencySpec((self.workspace_lock(".", "read", subtree=True, materialize=False),
                                *(self.workspace_lock(str(path), "write") for path in arguments.get("paths", [])),
                                ResourceLock("task", self.session.agent_id, "write")), exclusive=True)

    async def check_permissions(self, arguments: dict[str, Any], context: PermissionContext) -> PermissionResult:
        for path in arguments.get("paths", []):
            result = inspect_paths("write_text_file", {"path": path}, context)
            if result is not None:
                return result
        if arguments.get("preview", True):
            return PermissionResult.allow("checkpoint restore preview is read-only")
        return PermissionResult.ask("restoring checkpoint paths overwrites current source; review the preview",
                                    bypass_immune=True, classifier_approvable=False)

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        task, store = self.session.task_run, self.session.checkpoint_store
        if task is None or store is None:
            return ToolResult(self.name, "Task checkpoint store unavailable", ok=False)
        checkpoint = await asyncio.to_thread(store.load, str(arguments["checkpoint_id"]), task)
        current = await self.session.capture_revision()
        if checkpoint.workspace != current.workspace:
            raise ValueError("cannot restore a checkpoint from another workspace")
        paths = [str(path) for path in arguments["paths"]]
        planned = []
        total = 0
        for relative in paths:
            target = self.resolve_workspace_path(relative)
            if relative not in checkpoint.files and relative not in current.files:
                raise ValueError("path is outside checkpoint coverage: " + relative)
            if not path_allowed(self.session, relative, self.name):
                raise PermissionError("restore path is restricted by read policy")
            if task.contract.allowed_paths and not any(relative == allowed or relative.startswith(allowed.rstrip("/") + "/")
                                                       for allowed in task.contract.allowed_paths):
                raise ValueError("restore path is outside task scope")
            data = await asyncio.to_thread(store.read, checkpoint, relative)
            total += len(data) if data is not None else 0
            if total > store.MAX_BYTES:
                raise ValueError("restore byte budget exceeded")
            planned.append((target, data))
        preview = {"checkpoint_id": checkpoint.id, "expected_revision": current.digest, "paths": paths}
        if arguments.get("preview", True):
            import json
            return ToolResult(self.name, json.dumps(preview), metadata={"restore_preview": preview})
        if arguments.get("expected_revision") != current.digest:
            return ToolResult(self.name, "Workspace changed since restore preview", ok=False,
                              metadata={"error_type": "StaleRestorePreview"})
        def restore():
            for target, data in planned:
                if data is None:
                    target.unlink(missing_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
        await asyncio.to_thread(restore)
        # This is speculative task state only; final completion cannot use old proof.
        task.evidence.clear()
        task.reviews.clear()
        task.plan = [replace(step, status="pending") for step in task.plan]
        task.mutation_seen = True
        await self.session.persist_task_async()
        return ToolResult(self.name, "Restored selected paths; execute verification again.",
                          metadata={"changed_paths": paths, "restored_checkpoint": checkpoint.id})
