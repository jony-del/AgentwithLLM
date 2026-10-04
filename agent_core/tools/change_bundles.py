"""Merge owned worktree results without overwriting concurrent parent edits."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from agent_core.models import ToolRisk, ToolResult
from agent_core.merge import MergeConflict, three_way_merge
from agent_core.permission_safety import inspect_paths, ordinary_write_permission
from agent_core.permission_types import PermissionBehavior, PermissionContext, PermissionResult
from agent_core.session import SessionAwareMixin, SessionContext
from agent_core.tools.base import ConcurrencySpec, ExecutionSafety, Tool, WorkspacePathMixin, write_text_exact
from agent_core.tools.catalog import builtin_tool


@builtin_tool
class MergeChangeBundleTool(WorkspacePathMixin, SessionAwareMixin, Tool):
    name = "merge_change_bundle"
    description = "Merge a session-owned worktree bundle; optional three_way accepts disjoint text edits with Python AST ownership checks. Verify afterwards."
    input_schema = {"type": "object", "properties": {"bundle_id": {"type": "string", "pattern": "^[0-9a-f]{32}$"},
                    "strategy": {"enum": ["strict", "three_way"]}},
                    "required": ["bundle_id"]}
    risk = ToolRisk.WRITE
    accept_edits_safe = True
    execution_safety = ExecutionSafety.TRANSACTIONAL
    transaction_backend = "workspace"

    def __init__(self, workspace: str | Path | None = None) -> None:
        WorkspacePathMixin.__init__(self, workspace)
        SessionAwareMixin.__init__(self, SessionContext(workspace=self._workspace))

    def _bundle(self, arguments: dict[str, Any]):
        if self.session.bundle_store is None:
            raise ValueError("change bundle store unavailable")
        return self.session.bundle_store.load(str(arguments["bundle_id"]))

    def concurrency_spec(self, arguments: dict[str, Any]) -> ConcurrencySpec:
        try:
            bundle = self._bundle(arguments)
            return ConcurrencySpec(tuple(self.workspace_lock(path, "write") for path in bundle.contents))
        except (ValueError, OSError, KeyError):
            return ConcurrencySpec((self.workspace_lock(".", "write", subtree=True),), exclusive=True)

    async def check_permissions(self, arguments: dict[str, Any], context: PermissionContext) -> PermissionResult:
        try:
            bundle = self._bundle(arguments)
            if bundle.parent_workspace != str(context.workspace.resolve()):
                return PermissionResult.deny("bundle belongs to another parent workspace")
            for path in bundle.contents:
                result = inspect_paths("write_text_file", {"path": path}, context)
                if result is not None and result.behavior in {PermissionBehavior.ASK, PermissionBehavior.DENY}:
                    return result
            return ordinary_write_permission(self.name, arguments, context)
        except (ValueError, OSError, KeyError) as exc:
            return PermissionResult.deny(str(exc))

    def _invoke(self, arguments: dict[str, Any]) -> ToolResult:
        bundle = self._bundle(arguments)
        conflicts = []
        planned = []
        total = 0
        for relative, content in bundle.contents.items():
            path = self.resolve_workspace_path(relative)
            total += path.stat().st_size if path.is_file() else 0
            if total > 16 * 1024 * 1024:
                conflicts.append(relative)
                continue
            data = path.read_bytes() if path.is_file() else None
            current = hashlib.sha256(data).hexdigest() if data is not None else None
            if current != bundle.expected_hashes[relative] or (path.exists() and not path.is_file()):
                base = bundle.baseline_contents.get(relative)
                if arguments.get("strategy", "strict") == "three_way" and base is not None and data is not None and content is not None:
                    try:
                        content = three_way_merge(base, data.decode("utf-8"), content, python=relative.endswith(".py"))
                    except (MergeConflict, UnicodeError):
                        conflicts.append(relative)
                else:
                    conflicts.append(relative)
            planned.append((path, content))
        if conflicts:
            return ToolResult(self.name, "Resolve parent/source conflicts before merging: " + ", ".join(conflicts),
                              ok=False, metadata={"error_type": "MergeConflict", "conflicts": conflicts})
        for path, content in planned:
            if content is None:
                path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                write_text_exact(path, content)
        return ToolResult(self.name, "Merged bundle; run verification on the integrated workspace.",
                          metadata={"bundle_id": bundle.id, "changed_paths": list(bundle.contents),
                                    "strategy": arguments.get("strategy", "strict"),
                                    "precision": "Python AST ownership; other languages use conservative line merge"})
