"""Navigation and code-editing tools that round out the Claude-Code-style tool set.

- ``glob``        : fast file-name pattern matching (the name-search complement to
                    ``search_text``'s content search).
- ``multi_edit``  : a sequence of exact-string edits applied to one file atomically.
- ``apply_patch`` : apply a context-anchored unified diff across one or more files.

All are stdlib-only and confined to the workspace via ``WorkspacePathMixin``. They reuse
the shared ``_IGNORED_DIRS`` ignore-list and ``_apply_exact_edit`` helper from
``builtin`` so behaviour stays consistent with ``search_text`` / ``edit_file``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_core.models import ToolRisk, ToolResult
from agent_core.permission_safety import ordinary_read_permission, ordinary_write_permission
from agent_core.permission_types import PermissionContext, PermissionResult
from agent_core.tools.base import ConcurrencySpec, ExecutionSafety, Tool, WorkspacePathMixin, coerce_int
from agent_core.tools.builtin import (
    ExactEditError,
    _apply_exact_edit,
    _IGNORED_DIRS,
    unified_diff,
)
from agent_core.tools.catalog import builtin_tool
from agent_core.unified_diff import PatchError, PatchHunk, PatchOperation, parse_unified_diff

_MAX_GLOB_RESULTS = 200


@builtin_tool
class GlobTool(WorkspacePathMixin, Tool):
    name = "glob"
    description = (
        "Find files by glob pattern (e.g. '**/*.py' or 'src/*.ts'), returning workspace-"
        "relative paths newest-first. Common build/vcs dirs are skipped. Use this to locate "
        "files by name; use search_text to grep their contents."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py'."},
            "path": {"type": "string", "description": "Directory to search under; defaults to the workspace root."},
            "max_results": {"type": "integer", "description": "Cap the number of paths returned (default 200)."},
        },
        "required": ["pattern"],
    }
    risk = ToolRisk.READ
    execution_safety = ExecutionSafety.SPECULATIVE_SAFE

    async def check_permissions(
        self, arguments: dict[str, Any], context: PermissionContext
    ) -> PermissionResult:
        return ordinary_read_permission(self.name, arguments, context)

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        return ConcurrencySpec((self.workspace_lock(
            arguments.get("path", "."), "read", subtree=True, requires_success=True
        ),))

    def _invoke(self, arguments: dict[str, object]) -> ToolResult:
        pattern = str(arguments["pattern"])
        base = self.resolve_workspace_path(arguments.get("path", "."))
        max_results = coerce_int(arguments.get("max_results", _MAX_GLOB_RESULTS))
        if not base.exists():
            return ToolResult(self.name, f"No such path: {base}", ok=False, metadata={"error_type": "NotFound"})

        matches: list[Path] = []
        try:
            for path in base.glob(pattern):
                if not path.is_file():
                    continue
                rel = path.relative_to(self.workspace)
                if any(part in _IGNORED_DIRS for part in rel.parts[:-1]):
                    continue
                matches.append(path)
        except (ValueError, OSError) as exc:
            return ToolResult(self.name, f"Bad glob pattern: {exc}", ok=False, metadata={"error_type": "BadPattern"})

        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        truncated = len(matches) > max_results
        shown = matches[:max_results]
        if not shown:
            return ToolResult(self.name, "No files matched.")
        body = "\n".join(p.relative_to(self.workspace).as_posix() for p in shown)
        if truncated:
            body += f"\n[... stopped at {max_results} of {len(matches)} matches ...]"
        return ToolResult(self.name, body, metadata={"matches": len(shown), "truncated": truncated})


@builtin_tool
class MultiEditTool(WorkspacePathMixin, Tool):
    name = "multi_edit"
    description = (
        "Apply a sequence of precise string replacements to a single workspace file in one "
        "atomic operation. Edits run in order, each seeing the result of the previous; if any "
        "edit fails the file is left untouched. Each edit's `old_string` must be unique unless "
        "`replace_all` is true."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "edits": {
                "type": "array",
                "description": "Ordered list of edits to apply.",
                "items": {
                    "type": "object",
                    "properties": {
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                        "replace_all": {"type": "boolean"},
                    },
                    "required": ["old_string", "new_string"],
                },
            },
        },
        "required": ["path", "edits"],
    }
    risk = ToolRisk.WRITE
    accept_edits_safe = True
    execution_safety = ExecutionSafety.TRANSACTIONAL
    transaction_backend = "workspace"

    async def check_permissions(
        self, arguments: dict[str, Any], context: PermissionContext
    ) -> PermissionResult:
        return ordinary_write_permission(self.name, arguments, context)

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        return ConcurrencySpec((self.workspace_lock(arguments["path"], "write"),))

    def _invoke(self, arguments: dict[str, object]) -> ToolResult:
        path = self.resolve_workspace_path(arguments["path"])
        edits = arguments.get("edits")
        if not isinstance(edits, list) or not edits:
            return ToolResult(self.name, "edits must be a non-empty list", ok=False, metadata={"error_type": "BadArgs"})
        if not path.exists():
            return ToolResult(self.name, f"No such file: {path}", ok=False, metadata={"error_type": "NotFound"})

        original = path.read_text(encoding="utf-8")
        text = original
        total = 0
        for index, edit in enumerate(edits):
            if not isinstance(edit, dict):
                return ToolResult(self.name, f"edit #{index} is not an object", ok=False, metadata={"error_type": "BadArgs"})
            try:
                text, replaced = _apply_exact_edit(
                    text,
                    str(edit.get("old_string", "")),
                    str(edit.get("new_string", "")),
                    bool(edit.get("replace_all", False)),
                )
            except ExactEditError as exc:
                # Atomic: nothing is written if any edit fails.
                return ToolResult(
                    self.name,
                    f"edit #{index} failed: {exc} (no changes written)",
                    ok=False,
                    metadata={"error_type": exc.error_type, "failed_edit": index, **exc.metadata},
                )
            total += replaced
        path.write_text(text, encoding="utf-8")
        rel = str(arguments.get("path", ""))
        return ToolResult(
            self.name,
            f"Applied {len(edits)} edit(s) ({total} replacement(s)) to {path}",
            metadata={"diff": unified_diff(original, text, rel)},
        )

    def render_args(self, arguments: dict[str, object]) -> str | None:
        edits = arguments.get("edits")
        count = len(edits) if isinstance(edits, list) else 0
        return f"{arguments.get('path', '')}, {count} edits"

    def render_result(self, arguments: dict[str, object], result: ToolResult) -> str | None:
        return result.metadata.get("diff") or None


@builtin_tool
class ApplyPatchTool(WorkspacePathMixin, Tool):
    name = "apply_patch"
    description = (
        "Apply a unified-diff patch across one or more workspace files. Hunks are matched by "
        "their context lines (line numbers are advisory), so the patch need not be byte-exact on "
        "offsets. Use '--- /dev/null' as the old path to create a new file. If any hunk's context "
        "can't be located the whole patch is rejected and nothing is written."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "patch": {"type": "string", "description": "The unified diff text."},
        },
        "required": ["patch"],
    }
    risk = ToolRisk.WRITE
    accept_edits_safe = True
    execution_safety = ExecutionSafety.TRANSACTIONAL
    transaction_backend = "workspace"

    async def check_permissions(
        self, arguments: dict[str, Any], context: PermissionContext
    ) -> PermissionResult:
        return ordinary_write_permission(self.name, arguments, context)

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        try:
            patch_set = parse_unified_diff(str(arguments.get("patch", "")))
        except PatchError:
            return ConcurrencySpec((self.workspace_lock(".", "write", subtree=True),))
        locks = tuple(self.workspace_lock(file.target, "write") for file in patch_set.files)
        return ConcurrencySpec(locks)

    def _invoke(self, arguments: dict[str, object]) -> ToolResult:
        patch_text = str(arguments.get("patch", ""))
        try:
            patch_set = parse_unified_diff(patch_text)
        except PatchError as exc:
            return ToolResult(self.name, f"Could not parse patch: {exc}", ok=False, metadata={"error_type": "BadPatch"})

        # Pass 1: compute every new file body before touching disk, so a failure midway
        # leaves the workspace untouched (atomic across files).
        planned: list[tuple[Path, str | None, PatchOperation]] = []
        for patch_file in patch_set.files:
            target = patch_file.target
            path = self.resolve_workspace_path(target)
            if patch_file.operation is PatchOperation.CREATE:
                if path.exists():
                    return ToolResult(
                        self.name,
                        f"File already exists: {target}",
                        ok=False,
                        metadata={"error_type": "AlreadyExists"},
                    )
                new_text = _apply_hunks("", patch_file.hunks)
                if new_text and patch_text.endswith("\n"):
                    new_text += "\n"
                planned.append((path, new_text, patch_file.operation))
                continue
            if not path.exists():
                return ToolResult(self.name, f"No such file to patch: {target}", ok=False, metadata={"error_type": "NotFound"})
            original = path.read_text(encoding="utf-8")
            try:
                updated = _apply_hunks(original, patch_file.hunks)
            except PatchError as exc:
                return ToolResult(
                    self.name,
                    f"Hunk did not apply to {target}: {exc} (no changes written)",
                    ok=False,
                    metadata={"error_type": "HunkFailed"},
                )
            if patch_file.operation is PatchOperation.DELETE and updated:
                return ToolResult(
                    self.name,
                    f"Delete patch did not remove all content from {target} (no changes written)",
                    ok=False,
                    metadata={"error_type": "HunkFailed"},
                )
            planned.append(
                (path, None if patch_file.operation is PatchOperation.DELETE else updated, patch_file.operation)
            )

        # Pass 2: write everything.
        for path, planned_text, operation in planned:
            if operation is PatchOperation.DELETE:
                path.unlink()
                continue
            if operation is PatchOperation.CREATE:
                path.parent.mkdir(parents=True, exist_ok=True)
            assert planned_text is not None
            path.write_text(planned_text, encoding="utf-8")
        summary = ", ".join(p.relative_to(self.workspace).as_posix() for p, _, _ in planned)
        return ToolResult(self.name, f"Patched {len(planned)} file(s): {summary}")

    def render_args(self, arguments: dict[str, object]) -> str | None:
        try:
            patch_set = parse_unified_diff(str(arguments.get("patch", "")))
        except PatchError:
            return None
        names = ", ".join(file.target for file in patch_set.files)
        return names or None

    def render_result(self, arguments: dict[str, object], result: ToolResult) -> str | None:
        # The patch argument already *is* a unified diff — show it verbatim.
        return str(arguments.get("patch", "")) or None


# --- unified-diff parsing/application (stdlib only) ------------------------------

# Compatibility shim retained for callers that imported the old private parser;
# all authorization and execution paths above use ``parse_unified_diff``.
def _parse_unified_diff(patch_text: str):
    """Parse into ``[(target_path, hunks, is_new_file), ...]``.

    Each hunk is ``(old_lines, new_lines)``: the context+removed lines to find, and the
    context+added lines to substitute. Line numbers in ``@@`` headers are ignored (we
    anchor on context), which makes the patch resilient to small offset drift.
    """
    patch_set = parse_unified_diff(patch_text)
    return [
        (
            file.target,
            [(list(hunk.old_lines), list(hunk.new_lines)) for hunk in file.hunks],
            file.operation is PatchOperation.CREATE,
        )
        for file in patch_set.files
    ]


def _apply_hunks(original: str, hunks: tuple[PatchHunk, ...]) -> str:
    """Apply each hunk by locating its old-block as a contiguous run of lines."""
    had_trailing_newline = original.endswith("\n")
    file_lines = original.splitlines()
    cursor = 0
    for hunk in hunks:
        old_block = list(hunk.old_lines)
        new_block = list(hunk.new_lines)
        if not old_block:
            # Pure insertion with no context anchor: append at the cursor.
            file_lines[cursor:cursor] = new_block
            cursor += len(new_block)
            continue
        index = _find_block(file_lines, old_block, cursor)
        if index < 0:
            index = _find_block(file_lines, old_block, 0)  # fall back to a full scan
        if index < 0:
            raise PatchError("context lines not found")
        file_lines[index:index + len(old_block)] = new_block
        cursor = index + len(new_block)
    result = "\n".join(file_lines)
    if had_trailing_newline and result:
        result += "\n"
    return result


def _find_block(haystack: list[str], needle: list[str], start: int) -> int:
    if not needle:
        return -1
    last = len(haystack) - len(needle)
    for i in range(max(start, 0), last + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1
