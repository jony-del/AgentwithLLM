"""Local, dependency-free built-in tools: directory listing, precise editing,
full-text search, command execution, git diff, and test running.

All file access is confined to the workspace via ``WorkspacePathMixin``; the
command/test runners execute with the workspace as their working directory.
Everything here is stdlib only — no LSP/MCP or other external integrations.
"""

from __future__ import annotations

import difflib
import importlib.util
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from agent_core.models import ToolRisk, ToolResult
from agent_core.permission_safety import (
    ordinary_read_permission,
    ordinary_write_permission,
)
from agent_core.permission_types import (
    DecisionSource,
    PermissionContext,
    PermissionMode,
    PermissionResult,
)
from agent_core.sandbox import SandboxAwareMixin
from agent_core.session import SessionAwareMixin, SessionContext
from agent_core.tools.base import (
    ConcurrencySpec,
    ExecutionScope,
    ExecutionSafety,
    ResourceLock,
    Tool,
    WorkspacePathMixin,
    coerce_int,
    read_text_exact,
    write_text_exact,
)
from agent_core.tools.catalog import builtin_tool
from agent_core.codeintel.snapshots import edit_precondition, worktree_id


def unified_diff(before: str, after: str, path: str) -> str:
    """A unified diff (stdlib only) for a file edit, or ``""`` when nothing changed.

    Used by write/edit tools to hand the UI a ready-to-highlight diff via
    ``Tool.render_result`` — the diff is kept in result metadata so it doesn't
    enter the model's transcript.
    """
    if before == after:
        return ""
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(diff)

# Directories that are noise for search/listing and almost never what a user wants
# to grep through. Skipped when walking the tree.
_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        ".venv",
        "venv",
        "env",
        "dist",
        "build",
        ".idea",
        ".vscode",
        "runs",
        "memory",
    }
)

# Search guards so one giant or binary file can't wedge a scan.
_MAX_SEARCH_FILE_BYTES = 2_000_000
_MAX_COMMAND_TIMEOUT = 600  # hard cap (seconds) regardless of requested timeout
# Single (non-stacked) bound on one ripgrep subprocess; on expiry the child is
# killed and reaped (subprocess.run's TimeoutExpired handling), then the search
# falls back to the pure-Python scan.
_RG_TIMEOUT_SECONDS = 30

logger = logging.getLogger(__name__)
# Default line window for an unguided read_text_file (mirrors Claude Code's Read).
_DEFAULT_READ_LINES = 2000


def _is_probably_binary(sample: bytes) -> bool:
    return b"\x00" in sample


class ExactEditError(Exception):
    """A precise string-replace could not be applied (empty/duplicate/missing match).

    Carries an ``error_type`` so callers can surface it in ``ToolResult.metadata`` the
    same way the inline checks used to.
    """

    def __init__(self, message: str, error_type: str, **metadata: object) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.metadata = metadata


def _apply_exact_edit(text: str, old_string: str, new_string: str, replace_all: bool) -> tuple[str, int]:
    """Apply one exact-string replacement to ``text``; return ``(updated, replaced_count)``.

    Shared by ``edit_file`` (single edit) and ``multi_edit`` (a sequence applied in
    memory). Raises ``ExactEditError`` if the edit is empty, a no-op, missing, or
    ambiguous (more than one match without ``replace_all``).
    """
    if "\r\n" in text:
        old_string = re.sub(r"(?<!\r)\n", "\r\n", old_string)
        new_string = re.sub(r"(?<!\r)\n", "\r\n", new_string)
    if not old_string:
        raise ExactEditError("old_string must not be empty", "EmptyMatch")
    if old_string == new_string:
        raise ExactEditError("old_string and new_string are identical", "NoOp")
    count = text.count(old_string)
    if count == 0:
        raise ExactEditError("old_string not found in file", "NotFound")
    if count > 1 and not replace_all:
        raise ExactEditError(
            f"old_string is not unique ({count} matches); add surrounding context or pass replace_all=true",
            "Ambiguous",
            matches=count,
        )
    updated = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
    return updated, (count if replace_all else 1)


@builtin_tool
class ListDirTool(WorkspacePathMixin, Tool):
    name = "list_dir"
    description = "List the entries of a directory in the workspace (directories end with '/')."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Directory to list; defaults to the workspace root."}},
        "required": [],
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
        target = self.resolve_workspace_path(arguments.get("path", "."))
        if not target.exists():
            return ToolResult(self.name, f"No such path: {target}", ok=False, metadata={"error_type": "NotFound"})
        if target.is_file():
            return ToolResult(self.name, target.name)
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        lines = [f"{p.name}/" if p.is_dir() else p.name for p in entries]
        return ToolResult(self.name, "\n".join(lines) if lines else "(empty directory)")


@builtin_tool
class EditFileTool(WorkspacePathMixin, Tool):
    name = "edit_file"
    description = (
        "Make a precise edit by replacing an exact string in a workspace file. "
        "`old_string` must match verbatim and, unless `replace_all` is true, must be unique."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string", "description": "Exact text to replace (must be unique unless replace_all)."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of requiring uniqueness."},
            "expected_version": {"type": "object", "description": "File version returned by a verified read."},
        },
        "required": ["path", "old_string", "new_string"],
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
        old_string = str(arguments["old_string"])
        new_string = str(arguments["new_string"])
        replace_all = bool(arguments.get("replace_all", False))
        edit_precondition(self.workspace, path, arguments.get("expected_version"))

        if not path.exists():
            return ToolResult(self.name, f"No such file: {path}", ok=False, metadata={"error_type": "NotFound"})

        text = read_text_exact(path)
        try:
            updated, replaced = _apply_exact_edit(text, old_string, new_string, replace_all)
        except ExactEditError as exc:
            return ToolResult(self.name, str(exc), ok=False, metadata={"error_type": exc.error_type, **exc.metadata})
        write_text_exact(path, updated)
        rel = str(arguments.get("path", ""))
        return ToolResult(
            self.name,
            f"Replaced {replaced} occurrence(s) in {path}",
            metadata={"diff": unified_diff(text, updated, rel)},
        )

    def render_args(self, arguments: dict[str, object]) -> str | None:
        return str(arguments.get("path", "")) or None

    def render_result(self, arguments: dict[str, object], result: ToolResult) -> str | None:
        return result.metadata.get("diff") or None


@builtin_tool
class SearchTextTool(WorkspacePathMixin, Tool):
    name = "search_text"
    description = (
        "Full-text search across workspace files (plain substring by default, or a regex). "
        "Returns 'relpath:line: text' matches; common build/vcs dirs are skipped."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Substring (or regex if regex=true) to find."},
            "path": {"type": "string", "description": "File or directory to search; defaults to the workspace root."},
            "glob": {"type": "string", "description": "Only search files whose name matches this glob, e.g. '*.py'."},
            "regex": {"type": "boolean", "description": "Treat pattern as a regular expression."},
            "ignore_case": {"type": "boolean", "description": "Case-insensitive matching."},
            "max_results": {"type": "integer", "description": "Stop after this many matches (default 100)."},
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
            arguments.get("path", "."), "read", subtree=True, requires_success=True, materialize=False
        ),))

    def _invoke(self, arguments: dict[str, object]) -> ToolResult:
        from agent_core.codeintel.legacy import search
        return search(self, arguments)


@builtin_tool
class GitDiffTool(WorkspacePathMixin, Tool):
    name = "git_diff"
    description = "Show the git diff for the workspace. Set staged=true for the index; pass path to scope it."
    input_schema = {
        "type": "object",
        "properties": {
            "staged": {"type": "boolean", "description": "Diff the staged changes (git diff --staged)."},
            "path": {"type": "string", "description": "Limit the diff to this file or directory."},
        },
        "required": [],
    }
    risk = ToolRisk.READ

    async def check_permissions(
        self, arguments: dict[str, Any], context: PermissionContext
    ) -> PermissionResult:
        return ordinary_read_permission(self.name, arguments, context)

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        raw_path = arguments.get("path", ".")
        return ConcurrencySpec((self.workspace_lock(raw_path, "read", subtree=True),))

    def _invoke(self, arguments: dict[str, object]) -> ToolResult:
        cmd = ["git", "diff"]
        if arguments.get("staged"):
            cmd.append("--staged")
        if arguments.get("path"):
            cmd += ["--", str(self.resolve_workspace_path(arguments["path"]))]
        return _run_subprocess(self.name, cmd, cwd=self.workspace, timeout=30, shell=False)


_TEST_SHELL_META = re.compile(r"[;&|`\r\n]|\$\(|%COMSPEC%", re.IGNORECASE)
_TEST_UNSAFE_OPTIONS = {
    "-c",
    "-p",
    "--rootdir",
    "--confcutdir",
    "--basetemp",
    "--override-ini",
    "--import-mode",
}


def _unsafe_test_arguments(arguments: dict[str, Any]) -> str | None:
    target = str(arguments.get("target", ""))
    if _TEST_SHELL_META.search(target):
        return "test target contains shell control syntax"
    extra = arguments.get("args") or []
    if not isinstance(extra, list):
        return "test args must be an argv list"
    for value in extra:
        arg = str(value)
        if _TEST_SHELL_META.search(arg):
            return "test args contain shell control syntax"
        if arg.startswith("@"):
            return "pytest response-file arguments are not permitted"
        option = arg.split("=", 1)[0]
        if option in _TEST_UNSAFE_OPTIONS:
            return f"pytest option {option!r} can alter code/config loading and is not permitted"
    return None


@builtin_tool
class RunTestsTool(WorkspacePathMixin, SessionAwareMixin, SandboxAwareMixin, Tool):
    name = "run_tests"
    description = (
        "Run the test suite with pytest in the workspace. Optionally scope to `target` "
        "(file/node id) and pass extra `args`. DANGEROUS: executes test code."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "target": {"type": "string", "description": "A pytest path or node id, e.g. tests/test_x.py::test_y."},
            "args": {"type": "array", "items": {"type": "string"}, "description": "Extra pytest arguments."},
            "timeout": {"type": "integer", "description": "Seconds before the run is killed (default 300)."},
        },
        "required": [],
    }
    risk = ToolRisk.DANGEROUS
    safely_cancellable = True
    execution_timeout = 600.0

    def __init__(self, workspace: str | Path | None = None) -> None:
        WorkspacePathMixin.__init__(self, workspace)
        SessionAwareMixin.__init__(self, SessionContext(workspace=self._workspace))

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        return ConcurrencySpec(
            (
                ResourceLock(
                    "fs",
                    str(self.workspace.resolve()),
                    "write",
                    subtree=True,
                    requires_success=True,
                ),
                ResourceLock("env", "process", "write", requires_success=True),
            )
        )

    async def check_permissions(
        self, arguments: dict[str, Any], context: PermissionContext
    ) -> PermissionResult:
        invalid = _unsafe_test_arguments(arguments)
        if invalid is not None:
            return PermissionResult.deny(invalid, decision_source=DecisionSource.TOOL)
        allow_rule = context.rules.allow_match(self.name, arguments) if context.rules is not None else None
        if allow_rule is not None:
            return PermissionResult.allow(
                "test invocation allowed by rule",
                decision_source=DecisionSource.RULE,
                matched_rule=allow_rule,
            )
        if self.name in context.session_authorizations.tool_names:
            return PermissionResult.allow("tests allowed for this session", decision_source=DecisionSource.RULE)
        if context.sandbox.auto_allow_enabled and context.sandbox.will_sandbox:
            return PermissionResult.passthrough("sandbox policy may allow this test invocation")
        if context.mode is PermissionMode.BYPASS:
            return PermissionResult.passthrough("test invocation may be resolved by bypass mode")
        return PermissionResult.ask(
            "workspace tests execute project code and require review",
            classifier_approvable=True,
            metadata={"target": str(arguments.get("target", ""))[:200]},
        )

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        # Host pytest is irrelevant for a Linux guest; the prepared image manifest is
        # the authority in that mode.
        if not self.sandbox.uses_guest and importlib.util.find_spec("pytest") is None:
            return ToolResult(
                self.name,
                "pytest is not installed in this environment. Install the dev extras "
                "(pip install -e .[dev]) or pytest itself to run tests.",
                ok=False,
                metadata={"error_type": "MissingDependency"},
            )
        cmd = [sys.executable, "-m", "pytest"]
        guest_cmd = ["@python", "-m", "pytest"]
        if arguments.get("target"):
            target = str(arguments["target"])
            cmd.append(target)
            target_path = Path(target.split("::", 1)[0])
            if target_path.is_absolute():
                resolved = target_path.resolve()
                if resolved != self.workspace and self.workspace not in resolved.parents:
                    return ToolResult(
                        self.name, "test target escapes workspace", ok=False,
                        metadata={"error_type": "guest_capability_unavailable"},
                    )
                suffix = target[len(str(target_path)):]
                target = self.sandbox.translate_path(resolved) + suffix
            guest_cmd.append(target)
        extra = arguments.get("args") or []
        if isinstance(extra, list):
            cmd += [str(a) for a in extra]
            guest_cmd += [str(a) for a in extra]
        timeout = max(1, min(coerce_int(arguments.get("timeout", 300)), _MAX_COMMAND_TIMEOUT))
        # Sandbox the (argv) test invocation when active; no command string to exclude on.
        from agent_core.sandbox import SandboxInvocation

        invocation = SandboxInvocation.create(
            cmd,
            guest_argv=guest_cmd,
            required_guest_capabilities=("python",),
            scope=ExecutionScope.for_workspace(self.workspace, network="deny"),
        )
        try:
            spec, shell = self.sandbox.wrap_invocation(invocation)
        except RuntimeError as exc:
            return ToolResult(
                self.name, str(exc), ok=False,
                metadata={"error_type": getattr(exc, "error_type", "SandboxUnavailable")},
            )
        if shell or not isinstance(spec, (list, tuple)):
            return ToolResult(self.name, "test runner requires explicit argv", ok=False)
        from agent_core.process_supervisor import ProcessSupervisor
        from tempfile import TemporaryDirectory

        async def execute(supervisor: ProcessSupervisor) -> ToolResult:
            output = await supervisor.run_argv([str(item) for item in spec], self.workspace, timeout=timeout)
            content = str(output.pop("output"))
            return ToolResult(
                self.name, f"[exit {output['exit_code']}; state {output['state']}]\n{content}",
                ok=output["state"] == "completed", metadata={**output, "returncode": output["exit_code"]},
            )

        supervisor = self.session.process_supervisor
        if isinstance(supervisor, ProcessSupervisor):
            return await execute(supervisor)
        # Standalone tool embedding remains supported, with owned cleanup.
        with TemporaryDirectory(prefix="polaris-test-runner-") as root:
            from agent_core.tool_config import ShellToolConfig
            owned = ProcessSupervisor(ShellToolConfig(), root)
            try:
                return await execute(owned)
            finally:
                await owned.shutdown()


def _shell_invocation(command: str):
    """Pick how to run a free-form shell command line per platform.

    On Windows the default ``shell=True`` runs ``cmd.exe``, which lacks ``cat`` and
    every PowerShell cmdlet (``Get-Content`` etc.) the model naturally reaches for —
    so run PowerShell explicitly and force its output stream to UTF-8. On POSIX,
    ``shell=True`` (``/bin/sh``) is what's expected. Returns ``(spec, shell)``.
    """
    if os.name == "nt":
        # Force UTF-8 both ways: the output stream, and cmdlet file reads/writes
        # (Windows PowerShell 5.1 otherwise reads files in the ANSI codepage — GBK
        # on zh-CN — and garbles UTF-8 content like CJK).
        prelude = (
            "$OutputEncoding=[Console]::OutputEncoding=[Text.Encoding]::UTF8; "
            "$PSDefaultParameterValues['*:Encoding']='utf8'; "
        )
        return (["powershell", "-NoProfile", "-NonInteractive", "-Command", prelude + command], False)
    return (command, True)


def _utf8_child_env() -> dict[str, str]:
    """Environment that makes child processes emit UTF-8 (kills GBK encode errors)."""
    from agent_core.process_supervisor import safe_process_environment
    env = safe_process_environment()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run_subprocess(tool_name: str, command, *, cwd: Path, timeout: int, shell: bool) -> ToolResult:
    """Run a subprocess, capturing combined output and exit code into a ToolResult.

    A non-zero exit code is reported as ``ok=False`` but is not an error in itself —
    the output (e.g. failing tests, a diff) is still returned for the agent to read.
    Output is decoded as UTF-8 with ``errors="replace"`` so a narrow OS locale (e.g.
    GBK on zh-CN Windows) can't raise mid-decode.
    """
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            shell=shell,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=_utf8_child_env(),
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return ToolResult(tool_name, f"Command not found: {exc}", ok=False, metadata={"error_type": "NotFound"})
    except subprocess.TimeoutExpired:
        return ToolResult(tool_name, f"Timed out after {timeout}s", ok=False, metadata={"error_type": "Timeout"})

    output = (completed.stdout or "") + (completed.stderr or "")
    output = output.strip() or "(no output)"
    body = f"[exit {completed.returncode}]\n{output}"
    return ToolResult(tool_name, body, ok=completed.returncode == 0, metadata={"returncode": completed.returncode})


@builtin_tool
class EchoTool(Tool):
    name = "echo"
    description = "Echo text back to the agent."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    risk = ToolRisk.READ
    execution_safety = ExecutionSafety.SPECULATIVE_SAFE

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        return ConcurrencySpec()

    def _invoke(self, arguments: dict[str, object]) -> ToolResult:
        return ToolResult(name=self.name, content=str(arguments.get("text", "")))


@builtin_tool
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
    execution_safety = ExecutionSafety.SPECULATIVE_SAFE

    async def check_permissions(
        self, arguments: dict[str, Any], context: PermissionContext
    ) -> PermissionResult:
        return ordinary_read_permission(self.name, arguments, context)

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        return ConcurrencySpec((self.workspace_lock(
            arguments["path"], "read", requires_success=True
        ),))

    def _invoke(self, arguments: dict[str, object]) -> ToolResult:
        path = self.resolve_workspace_path(arguments["path"])
        if path.suffix.lower() == ".ipynb":
            from agent_core.notebook import format_notebook

            content, metadata = format_notebook(path)
            return ToolResult(name=self.name, content=content, metadata=metadata)
        session = getattr(self, "_code_session", None)
        session_config = getattr(session, "codeintel_config", None)
        if session_config is not None and not session_config.enabled:
            # Layer disabled: plain read, no versioned evidence is issued.
            text = path.read_text(encoding="utf-8")
            version_metadata: dict[str, Any] | None = None
        else:
            from agent_core.codeintel.config import CodeIntelConfig
            from agent_core.codeintel.budget import QueryBudget
            from agent_core.codeintel.snapshots import read_snapshot
            from agent_core.tools.base import current_execution_context, current_execution_scope
            # Bounded reads hash the exact bytes used to generate the observation.
            data, version = read_snapshot(self.workspace, str(path.relative_to(self.workspace)),
                QueryBudget(CodeIntelConfig(max_file_bytes=16 * 1024 * 1024), current_execution_scope()), allow_secret=True)
            context = current_execution_context()
            root = context.logical_workspace if context and context.logical_workspace else self.workspace
            version_metadata = {**version.to_dict(), "worktree_id": worktree_id(root)}
            text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        offset = arguments.get("offset")
        limit = arguments.get("limit")
        version_fields: dict[str, Any] = {}
        if version_metadata is not None:
            version_fields = {"file_version": version_metadata}
        # No paging requested: return up to DEFAULT_READ_LINES (like Claude Code's
        # Read), with a note to page on if the file is longer. This keeps an
        # unguided read predictable and bounded instead of dumping a huge file.
        if offset is None and limit is None:
            lines = text.splitlines()
            if len(lines) <= _DEFAULT_READ_LINES:
                return ToolResult(name=self.name, content=text,
                                  metadata={**version_fields, "start_line": 1, "end_line": len(lines)}
                                  if version_fields else {})
            shown = "\n".join(lines[:_DEFAULT_READ_LINES])
            note = (
                f"\n[file truncated: showing 1-{_DEFAULT_READ_LINES} of {len(lines)} lines; "
                f"pass offset={_DEFAULT_READ_LINES + 1} to continue]"
            )
            return ToolResult(
                name=self.name,
                content=shown + note,
                metadata={"total_lines": len(lines), "shown_lines": _DEFAULT_READ_LINES,
                          **version_fields, "start_line": 1, "end_line": _DEFAULT_READ_LINES},
            )
        # Explicit paging: honor offset/limit verbatim.
        lines = text.splitlines()
        start = max(coerce_int(offset) - 1, 0) if offset is not None else 0
        end = start + coerce_int(limit) if limit is not None else len(lines)
        if not version_fields:
            return ToolResult(name=self.name, content="\n".join(lines[start:end]))
        return ToolResult(name=self.name, content="\n".join(lines[start:end]), metadata={
            **version_fields, "start_line": start + 1, "end_line": min(end, len(lines))})


@builtin_tool
class WriteTextFileTool(WorkspacePathMixin, Tool):
    name = "write_text_file"
    description = "Write UTF-8 text to a file inside the current workspace."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "expected_version": {"type": "object", "description": "File version returned by a verified read."},
        },
        "required": ["path", "content"],
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
        before = read_text_exact(path) if path.exists() else ""
        edit_precondition(self.workspace, path, arguments.get("expected_version"))
        path.parent.mkdir(parents=True, exist_ok=True)
        content = str(arguments.get("content", ""))
        write_text_exact(path, content)
        rel = str(arguments.get("path", ""))
        verb = "Created" if before == "" else "Wrote"
        return ToolResult(
            name=self.name,
            content=f"{verb} {path}",
            metadata={"diff": unified_diff(before, content, rel)},
        )

    def render_args(self, arguments: dict[str, object]) -> str | None:
        return str(arguments.get("path", "")) or None

    def render_result(self, arguments: dict[str, object], result: ToolResult) -> str | None:
        return result.metadata.get("diff") or None
