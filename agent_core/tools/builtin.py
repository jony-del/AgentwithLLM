"""Local, dependency-free built-in tools: directory listing, precise editing,
full-text search, command execution, git diff, and test running.

All file access is confined to the workspace via ``WorkspacePathMixin``; the
command/test runners execute with the workspace as their working directory.
Everything here is stdlib only — no LSP/MCP or other external integrations.
"""

from __future__ import annotations

import difflib
import fnmatch
import os
import re
import subprocess
import sys
from pathlib import Path

from agent_core.models import ToolRisk, ToolResult
from agent_core.tools.base import ConcurrencySpec, Tool, WorkspacePathMixin
from agent_core.tools.catalog import builtin_tool


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

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        return ConcurrencySpec((self.workspace_lock(arguments.get("path", "."), "read", subtree=True),))

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
        },
        "required": ["path", "old_string", "new_string"],
    }
    risk = ToolRisk.WRITE

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        return ConcurrencySpec((self.workspace_lock(arguments["path"], "write"),))

    def _invoke(self, arguments: dict[str, object]) -> ToolResult:
        path = self.resolve_workspace_path(arguments["path"])
        old_string = str(arguments["old_string"])
        new_string = str(arguments["new_string"])
        replace_all = bool(arguments.get("replace_all", False))

        if not path.exists():
            return ToolResult(self.name, f"No such file: {path}", ok=False, metadata={"error_type": "NotFound"})

        text = path.read_text(encoding="utf-8")
        try:
            updated, replaced = _apply_exact_edit(text, old_string, new_string, replace_all)
        except ExactEditError as exc:
            return ToolResult(self.name, str(exc), ok=False, metadata={"error_type": exc.error_type, **exc.metadata})
        path.write_text(updated, encoding="utf-8")
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

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        return ConcurrencySpec((self.workspace_lock(arguments.get("path", "."), "read", subtree=True),))

    def _invoke(self, arguments: dict[str, object]) -> ToolResult:
        pattern = str(arguments["pattern"])
        base = self.resolve_workspace_path(arguments.get("path", "."))
        glob = arguments.get("glob")
        max_results = int(arguments.get("max_results", 100))
        flags = re.IGNORECASE if arguments.get("ignore_case") else 0
        try:
            matcher = re.compile(pattern if arguments.get("regex") else re.escape(pattern), flags)
        except re.error as exc:
            return ToolResult(self.name, f"Invalid regex: {exc}", ok=False, metadata={"error_type": "BadRegex"})

        if not base.exists():
            return ToolResult(self.name, f"No such path: {base}", ok=False, metadata={"error_type": "NotFound"})

        results: list[str] = []
        truncated = False
        for file in self._iter_files(base, glob):
            if len(results) >= max_results:
                truncated = True
                break
            for lineno, line in self._matches_in_file(file, matcher):
                rel = file.relative_to(self.workspace)
                results.append(f"{rel.as_posix()}:{lineno}: {line.strip()}")
                if len(results) >= max_results:
                    truncated = True
                    break

        if not results:
            return ToolResult(self.name, "No matches.")
        body = "\n".join(results)
        if truncated:
            body += f"\n[... stopped at {max_results} matches ...]"
        return ToolResult(self.name, body, metadata={"matches": len(results), "truncated": truncated})

    def _iter_files(self, base: Path, glob: object):
        candidates = [base] if base.is_file() else base.rglob("*")
        for path in candidates:
            if not path.is_file():
                continue
            if any(part in _IGNORED_DIRS for part in path.relative_to(self.workspace).parts[:-1]):
                continue
            if glob and not fnmatch.fnmatch(path.name, str(glob)):
                continue
            yield path

    @staticmethod
    def _matches_in_file(file: Path, matcher):
        try:
            if file.stat().st_size > _MAX_SEARCH_FILE_BYTES:
                return
            raw = file.read_bytes()
        except OSError:
            return
        if _is_probably_binary(raw[:1024]):
            return
        text = raw.decode("utf-8", errors="replace")
        for index, line in enumerate(text.splitlines(), start=1):
            if matcher.search(line):
                yield index, line


@builtin_tool
class RunCommandTool(WorkspacePathMixin, Tool):
    name = "run_command"
    description = (
        "Run a shell command in the workspace and return its combined stdout/stderr and exit code. "
        "DANGEROUS: executes arbitrary commands; requires permission."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command line to execute."},
            "timeout": {"type": "integer", "description": "Seconds before the command is killed (default 30)."},
        },
        "required": ["command"],
    }
    risk = ToolRisk.DANGEROUS

    def _invoke(self, arguments: dict[str, object]) -> ToolResult:
        command = str(arguments["command"])
        timeout = min(int(arguments.get("timeout", 30)), _MAX_COMMAND_TIMEOUT)
        spec, shell = _shell_invocation(command)
        return _run_subprocess(self.name, spec, cwd=self.workspace, timeout=timeout, shell=shell)


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


@builtin_tool
class RunTestsTool(WorkspacePathMixin, Tool):
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

    def _invoke(self, arguments: dict[str, object]) -> ToolResult:
        cmd = [sys.executable, "-m", "pytest"]
        if arguments.get("target"):
            cmd.append(str(arguments["target"]))
        extra = arguments.get("args") or []
        if isinstance(extra, list):
            cmd += [str(a) for a in extra]
        timeout = min(int(arguments.get("timeout", 300)), _MAX_COMMAND_TIMEOUT)
        return _run_subprocess(self.name, cmd, cwd=self.workspace, timeout=timeout, shell=False)


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
    env = dict(os.environ)
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

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        return ConcurrencySpec((self.workspace_lock(arguments["path"], "read"),))

    def _invoke(self, arguments: dict[str, object]) -> ToolResult:
        path = self.resolve_workspace_path(arguments["path"])
        text = path.read_text(encoding="utf-8")
        offset = arguments.get("offset")
        limit = arguments.get("limit")
        # No paging requested: return up to DEFAULT_READ_LINES (like Claude Code's
        # Read), with a note to page on if the file is longer. This keeps an
        # unguided read predictable and bounded instead of dumping a huge file.
        if offset is None and limit is None:
            lines = text.splitlines()
            if len(lines) <= _DEFAULT_READ_LINES:
                return ToolResult(name=self.name, content=text)
            shown = "\n".join(lines[:_DEFAULT_READ_LINES])
            note = (
                f"\n[file truncated: showing 1-{_DEFAULT_READ_LINES} of {len(lines)} lines; "
                f"pass offset={_DEFAULT_READ_LINES + 1} to continue]"
            )
            return ToolResult(
                name=self.name,
                content=shown + note,
                metadata={"total_lines": len(lines), "shown_lines": _DEFAULT_READ_LINES},
            )
        # Explicit paging: honor offset/limit verbatim.
        lines = text.splitlines()
        start = max(int(offset) - 1, 0) if offset is not None else 0
        end = start + int(limit) if limit is not None else len(lines)
        return ToolResult(name=self.name, content="\n".join(lines[start:end]))


@builtin_tool
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

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        return ConcurrencySpec((self.workspace_lock(arguments["path"], "write"),))

    def _invoke(self, arguments: dict[str, object]) -> ToolResult:
        path = self.resolve_workspace_path(arguments["path"])
        before = path.read_text(encoding="utf-8") if path.exists() else ""
        path.parent.mkdir(parents=True, exist_ok=True)
        content = str(arguments.get("content", ""))
        path.write_text(content, encoding="utf-8")
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
