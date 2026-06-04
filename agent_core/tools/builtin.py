"""Local, dependency-free built-in tools: directory listing, precise editing,
full-text search, command execution, git diff, and test running.

All file access is confined to the workspace via ``WorkspacePathMixin``; the
command/test runners execute with the workspace as their working directory.
Everything here is stdlib only — no LSP/MCP or other external integrations.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
import sys
from pathlib import Path

from agent_core.models import ToolRisk, ToolResult
from agent_core.tools.base import Tool, WorkspacePathMixin
from agent_core.tools.catalog import builtin_tool

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


def _is_probably_binary(sample: bytes) -> bool:
    return b"\x00" in sample


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

    def run(self, arguments: dict[str, object]) -> ToolResult:
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

    def run(self, arguments: dict[str, object]) -> ToolResult:
        path = self.resolve_workspace_path(arguments["path"])
        old_string = str(arguments["old_string"])
        new_string = str(arguments["new_string"])
        replace_all = bool(arguments.get("replace_all", False))

        if not old_string:
            return ToolResult(self.name, "old_string must not be empty", ok=False, metadata={"error_type": "EmptyMatch"})
        if old_string == new_string:
            return ToolResult(self.name, "old_string and new_string are identical", ok=False, metadata={"error_type": "NoOp"})
        if not path.exists():
            return ToolResult(self.name, f"No such file: {path}", ok=False, metadata={"error_type": "NotFound"})

        text = path.read_text(encoding="utf-8")
        count = text.count(old_string)
        if count == 0:
            return ToolResult(self.name, "old_string not found in file", ok=False, metadata={"error_type": "NotFound"})
        if count > 1 and not replace_all:
            return ToolResult(
                self.name,
                f"old_string is not unique ({count} matches); add surrounding context or pass replace_all=true",
                ok=False,
                metadata={"error_type": "Ambiguous", "matches": count},
            )

        updated = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        path.write_text(updated, encoding="utf-8")
        replaced = count if replace_all else 1
        return ToolResult(self.name, f"Replaced {replaced} occurrence(s) in {path}")


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

    def run(self, arguments: dict[str, object]) -> ToolResult:
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

    def run(self, arguments: dict[str, object]) -> ToolResult:
        command = str(arguments["command"])
        timeout = min(int(arguments.get("timeout", 30)), _MAX_COMMAND_TIMEOUT)
        return _run_subprocess(self.name, command, cwd=self.workspace, timeout=timeout, shell=True)


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

    def run(self, arguments: dict[str, object]) -> ToolResult:
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

    def run(self, arguments: dict[str, object]) -> ToolResult:
        cmd = [sys.executable, "-m", "pytest"]
        if arguments.get("target"):
            cmd.append(str(arguments["target"]))
        extra = arguments.get("args") or []
        if isinstance(extra, list):
            cmd += [str(a) for a in extra]
        timeout = min(int(arguments.get("timeout", 300)), _MAX_COMMAND_TIMEOUT)
        return _run_subprocess(self.name, cmd, cwd=self.workspace, timeout=timeout, shell=False)


def _run_subprocess(tool_name: str, command, *, cwd: Path, timeout: int, shell: bool) -> ToolResult:
    """Run a subprocess, capturing combined output and exit code into a ToolResult.

    A non-zero exit code is reported as ``ok=False`` but is not an error in itself —
    the output (e.g. failing tests, a diff) is still returned for the agent to read.
    """
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            shell=shell,
            capture_output=True,
            text=True,
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
