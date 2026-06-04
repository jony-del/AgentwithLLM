import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agent_core.models import ToolRisk
from agent_core.tools.builtin import (
    EditFileTool,
    GitDiffTool,
    ListDirTool,
    RunCommandTool,
    RunTestsTool,
    SearchTextTool,
    _run_subprocess,
)


# --- list_dir ----------------------------------------------------------------


def test_list_dir_marks_directories(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    out = ListDirTool(tmp_path).run({"path": "."}).content
    assert "sub/" in out
    assert "a.txt" in out


def test_list_dir_missing_path_is_not_ok(tmp_path: Path) -> None:
    result = ListDirTool(tmp_path).run({"path": "nope"})
    assert not result.ok


# --- edit_file ---------------------------------------------------------------


def test_edit_file_replaces_unique_string(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("hello world", encoding="utf-8")
    result = EditFileTool(tmp_path).run({"path": "f.txt", "old_string": "world", "new_string": "there"})
    assert result.ok
    assert f.read_text(encoding="utf-8") == "hello there"


def test_edit_file_rejects_ambiguous_match(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("a a a", encoding="utf-8")
    result = EditFileTool(tmp_path).run({"path": "f.txt", "old_string": "a", "new_string": "b"})
    assert not result.ok
    assert result.metadata["error_type"] == "Ambiguous"
    assert f.read_text(encoding="utf-8") == "a a a"  # unchanged


def test_edit_file_replace_all(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("a a a", encoding="utf-8")
    result = EditFileTool(tmp_path).run(
        {"path": "f.txt", "old_string": "a", "new_string": "b", "replace_all": True}
    )
    assert result.ok
    assert f.read_text(encoding="utf-8") == "b b b"


def test_edit_file_missing_string_is_not_ok(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("hello", encoding="utf-8")
    result = EditFileTool(tmp_path).run({"path": "f.txt", "old_string": "absent", "new_string": "x"})
    assert not result.ok
    assert result.metadata["error_type"] == "NotFound"


def test_edit_file_rejects_path_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes workspace"):
        EditFileTool(tmp_path).run({"path": "../evil.txt", "old_string": "a", "new_string": "b"})


# --- search_text -------------------------------------------------------------


def test_search_text_finds_substring_with_location(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("import os\nresult = compute()\n", encoding="utf-8")
    out = SearchTextTool(tmp_path).run({"pattern": "compute"}).content
    assert "code.py:2:" in out
    assert "compute()" in out


def test_search_text_glob_and_regex(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("def foo(): docs\n", encoding="utf-8")
    out = SearchTextTool(tmp_path).run({"pattern": r"def \w+\(", "regex": True, "glob": "*.py"}).content
    assert "a.py:1:" in out
    assert "b.md" not in out  # excluded by glob


def test_search_text_skips_ignored_dirs(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("needle", encoding="utf-8")
    (tmp_path / "keep.txt").write_text("needle", encoding="utf-8")
    out = SearchTextTool(tmp_path).run({"pattern": "needle"}).content
    assert "keep.txt:1:" in out
    assert ".git" not in out


def test_search_text_no_matches(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("nothing here", encoding="utf-8")
    assert SearchTextTool(tmp_path).run({"pattern": "zzz"}).content == "No matches."


# --- command execution -------------------------------------------------------


def test_run_subprocess_success(tmp_path: Path) -> None:
    result = _run_subprocess("t", [sys.executable, "-c", "print('hi')"], cwd=tmp_path, timeout=30, shell=False)
    assert result.ok
    assert "hi" in result.content
    assert result.metadata["returncode"] == 0


def test_run_subprocess_nonzero_exit_is_not_ok(tmp_path: Path) -> None:
    result = _run_subprocess(
        "t", [sys.executable, "-c", "import sys; sys.exit(3)"], cwd=tmp_path, timeout=30, shell=False
    )
    assert not result.ok
    assert result.metadata["returncode"] == 3


def test_run_subprocess_timeout(tmp_path: Path) -> None:
    result = _run_subprocess(
        "t", [sys.executable, "-c", "import time; time.sleep(5)"], cwd=tmp_path, timeout=1, shell=False
    )
    assert not result.ok
    assert result.metadata["error_type"] == "Timeout"


def test_run_command_shell_echo(tmp_path: Path) -> None:
    result = RunCommandTool(tmp_path).run({"command": "echo hello"})
    assert "hello" in result.content


def test_run_command_is_dangerous() -> None:
    assert RunCommandTool().risk is ToolRisk.DANGEROUS


# --- run_tests ---------------------------------------------------------------


def test_run_tests_runs_pytest(tmp_path: Path) -> None:
    (tmp_path / "test_sample.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8")
    result = RunTestsTool(tmp_path).run({"args": ["-q"]})
    assert result.ok
    assert "passed" in result.content


# --- git_diff ----------------------------------------------------------------


def test_git_diff_reports_changes(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git not installed")

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "Test")
    target = tmp_path / "f.txt"
    target.write_text("one\n", encoding="utf-8")
    git("add", "f.txt")
    git("commit", "-m", "init")
    target.write_text("two\n", encoding="utf-8")

    result = GitDiffTool(tmp_path).run({})
    assert "-one" in result.content
    assert "+two" in result.content
