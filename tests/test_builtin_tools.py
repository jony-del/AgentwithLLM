import os
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
    ReadTextFileTool,
    RunCommandTool,
    RunTestsTool,
    SearchTextTool,
    _DEFAULT_READ_LINES,
    _run_subprocess,
    _shell_invocation,
)


# --- list_dir ----------------------------------------------------------------


async def test_list_dir_marks_directories(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    out = (await ListDirTool(tmp_path).run({"path": "."})).content
    assert "sub/" in out
    assert "a.txt" in out


async def test_list_dir_missing_path_is_not_ok(tmp_path: Path) -> None:
    result = await ListDirTool(tmp_path).run({"path": "nope"})
    assert not result.ok


# --- edit_file ---------------------------------------------------------------


async def test_edit_file_replaces_unique_string(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("hello world", encoding="utf-8")
    result = await EditFileTool(tmp_path).run({"path": "f.txt", "old_string": "world", "new_string": "there"})
    assert result.ok
    assert f.read_text(encoding="utf-8") == "hello there"


async def test_edit_file_rejects_ambiguous_match(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("a a a", encoding="utf-8")
    result = await EditFileTool(tmp_path).run({"path": "f.txt", "old_string": "a", "new_string": "b"})
    assert not result.ok
    assert result.metadata["error_type"] == "Ambiguous"
    assert f.read_text(encoding="utf-8") == "a a a"  # unchanged


async def test_edit_file_replace_all(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("a a a", encoding="utf-8")
    result = await EditFileTool(tmp_path).run(
        {"path": "f.txt", "old_string": "a", "new_string": "b", "replace_all": True}
    )
    assert result.ok
    assert f.read_text(encoding="utf-8") == "b b b"


async def test_edit_file_missing_string_is_not_ok(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("hello", encoding="utf-8")
    result = await EditFileTool(tmp_path).run({"path": "f.txt", "old_string": "absent", "new_string": "x"})
    assert not result.ok
    assert result.metadata["error_type"] == "NotFound"


async def test_edit_file_rejects_path_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes workspace"):
        await EditFileTool(tmp_path).run({"path": "../evil.txt", "old_string": "a", "new_string": "b"})


# --- search_text -------------------------------------------------------------


async def test_search_text_finds_substring_with_location(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("import os\nresult = compute()\n", encoding="utf-8")
    out = (await SearchTextTool(tmp_path).run({"pattern": "compute"})).content
    assert "code.py:2:" in out
    assert "compute()" in out


async def test_search_text_glob_and_regex(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("def foo(): docs\n", encoding="utf-8")
    out = (await SearchTextTool(tmp_path).run({"pattern": r"def \w+\(", "regex": True, "glob": "*.py"})).content
    assert "a.py:1:" in out
    assert "b.md" not in out  # excluded by glob


async def test_search_text_skips_ignored_dirs(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("needle", encoding="utf-8")
    (tmp_path / "keep.txt").write_text("needle", encoding="utf-8")
    out = (await SearchTextTool(tmp_path).run({"pattern": "needle"})).content
    assert "keep.txt:1:" in out
    assert ".git" not in out


async def test_search_text_no_matches(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("nothing here", encoding="utf-8")
    assert (await SearchTextTool(tmp_path).run({"pattern": "zzz"})).content == "No matches."


# --- search_text ripgrep backend (R3): probe, parity, degradation --------------


def _search_fixture(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("import os\nvalue = compute()\nCOMPUTE = 1\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("compute the answer\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("compute()\n", encoding="utf-8")


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
async def test_search_text_rg_and_python_backends_agree(tmp_path: Path, monkeypatch) -> None:
    _search_fixture(tmp_path)
    for args in (
        {"pattern": "compute"},
        {"pattern": "compute", "ignore_case": True},
        {"pattern": r"compute\(\)", "regex": True},
        {"pattern": "compute", "glob": "*.py"},
        {"pattern": "compute", "path": "pkg"},
    ):
        with_rg = (await SearchTextTool(tmp_path).run(dict(args))).content
        monkeypatch.setattr(shutil, "which", lambda name: None)
        pure = (await SearchTextTool(tmp_path).run(dict(args))).content
        monkeypatch.undo()
        assert sorted(with_rg.splitlines()) == sorted(pure.splitlines()), args


async def test_search_text_parses_rg_output_shape(tmp_path: Path, monkeypatch) -> None:
    # A stubbed rg (runs everywhere): output must be re-rendered into the exact
    # `relpath:line: text` shape of the pure-Python backend.
    _search_fixture(tmp_path)
    from agent_core.tools import builtin

    fake_stdout = f"pkg{os.sep}mod.py:2:value = compute()\nnotes.md:1:compute the answer\n"

    class _Proc:
        returncode = 0
        stdout = fake_stdout.encode("utf-8")
        stderr = b""

    monkeypatch.setattr(shutil, "which", lambda name: "C:/fake/rg.exe")
    monkeypatch.setattr(builtin.subprocess, "run", lambda *a, **k: _Proc())
    out = (await SearchTextTool(tmp_path).run({"pattern": "compute"})).content
    assert "pkg/mod.py:2: value = compute()" in out
    assert "notes.md:1: compute the answer" in out


async def test_search_text_falls_back_when_rg_errors(tmp_path: Path, monkeypatch, caplog) -> None:
    _search_fixture(tmp_path)
    from agent_core.tools import builtin

    class _Broken:
        returncode = 2
        stdout = b""
        stderr = b"rg: bad flag"

    monkeypatch.setattr(shutil, "which", lambda name: "C:/fake/rg.exe")
    monkeypatch.setattr(builtin.subprocess, "run", lambda *a, **k: _Broken())
    with caplog.at_level("DEBUG", logger="agent_core.tools.builtin"):
        out = (await SearchTextTool(tmp_path).run({"pattern": "compute"})).content
    # The pure-Python fallback still finds everything (and skips ignored dirs).
    assert "pkg/mod.py:2:" in out and "notes.md:1:" in out
    assert "node_modules" not in out
    assert any("falling back" in record.getMessage() for record in caplog.records)


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


async def test_run_command_shell_echo(tmp_path: Path) -> None:
    result = await RunCommandTool(tmp_path).run({"command": "echo hello"})
    assert "hello" in result.content


def test_run_command_is_dangerous() -> None:
    assert RunCommandTool().risk is ToolRisk.DANGEROUS


def test_shell_invocation_per_platform() -> None:
    spec, shell = _shell_invocation("Get-Content x")
    if sys.platform.startswith("win"):
        assert shell is False
        assert spec[0] == "powershell" and spec[-1].endswith("Get-Content x")
    else:
        assert shell is True
        assert spec == "Get-Content x"


async def test_run_command_child_python_emits_utf8(tmp_path: Path) -> None:
    # The GBK regression: a child printing non-ASCII must not crash and must
    # round-trip as UTF-8 (PYTHONUTF8/PYTHONIOENCODING are forced for the child).
    code = "print('\\u2705 完成')"
    result = await RunCommandTool(tmp_path).run({"command": f'{sys.executable} -c "{code}"'})
    assert result.ok
    assert "完成" in result.content


@pytest.mark.skipif(not sys.platform.startswith("win"), reason="PowerShell cmdlet test is Windows-only")
async def test_run_command_powershell_cmdlet_reads_utf8(tmp_path: Path) -> None:
    # Get-Content (a PowerShell cmdlet, absent in cmd.exe) must resolve AND read a
    # UTF-8 file's CJK correctly under Windows PowerShell 5.1.
    (tmp_path / "u.txt").write_text("你好世界\n", encoding="utf-8")
    result = await RunCommandTool(tmp_path).run({"command": "Get-Content u.txt"})
    assert result.ok
    assert "你好世界" in result.content


# --- read_text_file ----------------------------------------------------------


async def test_read_text_file_returns_small_file_whole(tmp_path: Path) -> None:
    (tmp_path / "s.txt").write_text("a\nb\nc\n", encoding="utf-8")
    result = await ReadTextFileTool(tmp_path).run({"path": "s.txt"})
    assert result.content == "a\nb\nc\n"  # returned verbatim, including trailing newline
    assert "file truncated" not in result.content


async def test_read_text_file_caps_large_file_with_note(tmp_path: Path) -> None:
    total = _DEFAULT_READ_LINES + 500
    (tmp_path / "big.txt").write_text("\n".join(str(i) for i in range(total)), encoding="utf-8")
    result = await ReadTextFileTool(tmp_path).run({"path": "big.txt"})
    body, _, note = result.content.rpartition("\n")
    assert len(body.splitlines()) == _DEFAULT_READ_LINES
    assert f"of {total} lines" in note
    assert f"offset={_DEFAULT_READ_LINES + 1}" in note
    assert result.metadata["total_lines"] == total


async def test_read_text_file_explicit_paging_is_exact(tmp_path: Path) -> None:
    (tmp_path / "p.txt").write_text("\n".join(f"L{i}" for i in range(100)), encoding="utf-8")
    result = await ReadTextFileTool(tmp_path).run({"path": "p.txt", "offset": 10, "limit": 3})
    assert result.content == "L9\nL10\nL11"  # offset is 1-based


# --- run_tests ---------------------------------------------------------------


async def test_run_tests_runs_pytest(tmp_path: Path) -> None:
    (tmp_path / "test_sample.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8")
    result = await RunTestsTool(tmp_path).run({"args": ["-q"]})
    assert result.ok
    assert "passed" in result.content


# --- git_diff ----------------------------------------------------------------


async def test_git_diff_reports_changes(tmp_path: Path) -> None:
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

    result = await GitDiffTool(tmp_path).run({})
    assert "-one" in result.content
    assert "+two" in result.content
