"""Import-boundary tests for the layered dependencies (revision guide D1/D9).

``import agent_core`` — and constructing a working NullUI agent — must succeed with
only the CORE dependencies installed (httpx + pyyaml): no rich/prompt_toolkit (the
[terminal] extra), no mcp (the [mcp] extra), no bs4/ddgs/markdownify (the [web]
extra). Each check runs in a subprocess with a meta_path blocker standing in for a
core-only environment, so the test works even though this dev environment has all
extras installed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Installed distributions of the optional extras, blocked at import time.
_BLOCKER = """
import sys

BLOCKED = {"rich", "prompt_toolkit", "mcp", "bs4", "ddgs", "markdownify"}


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {name!r} (blocked)", name=name)
        return None


sys.meta_path.insert(0, _Blocker())
"""


def _run_blocked(script: str, tmp_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER + script],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        env={"PYTHONPATH": str(_REPO_ROOT), "TMPDIR": str(tmp_path), "SYSTEMROOT": "C:\\Windows"}
        if sys.platform == "win32"
        else {"PYTHONPATH": str(_REPO_ROOT), "TMPDIR": str(tmp_path)},
    )


def test_import_agent_core_needs_only_core_deps(tmp_path: Path) -> None:
    result = _run_blocked(
        """
import agent_core

print("import-ok")
""",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "import-ok" in result.stdout


def test_agent_constructs_and_degrades_without_extras(tmp_path: Path) -> None:
    run_dir = str(tmp_path / "runs").replace("\\", "/")
    result = _run_blocked(
        f"""
import logging

logging.basicConfig(level=logging.WARNING)

from agent_core.providers.fake import FakeProvider
from agent_core.react import ReActAgent, ReActConfig
from agent_core.tools.catalog import default_tools

agent = ReActAgent(
    provider=FakeProvider(),
    config=ReActConfig(run_dir={run_dir!r}, session_dir=""),
)
names = {{tool.name for tool in default_tools()}}
assert "web_fetch" not in names and "web_search" not in names, names  # [web] skipped
assert "read_text_file" in names and "run_command" in names  # core tools intact
print("construct-ok")
""",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "construct-ok" in result.stdout
    # The degradation is observable, not silent (logged skip with the install hint).
    assert "web tools disabled" in result.stderr


def test_console_ui_gives_actionable_error_without_terminal_extra(tmp_path: Path) -> None:
    result = _run_blocked(
        """
from agent_core.ui import ConsoleUI

try:
    ConsoleUI()
except RuntimeError as exc:
    assert "[terminal]" in str(exc), exc
    print("actionable-ok")
""",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "actionable-ok" in result.stdout
