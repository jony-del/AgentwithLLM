"""Offline CLI integration chain (audit phase 5): real subprocess, real disk.

One fake-provider run persists a resumable transcript; a second run resumes it
in the same project; a different project is refused (or forks on request).
Every invocation is a full ``python -m agent_core`` process against a temp
workspace and an isolated user-state directory — no real model, no network.

The sandbox uses its own short-lived mkdtemp root rather than pytest's tmp_path:
pytest nests tmp_path deeply enough that the transcript project slug plus the
``.write.lock`` sidecar can exceed the classic Windows MAX_PATH — a separate
path-length limitation this chain is not about.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

SESSION_LINE = re.compile(r"Session:\s+([0-9A-Za-z-]+)\s+\(resume with --resume")
CLI_TIMEOUT_SECONDS = 240


@pytest.fixture
def cli_root() -> Path:
    root = Path(tempfile.mkdtemp(prefix="cli-itg-"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _workspace(root: Path, name: str) -> tuple[Path, dict[str, str]]:
    workspace = root / name
    workspace.mkdir()
    state = root / "user-state"
    if not state.exists():
        # mode=0o700 gives the state root a clean owner-only DACL. A default
        # mkdir would inherit the temp directory's ACL, which on some hosts
        # carries write grants for foreign principals — correctly rejected by
        # the recovery-state security check the CLI runs at startup.
        state.mkdir(mode=0o700)
    env = {
        "POLARIS_HOME": str(state),
        "AGENT_TRUST_STORE": str(state / "trusted.json"),
        "AGENT_SESSION_DIR": str(root / "sessions"),
        "AGENT_SANDBOX_ALLOW_UNATTENDED": "1",
    }
    return workspace, env


def _run_cli(*args: str, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        [sys.executable, "-m", "agent_core", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=CLI_TIMEOUT_SECONDS,
        env={**os.environ, **env},
    )
    assert completed.returncode == 0, (
        f"CLI {' '.join(args[:2])} failed rc={completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return completed


def _session_id(stdout: str) -> str:
    match = SESSION_LINE.search(stdout)
    assert match, f"CLI did not report a resumable session id; stdout:\n{stdout}"
    return match.group(1)


def _transcript(root: Path, session_id: str) -> Path:
    matches = list((root / "sessions").rglob(f"{session_id}.jsonl"))
    assert len(matches) == 1, f"expected one transcript for {session_id}, got {matches}"
    return matches[0]


def _run_args() -> list[str]:
    return ["--provider", "fake", "--permission", "auto", "--no-memory"]


def test_fake_run_persists_and_resumes_in_same_project(cli_root: Path) -> None:
    workspace, env = _workspace(cli_root, "ws-a")

    first = _run_cli("run", "hello integration", *_run_args(), cwd=workspace, env=env)
    session_id = _session_id(first.stdout)
    transcript = _transcript(cli_root, session_id)
    before = transcript.read_text(encoding="utf-8")
    assert "Final answer" in first.stdout

    second = _run_cli(
        "run",
        "continue integration",
        *_run_args(),
        "--resume",
        session_id,
        cwd=workspace,
        env=env,
    )
    # Same project resume keeps the session (and the transcript keeps growing).
    assert _session_id(second.stdout) == session_id
    grown = _transcript(cli_root, session_id).read_text(encoding="utf-8")
    assert len(grown) > len(before)


def test_resume_rejects_other_project_and_forks_on_request(cli_root: Path) -> None:
    workspace, env = _workspace(cli_root, "ws-a")
    other = cli_root / "ws-b"
    other.mkdir()

    first = _run_cli("run", "owned by project a", *_run_args(), cwd=workspace, env=env)
    session_id = _session_id(first.stdout)
    transcript = _transcript(cli_root, session_id)
    before_fork = transcript.read_bytes()

    denied = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_core",
            "run",
            "intruder",
            *_run_args(),
            "--resume",
            session_id,
        ],
        cwd=str(other),
        capture_output=True,
        text=True,
        timeout=CLI_TIMEOUT_SECONDS,
        env={**os.environ, **env},
    )
    assert denied.returncode == 1, (
        f"cross-project resume must fail:\nstdout:\n{denied.stdout}\nstderr:\n{denied.stderr}"
    )
    assert "belongs to project" in denied.stderr

    forked = _run_cli(
        "run",
        "forked continuation",
        *_run_args(),
        "--resume",
        session_id,
        "--fork-session",
        cwd=other,
        env=env,
    )
    forked_id = _session_id(forked.stdout)
    assert forked_id != session_id
    # The fork leaves the source transcript byte-identical.
    assert transcript.read_bytes() == before_fork
