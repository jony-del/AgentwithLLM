"""Process-level crash matrix (audit §9.1): the child really dies at each boundary.

Every case launches ``crash_child.py`` as a separate interpreter that builds the
same one-tool agent and ``os._exit(1)`` immediately after the armed durable
boundary. The parent then runs real startup recovery over the same on-disk
state and applies the same truth assertions as the in-process matrix. The
compaction boundary row is the exception: no journal exists, so the parent pins
the last-boundary-wins resume truth against the expected chain the child
durably wrote BEFORE dying — never against a re-parse of the snapshot under
test. A wedged child is killed and reaped, and every failure message names the
injection point the child actually reached.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_core.tools.transaction import JournalStorage

import crashkit
from crashkit import CrashPoint

CHILD = Path(__file__).with_name("crash_child.py")
CHILD_TIMEOUT_SECONDS = 30


def _reached_point(stderr: str) -> str:
    """The injection point the child actually reached, for failure messages."""

    for line in (stderr or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("crash point reached:"):
            return stripped.removeprefix("crash point reached:").strip()
    return "<no injection point was reached>"


def _child_diagnostic(completed: subprocess.CompletedProcess, point: CrashPoint) -> str:
    return (
        f"child did not die at {point.value} (actually reached: "
        f"{_reached_point(completed.stderr)}): rc={completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def _run_child(
    point: CrashPoint, polaris_home: Path, workspace: Path, session_dir: Path
) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        str(CHILD),
        point.value,
        str(polaris_home),
        str(workspace),
        str(session_dir),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            **os.environ,
            "POLARIS_HOME": str(polaris_home),
            "AGENT_SANDBOX_ALLOW_UNATTENDED": "1",
        },
    )
    try:
        stdout, stderr = process.communicate(timeout=CHILD_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        # A timeout must terminate the child AND wait for the exit to be reaped,
        # exactly like the recovery contract expects of a supervised process.
        process.kill()
        try:
            stdout, stderr = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            stdout = stderr = ""
        raise AssertionError(
            f"child at {point.value} exceeded {CHILD_TIMEOUT_SECONDS}s and was killed; "
            f"injection point reached: {_reached_point(stderr)}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        ) from None
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


@pytest.mark.parametrize("point", crashkit.JOURNAL_BACKED_POINTS, ids=lambda point: point.value)
def test_process_crash_recovery(point: CrashPoint, tmp_path, monkeypatch):
    polaris_home = tmp_path / "private-state"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_dir = tmp_path / "transcripts"
    completed = _run_child(point, polaris_home, workspace, session_dir)
    assert completed.returncode == 1, _child_diagnostic(completed, point)

    monkeypatch.setenv("POLARIS_HOME", str(polaris_home))
    storage = JournalStorage.user_state(
        workspace, crashkit.SESSION_ID, "run-recovery"
    )
    journal_path = crashkit.find_single_journal(storage)
    env = crashkit.CrashWorkspace(root=tmp_path, workspace=storage.workspace)

    before = crashkit.snapshot(env.workspace, env.transcript_path, journal_path)
    outcomes, again, _ = crashkit.run_recovery(
        storage, env.session_dir, env.workspace, env.session_id
    )
    after = crashkit.snapshot(env.workspace, env.transcript_path, journal_path)
    crashkit.assert_no_replay_and_truthful(
        point, before=before, outcomes=outcomes, outcomes_again=again, after=after
    )


def test_process_crash_at_compaction_boundary(tmp_path):
    point = CrashPoint.COMPACTION_BOUNDARY_COMMITTED
    polaris_home = tmp_path / "private-state"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_dir = tmp_path / "transcripts"
    completed = _run_child(point, polaris_home, workspace, session_dir)
    assert completed.returncode == 1, _child_diagnostic(completed, point)

    env = crashkit.CrashWorkspace(root=tmp_path, workspace=workspace)
    # No journal and no side effects exist for this point: the expectation is the
    # chain the child durably recorded before dying, independent of the
    # transcript that recovery is about to be judged against.
    crashkit.assert_compaction_boundary_truthful(
        expected=crashkit.load_expected_chain(session_dir.parent / "expected_chain.json"),
        transcript_path=env.transcript_path,
    )
