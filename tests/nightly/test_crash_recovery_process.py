"""Process-level crash matrix (audit §9.1): the child really dies at each boundary.

Every case launches ``crash_child.py`` as a separate interpreter that builds the
same one-tool agent and ``os._exit(1)`` immediately after the armed durable
boundary. The parent then runs real startup recovery over the same on-disk
state and applies the same truth assertions as the in-process matrix.
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


@pytest.mark.parametrize("point", list(CrashPoint), ids=lambda point: point.value)
def test_process_crash_recovery(point: CrashPoint, tmp_path, monkeypatch):
    polaris_home = tmp_path / "private-state"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_dir = tmp_path / "transcripts"
    completed = subprocess.run(
        [
            sys.executable,
            str(CHILD),
            point.value,
            str(polaris_home),
            str(workspace),
            str(session_dir),
        ],
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_SECONDS,
        env={
            **os.environ,
            "POLARIS_HOME": str(polaris_home),
            "AGENT_SANDBOX_ALLOW_UNATTENDED": "1",
        },
    )
    assert completed.returncode == 1, (
        f"child did not die at {point.value}: rc={completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )

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
