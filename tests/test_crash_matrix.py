"""In-process crash-injection matrix for the second-round audit §9.1.

Each test arms one durable boundary (``external_intent`` / ``external_outcome``
/ ``history_ready`` / transcript append / ``history_persisted``), lets a real
agent turn die on it, releases the journal the way a process crash would, and
then pins the recovery contract: external side effects are never replayed and
the recovered transcript never reports a false failure.
"""

from __future__ import annotations

import pytest

from agent_core.tools.transaction import JournalStorage

import crashkit
from crashkit import CrashPoint

pytestmark = pytest.mark.crash


@pytest.fixture
def crash_env(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("POLARIS_HOME", str(tmp_path / "private-state"))
    monkeypatch.chdir(workspace)
    storage = JournalStorage.user_state(workspace, crashkit.SESSION_ID, "run-recovery")
    return crashkit.CrashWorkspace(root=tmp_path, workspace=storage.workspace), storage


@pytest.mark.parametrize("point", list(CrashPoint), ids=lambda point: point.value)
async def test_crash_at_boundary_never_replays_or_lies(crash_env, point):
    env, storage = crash_env
    agent, _tool, batches = crashkit.build_crash_agent(env)
    injector = crashkit.CrashInjector(point).install()
    try:
        await crashkit.drive_to_crash(agent, injector, batches)
    finally:
        injector.restore()
        agent.logger.close()

    assert len(batches) == 1
    journal = batches[0].journal
    # Simulated crash: ownership is dropped without a terminal record.
    journal._release_for_later_recovery()

    before = crashkit.snapshot(env.workspace, env.transcript_path, journal.path)
    outcomes, again, _ = crashkit.run_recovery(
        storage, env.session_dir, env.workspace, env.session_id
    )
    after = crashkit.snapshot(env.workspace, env.transcript_path, journal.path)
    crashkit.assert_no_replay_and_truthful(
        point, before=before, outcomes=outcomes, outcomes_again=again, after=after
    )
