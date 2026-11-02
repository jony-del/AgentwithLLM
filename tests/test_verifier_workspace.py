import asyncio
from dataclasses import replace
import os
from types import SimpleNamespace

import pytest

from agent_core.permission_rules import RuleSet
from agent_core.task_runtime import capture_revision
from agent_core.verifier import _copy_source, VerifierConfig
from agent_core.verifier_workspace import VerifierWorkspace


def setup(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("answer = 42\n")
    # Synthetic ID enables the reuse path; the copy function still checks real bytes.
    revision = replace(capture_revision(root), snapshot_id="test-mode-manifest")
    session = SimpleNamespace(workspace=root, task_run=SimpleNamespace(id="task"), code_permission_rules=None)
    return session, revision


async def test_clean_reuse_dirty_extra_file_task_and_policy_invalidation(tmp_path):
    session, revision = setup(tmp_path)
    pool = VerifierWorkspace()
    try:
        async with pool.lease(session, revision, _copy_source) as (first, metrics):
            assert not metrics["reused"]
        async with pool.lease(session, revision, _copy_source) as (same, metrics):
            assert same == first and metrics["files_reused"] == 1
            (same / "a.py").write_text("modified\n")
        async with pool.lease(session, revision, _copy_source) as (fresh, metrics):
            assert fresh != first and not metrics["reused"]
            assert (fresh / "a.py").read_text() == "answer = 42\n"
            (fresh / "cache").mkdir()
        async with pool.lease(session, revision, _copy_source) as (clean, metrics):
            assert not metrics["reused"] and not (clean / "cache").exists()
        session.task_run.id = "other"
        async with pool.lease(session, revision, _copy_source) as (_, metrics):
            assert not metrics["reused"]
        session.code_permission_rules = lambda: RuleSet.from_lists(deny=["read_text_file(a.py)"])
        with pytest.raises(PermissionError):
            async with pool.lease(session, revision, _copy_source):
                pytest.fail("denied copy leased")
    finally:
        saved = pool.path
        await pool.close()
        assert saved is None or not saved.exists()


async def test_hardlink_is_never_reused_or_followed_for_writes(tmp_path):
    session, revision = setup(tmp_path)
    external = tmp_path / "external.py"
    external.write_text("answer = 42\n")
    pool = VerifierWorkspace()
    try:
        async with pool.lease(session, revision, _copy_source) as (copy, _):
            (copy / "a.py").unlink()
            os.link(external, copy / "a.py")
        async with pool.lease(session, revision, _copy_source) as (_, metrics):
            assert not metrics["reused"]
        assert external.read_text() == "answer = 42\n"
    finally:
        await pool.close()


async def test_exclusive_lease_and_failure_cleanup(tmp_path):
    session, revision = setup(tmp_path)
    pool = VerifierWorkspace()
    entered = asyncio.Event()
    async def second():
        async with pool.lease(session, revision, _copy_source):
            entered.set()
    async with pool.lease(session, revision, _copy_source):
        waiter = asyncio.create_task(second())
        await asyncio.sleep(0)
        assert not entered.is_set()
    await waiter
    with pytest.raises(RuntimeError):
        async with pool.lease(session, revision, _copy_source) as (copy, _):
            raise RuntimeError("probe failed")
    assert not copy.exists()
    await pool.close()


def test_reuse_config_is_explicit_boolean():
    assert not VerifierConfig().reuse_workspace
    assert VerifierConfig.from_dict({"reuse_workspace": True}).reuse_workspace
    with pytest.raises(ValueError, match="boolean"):
        VerifierConfig.from_dict({"reuse_workspace": "true"})


async def test_disabled_reuse_keeps_transient_copy_without_cache_inventory(tmp_path, monkeypatch):
    from agent_core.verifier_workspace import verification_copy
    session, revision = setup(tmp_path)
    def forbidden(*args):
        pytest.fail("disabled reuse scanned cache metadata")
    monkeypatch.setattr(VerifierWorkspace, "_metadata", forbidden)
    async with verification_copy(session, revision, _copy_source, reuse=False) as (copy, metrics):
        assert not metrics["reused"] and copy.exists()
    assert not copy.exists() and not hasattr(session, "verifier_workspace")


async def test_real_snapshot_reuse_and_changed_revision_rebuild(tmp_path):
    from tests.test_source_snapshots import setup as snapshot_setup
    root, _, checkpoints, tracker = snapshot_setup(tmp_path)
    current = await tracker.capture()
    session = SimpleNamespace(workspace=root, checkpoint_store=checkpoints,
                              task_run=SimpleNamespace(id="task"), code_permission_rules=None)
    pool = VerifierWorkspace()
    try:
        async with pool.lease(session, current, _copy_source) as (first, _):
            assert (first / "a.py").read_bytes() == b"value = 1\r\n"
        async with pool.lease(session, current, _copy_source) as (copy, metrics):
            assert metrics["reused"] and copy == first
        (root / "a.py").write_bytes(b"value = 2\r\n")
        changed = await tracker.capture()
        async with pool.lease(session, changed, _copy_source) as (copy, metrics):
            assert not metrics["reused"] and copy != first
            assert (copy / "a.py").read_bytes() == b"value = 2\r\n"
    finally:
        await pool.close()


async def test_cancelled_copy_worker_is_drained_before_cleanup(tmp_path):
    import threading
    session, current = setup(tmp_path)
    pool = VerifierWorkspace()
    started, finish = threading.Event(), threading.Event()
    destinations = []
    def slow_copy(session, revision, destination):
        destinations.append(destination)
        started.set()
        assert finish.wait(5)
        _copy_source(session, revision, destination)
    async def run():
        async with pool.lease(session, current, slow_copy):
            pytest.fail("cancelled lease yielded")
    task = asyncio.create_task(run())
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert destinations[0].exists() and not task.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not destinations[0].exists()
    await pool.close()


async def test_behavior_verifier_reuses_copy_but_runs_fresh_probes(tmp_path):
    from agent_core.models import LLMResult
    from agent_core.providers.base import ProviderConfig
    from agent_core.tools.verifier import probe_executor
    from agent_core.verifier import verify_behavior
    from tests.test_verifier import setup_agent, close, probes, PASS
    batches = [probes(), probes()]
    for batch in batches:
        for call in batch:
            call.arguments["argv"].insert(1, "-B")  # Keep this test's copy clean.
    agent, _ = setup_agent(tmp_path, [LLMResult("", tool_calls=batches[0]), LLMResult(PASS),
                                    LLMResult("", tool_calls=batches[1]), LLMResult(PASS)])
    try:
        revision = await agent.session.capture_revision()
        assert revision.snapshot_id
        records = []
        for _ in range(2):
            records.append(await verify_behavior(agent.session, agent.provider, ProviderConfig(),
                VerifierConfig(reuse_workspace=True), revision,
                lambda copy, rev, output: probe_executor(agent, copy, rev, output)))
        assert all(r["status"] == "passed" for r in records), records
        assert not records[0]["workspace_copy"]["reused"] and records[1]["workspace_copy"]["reused"]
        assert len(records[0]["probes"]) == len(records[1]["probes"]) == 2
        assert {p["id"] for p in records[0]["probes"]}.isdisjoint(p["id"] for p in records[1]["probes"])
        copy = agent.session.verifier_workspace.path
        assert copy.exists()
    finally:
        await close(agent)
    assert not copy.exists()
