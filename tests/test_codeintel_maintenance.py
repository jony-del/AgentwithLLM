"""Maintenance must yield, retain fresh evidence and drain workers on shutdown."""
import asyncio
from dataclasses import replace
import threading

import pytest

from agent_core.codeintel.budget import QueryBudget
from agent_core.codeintel.config import CodeIntelConfig
from agent_core.codeintel.models import SearchRequest
from agent_core.codeintel.service import CodeIntelligenceService
from tests.test_codeintel import built


async def test_unchanged_audit_hashes_without_reextracting_or_revising(tmp_path, monkeypatch):
    import agent_core.codeintel.indexer as indexer
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def target(): return 1\n")
    service = await built(tmp_path)
    try:
        before = (await service.search(SearchRequest("target", kind="symbol"))).hits[0]
        def forbidden(*args):
            pytest.fail("unchanged audit reparsed source")
        monkeypatch.setattr(indexer, "python_structure", forbidden)
        status = await service.ensure_index(audit=True)
        after = (await service.search(SearchRequest("target", kind="symbol"))).hits[0]
        assert status["usage"]["bytes"] > 0 and status["maintenance"]["unchanged_files"] == 1
        assert before.version == after.version
        assert not status["pending"]
    finally:
        await service.close()


async def test_maintenance_file_cap_progresses_across_calls(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for i in range(9):
        (root / f"f{i}.py").write_text(f"def target_{i}(): return {i}\n")
    service = CodeIntelligenceService(root, CodeIntelConfig(watch=False, maintenance_files=2),
                                      database=tmp_path / "index.db")
    try:
        statuses = []
        for _ in range(50):
            statuses.append(await service.ensure_index())
            if statuses[-1]["catalog_complete"] and not statuses[-1]["pending"]:
                break
        assert len(statuses) > 1 and not statuses[-1]["pending"]
        assert all(s["maintenance"]["processed_files"] <= 2 for s in statuses)
        assert (await service.search(SearchRequest("target_8", kind="symbol"))).hits
    finally:
        await service.close()


async def test_cancelled_writer_waiter_exits_before_lock_owner_releases(tmp_path):
    from agent_core.codeintel.models import ChangeSet
    service = await built(tmp_path)
    service._lock.acquire()
    task = asyncio.create_task(service.publish_changes(ChangeSet("wait", ("a.py",))))
    try:
        await asyncio.sleep(.03)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, .5)
    finally:
        service._lock.release()
        await asyncio.gather(task, return_exceptions=True)
        await service.close()


async def test_close_drains_active_workers_and_rejects_waiters(tmp_path, monkeypatch):
    service = await built(tmp_path, max_concurrent_requests=1)
    entered, finish = threading.Event(), threading.Event()
    closed = []
    original = service.store.close
    monkeypatch.setattr(service.store, "close", lambda: (closed.append(True), original()))
    def work():
        entered.set()
        assert finish.wait(5)
        return "done"
    first = asyncio.create_task(service._run(work, QueryBudget(service.config)))
    assert await asyncio.to_thread(entered.wait, 2)
    second = asyncio.create_task(service._run(lambda: "unexpected", QueryBudget(service.config)))
    closing = asyncio.create_task(service.close())
    try:
        await asyncio.sleep(.03)
        assert not closed and not closing.done()
        finish.set()
        assert await first == "done"
        with pytest.raises(RuntimeError, match="closed"):
            await second
        await closing
        assert closed
    finally:
        finish.set()
        await asyncio.gather(first, second, closing, return_exceptions=True)


async def test_admission_observes_scope_cancellation_before_worker_submission(tmp_path):
    from agent_core.execution import ExecutionScope
    service = await built(tmp_path, max_concurrent_requests=1)
    await service._admission.acquire()
    scope = ExecutionScope.for_workspace(service.workspace)
    def forbidden():
        pytest.fail("cancelled queued request submitted IO")
    task = asyncio.create_task(service._run(forbidden, QueryBudget(service.config, scope)))
    try:
        await asyncio.sleep(.03)
        scope.cancellation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, .5)
    finally:
        service._admission.release()
        await service.close()
        await scope.close()


async def test_writer_file_lock_wait_is_limited_by_maintenance_budget(tmp_path):
    from agent_core.file_lock import FileLock
    service = await built(tmp_path, maintenance_seconds=.1)
    try:
        with FileLock(service.store.path.with_suffix(".writer.lock")):
            status = await asyncio.wait_for(service.ensure_index(), .8)
            assert status["stop_reason"] == "deadline"
    finally:
        await service.close()


async def test_slow_file_finishes_without_repeated_slice_starvation(tmp_path, monkeypatch):
    import time
    import agent_core.codeintel.indexer as indexer
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def slow(): return 1\n")
    parse = indexer.python_structure
    calls = []
    def slow(text):
        calls.append(True)
        time.sleep(.08)
        return parse(text)
    monkeypatch.setattr(indexer, "python_structure", slow)
    service = CodeIntelligenceService(root, CodeIntelConfig(watch=False, maintenance_seconds=.03,
                                                           query_seconds=2), database=tmp_path / "index.db")
    try:
        for _ in range(30):
            status = await service.ensure_index()
            if status["catalog_complete"] and not status["pending"]:
                break
        assert not status["pending"] and len(calls) == 1
        assert (await service.search(SearchRequest("slow", kind="symbol"))).hits
    finally:
        await service.close()


async def test_repeated_cancel_and_cancelled_close_do_not_detach_worker(tmp_path, monkeypatch):
    service = await built(tmp_path)
    entered, finish = threading.Event(), threading.Event()
    closed = []
    original = service.store.close
    monkeypatch.setattr(service.store, "close", lambda: (closed.append(True), original()))
    def worker():
        entered.set()
        assert finish.wait(5)
    running = asyncio.create_task(service._run(worker, QueryBudget(service.config)))
    assert await asyncio.to_thread(entered.wait, 2)
    running.cancel()
    await asyncio.sleep(.01)
    running.cancel()
    closing = asyncio.create_task(service.close())
    await asyncio.sleep(.01)
    closing.cancel()
    try:
        await asyncio.sleep(.02)
        assert service._active and not closed and not running.done() and not closing.done()
    finally:
        finish.set()
    for task in (running, closing):
        with pytest.raises(asyncio.CancelledError):
            await task
    assert closed and not service._active


async def test_light_maintenance_avoids_full_status_statistics(tmp_path, monkeypatch):
    service = await built(tmp_path)
    try:
        monkeypatch.setattr(service.store, "status", lambda: pytest.fail("full statistics in light slice"))
        status = await service.ensure_index(detailed=False)
        assert status["catalog_complete"] and not status["pending"]
    finally:
        await service.close()


def test_maintenance_config_requires_positive_limits():
    config = CodeIntelConfig.from_dict({"maintenance_seconds": .2, "maintenance_files": 16,
                                        "max_concurrent_requests": 2})
    assert replace(config).maintenance_files == 16
    for key in ("maintenance_seconds", "maintenance_files", "max_concurrent_requests"):
        with pytest.raises(ValueError):
            CodeIntelConfig.from_dict({key: 0})


async def test_bounded_replay_advances_sequence_atomically_and_reports_backlog(tmp_path, monkeypatch):
    from agent_core.codeintel.changes import record_commit, replay_commits
    from agent_core.codeintel.store import default_store
    monkeypatch.setenv("POLARIS_HOME", str(tmp_path / "home"))
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def target(): return 1\n")
    service = CodeIntelligenceService(root, CodeIntelConfig(watch=False), database=default_store(root))
    try:
        await service.ensure_index()
        for i in range(7):
            record_commit(root, f"event-{i}", ["a.py"])
        assert replay_commits(service.store, limit=3, budget=QueryBudget(service.config))
        with service.store.connect() as db:
            assert int(service.store.get(db, "commit_sequence")) == 3
            assert db.execute("SELECT reason FROM pending WHERE path='a.py'").fetchone()[0].endswith("event-2")
        page = await service.search(SearchRequest("target", kind="symbol"))
        assert not page.hits and "pending_updates" in page.coverage.reasons
        await service.ensure_index()
        assert (await service.search(SearchRequest("target", kind="symbol"))).hits
    finally:
        await service.close()


async def test_replay_batch_failure_rolls_back_invalidations_and_sequence(tmp_path, monkeypatch):
    import sqlite3
    from agent_core.codeintel.changes import record_commit, replay_commits
    from agent_core.codeintel.store import default_store
    monkeypatch.setenv("POLARIS_HOME", str(tmp_path / "home"))
    root = tmp_path / "repo"
    root.mkdir()
    service = CodeIntelligenceService(root, CodeIntelConfig(watch=False), database=default_store(root))
    try:
        await service.ensure_index()
        record_commit(root, "first", ["a.py"])
        record_commit(root, "second", ["bad.py"])
        with service.store.connect() as db:
            db.execute("""CREATE TRIGGER fail_path BEFORE INSERT ON pending
                WHEN NEW.path='bad.py' BEGIN SELECT RAISE(ABORT, 'injected'); END""")
        with pytest.raises(sqlite3.IntegrityError, match="injected"):
            replay_commits(service.store, budget=QueryBudget(service.config))
        with service.store.connect() as db:
            assert service.store.get(db, "commit_sequence") == "0"
            assert not db.execute("SELECT 1 FROM pending").fetchone()
            assert not db.execute("SELECT 1 FROM events").fetchone()
    finally:
        await service.close()
