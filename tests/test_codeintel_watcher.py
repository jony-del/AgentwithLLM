import asyncio
from types import SimpleNamespace

import pytest

from agent_core.codeintel.watcher import WatchChanges, maintain, publish_batch


def test_watch_queue_filters_paths_and_requests_directory_reconciliation(tmp_path):
    changed = WatchChanges(tmp_path, ("build", "runs"))
    for raw in ("a.py", "pkg/memory/b.py", "src/build/c.py"):
        changed.event(str(tmp_path / raw), False)
    for raw in ("build/out.py", "runs/log.txt", ".env", "pkg/.venv/hidden.py", "../escape.py"):
        changed.event(str(tmp_path / raw), False)
    changed.event(str(tmp_path.parent / "foreign.py"), False)
    changed.event(str(tmp_path / "build"), True)
    assert not changed.rescan.is_set()
    assert [p for p, _ in changed.batch()] == ["a.py", "pkg/memory/b.py", "src/build/c.py"]
    changed.event(str(tmp_path / "pkg"), True)
    assert changed.rescan.is_set()


def test_watch_queue_bounds_batches_overflow_and_preserves_newer_events(tmp_path):
    changed = WatchChanges(tmp_path, (), capacity=3, batch_size=2)
    for index in range(4):
        changed.event(str(tmp_path / f"f{index}.py"), False)
    assert changed.rescan.is_set()
    batch = changed.batch()
    assert [p for p, _ in batch] == ["f0.py", "f1.py"]
    # Existing paths are updated even while full; old acknowledgements cannot
    # erase notifications that arrived while the publisher was awaiting IO.
    changed.event(str(tmp_path / "f0.py"), False)
    changed.acknowledge(batch)
    assert [p for p, _ in changed.batch()] == ["f0.py", "f2.py"]
    assert len(changed.changed) == 2


async def test_failed_publication_retries_without_erasing_concurrent_updates(tmp_path):
    changed = WatchChanges(tmp_path, ())
    changed.event(str(tmp_path / "a.py"), False)
    attempts = []
    class Service:
        async def publish_changes(self, change):
            attempts.append(change.paths)
            if len(attempts) == 1:
                changed.event(str(tmp_path / "a.py"), False)
                changed.event(str(tmp_path / "b.py"), False)
                raise OSError("writer unavailable")
    with pytest.raises(OSError, match="writer unavailable"):
        await publish_batch(Service(), changed)
    assert [p for p, _ in changed.batch()] == ["a.py", "b.py"]
    await publish_batch(Service(), changed)
    assert attempts == [("a.py",), ("a.py", "b.py")]
    assert not changed.batch()


async def test_cancelled_publication_does_not_acknowledge_pending_events(tmp_path):
    changed = WatchChanges(tmp_path, ())
    changed.event(str(tmp_path / "a.py"), False)
    pending = changed.batch()
    started = asyncio.Event()
    class Service:
        async def publish_changes(self, change):
            started.set()
            await asyncio.Future()
    task = asyncio.create_task(publish_batch(Service(), changed))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert changed.batch() == pending


async def test_maintenance_retries_failed_batches_and_reconciliation_with_bounded_rounds(tmp_path, monkeypatch):
    import agent_core.codeintel.watcher as watcher
    changed = WatchChanges(tmp_path, ())
    for index in range(600):
        changed.event(str(tmp_path / f"f{index}.py"), False)
    changed.rescan.set()
    monkeypatch.setattr(watcher, "WatchChanges", lambda *args: changed)
    class Service:
        workspace = tmp_path
        config = SimpleNamespace(watch=False, exclude_paths=(), reconcile_seconds=10**12)
        _closed = False
        _observer = None
        watch_state = "not_started"
        publications = []
        reconciliations = []
        async def publish_changes(self, change):
            self.publications.append(change.paths)
            if len(self.publications) == 1:
                raise OSError("writer busy")
        async def has_pending_work(self):
            return False
        async def ensure_index(self, **arguments):
            self.reconciliations.append(arguments["reconcile"])
            if len(self.reconciliations) == 1:
                raise OSError("reconciliation interrupted")
            return {"stop_reason": None}
    service = Service()
    rounds = 0
    async def tick(seconds):
        nonlocal rounds
        assert seconds == 1
        rounds += 1
        if rounds == 4:
            service._closed = True
    monkeypatch.setattr(watcher.asyncio, "sleep", tick)
    await maintain(service)
    assert [len(batch) for batch in service.publications] == [256, 256, 256, 88]
    assert service.publications[0] == service.publications[1]
    assert service.reconciliations == [True, True]
    assert not changed.batch()
    assert "degraded:" in service.watch_state
