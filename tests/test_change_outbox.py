"""Per-path recoverable consumption of giant change-outbox events (S2)."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing

import pytest

from agent_core.codeintel.budget import BudgetExceeded, QueryBudget
from agent_core.codeintel.changes import (LEGACY_PAYLOAD_CAP, PATH_BATCH, UnsupportedChangePayload,
                                          record_commit, replay_commits)
from agent_core.codeintel.config import CodeIntelConfig
from agent_core.codeintel.store import CodeStore, default_store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("POLARIS_HOME", str(tmp_path / "home"))
    root = tmp_path / "repo"
    root.mkdir()
    store = CodeStore(root, default_store(root))
    store.initialize()
    yield store, root
    store.close()


def pending_paths(store):
    with store.connect() as db:
        return {row[0] for row in db.execute("SELECT path FROM pending")}


def cursor_state(store):
    with store.connect() as db:
        return (store.get(db, "commit_sequence"), store.get(db, "commit_partial"),
                store.get(db, "generation"))


def outbox(store):
    return default_store(store.root).parent / "changes.sqlite3"


def write_legacy_event(store, event_id, paths, *, version=1):
    with closing(sqlite3.connect(outbox(store), timeout=0.5)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS changes(seq INTEGER PRIMARY KEY,event_id TEXT UNIQUE,paths TEXT,at REAL,v INTEGER)")
        db.execute("INSERT INTO changes(event_id,paths,at,v) VALUES(?,?,?,?)",
                   (event_id, json.dumps(paths), 1.0, version))


def closing_connect(path):
    return closing(sqlite3.connect(path, timeout=0.5))


def drain(store, *, candidates=10000, limit=64):
    calls = 0
    while replay_commits(store, limit=limit, budget=QueryBudget(CodeIntelConfig(watch=False, max_candidates=candidates))):
        calls += 1
        assert calls < 10000, "replay did not converge"
    return calls


def test_giant_event_progresses_under_short_budget(store):
    store, root = store
    record_commit(root, "giant", [f"f{i:05d}.py" for i in range(5000)])
    seen = 0
    for _ in range(100):
        budget = QueryBudget(CodeIntelConfig(watch=False, max_candidates=300))
        more = replay_commits(store, budget=budget)
        now = len(pending_paths(store))
        assert now > seen  # Every call commits a bounded batch; no call is starved.
        seen = now
        if not more:
            break
    assert seen == 5000
    assert cursor_state(store)[0] == "1"
    with store.connect() as db:
        assert db.execute("SELECT 1 FROM events WHERE id='commit:giant'").fetchone()
    # The dedupe row exists only after full consumption; extra calls are no-ops.
    generation = cursor_state(store)[2]
    assert not replay_commits(store, budget=QueryBudget(CodeIntelConfig(watch=False)))
    assert cursor_state(store) == ("1", "0", generation)


def test_replay_resumes_at_exact_ordinal_after_batch_failure(store):
    store, root = store
    paths = [f"f{i:05d}.py" for i in range(1000)]
    record_commit(root, "giant", paths)
    failing = paths[PATH_BATCH * 2 + 88]
    with store.connect() as db:
        db.execute(f"""CREATE TRIGGER fail_path BEFORE INSERT ON pending
            WHEN NEW.path='{failing}' BEGIN SELECT RAISE(ABORT, 'injected'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        replay_commits(store, budget=QueryBudget(CodeIntelConfig(watch=False)))
    # Batches before the failing one persist; the failed batch rolled back atomically.
    assert len(pending_paths(store)) == PATH_BATCH * 2
    assert cursor_state(store)[:2] == ("0", f"1:{PATH_BATCH * 2}")
    with store.connect() as db:
        assert not db.execute("SELECT 1 FROM events").fetchone()
        db.execute("DROP TRIGGER fail_path")
    assert not replay_commits(store, budget=QueryBudget(CodeIntelConfig(watch=False)))
    assert pending_paths(store) == set(paths)  # No skipped, duplicated or lost paths.
    assert cursor_state(store)[:2] == ("1", "0")


def test_budget_exhaustion_before_any_progress_propagates(store):
    store, root = store
    record_commit(root, "giant", [f"f{i}.py" for i in range(10)])
    budget = QueryBudget(CodeIntelConfig(watch=False, max_candidates=1))
    budget.consume(candidates=1)
    with pytest.raises(BudgetExceeded):
        replay_commits(store, budget=budget)
    assert cursor_state(store)[:2] == ("0", "0")
    assert not pending_paths(store)


def test_legacy_giant_json_migrates_then_consumes_per_path(store):
    store, root = store
    paths = [f"legacy/{i:05d}.py" for i in range(1500)]
    write_legacy_event(store, "legacy-giant", paths)
    drain(store, candidates=300)
    assert pending_paths(store) == set(paths)
    with closing_connect(outbox(store)) as db:
        count = db.execute("SELECT path_count FROM change_migrated").fetchone()[0]
        assert count == 1500
        assert db.execute("SELECT value FROM outbox_meta WHERE key='format'").fetchone()[0] == "2"
    assert cursor_state(store)[:2] == ("1", "0")


def test_unsupported_event_version_blocks_without_skipping(store):
    store, root = store
    write_legacy_event(store, "future-format", ["a.py"], version=2)
    record_commit(root, "valid", ["b.py"])
    with pytest.raises(ValueError, match="unsupported code change schema"):
        replay_commits(store, budget=QueryBudget(CodeIntelConfig(watch=False)))
    # The unsupported event is neither advanced past nor skipped: nothing was consumed.
    assert cursor_state(store)[:2] == ("0", "0")
    assert not pending_paths(store)


def test_oversized_legacy_payload_reports_unsupported(store, monkeypatch):
    store, root = store
    monkeypatch.setattr("agent_core.codeintel.changes.LEGACY_PAYLOAD_CAP", 64)
    write_legacy_event(store, "huge-legacy", [f"f{i}.py" for i in range(10)])
    with pytest.raises(UnsupportedChangePayload, match="unsupported"):
        replay_commits(store, budget=QueryBudget(CodeIntelConfig(watch=False)))
    assert cursor_state(store)[:2] == ("0", "0")
    assert not pending_paths(store)


def test_oversized_cap_default_is_generous():
    assert LEGACY_PAYLOAD_CAP == 64 * 1024 * 1024


def test_duplicate_event_records_one_normalized_copy(store):
    store, root = store
    record_commit(root, "dup", ["a.py", "b.py"])
    record_commit(root, "dup", ["a.py", "b.py"])
    with closing_connect(outbox(store)) as db:
        assert db.execute("SELECT count(*) FROM change_paths").fetchone()[0] == 2
    drain(store)
    assert pending_paths(store) == {"a.py", "b.py"}
    generation = cursor_state(store)[2]
    assert not replay_commits(store, budget=QueryBudget(CodeIntelConfig(watch=False)))
    assert cursor_state(store)[2] == generation


def test_deleted_or_unknown_paths_are_still_invalidated(store):
    store, root = store
    record_commit(root, "deleted", ["gone/removed.py", "also/missing.txt"])
    assert not replay_commits(store, budget=QueryBudget(CodeIntelConfig(watch=False)))
    assert pending_paths(store) == {"gone/removed.py", "also/missing.txt"}


def test_concurrent_replay_never_loses_or_duplicates_paths(store):
    store, root = store
    paths = [f"c{i:04d}.py" for i in range(1200)]
    record_commit(root, "race", paths)
    errors = []

    def worker():
        for _ in range(200):
            try:
                if not replay_commits(store, budget=QueryBudget(CodeIntelConfig(watch=False, max_candidates=64))):
                    return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc):
                    errors.append(exc)
                    return
            except ValueError:
                errors.append("value-error")
                return
        errors.append("did not converge")

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert pending_paths(store) == set(paths)
    assert cursor_state(store)[:2] == ("1", "0")


def test_event_sequence_order_and_limit(store):
    store, root = store
    for i in range(5):
        record_commit(root, f"event-{i}", [f"f{i}.py"])
    assert replay_commits(store, limit=2, budget=QueryBudget(CodeIntelConfig(watch=False)))
    assert cursor_state(store)[0] == "2"
    drain(store)
    assert cursor_state(store)[0] == "5"
    assert pending_paths(store) == {f"f{i}.py" for i in range(5)}
