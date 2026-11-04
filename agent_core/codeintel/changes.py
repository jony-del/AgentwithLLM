"""Post-commit outbox shared by every policy-isolated index for a worktree.

Outbox format v2 stores one normalized row per changed path, so replay consumes
a bounded number of paths per transaction and resumes mid-event after cancel,
deadline or failure. Legacy v1 rows (one JSON array per event) remain readable:
they are size-checked and streamed into the normalized table through SQLite
json_each (verified driver capability), never silently advanced or skipped.
"""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time
from typing import Any
from agent_core.codeintel.budget import BudgetExceeded, QueryBudget

from agent_core.codeintel.store import CodeStore, default_store

# Per-call bounds: events completed and paths written per store transaction.
PATH_BATCH = 256
# Legacy single-JSON payloads above this size are reported unsupported instead of
# being parsed; the normalized format makes such events impossible going forward.
LEGACY_PAYLOAD_CAP = 64 * 1024 * 1024
OUTBOX_FORMAT = 2


class UnsupportedChangePayload(ValueError):
    """A legacy event that cannot be safely split; replay neither advances nor skips it."""


def _valid_relative_path(value: Any) -> bool:
    return isinstance(value, str) and not Path(value).is_absolute() and ".." not in Path(value).parts


def _parse_cursor(value: str) -> tuple[int, int] | None:
    if not value or value == "0":
        return None
    parts = value.split(":")
    if len(parts) != 2 or any(not p.isdigit() for p in parts):
        raise ValueError("corrupt commit replay cursor")
    parsed = (int(parts[0]), int(parts[1]))
    return parsed if parsed[0] > 0 else None


def _outbox_path(store: CodeStore) -> Path:
    return default_store(store.root).parent / "changes.sqlite3"


def _ensure_schema(db: sqlite3.Connection) -> None:
    db.execute("CREATE TABLE IF NOT EXISTS changes(seq INTEGER PRIMARY KEY,event_id TEXT UNIQUE,paths TEXT,at REAL,v INTEGER)")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS change_paths(
            seq INTEGER NOT NULL, ordinal INTEGER NOT NULL, path TEXT NOT NULL,
            PRIMARY KEY(seq, ordinal)) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS change_migrated(seq INTEGER PRIMARY KEY, path_count INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS outbox_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    row = db.execute("SELECT value FROM outbox_meta WHERE key='format'").fetchone()
    if row is not None and (not row[0].isdigit() or int(row[0]) > OUTBOX_FORMAT):
        raise ValueError("unsupported code change schema")


def record_commit(workspace: Path, event_id: str, paths: list[str]) -> None:
    root = default_store(workspace).parent
    if not root.exists():
        return  # No index exists; its first inventory will observe these files.
    path = root / "changes.sqlite3"
    if path.is_symlink() or root.resolve() != root.absolute():
        raise ValueError("redirected code change outbox")
    with closing(sqlite3.connect(path, timeout=0.5)) as db, db:
        _ensure_schema(db)
        # The legacy single-JSON row stays for readers predating outbox format 2;
        # normalized rows are authoritative for new consumers.
        inserted = db.execute("INSERT OR IGNORE INTO changes(event_id,paths,at,v) VALUES(?,?,?,1)",
                              (event_id, json.dumps(paths), time.time())).rowcount
        if inserted:
            seq = db.execute("SELECT seq FROM changes WHERE event_id=?", (event_id,)).fetchone()[0]
            db.executemany("INSERT INTO change_paths VALUES(?,?,?)",
                           ((seq, ordinal, p) for ordinal, p in enumerate(paths)))
            db.execute("INSERT INTO change_migrated VALUES(?,?)", (seq, len(paths)))
            db.execute("INSERT OR REPLACE INTO outbox_meta VALUES('format',?)", (str(OUTBOX_FORMAT),))


def _migrate_legacy(source: sqlite3.Connection, seq: int, raw: Any, budget: QueryBudget | None) -> int:
    """Split a legacy v1 JSON event into normalized path rows; one bounded C-speed pass."""
    if not isinstance(raw, str):
        raise ValueError("invalid code change paths")
    size = len(raw.encode("utf-8"))
    if size > LEGACY_PAYLOAD_CAP:
        raise UnsupportedChangePayload(
            f"legacy change event {seq} payload {size} bytes exceeds {LEGACY_PAYLOAD_CAP}; unsupported")
    if budget is not None:
        budget.check()
    kind = source.execute("SELECT json_type(?)", (raw,)).fetchone()[0]
    if kind != "array":
        raise ValueError("invalid code change paths")
    count = 0
    batch: list[tuple[int, int, str]] = []
    # json_each streams array elements; inserts land in the same transaction, so an
    # interruption rolls the migration back and the next call retries it from scratch.
    with source:
        cursor = source.execute(
            "SELECT CAST(key AS INTEGER), value FROM json_each(?) ORDER BY CAST(key AS INTEGER)", (raw,))
        for ordinal, value in cursor:
            if not _valid_relative_path(value):
                raise ValueError("invalid code change paths")
            batch.append((seq, ordinal, value))
            count += 1
            if len(batch) >= 1000:
                source.executemany("INSERT INTO change_paths VALUES(?,?,?)", batch)
                batch.clear()
        if batch:
            source.executemany("INSERT INTO change_paths VALUES(?,?,?)", batch)
        source.execute("INSERT INTO change_migrated VALUES(?,?)", (seq, count))
        source.execute("INSERT OR REPLACE INTO outbox_meta VALUES('format',?)", (str(OUTBOX_FORMAT),))
    return count


class _ConcurrentReplay(RuntimeError):
    """Another replay advanced the cursor between the read and the write transaction."""


def _remaining(store: CodeStore) -> tuple[int, tuple[int, int] | None]:
    with store.connect() as db:
        return int(store.get(db, "commit_sequence")), _parse_cursor(store.get(db, "commit_partial"))


def replay_commits(store: CodeStore, limit: int = 64, *, budget: QueryBudget | None = None) -> bool:
    """Replay a bounded prefix; True means uncommitted events or paths remain.

    Each store transaction re-reads the consumption cursor, writes one bounded path
    batch and updates the cursor atomically. Cancellation, deadline or failure rolls
    back only the in-flight batch; completed batches persist and the next call resumes
    at the recorded ordinal without skipping or repeating paths. An event is marked
    consumed (its dedupe row) only after every one of its paths has been replayed.
    """
    if budget is not None:
        budget.check()
    path = _outbox_path(store)
    if not path.exists():
        return False
    if path.is_symlink():
        raise ValueError("redirected code change outbox")
    events_done = 0
    progressed = False
    with closing(sqlite3.connect(path, timeout=0.05)) as source:
        with source:
            _ensure_schema(source)
        while events_done < limit:
            if budget is not None:
                budget.check()
            sequence, partial = _remaining(store)
            if partial is not None:
                row = source.execute(
                    "SELECT c.event_id, m.path_count FROM changes c "
                    "JOIN change_migrated m ON m.seq=c.seq WHERE c.seq=?", (partial[0],)).fetchone()
                if row is None:
                    raise ValueError("commit replay cursor references an unknown event")
                seq, event_id, start = partial[0], row[0], partial[1]
                expected_partial: tuple[int, int] | None = partial
            else:
                row = source.execute(
                    "SELECT c.seq, c.event_id, c.paths, c.v, m.path_count FROM changes c "
                    "LEFT JOIN change_migrated m ON m.seq=c.seq WHERE c.seq>? ORDER BY c.seq LIMIT 1",
                    (sequence,)).fetchone()
                if row is None:
                    return False
                seq, event_id, raw, version, total = row
                if total is None:
                    if version != 1:
                        raise ValueError("unsupported code change schema")
                    _migrate_legacy(source, seq, raw, budget)
                start = 0
                expected_partial = None
            while True:  # Bounded path batches of the current event.
                take = PATH_BATCH
                if budget is not None:
                    take = min(take, budget.config.max_candidates - budget.candidates)
                    if take <= 0:
                        if progressed:
                            return True  # Committed batches persist; the next call resumes.
                        raise BudgetExceeded("max_candidates")
                rows = source.execute(
                    "SELECT ordinal, path FROM change_paths WHERE seq=? AND ordinal>=? "
                    "ORDER BY ordinal LIMIT ?", (seq, start, take)).fetchall()
                complete = len(rows) < take
                try:
                    with store.connect(timeout=0.05) as db:
                        # Take the write reservation before re-reading the cursor so a
                        # concurrent replay cannot regress or double-consume the batch.
                        db.execute("BEGIN IMMEDIATE")
                        current_sequence = int(store.get(db, "commit_sequence"))
                        current_partial = _parse_cursor(store.get(db, "commit_partial"))
                        if current_sequence != sequence or current_partial != expected_partial:
                            raise _ConcurrentReplay
                        for _, changed in rows:
                            if not _valid_relative_path(changed):
                                raise ValueError("invalid code change paths")
                            if budget is not None:
                                budget.consume(candidates=1)
                            reason = f"transaction:commit:{event_id}"
                            db.execute("INSERT OR REPLACE INTO pending VALUES(?,?)", (changed, reason))
                            db.execute("UPDATE files SET state='pending' WHERE path=?", (changed,))
                        if rows:
                            store.bump(db)
                            progressed = True
                        if complete:
                            inserted = db.execute("INSERT OR IGNORE INTO events VALUES(?,?,?,?)",
                                (f"commit:{event_id}", "transaction",
                                 json.dumps({"paths": len(rows) + start}), time.time())).rowcount
                            if inserted:
                                store.bump(db)
                            db.execute("INSERT OR REPLACE INTO meta VALUES('commit_sequence',?)", (str(seq),))
                            db.execute("INSERT OR REPLACE INTO meta VALUES('commit_partial','0')")
                            db.execute("DELETE FROM events WHERE at < ?", (time.time() - 7 * 86400,))
                        else:
                            db.execute("INSERT OR REPLACE INTO meta VALUES('commit_partial',?)",
                                       (f"{seq}:{rows[-1][0] + 1}",))
                except _ConcurrentReplay:
                    break  # Re-read the cursor below; another replay made progress.
                if complete:
                    events_done += 1
                    break
                start = rows[-1][0] + 1
                expected_partial = (seq, start)
            if not complete:
                continue  # Concurrent replay moved the cursor; recompute from the top.
        sequence, partial = _remaining(store)
        if partial is not None:
            return True
        return source.execute("SELECT 1 FROM changes WHERE seq>? LIMIT 1", (sequence,)).fetchone() is not None
