"""Post-commit outbox shared by every policy-isolated index for a worktree."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time
from agent_core.codeintel.budget import QueryBudget

from agent_core.codeintel.store import CodeStore, default_store


def record_commit(workspace: Path, event_id: str, paths: list[str]) -> None:
    root = default_store(workspace).parent
    if not root.exists():
        return  # No index exists; its first inventory will observe these files.
    path = root / "changes.sqlite3"
    if path.is_symlink() or root.resolve() != root.absolute():
        raise ValueError("redirected code change outbox")
    with closing(sqlite3.connect(path, timeout=0.5)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS changes(seq INTEGER PRIMARY KEY,event_id TEXT UNIQUE,paths TEXT,at REAL,v INTEGER)")
        db.execute("INSERT OR IGNORE INTO changes(event_id,paths,at,v) VALUES(?,?,?,1)",
                   (event_id, json.dumps(paths), time.time()))


def replay_commits(store: CodeStore, limit: int = 64, *, budget: QueryBudget | None = None) -> bool:
    """Atomically replay a bounded prefix; True means more events remain.

    The sequence and invalidations commit together. A cancelled/invalid batch
    leaves both untouched, so the next call can safely retry the same prefix.
    """
    if budget is not None:
        budget.check()
        limit = min(limit, max(1, budget.config.max_candidates - budget.candidates))
    path = default_store(store.root).parent / "changes.sqlite3"
    if not path.exists():
        return False
    if path.is_symlink():
        raise ValueError("redirected code change outbox")
    with store.connect() as db:
        sequence = int(store.get(db, "commit_sequence"))
    with closing(sqlite3.connect(path, timeout=0.05)) as source:
        rows = source.execute("SELECT seq,event_id,paths,v FROM changes WHERE seq>? ORDER BY seq LIMIT ?",
                              (sequence, limit + 1)).fetchall()
    if not rows:
        return False
    events = []
    for seq, event_id, raw, version in rows[:limit]:
        if budget is not None:
            budget.consume(candidates=1)
        if version != 1:
            raise ValueError("unsupported code change schema")
        paths = json.loads(raw)
        if not isinstance(paths, list) or not all(isinstance(p, str) and not Path(p).is_absolute() and ".." not in Path(p).parts for p in paths):
            raise ValueError("invalid code change paths")
        events.append((seq, event_id, paths))
    with store.connect(timeout=0.05) as db:
        # Another query may have advanced the sequence since the source lookup.
        # Taking the SQLite write reservation before rereading avoids regression.
        db.execute("BEGIN IMMEDIATE")
        sequence = int(store.get(db, "commit_sequence"))
        for seq, event_id, paths in events:
            if budget is not None:
                budget.check()
            if seq <= sequence:
                continue
            inserted = db.execute("INSERT OR IGNORE INTO events VALUES(?,?,?,?)",
                (f"commit:{event_id}", "transaction", json.dumps(paths), time.time())).rowcount
            if inserted:
                reason = f"transaction:commit:{event_id}"
                def checked_paths():
                    for item in paths:
                        if budget is not None:
                            budget.check()
                        yield item
                db.executemany("INSERT OR REPLACE INTO pending VALUES(?,?)", ((p, reason) for p in checked_paths()))
                db.executemany("UPDATE files SET state='pending' WHERE path=?", ((p,) for p in checked_paths()))
                store.bump(db)
            sequence = seq
        if budget is not None:
            budget.check()
        db.execute("INSERT OR REPLACE INTO meta VALUES('commit_sequence',?)", (str(sequence),))
        db.execute("DELETE FROM events WHERE at < ?", (time.time() - 7 * 86400,))
    return len(rows) > limit
