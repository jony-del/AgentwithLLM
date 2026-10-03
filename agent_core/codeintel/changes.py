"""Post-commit outbox shared by every policy-isolated index for a worktree."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time

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


def replay_commits(store: CodeStore, limit: int = 1000) -> None:
    path = default_store(store.root).parent / "changes.sqlite3"
    if not path.exists():
        return
    if path.is_symlink():
        raise ValueError("redirected code change outbox")
    with store.connect() as db:
        sequence = int(store.get(db, "commit_sequence"))
    with closing(sqlite3.connect(path, timeout=0.5)) as source:
        rows = source.execute("SELECT seq,event_id,paths,v FROM changes WHERE seq>? ORDER BY seq LIMIT ?",
                              (sequence, limit)).fetchall()
    for sequence, event_id, raw, version in rows:
        if version != 1:
            raise ValueError("unsupported code change schema")
        paths = json.loads(raw)
        if not isinstance(paths, list) or not all(isinstance(p, str) and not Path(p).is_absolute() and ".." not in Path(p).parts for p in paths):
            raise ValueError("invalid code change paths")
        store.enqueue(f"commit:{event_id}", tuple(paths), "transaction")
        with store.connect() as db:
            db.execute("INSERT OR REPLACE INTO meta VALUES('commit_sequence',?)", (str(sequence),))
