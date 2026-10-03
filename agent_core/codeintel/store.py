from __future__ import annotations

import json
from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
import time
import threading
from typing import Any, Iterator

from agent_core.codeintel.models import SCHEMA_VERSION
from agent_core.codeintel.snapshots import worktree_id


def default_store(root: Path) -> Path:
    # Resolve the state home once so the per-connection redirect guard compares
    # canonical paths even when POLARIS_HOME itself sits behind a symlink/junction.
    base = Path(os.environ.get("POLARIS_HOME", str(Path.home() / ".polaris"))).expanduser().resolve()
    return base / "codeintel" / worktree_id(root) / "code.sqlite3"


class CodeStore:
    """Short-lived SQLite connections; a single atomic file revision owns all facts."""

    def __init__(self, root: Path, path: Path | None = None) -> None:
        self.root = root
        self.path = path or default_store(root)
        self._local = threading.local()
        self._anchor: sqlite3.Connection | None = None

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        current = getattr(self._local, "connection", None)
        if current is not None:
            yield current
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink() or self.path.parent.resolve() != self.path.parent.absolute():
            raise ValueError("redirected code index path")
        connection = sqlite3.connect(self.path, timeout=0.5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS files(
                    id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, language TEXT NOT NULL,
                    module TEXT NOT NULL, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
                    hash TEXT NOT NULL DEFAULT '', revision INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL DEFAULT 'pending',
                    diagnostic TEXT NOT NULL DEFAULT '', seen INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS files_module ON files(module, path);
                CREATE INDEX IF NOT EXISTS files_language ON files(language, path);
                CREATE INDEX IF NOT EXISTS files_mtime ON files(mtime_ns DESC, path);
                CREATE TABLE IF NOT EXISTS symbols(
                    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                    name TEXT NOT NULL, qualified TEXT NOT NULL, kind TEXT NOT NULL,
                    line INTEGER NOT NULL, end_line INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS symbol_name ON symbols(name, file_id);
                CREATE INDEX IF NOT EXISTS symbol_qualified ON symbols(qualified, file_id);
                CREATE INDEX IF NOT EXISTS symbol_file ON symbols(file_id);
                CREATE TABLE IF NOT EXISTS edges(
                    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                    source TEXT NOT NULL, target TEXT NOT NULL, kind TEXT NOT NULL,
                    line INTEGER NOT NULL, precision TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS edge_target ON edges(target, kind, file_id);
                CREATE INDEX IF NOT EXISTS edge_source ON edges(source, kind, file_id);
                CREATE INDEX IF NOT EXISTS edge_file ON edges(file_id);
                CREATE TABLE IF NOT EXISTS pending(path TEXT PRIMARY KEY, reason TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, source TEXT NOT NULL, paths TEXT NOT NULL, at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS scan_dirs(path TEXT PRIMARY KEY);
            """)
            version = db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
            if version and version[0] != str(SCHEMA_VERSION):
                raise RuntimeError("unsupported code index schema; rebuild into a new index directory")
            db.execute("INSERT OR IGNORE INTO meta VALUES('schema', ?)", (str(SCHEMA_VERSION),))
            db.execute("INSERT OR IGNORE INTO meta VALUES('generation', '0')")
            db.execute("INSERT OR IGNORE INTO meta VALUES('catalog_complete', '0')")
            db.execute("INSERT OR IGNORE INTO meta VALUES('scan_epoch', '1')")
            db.execute("INSERT OR IGNORE INTO meta VALUES('last_reconcile', '0')")
            if not db.execute("SELECT 1 FROM meta WHERE key='initialized'").fetchone():
                db.execute("INSERT OR IGNORE INTO scan_dirs VALUES('.')")
                db.execute("INSERT INTO meta VALUES('initialized','1')")
            try:
                db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS text_fts USING fts5(text, tokenize='trigram case_sensitive 1')")
                db.execute("INSERT OR REPLACE INTO meta VALUES('fts', '1')")
            except sqlite3.OperationalError:
                db.execute("INSERT OR REPLACE INTO meta VALUES('fts', '0')")
            # Keep WAL alive between short reader connections. Otherwise the final
            # connection closing checkpoints on every indexed file on Windows.
            if self._anchor is None:
                self._anchor = sqlite3.connect(self.path, timeout=0.5, check_same_thread=False)

    @contextmanager
    def batch(self) -> Iterator[None]:
        with self.connect() as db:
            self._local.connection = db
            try:
                yield
            finally:
                self._local.connection = None

    def close(self) -> None:
        if self._anchor is not None:
            self._anchor.close()
            self._anchor = None

    @staticmethod
    def get(db: sqlite3.Connection, key: str) -> str:
        row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return str(row[0]) if row else "0"

    @staticmethod
    def bump(db: sqlite3.Connection) -> None:
        db.execute("UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='generation'")

    def enqueue(self, event_id: str, paths: tuple[str, ...], source: str) -> None:
        with self.connect() as db:
            inserted = db.execute("INSERT OR IGNORE INTO events VALUES(?,?,?,?)",
                                  (event_id, source, json.dumps(paths), time.time())).rowcount
            if inserted:
                db.executemany("INSERT OR REPLACE INTO pending VALUES(?,?)", ((p, f"{source}:{event_id}") for p in paths))
                # Dirty rows are never eligible for structural/text answers.
                db.executemany("UPDATE files SET state='pending' WHERE path=?", ((p,) for p in paths))
                self.bump(db)
            db.execute("DELETE FROM events WHERE at < ?", (time.time() - 7 * 86400,))

    def status(self) -> dict[str, Any]:
        with self.connect() as db:
            return {"schema_version": SCHEMA_VERSION, "workspace": str(self.root),
                    "worktree_id": worktree_id(self.root), "generation": int(self.get(db, "generation")),
                    "catalog_complete": self.get(db, "catalog_complete") == "1",
                    "last_reconcile": float(self.get(db, "last_reconcile")),
                    "fts": self.get(db, "fts") == "1",
                    "files": db.execute("SELECT count(*) FROM files").fetchone()[0],
                    "pending": db.execute("SELECT count(*) FROM pending").fetchone()[0],
                    "states": dict(db.execute("SELECT state,count(*) FROM files GROUP BY state")),
                    "modules": dict(db.execute("SELECT module,count(*) FROM files GROUP BY module")),
                    "structured_languages": ["python"], "reference_precision": "syntactic"}
