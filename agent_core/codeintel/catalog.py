from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

from agent_core.codeintel.backends import language, module
from agent_core.codeintel.budget import BudgetExceeded, QueryBudget
from agent_core.codeintel.snapshots import IGNORED_DIRS, contained
from agent_core.codeintel.store import CodeStore
from agent_core.permission_safety import is_secret_path


class FileCatalog:
    """Durable directory queue with a bounded, resumable in-process scandir cursor.

    A crash restarts only the unfinished directory; upserts are idempotent.
    Enumeration is never part of a warm search.
    """

    def __init__(self, store: CodeStore) -> None:
        self.store = store
        self._iterator: Any = None
        self._directory: str | None = None

    def close(self) -> None:
        if self._iterator is not None:
            self._iterator.close()
            self._iterator = None

    def start_reconcile(self) -> None:
        with self.store.connect() as db:
            if db.execute("SELECT 1 FROM scan_dirs LIMIT 1").fetchone():
                return
            db.execute("UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='scan_epoch'")
            db.execute("UPDATE meta SET value='0' WHERE key='catalog_complete'")
            db.execute("INSERT OR REPLACE INTO meta VALUES('scan_error','0')")
            db.execute("INSERT OR IGNORE INTO scan_dirs VALUES('.')")
            self.store.bump(db)

    def advance(self, budget: QueryBudget) -> None:
        # One short transaction per slice, with a checkpoint even on budget exhaustion.
        with self.store.connect() as db:
            epoch = int(self.store.get(db, "scan_epoch"))
            if self.store.get(db, "catalog_complete") == "1" and self._iterator is None:
                return
            try:
                while True:
                    budget.check()
                    if self._iterator is None:
                        row = db.execute("SELECT path FROM scan_dirs ORDER BY path LIMIT 1").fetchone()
                        if row is None:
                            # Deletions are queued only after the entire inventory pass.
                            db.execute("INSERT OR IGNORE INTO pending SELECT path,'deleted' FROM files WHERE seen<>?", (epoch,))
                            db.execute("UPDATE files SET state='pending' WHERE seen<>?", (epoch,))
                            db.execute("UPDATE meta SET value='1' WHERE key='catalog_complete'")
                            db.execute("UPDATE meta SET value=? WHERE key='last_reconcile'", (str(time.time()),))
                            return
                        self._directory = str(row[0])
                        try:
                            directory = contained(self.store.root, self._directory)
                            self._iterator = os.scandir(directory)
                        except (OSError, ValueError):
                            db.execute("DELETE FROM scan_dirs WHERE path=?", (self._directory,))
                            db.execute("INSERT OR REPLACE INTO meta VALUES('scan_error',?)", (self._directory,))
                            continue
                    budget.consume(candidates=1)
                    try:
                        entry = next(self._iterator, None)
                    except OSError:
                        # The directory vanished mid-scan; the end-of-pass deletion
                        # sweep (seen<>epoch) covers its files.
                        self.close()
                        db.execute("DELETE FROM scan_dirs WHERE path=?", (self._directory,))
                        continue
                    if entry is None:
                        self.close()
                        db.execute("DELETE FROM scan_dirs WHERE path=?", (self._directory,))
                        continue
                    path = Path(entry.path)
                    if entry.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()) or is_secret_path(path):
                        continue
                    rel = path.relative_to(self.store.root).as_posix()
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name not in IGNORED_DIRS:
                                db.execute("INSERT OR IGNORE INTO scan_dirs VALUES(?)", (rel,))
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        stat = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    old = db.execute("SELECT size,mtime_ns FROM files WHERE path=?", (rel,)).fetchone()
                    changed = old is None or (old[0], old[1]) != (stat.st_size, stat.st_mtime_ns)
                    db.execute("""INSERT INTO files(path,language,module,size,mtime_ns,seen) VALUES(?,?,?,?,?,?)
                        ON CONFLICT(path) DO UPDATE SET size=excluded.size,mtime_ns=excluded.mtime_ns,seen=excluded.seen""",
                               (rel, language(rel), module(rel), stat.st_size, stat.st_mtime_ns, epoch))
                    if changed:
                        db.execute("INSERT OR REPLACE INTO pending VALUES(?,'catalog')", (rel,))
                        db.execute("UPDATE files SET state='pending' WHERE path=?", (rel,))
                        self.store.bump(db)
            except (BudgetExceeded, TimeoutError):
                # The iterator already advanced; commit all observations before yielding.
                return
            except BaseException:
                # The surrounding transaction rolls back. Restart this directory on
                # resumption so its advanced in-memory cursor cannot hide entries.
                self.close()
                raise
