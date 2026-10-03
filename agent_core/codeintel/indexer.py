from __future__ import annotations

import logging
from typing import Callable, Any

from agent_core.codeintel.backends import language, module, python_structure, manifest_dependencies
from agent_core.codeintel.budget import BudgetExceeded, QueryBudget
from agent_core.codeintel.models import StaleEvidence
from agent_core.codeintel.snapshots import BudgetError, contained, read_snapshot
from agent_core.codeintel.store import CodeStore

logger = logging.getLogger(__name__)


def update_pending(store: CodeStore, budget: QueryBudget, allowed: Callable[[str], bool] | None = None,
                   *, chunk: int | None = None) -> bool:
    """Drain the pending queue; returns True when the chunk limit cut the pass short.

    Chunking lets the caller commit intermediate transactions so a bounded slice does
    not hold the SQLite write lock for its whole duration.
    """
    processed = 0
    while True:
        budget.check()
        with store.connect() as db:
            item = db.execute("SELECT path,reason FROM pending ORDER BY rowid LIMIT 1").fetchone()
        if item is None:
            return False
        raw = str(item[0])
        state, diagnostic, text, digest = "ready", "", "", ""
        symbols: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        try:
            if allowed is not None and not allowed(raw):
                raise BudgetError("permission_filtered")
            path = contained(store.root, raw)
            observed_stat = path.stat()
            content, version = read_snapshot(store.root, raw, budget)
            digest = version.sha256
            if b"\x00" in content[:8192]:
                raise BudgetError("binary")
            text = content.decode("utf-8")
            if language(raw) == "python":
                # ast.parse does not execute repository code.
                symbols, edges = python_structure(text)
            else:
                try:
                    edges = manifest_dependencies(raw, text)
                except (ValueError, TypeError, AttributeError) as exc:
                    # A malformed/partially edited manifest remains searchable text.
                    state, diagnostic = "text_only", f"manifest_parse_failed:{type(exc).__name__}"
            stat = path.stat()
            if (observed_stat.st_size, observed_stat.st_mtime_ns, observed_stat.st_ino) != (stat.st_size, stat.st_mtime_ns, stat.st_ino):
                raise StaleEvidence("file changed while parsing")
        except FileNotFoundError:
            with store.connect() as db:
                current = db.execute("SELECT reason FROM pending WHERE path=?", (raw,)).fetchone()
                if current is None or current[0] != item[1]:
                    return False
                row = db.execute("SELECT id FROM files WHERE path=?", (raw,)).fetchone()
                if row and store.get(db, "fts") == "1":
                    db.execute("DELETE FROM text_fts WHERE rowid=?", (row[0],))
                db.execute("DELETE FROM files WHERE path=?", (raw,))
                db.execute("DELETE FROM pending WHERE path=?", (raw,))
                store.bump(db)
            processed += 1
            if chunk is not None and processed >= chunk:
                return True
            continue
        except BudgetExceeded:
            raise
        except StaleEvidence:
            # Keep queued; never publish a parse of moving content.
            return False
        except (SyntaxError, RecursionError) as exc:
            state, diagnostic = "text_only", f"parse_failed:{type(exc).__name__}"
            stat = path.stat()
            if (observed_stat.st_size, observed_stat.st_mtime_ns, observed_stat.st_ino) != (stat.st_size, stat.st_mtime_ns, stat.st_ino):
                return False
        except (OSError, ValueError, UnicodeError) as exc:
            state, diagnostic = "skipped", str(exc)[:200]
            text = ""
            try:
                stat = contained(store.root, raw).stat()
            except (OSError, ValueError):
                stat = None
        budget.check()
        with store.connect() as db:
            # Each file's facts publish atomically inside the caller's chunk transaction.
            current = db.execute("SELECT reason FROM pending WHERE path=?", (raw,)).fetchone()
            if current is None or current[0] != item[1]:
                # Another producer reported a newer revision while this file was parsed.
                return False
            epoch = int(store.get(db, "scan_epoch"))
            db.execute("""INSERT INTO files(path,language,module,size,mtime_ns,seen) VALUES(?,?,?,?,?,?)
                ON CONFLICT(path) DO NOTHING""", (raw, language(raw), module(raw),
                stat.st_size if stat else 0, stat.st_mtime_ns if stat else 0, epoch))
            row = db.execute("SELECT id,hash FROM files WHERE path=?", (raw,)).fetchone()
            file_id = row[0]
            db.execute("DELETE FROM symbols WHERE file_id=?", (file_id,))
            db.execute("DELETE FROM edges WHERE file_id=?", (file_id,))
            if store.get(db, "fts") == "1":
                db.execute("DELETE FROM text_fts WHERE rowid=?", (file_id,))
                if text:
                    db.execute("INSERT INTO text_fts(rowid,text) VALUES(?,?)", (file_id, text))
            db.executemany("INSERT INTO symbols VALUES(?,?,?,?,?,?)", ((file_id, s["name"], s["qualified"], s["kind"],
                              s["line"], s["end_line"]) for s in symbols))
            db.executemany("INSERT INTO edges VALUES(?,?,?,?,?,?)", ((file_id, e["source"], e["target"], e["kind"],
                              e["line"], e["precision"]) for e in edges))
            db.execute("""UPDATE files SET hash=?,revision=revision+1,state=?,diagnostic=?,size=?,mtime_ns=? WHERE id=?""",
                       (digest, state, diagnostic, stat.st_size if stat else 0, stat.st_mtime_ns if stat else 0, file_id))
            db.execute("DELETE FROM pending WHERE path=?", (raw,))
            store.bump(db)
        processed += 1
        if chunk is not None and processed >= chunk:
            return True
