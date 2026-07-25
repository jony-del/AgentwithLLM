from __future__ import annotations

import array
import hashlib
import json
import math
import os
import sqlite3
import threading
import time
from contextlib import closing, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent_core.file_lock import FileLock
from agent_core.memory.chunking import CHUNKER_VERSION, chunk_markdown
from agent_core.memory.config import MemoryRetrievalConfig
from agent_core.memory.models import EmbeddingBackend, MemoryDocument
from agent_core.memory.repository import MemoryRepository
from agent_core.memory.text import (
    exact_atoms,
    field_atoms,
    fts_document,
    fts_query,
    normalize_text,
)

INDEX_SCHEMA_VERSION = 3
LEXICAL_VERSION = "unicode-nfkc-cjk-code-v1"
DEFAULT_EMBEDDING_FINGERPRINT = "BAAI-bge-m3-int8-onnx-v1"
_BACKGROUND_LOCK = threading.Lock()
_BACKGROUND_JOBS: set[str] = set()


@dataclass(slots=True)
class IndexCandidate:
    chunk_id: str
    document_id: str
    ordinal: int
    heading: str
    content: str
    start_char: int
    end_char: int
    name: str
    description: str
    type: str
    updated_at: str
    confidence: float
    explicit: bool
    verified_at: str | None
    tags: list[str]
    sources: list[str]
    rank: int = 0
    raw_score: float = 0.0
    explicit_filter: bool = False


@dataclass(slots=True)
class MemoryIndexStatus:
    path: str
    schema_version: int
    index_fingerprint: str
    documents: int
    chunks: int
    embedded_chunks: int
    pending_embeddings: int
    coverage: float
    fts5: bool
    diagnostics: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_index_base() -> Path:
    override = os.getenv("POLARIS_INDEX_DIR")
    if override:
        return Path(override).expanduser()
    polaris_home = os.getenv("POLARIS_HOME")
    if polaris_home:
        return Path(polaris_home).expanduser() / "indexes"
    return Path.home() / ".polaris" / "indexes"


def memory_root_hash(root: Path) -> str:
    canonical = os.path.normcase(str(root.resolve(strict=False)))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def retrieval_fingerprint(
    config: MemoryRetrievalConfig,
    embedding_fingerprint: str = DEFAULT_EMBEDDING_FINGERPRINT,
) -> str:
    payload = {
        "schema": INDEX_SCHEMA_VERSION,
        "chunker": CHUNKER_VERSION,
        "lexical": LEXICAL_VERSION,
        "chunk_tokens": config.chunk_tokens,
        "chunk_overlap_tokens": config.chunk_overlap_tokens,
        "embedding": embedding_fingerprint,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _document_sha(document: MemoryDocument) -> str:
    path = Path(document.path) if document.path else None
    if path is not None:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            pass
    payload = json.dumps(
        {**document.header(), "content": document.content},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _decode_json_list(raw: str) -> list[str]:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


class MemoryIndex:
    """Rebuildable SQLite exact/BM25/embedding index over Markdown topics."""

    def __init__(
        self,
        repository: MemoryRepository,
        config: MemoryRetrievalConfig,
        *,
        index_base: str | Path | None = None,
        embedding_fingerprint: str = DEFAULT_EMBEDDING_FINGERPRINT,
    ) -> None:
        self.repository = repository
        self.config = config
        self.embedding_fingerprint = embedding_fingerprint
        self.fingerprint = retrieval_fingerprint(config, embedding_fingerprint)
        base = Path(index_base) if index_base is not None else default_index_base()
        self.directory = (
            base.expanduser()
            / memory_root_hash(repository.root)
            / self.fingerprint
        )
        self.path = self.directory / "memory.sqlite3"
        self.lock_path = self.directory.parent / ".index.lock"
        self.last_sync_diagnostics: list[str] = []

    def _connect(self, path: Path | None = None, *, write: bool = False) -> sqlite3.Connection:
        target = path or self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            target,
            timeout=max(0.1, self.config.timeout_seconds),
            check_same_thread=False,
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.execute(
                f"PRAGMA busy_timeout={max(100, int(self.config.timeout_seconds * 1000))}"
            )
            if write:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=NORMAL")
                connection.execute("PRAGMA foreign_keys=ON")
        except BaseException:
            connection.close()
            raise
        return connection

    def _create_schema(self, connection: sqlite3.Connection) -> bool:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                normalized_id TEXT NOT NULL,
                path TEXT NOT NULL,
                mtime_ns INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                name TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                description TEXT NOT NULL,
                normalized_description TEXT NOT NULL,
                type TEXT NOT NULL,
                normalized_type TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                confidence REAL NOT NULL,
                explicit INTEGER NOT NULL,
                verified_at TEXT,
                tags_json TEXT NOT NULL,
                sources_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS documents_type_idx ON documents(normalized_type);
            CREATE TABLE IF NOT EXISTS document_tags (
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                value TEXT NOT NULL,
                PRIMARY KEY(document_id, value)
            );
            CREATE INDEX IF NOT EXISTS document_tags_value_idx ON document_tags(value);
            CREATE TABLE IF NOT EXISTS document_sources (
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                value TEXT NOT NULL,
                PRIMARY KEY(document_id, value)
            );
            CREATE INDEX IF NOT EXISTS document_sources_value_idx ON document_sources(value);
            CREATE TABLE IF NOT EXISTS chunks (
                rowid INTEGER PRIMARY KEY,
                chunk_id TEXT NOT NULL UNIQUE,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                heading TEXT NOT NULL,
                content TEXT NOT NULL,
                normalized_content TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                embedding BLOB,
                embedding_dim INTEGER,
                embedding_fingerprint TEXT
            );
            CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks(document_id, ordinal);
            CREATE TABLE IF NOT EXISTS exact_atoms (
                atom TEXT NOT NULL,
                kind TEXT NOT NULL,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                field TEXT NOT NULL,
                PRIMARY KEY(atom, kind, chunk_id, field)
            );
            CREATE INDEX IF NOT EXISTS exact_atoms_lookup_idx ON exact_atoms(atom, kind);
            CREATE TABLE IF NOT EXISTS embedding_queue (
                chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT NOT NULL DEFAULT '',
                updated_at REAL NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS embedding_queue_status_idx
                ON embedding_queue(status, updated_at);
            CREATE TABLE IF NOT EXISTS diagnostics (
                path TEXT PRIMARY KEY,
                mtime_ns INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL,
                reason TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            """
        )
        try:
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    document_id UNINDEXED,
                    name,
                    tags,
                    description,
                    content
                )
                """
            )
        except sqlite3.OperationalError:
            fts5 = False
        else:
            fts5 = True
        metadata = {
            "schema_version": str(INDEX_SCHEMA_VERSION),
            "index_fingerprint": self.fingerprint,
            "embedding_fingerprint": self.embedding_fingerprint,
            "chunker_version": CHUNKER_VERSION,
            "lexical_version": LEXICAL_VERSION,
            "fts5": "1" if fts5 else "0",
            "updated_at": str(time.time()),
        }
        connection.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            list(metadata.items()),
        )
        return fts5

    def _schema_valid(self, connection: sqlite3.Connection) -> bool:
        try:
            rows = dict(connection.execute("SELECT key, value FROM metadata"))
        except sqlite3.DatabaseError:
            return False
        return (
            rows.get("schema_version") == str(INDEX_SCHEMA_VERSION)
            and rows.get("index_fingerprint") == self.fingerprint
        )

    def ensure_current(self) -> MemoryIndexStatus:
        try:
            if not self.path.exists():
                self.rebuild()
            else:
                needs_rebuild = False
                # Acquire the cross-process writer lock before opening SQLite.
                # On Windows this avoids holding the database open while another
                # writer attempts an atomic generation replacement.
                with FileLock(self.lock_path, timeout=self.config.timeout_seconds):
                    with closing(self._connect(write=True)) as connection:
                        if self._schema_valid(connection):
                            self._sync_incremental(connection, already_locked=True)
                        else:
                            needs_rebuild = True
                if needs_rebuild:
                    self.rebuild()
            return self.status()
        except sqlite3.DatabaseError:
            self.rebuild()
            return self.status()

    def rebuild(self) -> MemoryIndexStatus:
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.directory / f".memory-{os.getpid()}-{time.time_ns()}.sqlite3"
        with FileLock(self.lock_path, timeout=self.config.timeout_seconds):
            try:
                temporary.unlink(missing_ok=True)
                with closing(self._connect(temporary, write=True)) as connection:
                    self._create_schema(connection)
                    self._index_all(connection)
                    connection.execute(
                        "INSERT OR REPLACE INTO metadata(key, value) VALUES ('updated_at', ?)",
                        (str(time.time()),),
                    )
                    connection.commit()
                    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                deadline = time.monotonic() + self.config.timeout_seconds
                while True:
                    try:
                        os.replace(temporary, self.path)
                        break
                    except PermissionError:
                        # Windows does not replace an open SQLite file. Readers
                        # are short-lived, so retry within the shared index timeout.
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.025)
            finally:
                temporary.unlink(missing_ok=True)
                Path(str(temporary) + "-wal").unlink(missing_ok=True)
                Path(str(temporary) + "-shm").unlink(missing_ok=True)
        return self.status()

    def schedule_rebuild(self) -> None:
        key = f"rebuild:{self.path}"
        with _BACKGROUND_LOCK:
            if key in _BACKGROUND_JOBS:
                return
            _BACKGROUND_JOBS.add(key)

        def worker() -> None:
            try:
                self.rebuild()
            except Exception:
                return
            finally:
                with _BACKGROUND_LOCK:
                    _BACKGROUND_JOBS.discard(key)

        threading.Thread(target=worker, name="polaris-memory-index", daemon=True).start()

    def _source_states(self) -> dict[str, tuple[Path, int, int, str | None]]:
        """Enumerate topic metadata without opening unchanged Markdown files."""
        if not self.repository.root.exists():
            return {}
        current: dict[str, tuple[Path, int, int, str | None]] = {}
        with os.scandir(self.repository.root) as entries:
            for entry in entries:
                if (
                    entry.name == "MEMORY.md"
                    or entry.name.startswith(".")
                    or not entry.name.casefold().endswith(".md")
                ):
                    continue
                path = Path(entry.path)
                stat_error: str | None
                try:
                    stat = entry.stat(follow_symlinks=True)
                except OSError as exc:
                    mtime_ns, size_bytes = 0, 0
                    stat_error = f"{path}: {type(exc).__name__}: {exc}"
                else:
                    mtime_ns = int(stat.st_mtime_ns)
                    size_bytes = int(stat.st_size)
                    stat_error = None
                current[str(path)] = (path, mtime_ns, size_bytes, stat_error)
        return current

    def _scan_documents(
        self,
    ) -> tuple[
        list[tuple[MemoryDocument, int, int]],
        list[tuple[str, int, int, str]],
    ]:
        documents: list[tuple[MemoryDocument, int, int]] = []
        diagnostics: list[tuple[str, int, int, str]] = []
        seen: set[str] = set()
        sources = self._source_states()
        for path_key in sorted(sources):
            path, mtime_ns, size_bytes, stat_error = sources[path_key]
            if stat_error is not None:
                diagnostics.append((path_key, mtime_ns, size_bytes, stat_error))
                continue
            try:
                document = self.repository.load_document_path(path)
                if document.archived:
                    raise ValueError("archived document is outside the archive directory")
                if document.id in seen:
                    raise ValueError(f"duplicate id {document.id}")
            except Exception as exc:  # noqa: BLE001 - diagnostics isolate malformed sources
                diagnostics.append(
                    (
                        str(path),
                        mtime_ns,
                        size_bytes,
                        f"{path}: {type(exc).__name__}: {exc}",
                    )
                )
                continue
            seen.add(document.id)
            documents.append((document, mtime_ns, size_bytes))
        return documents, diagnostics

    def _index_all(self, connection: sqlite3.Connection) -> None:
        documents, diagnostics = self._scan_documents()
        for document, mtime_ns, size_bytes in documents:
            self._insert_document(
                connection,
                document,
                mtime_ns=mtime_ns,
                size_bytes=size_bytes,
            )
        self._replace_diagnostics(connection, diagnostics)
        self.last_sync_diagnostics = [item[3] for item in diagnostics]

    def _sync_incremental(
        self,
        connection: sqlite3.Connection,
        *,
        already_locked: bool = False,
    ) -> None:
        lock = (
            nullcontext()
            if already_locked
            else FileLock(self.lock_path, timeout=self.config.timeout_seconds)
        )
        with lock:
            indexed = {
                str(row["path"]): {
                    "id": str(row["id"]),
                    "sha256": str(row["sha256"]),
                    "mtime_ns": int(row["mtime_ns"]),
                    "size_bytes": int(row["size_bytes"]),
                }
                for row in connection.execute(
                    "SELECT id, path, sha256, mtime_ns, size_bytes FROM documents"
                )
            }
            old_diagnostics = {
                str(row["path"]): (
                    int(row["mtime_ns"]),
                    int(row["size_bytes"]),
                    str(row["reason"]),
                )
                for row in connection.execute(
                    "SELECT path, mtime_ns, size_bytes, reason FROM diagnostics"
                )
            }
            current = self._source_states()

            known_paths = set(indexed).union(old_diagnostics)
            changed_paths = {
                candidate_key
                for candidate_key, (_, mtime_ns, size_bytes, _) in current.items()
                if not (
                    (
                        candidate_key in indexed
                        and indexed[candidate_key]["mtime_ns"] == mtime_ns
                        and indexed[candidate_key]["size_bytes"] == size_bytes
                    )
                    or (
                        candidate_key in old_diagnostics
                        and old_diagnostics[candidate_key][:2] == (mtime_ns, size_bytes)
                    )
                )
            }
            removed_paths = known_paths.difference(current)
            if removed_paths or changed_paths.intersection(indexed):
                # A previously skipped duplicate can become valid when the
                # document owning its id is removed or changes identity.
                changed_paths.update(
                    path_key
                    for path_key, (_, _, reason) in old_diagnostics.items()
                    if path_key in current and "duplicate id " in reason
                )
            if not changed_paths and not removed_paths:
                self.last_sync_diagnostics = [
                    old_diagnostics[diagnostic_key][2]
                    for diagnostic_key in sorted(old_diagnostics)
                ]
                return

            parsed: dict[str, tuple[MemoryDocument, str] | str] = {}
            metadata_only: set[str] = set()
            for path_key in sorted(changed_paths):
                source, mtime_ns, size_bytes, stat_error = current[path_key]
                if stat_error is not None:
                    parsed[path_key] = stat_error
                    continue
                try:
                    document = self.repository.load_document_path(source)
                    if document.archived:
                        raise ValueError("archived document is outside the archive directory")
                    digest = _document_sha(document)
                except Exception as exc:  # noqa: BLE001 - one bad topic must not sink search
                    parsed[path_key] = f"{source}: {type(exc).__name__}: {exc}"
                    continue
                parsed[path_key] = (document, digest)
                old = indexed.get(path_key)
                if (
                    old is not None
                    and old["id"] == document.id
                    and old["sha256"] == digest
                ):
                    metadata_only.add(path_key)
                    connection.execute(
                        "UPDATE documents SET mtime_ns=?, size_bytes=? WHERE id=?",
                        (mtime_ns, size_bytes, document.id),
                    )

            for path_key in sorted(
                removed_paths.union(changed_paths.difference(metadata_only))
            ):
                old = indexed.get(path_key)
                if old is not None:
                    self._delete_document(connection, str(old["id"]))
            connection.executemany(
                "DELETE FROM diagnostics WHERE path=?",
                [
                    (path_key,)
                    for path_key in sorted(removed_paths.union(changed_paths))
                ],
            )

            existing_ids = {
                str(row["id"]): str(row["path"])
                for row in connection.execute("SELECT id, path FROM documents")
            }
            new_diagnostics: list[tuple[str, int, int, str]] = []
            for path_key in sorted(changed_paths.difference(metadata_only)):
                source, mtime_ns, size_bytes, _ = current[path_key]
                result = parsed[path_key]
                if isinstance(result, str):
                    new_diagnostics.append((path_key, mtime_ns, size_bytes, result))
                    continue
                document, digest = result
                if document.id in existing_ids:
                    new_diagnostics.append(
                        (
                            path_key,
                            mtime_ns,
                            size_bytes,
                            f"{source}: ValueError: duplicate id {document.id}",
                        )
                    )
                    continue
                self._insert_document(
                    connection,
                    document,
                    digest=digest,
                    mtime_ns=mtime_ns,
                    size_bytes=size_bytes,
                )
                existing_ids[document.id] = path_key
            connection.executemany(
                """
                INSERT OR REPLACE INTO diagnostics(
                    path, mtime_ns, size_bytes, reason, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (path_key, mtime_ns, size_bytes, reason[:2000], time.time())
                    for path_key, mtime_ns, size_bytes, reason in new_diagnostics
                ],
            )
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES ('updated_at', ?)",
                (str(time.time()),),
            )
            connection.commit()
            self.last_sync_diagnostics = [
                str(row["reason"])
                for row in connection.execute(
                    "SELECT reason FROM diagnostics ORDER BY path"
                )
            ]

    def _replace_diagnostics(
        self,
        connection: sqlite3.Connection,
        diagnostics: list[tuple[str, int, int, str]],
    ) -> None:
        connection.execute("DELETE FROM diagnostics")
        connection.executemany(
            """
            INSERT INTO diagnostics(path, mtime_ns, size_bytes, reason, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (path, mtime_ns, size_bytes, reason[:2000], time.time())
                for path, mtime_ns, size_bytes, reason in diagnostics
            ],
        )

    def _delete_document(self, connection: sqlite3.Connection, document_id: str) -> None:
        try:
            connection.execute("DELETE FROM chunks_fts WHERE document_id = ?", (document_id,))
        except sqlite3.OperationalError:
            pass
        connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def _insert_document(
        self,
        connection: sqlite3.Connection,
        document: MemoryDocument,
        *,
        digest: str | None = None,
        mtime_ns: int,
        size_bytes: int,
    ) -> None:
        connection.execute(
            """
            INSERT INTO documents(
                id, normalized_id, path, mtime_ns, size_bytes, sha256,
                name, normalized_name,
                description, normalized_description, type,
                normalized_type, updated_at, confidence, explicit, verified_at,
                tags_json, sources_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document.id,
                normalize_text(document.id),
                str(document.path or ""),
                int(mtime_ns),
                int(size_bytes),
                digest or _document_sha(document),
                document.name,
                normalize_text(document.name),
                document.description,
                normalize_text(document.description),
                document.type,
                normalize_text(document.type),
                document.updated_at,
                float(document.confidence),
                int(document.explicit),
                document.verified_at,
                json.dumps(document.tags, ensure_ascii=False),
                json.dumps(document.sources, ensure_ascii=False),
            ),
        )
        connection.executemany(
            "INSERT OR IGNORE INTO document_tags(document_id, value) VALUES (?, ?)",
            [(document.id, normalize_text(tag)) for tag in document.tags if normalize_text(tag)],
        )
        connection.executemany(
            "INSERT OR IGNORE INTO document_sources(document_id, value) VALUES (?, ?)",
            [(document.id, normalize_text(source)) for source in document.sources if normalize_text(source)],
        )
        chunks = chunk_markdown(
            document.id,
            document.content,
            chunk_tokens=self.config.chunk_tokens,
            overlap_tokens=self.config.chunk_overlap_tokens,
        )
        if not chunks:
            chunks = chunk_markdown(
                document.id,
                document.description or document.name,
                chunk_tokens=self.config.chunk_tokens,
                overlap_tokens=self.config.chunk_overlap_tokens,
            )
        common_atoms = [
            *field_atoms("id", [document.id]),
            *field_atoms("tag", document.tags),
            *field_atoms("type", [document.type]),
            *field_atoms("source", document.sources),
            *[(kind, value) for kind, value in exact_atoms(document.name)],
            *[(kind, value) for kind, value in exact_atoms(document.description)],
        ]
        for chunk in chunks:
            cursor = connection.execute(
                """
                INSERT INTO chunks(
                    chunk_id, document_id, ordinal, heading, content,
                    normalized_content, token_count,
                    start_char, end_char
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    document.id,
                    chunk.ordinal,
                    chunk.heading,
                    chunk.content,
                    normalize_text(chunk.content),
                    chunk.token_count,
                    chunk.start_char,
                    chunk.end_char,
                ),
            )
            atoms = [
                *((kind, value, "metadata") for kind, value in common_atoms),
                *((kind, value, "content") for kind, value in exact_atoms(chunk.content)),
            ]
            connection.executemany(
                """
                INSERT OR IGNORE INTO exact_atoms(atom, kind, document_id, chunk_id, field)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (value, kind, document.id, chunk.chunk_id, field)
                    for kind, value, field in atoms
                ],
            )
            connection.execute(
                """
                INSERT INTO embedding_queue(chunk_id, status, attempts, last_error, updated_at)
                VALUES (?, 'pending', 0, '', ?)
                """,
                (chunk.chunk_id, time.time()),
            )
            try:
                connection.execute(
                    """
                    INSERT INTO chunks_fts(
                        rowid, chunk_id, document_id, name, tags, description, content
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cursor.lastrowid,
                        chunk.chunk_id,
                        document.id,
                        fts_document(document.name),
                        fts_document(" ".join(document.tags)),
                        fts_document(document.description),
                        fts_document(f"{chunk.heading}\n{chunk.content}"),
                    ),
                )
            except sqlite3.OperationalError:
                pass

    def _filter_sql(
        self,
        filters: dict[str, list[str]],
        *,
        alias: str = "d",
    ) -> tuple[str, list[str]]:
        allowed = {"id", "tag", "type", "source"}
        unknown = set(filters).difference(allowed)
        if unknown:
            raise ValueError("unsupported memory filter(s): " + ", ".join(sorted(unknown)))
        clauses: list[str] = []
        parameters: list[str] = []
        for dimension in ("id", "type"):
            values = [normalize_text(value) for value in filters.get(dimension, []) if normalize_text(value)]
            if not values:
                continue
            column = "normalized_id" if dimension == "id" else "normalized_type"
            clauses.append(f"{alias}.{column} IN ({','.join('?' for _ in values)})")
            parameters.extend(values)
        for dimension, table in (("tag", "document_tags"), ("source", "document_sources")):
            values = [normalize_text(value) for value in filters.get(dimension, []) if normalize_text(value)]
            if not values:
                continue
            clauses.append(
                f"EXISTS (SELECT 1 FROM {table} f_{dimension} "
                f"WHERE f_{dimension}.document_id = {alias}.id "
                f"AND f_{dimension}.value IN ({','.join('?' for _ in values)}))"
            )
            parameters.extend(values)
        return (" AND " + " AND ".join(clauses)) if clauses else "", parameters

    @staticmethod
    def _candidate(row: sqlite3.Row, *, rank: int, raw_score: float = 0.0) -> IndexCandidate:
        return IndexCandidate(
            chunk_id=str(row["chunk_id"]),
            document_id=str(row["document_id"]),
            ordinal=int(row["ordinal"]),
            heading=str(row["heading"]),
            content=str(row["content"]),
            start_char=int(row["start_char"]),
            end_char=int(row["end_char"]),
            name=str(row["name"]),
            description=str(row["description"]),
            type=str(row["type"]),
            updated_at=str(row["updated_at"]),
            confidence=float(row["confidence"]),
            explicit=bool(row["explicit"]),
            verified_at=str(row["verified_at"]) if row["verified_at"] else None,
            tags=_decode_json_list(str(row["tags_json"])),
            sources=_decode_json_list(str(row["sources_json"])),
            rank=rank,
            raw_score=raw_score,
        )

    def exact_candidates(
        self,
        query: str,
        filters: dict[str, list[str]],
        *,
        limit: int,
    ) -> list[IndexCandidate]:
        if limit <= 0:
            return []
        filter_sql, filter_values = self._filter_sql(filters)
        columns = """
            c.chunk_id, c.document_id, c.ordinal, c.heading, c.content,
            c.start_char, c.end_char, d.name, d.description, d.type,
            d.updated_at, d.confidence, d.explicit, d.verified_at,
            d.tags_json, d.sources_json
        """
        result: list[IndexCandidate] = []
        per_document: dict[str, int] = {}
        with closing(self._connect()) as connection:
            if any(filters.values()):
                rows = connection.execute(
                    f"""
                    WITH ranked AS (
                        SELECT {columns},
                               ROW_NUMBER() OVER (
                                   PARTITION BY d.id ORDER BY c.ordinal, c.chunk_id
                               ) AS document_rank
                        FROM chunks c JOIN documents d ON d.id = c.document_id
                        WHERE 1=1 {filter_sql}
                    )
                    SELECT * FROM ranked
                    WHERE document_rank <= 3
                    ORDER BY updated_at DESC, chunk_id ASC LIMIT ?
                    """,
                    (*filter_values, limit),
                )
                for rank, row in enumerate(rows, 1):
                    candidate = self._candidate(row, rank=rank, raw_score=1.0)
                    candidate.explicit_filter = True
                    result.append(candidate)
                    per_document[candidate.document_id] = (
                        per_document.get(candidate.document_id, 0) + 1
                    )
            atoms = exact_atoms(query)
            stripped = normalize_text(query)
            if stripped and " " not in stripped and len(stripped) <= 128:
                atoms.append(("id", stripped))
            atoms = list(dict.fromkeys(atoms))
            phrases = [value for kind, value in atoms if kind == "phrase"]
            atoms = [(kind, value) for kind, value in atoms if kind != "phrase"]
            if phrases and len(result) < limit:
                phrase_sql = " OR ".join(
                    (
                        "instr(c.normalized_content, ?) > 0 "
                        "OR instr(d.normalized_name, ?) > 0 "
                        "OR instr(d.normalized_description, ?) > 0"
                    )
                    for _ in phrases
                )
                phrase_parameters = [
                    value
                    for phrase in phrases
                    for value in (phrase, phrase, phrase)
                ]
                seen = {candidate.chunk_id for candidate in result}
                rows = connection.execute(
                    f"""
                    SELECT {columns}
                    FROM chunks c JOIN documents d ON d.id = c.document_id
                    WHERE ({phrase_sql}) {filter_sql}
                    ORDER BY d.updated_at DESC, c.chunk_id ASC
                    LIMIT ?
                    """,
                    (
                        *phrase_parameters,
                        *filter_values,
                        max(0, (limit - len(result)) * 8),
                    ),
                )
                for row in rows:
                    if str(row["chunk_id"]) in seen:
                        continue
                    document_id = str(row["document_id"])
                    if per_document.get(document_id, 0) >= 3:
                        continue
                    candidate = self._candidate(
                        row,
                        rank=len(result) + 1,
                        raw_score=1.0,
                    )
                    result.append(candidate)
                    per_document[document_id] = per_document.get(document_id, 0) + 1
                    if len(result) >= limit:
                        break
            if not atoms or len(result) >= limit:
                return result[:limit]
            pairs = " OR ".join("(ea.kind = ? AND ea.atom = ?)" for _ in atoms)
            atom_parameters = [value for pair in atoms for value in pair]
            rows = connection.execute(
                f"""
                SELECT {columns}, COUNT(*) AS atom_matches
                FROM exact_atoms ea
                JOIN chunks c ON c.chunk_id = ea.chunk_id
                JOIN documents d ON d.id = c.document_id
                WHERE ({pairs}) {filter_sql}
                GROUP BY c.chunk_id
                ORDER BY atom_matches DESC, d.updated_at DESC, c.chunk_id ASC
                LIMIT ?
                """,
                (
                    *atom_parameters,
                    *filter_values,
                    max(0, (limit - len(result)) * 8),
                ),
            )
            seen = {candidate.chunk_id for candidate in result}
            for row in rows:
                if str(row["chunk_id"]) in seen:
                    continue
                document_id = str(row["document_id"])
                if per_document.get(document_id, 0) >= 3:
                    continue
                candidate = self._candidate(
                    row,
                    rank=len(result) + 1,
                    raw_score=float(row["atom_matches"]),
                )
                result.append(candidate)
                per_document[document_id] = per_document.get(document_id, 0) + 1
                if len(result) >= limit:
                    break
        return result[:limit]

    def bm25_candidates(
        self,
        query: str,
        filters: dict[str, list[str]],
        *,
        limit: int,
    ) -> list[IndexCandidate]:
        expression = fts_query(query)
        if not expression or limit <= 0:
            return []
        filter_sql, filter_values = self._filter_sql(filters)
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    f"""
                    SELECT c.chunk_id, c.document_id, c.ordinal, c.heading, c.content,
                           c.start_char, c.end_char, d.name, d.description, d.type,
                           d.updated_at, d.confidence, d.explicit, d.verified_at,
                           d.tags_json, d.sources_json,
                           bm25(chunks_fts, 0.0, 0.0, 4.0, 3.0, 2.0, 1.0) AS bm
                    FROM chunks_fts
                    JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
                    JOIN documents d ON d.id = c.document_id
                    WHERE chunks_fts MATCH ? {filter_sql}
                    ORDER BY bm ASC, c.chunk_id ASC LIMIT ?
                    """,
                    (expression, *filter_values, max(limit, limit * 8)),
                )
                result: list[IndexCandidate] = []
                per_document: dict[str, int] = {}
                for row in rows:
                    document_id = str(row["document_id"])
                    if per_document.get(document_id, 0) >= 3:
                        continue
                    per_document[document_id] = per_document.get(document_id, 0) + 1
                    result.append(
                        self._candidate(
                            row,
                            rank=len(result) + 1,
                            raw_score=-float(row["bm"]),
                        )
                    )
                    if len(result) >= limit:
                        break
                return result
        except sqlite3.OperationalError:
            return []

    def eligible_document_ids(self, filters: dict[str, list[str]]) -> set[str] | None:
        if not any(filters.values()):
            return None
        filter_sql, parameters = self._filter_sql(filters)
        with closing(self._connect()) as connection:
            return {
                str(row[0])
                for row in connection.execute(
                    f"SELECT d.id FROM documents d WHERE 1=1 {filter_sql}",
                    parameters,
                )
            }

    def embedded_rows(self, filters: dict[str, list[str]]) -> list[sqlite3.Row]:
        eligible = self.eligible_document_ids(filters)
        with closing(self._connect()) as connection:
            rows = list(
                connection.execute(
                    """
                    SELECT c.chunk_id, c.document_id, c.ordinal, c.heading, c.content,
                           c.start_char, c.end_char, c.embedding, c.embedding_dim,
                           d.name, d.description, d.type, d.updated_at, d.confidence,
                           d.explicit, d.verified_at, d.tags_json, d.sources_json
                    FROM chunks c JOIN documents d ON d.id = c.document_id
                    WHERE c.embedding IS NOT NULL
                      AND c.embedding_fingerprint = ?
                    ORDER BY c.chunk_id
                    """,
                    (self.embedding_fingerprint,),
                )
            )
        return rows if eligible is None else [row for row in rows if str(row["document_id"]) in eligible]

    def populate_embeddings(
        self,
        backend: EmbeddingBackend,
        *,
        batch_size: int = 32,
        deadline: float | None = None,
        max_batches: int | None = None,
    ) -> int:
        completed = 0
        batches = 0
        while max_batches is None or batches < max_batches:
            if deadline is not None and time.monotonic() >= deadline:
                break
            with FileLock(self.lock_path, timeout=self.config.timeout_seconds):
                with closing(self._connect(write=True)) as connection:
                    # Interrupted workers are resumable after a conservative minute.
                    connection.execute(
                        """
                        UPDATE embedding_queue SET status='pending'
                        WHERE status='working' AND updated_at < ?
                        """,
                        (time.time() - 60,),
                    )
                    rows = list(
                        connection.execute(
                            """
                            SELECT q.chunk_id, c.content
                            FROM embedding_queue q JOIN chunks c ON c.chunk_id = q.chunk_id
                            WHERE q.status='pending'
                            ORDER BY q.attempts, q.chunk_id LIMIT ?
                            """,
                            (max(1, batch_size),),
                        )
                    )
                    if not rows:
                        connection.commit()
                        break
                    ids = [str(row["chunk_id"]) for row in rows]
                    connection.executemany(
                        """
                        UPDATE embedding_queue
                        SET status='working', attempts=attempts+1, updated_at=?
                        WHERE chunk_id=?
                        """,
                        [(time.time(), chunk_id) for chunk_id in ids],
                    )
                    connection.commit()
            try:
                vectors = backend.embed(
                    [str(row["content"]) for row in rows],
                    deadline=deadline,
                )
                if len(vectors) != len(rows):
                    raise ValueError("embedding backend returned the wrong vector count")
                encoded: list[tuple[bytes, int, str]] = []
                for chunk_id, raw in zip(ids, vectors, strict=True):
                    values = [float(value) for value in raw]
                    norm = math.sqrt(sum(value * value for value in values))
                    if not values or not math.isfinite(norm) or norm <= 0:
                        raise ValueError("embedding backend returned an invalid vector")
                    normalized = array.array("f", (value / norm for value in values))
                    encoded.append((normalized.tobytes(), len(normalized), chunk_id))
            except Exception as exc:
                with FileLock(self.lock_path, timeout=self.config.timeout_seconds):
                    with closing(self._connect(write=True)) as connection:
                        connection.executemany(
                            """
                            UPDATE embedding_queue
                            SET status='pending', last_error=?, updated_at=?
                            WHERE chunk_id=?
                            """,
                            [(f"{type(exc).__name__}: {exc}"[:500], time.time(), item) for item in ids],
                        )
                        connection.commit()
                break
            with FileLock(self.lock_path, timeout=self.config.timeout_seconds):
                with closing(self._connect(write=True)) as connection:
                    connection.executemany(
                        """
                        UPDATE chunks
                        SET embedding=?, embedding_dim=?, embedding_fingerprint=?
                        WHERE chunk_id=?
                        """,
                        [
                            (blob, dimension, self.embedding_fingerprint, chunk_id)
                            for blob, dimension, chunk_id in encoded
                        ],
                    )
                    connection.executemany(
                        """
                        UPDATE embedding_queue
                        SET status='done', last_error='', updated_at=?
                        WHERE chunk_id=?
                        """,
                        [(time.time(), item) for item in ids],
                    )
                    connection.commit()
            completed += len(rows)
            batches += 1
        return completed

    def schedule_embeddings(self, backend: EmbeddingBackend) -> None:
        key = f"embedding:{self.path}:{backend.fingerprint}"
        with _BACKGROUND_LOCK:
            if key in _BACKGROUND_JOBS:
                return
            _BACKGROUND_JOBS.add(key)

        def worker() -> None:
            try:
                self.populate_embeddings(backend)
            except Exception:
                return
            finally:
                with _BACKGROUND_LOCK:
                    _BACKGROUND_JOBS.discard(key)

        threading.Thread(target=worker, name="polaris-memory-embedding", daemon=True).start()

    def status(self) -> MemoryIndexStatus:
        if not self.path.exists():
            return MemoryIndexStatus(
                str(self.path),
                INDEX_SCHEMA_VERSION,
                self.fingerprint,
                0,
                0,
                0,
                0,
                0.0,
                False,
                ["index is missing"],
            )
        try:
            with closing(self._connect()) as connection:
                metadata = dict(connection.execute("SELECT key, value FROM metadata"))
                documents = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
                chunks = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
                embedded = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
                    ).fetchone()[0]
                )
                pending = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM embedding_queue WHERE status != 'done'"
                    ).fetchone()[0]
                )
                diagnostics = [
                    str(row[0])
                    for row in connection.execute(
                        "SELECT reason FROM diagnostics ORDER BY created_at LIMIT 100"
                    )
                ]
            return MemoryIndexStatus(
                str(self.path),
                int(metadata.get("schema_version", 0)),
                str(metadata.get("index_fingerprint", "")),
                documents,
                chunks,
                embedded,
                pending,
                embedded / chunks if chunks else 1.0,
                metadata.get("fts5") == "1",
                diagnostics,
            )
        except sqlite3.DatabaseError as exc:
            return MemoryIndexStatus(
                str(self.path),
                0,
                self.fingerprint,
                0,
                0,
                0,
                0,
                0.0,
                False,
                [f"corrupt index: {type(exc).__name__}"],
            )
