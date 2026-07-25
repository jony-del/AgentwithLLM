from __future__ import annotations

import hashlib
import builtins
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import yaml

from agent_core.file_lock import FileLock
from agent_core.memory.models import MEMORY_TYPES, MemoryChange, MemoryDocument, MemoryRecord, utc_now
from agent_core.memory.paths import validate_memory_root
from agent_core.memory.security import require_secret_free

if TYPE_CHECKING:
    from agent_core.memory.config import MemoryConfig
    from agent_core.memory.models import (
        EmbeddingBackend,
        MemorySearchHit,
        MemorySearchRequest,
        RerankerBackend,
    )

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_CONFLICT_MARKERS = ("<<<<<<<", "=======", ">>>>>>>")
_INDEX_MAX_LINES = 200
_INDEX_MAX_BYTES = 25 * 1024
_CONTENT_MAX_BYTES = 64 * 1024


@dataclass(slots=True)
class ValidationReport:
    scanned: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    index_rebuilt: bool = False

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass(slots=True)
class MigrationReport:
    source: str
    checksum: str
    total: int = 0
    imported: int = 0
    skipped: int = 0
    corrupt_lines: list[int] = field(default_factory=list)
    already_complete: bool = False


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._").lower()
    return (text or "memory")[:64]


def _parse_document(path: Path, *, headers_only: bool = False) -> MemoryDocument:
    text = path.read_text(encoding="utf-8")
    if any(marker in text for marker in _CONFLICT_MARKERS):
        raise ValueError("contains Git conflict markers")
    if not text.startswith("---\n"):
        raise ValueError("missing YAML frontmatter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("unterminated YAML frontmatter")
    loaded = yaml.safe_load(text[4:end]) or {}
    if not isinstance(loaded, dict):
        raise ValueError("frontmatter must be a mapping")
    content = "" if headers_only else text[end + 5 :]
    return MemoryDocument.from_parts(loaded, content, path=str(path))


def _render_document(document: MemoryDocument) -> str:
    header = yaml.safe_dump(
        document.header(),
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).strip()
    return f"---\n{header}\n---\n\n{document.content.rstrip()}\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


class MemoryRepository:
    """Authoritative Markdown memory repository with locked atomic mutations."""

    def __init__(self, root: str | Path, *, scope: str = "private") -> None:
        self.root = validate_memory_root(root)
        self.scope = scope
        self.index_path = self.root / "MEMORY.md"
        self.lock_path = self.root / ".memory.lock"

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise ValueError("memory repository root may not be a symbolic link")

    def _paths(self, *, include_archived: bool = False) -> list[Path]:
        if not self.root.exists():
            return []
        paths = [
            path
            for path in self.root.glob("*.md")
            if path.name != "MEMORY.md" and not path.name.startswith(".")
        ]
        if include_archived and (self.root / "archive").exists():
            paths.extend((self.root / "archive").glob("*.md"))
        # MEMORY.md is a bounded human summary, not an enumeration source. The
        # authoritative Markdown corpus must remain complete beyond 200 topics.
        return sorted(paths, key=lambda item: item.name)

    def document_paths(self, *, include_archived: bool = False) -> list[Path]:
        """Return every authoritative topic path (never the bounded MEMORY.md view)."""
        return list(self._paths(include_archived=include_archived))

    def load_document_path(
        self,
        path: str | Path,
        *,
        headers_only: bool = False,
    ) -> MemoryDocument:
        """Parse and validate one contained authoritative Markdown topic."""
        candidate = self._contained(Path(path))
        if candidate.name == "MEMORY.md" or candidate.suffix.casefold() != ".md":
            raise ValueError("path is not an authoritative memory topic")
        document = _parse_document(candidate, headers_only=headers_only)
        self._validate_document(document, writing=False)
        return document

    def list(self, *, include_archived: bool = False, headers_only: bool = False) -> list[MemoryDocument]:
        documents: list[MemoryDocument] = []
        for path in self._paths(include_archived=include_archived):
            try:
                document = _parse_document(path, headers_only=headers_only)
            except (OSError, UnicodeError, ValueError, yaml.YAMLError):
                continue
            if document.archived and not include_archived:
                continue
            documents.append(document)
        return documents

    def scan_documents(
        self,
        *,
        include_archived: bool = False,
        headers_only: bool = False,
    ) -> tuple[builtins.list[MemoryDocument], ValidationReport]:
        """Parse the corpus once, returning valid documents and visible diagnostics."""
        documents: builtins.list[MemoryDocument] = []
        report = ValidationReport()
        seen: set[str] = set()
        for path in self._paths(include_archived=include_archived):
            report.scanned += 1
            try:
                document = _parse_document(path, headers_only=headers_only)
                self._validate_document(document, writing=False)
            except Exception as exc:  # noqa: BLE001 - every malformed source is diagnosed
                report.errors.append(f"{path}: {type(exc).__name__}: {exc}")
                continue
            if document.id in seen:
                report.errors.append(f"{path}: duplicate id {document.id}")
                continue
            seen.add(document.id)
            if document.archived and not include_archived:
                continue
            documents.append(document)
        return documents, report

    def get(self, memory_id: str) -> MemoryDocument | None:
        self._validate_id(memory_id)
        path = self._find_path(memory_id)
        if path is None:
            return None
        try:
            return _parse_document(path)
        except (OSError, UnicodeError, ValueError, yaml.YAMLError):
            return None

    def create(self, document: MemoryDocument) -> MemoryDocument:
        self._validate_document(document, writing=True)
        self._ensure_root()
        with FileLock(self.lock_path):
            if self._find_path(document.id) is not None:
                raise ValueError(f"memory id already exists: {document.id}")
            document = replace(document, updated_at=utc_now(), tags=list(document.tags), sources=list(document.sources))
            filename = (
                f"legacy-{document.id}.md"
                if document.type == "legacy"
                else f"{_slug(document.name)}-{document.id}.md"
            )
            path = self._contained(self.root / filename)
            _atomic_write(path, _render_document(document))
            self._rebuild_index_locked()
        return replace(document, path=str(path))

    def write(
        self,
        *,
        name: str,
        description: str,
        type: str,
        content: str,
        target_id: str | None = None,
        confidence: float = 0.5,
        tags: Iterable[str] = (),
        sources: Iterable[str] = (),
        explicit: bool = False,
    ) -> MemoryDocument:
        if target_id is None:
            document = MemoryDocument(
                name=name,
                description=description,
                type=type,  # type: ignore[arg-type]
                content=content,
                confidence=confidence,
                tags=list(tags),
                sources=list(sources),
                explicit=explicit,
            )
            return self.create(document)
        return self.update(
            target_id,
            name=name,
            description=description,
            type=type,
            content=content,
            confidence=confidence,
            tags=list(tags),
            sources=list(sources),
            explicit=explicit,
        )

    def update(self, memory_id: str, **changes: Any) -> MemoryDocument:
        self._validate_id(memory_id)
        self._ensure_root()
        with FileLock(self.lock_path):
            path = self._find_path(memory_id)
            if path is None:
                raise KeyError(memory_id)
            current = _parse_document(path)
            allowed = {
                "name", "description", "type", "content", "confidence", "tags",
                "sources", "explicit", "archived", "verified_at", "legacy",
            }
            unknown = set(changes) - allowed
            if unknown:
                raise ValueError("unsupported memory fields: " + ", ".join(sorted(unknown)))
            updated = replace(current, **changes, updated_at=utc_now(), path=str(path))
            self._validate_document(updated, writing=True)
            _atomic_write(path, _render_document(updated))
            self._rebuild_index_locked()
        return updated

    def archive(self, memory_id: str) -> MemoryDocument:
        self._ensure_root()
        with FileLock(self.lock_path):
            path = self._find_path(memory_id)
            if path is None:
                raise KeyError(memory_id)
            document = replace(_parse_document(path), archived=True, updated_at=utc_now())
            archive_path = self._contained(self.root / "archive" / path.name)
            _atomic_write(archive_path, _render_document(document))
            path.unlink()
            self._rebuild_index_locked()
        return replace(document, path=str(archive_path))

    def forget(self, memory_id: str) -> bool:
        self._validate_id(memory_id)
        self._ensure_root()
        with FileLock(self.lock_path):
            path = self._find_path(memory_id)
            if path is None:
                return False
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            trash = self._contained(self.root / ".trash" / f"{path.stem}-{stamp}.md")
            trash.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, trash)
            self._rebuild_index_locked()
        return True

    def apply(self, changes: Iterable[MemoryChange]) -> builtins.list[MemoryDocument]:
        results: builtins.list[MemoryDocument] = []
        for change in changes:
            if change.scope != self.scope:
                raise ValueError(f"change scope {change.scope!r} does not match repository {self.scope!r}")
            if change.operation == "create":
                results.append(
                    self.write(
                        name=change.name or "memory",
                        description=change.description or "",
                        type=change.type or "project",
                        content=change.content or "",
                        confidence=change.confidence if change.confidence is not None else 0.5,
                        tags=change.tags,
                        sources=change.sources,
                        explicit=change.explicit,
                    )
                )
            elif change.operation == "update":
                if not change.target_id:
                    raise ValueError("update requires target_id")
                values = {
                    key: value
                    for key, value in {
                        "name": change.name,
                        "description": change.description,
                        "type": change.type,
                        "content": change.content,
                        "confidence": change.confidence,
                        "tags": change.tags or None,
                        "sources": change.sources or None,
                        "explicit": change.explicit if change.explicit else None,
                    }.items()
                    if value is not None
                }
                results.append(self.update(change.target_id, **values))
            elif change.operation == "archive":
                if not change.target_id:
                    raise ValueError("archive requires target_id")
                results.append(self.archive(change.target_id))
            elif change.operation == "forget":
                if not change.target_id:
                    raise ValueError("forget requires target_id")
                self.forget(change.target_id)
            else:
                raise ValueError(f"unsupported memory operation: {change.operation}")
        return results

    def replace_records(self, records: builtins.list[MemoryRecord]) -> None:
        """Transactionally replace active topics for consolidation, archiving removals."""
        self._ensure_root()
        with FileLock(self.lock_path):
            current = {document.id: document for document in self.list()}
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            stage = self.root / f".dream-staging-{stamp}"
            backup = self.root / f".dream-backup-{stamp}"
            stage.mkdir()
            backup.mkdir()
            try:
                for record in records:
                    existing = current.get(record.id)
                    if existing is None:
                        content_preview = " ".join(record.content.split())
                        document = MemoryDocument(
                            id=record.id,
                            name=content_preview[:60] or "memory",
                            description=content_preview[:160] or "Consolidated memory",
                            type="reference" if record.kind in {"insight", "summary"} else "project",
                            content=record.content,
                            confidence=record.importance,
                            tags=list(record.tags),
                            sources=[record.source_run_id] if record.source_run_id else [],
                        )
                        filename = f"{_slug(document.name)}-{document.id}.md"
                    else:
                        document = replace(
                            existing,
                            content=record.content,
                            confidence=record.importance,
                            tags=list(record.tags),
                            sources=[record.source_run_id] if record.source_run_id else existing.sources,
                            updated_at=utc_now(),
                            legacy={
                                **existing.legacy,
                                "kind": record.kind,
                                "importance": record.importance,
                                "last_accessed_at": record.last_accessed_at,
                                "access_count": record.access_count,
                                "source_run_id": record.source_run_id,
                            },
                        )
                        filename = Path(existing.path or "").name
                    self._validate_document(document, writing=True)
                    _atomic_write(stage / filename, _render_document(document))
                active_paths = self._paths()
                for path in active_paths:
                    os.replace(path, backup / path.name)
                try:
                    for path in stage.glob("*.md"):
                        os.replace(path, self.root / path.name)
                    self._rebuild_index_locked()
                except BaseException:
                    for path in self._paths():
                        path.unlink()
                    for path in backup.glob("*.md"):
                        os.replace(path, self.root / path.name)
                    self._rebuild_index_locked()
                    raise
                survivor_ids = {record.id for record in records}
                archive_root = self.root / "archive"
                archived_paths: builtins.list[Path] = []
                try:
                    for memory_id, document in current.items():
                        if memory_id in survivor_ids:
                            continue
                        old_path = backup / Path(document.path or "").name
                        if not old_path.exists():
                            continue
                        archived = replace(document, archived=True, updated_at=utc_now())
                        archive_path = archive_root / old_path.name
                        _atomic_write(archive_path, _render_document(archived))
                        archived_paths.append(archive_path)
                except BaseException:
                    for path in self._paths():
                        path.unlink()
                    for path in backup.glob("*.md"):
                        os.replace(path, self.root / path.name)
                    for path in archived_paths:
                        path.unlink(missing_ok=True)
                    self._rebuild_index_locked()
                    raise
            finally:
                shutil.rmtree(stage, ignore_errors=True)
                shutil.rmtree(backup, ignore_errors=True)

    def search(self, query: str, limit: int = 5) -> builtins.list[MemoryDocument]:
        """Compatibility document view over the passage-first retrieval contract."""
        from agent_core.memory.models import MemorySearchRequest

        hits = self.search_hits(
            MemorySearchRequest.from_values(
                query,
                scope=self.scope,
                limit=limit,
                include_content=True,
            )
        )
        documents: builtins.list[MemoryDocument] = []
        for hit in hits:
            document = self.get(hit.id)
            if document is not None:
                documents.append(document)
        return documents

    def search_hits(
        self,
        request: "MemorySearchRequest",
        *,
        config: "MemoryConfig | None" = None,
        embedding_backend: "EmbeddingBackend | None" = None,
        reranker_backend: "RerankerBackend | None" = None,
        index_base: str | Path | None = None,
    ) -> builtins.list["MemorySearchHit"]:
        """Search passages without making SQLite an authoritative data source."""
        from agent_core.memory.retrieval import HybridMemoryRetriever

        retriever = HybridMemoryRetriever(
            self,
            config,
            embedding_backend=embedding_backend,
            reranker_backend=reranker_backend,
            index_base=index_base,
        )
        return retriever.search(request)

    def rebuild_index(self) -> str:
        self._ensure_root()
        with FileLock(self.lock_path):
            return self._rebuild_index_locked()

    def _rebuild_index_locked(self) -> str:
        documents = self.list(headers_only=True)
        lines = [
            "# Long-term memory",
            "",
            "Historical project context only. Verify current files, configuration, and runtime state before use.",
            "",
        ]
        for document in sorted(documents, key=lambda item: item.updated_at, reverse=True):
            description = " ".join(document.description.split())[:240]
            filename = Path(document.path or "").name
            line = f"- [{document.name}]({filename}) — {description}"
            candidate = "\n".join([*lines, line, ""])
            if len(lines) + 2 > _INDEX_MAX_LINES or len(candidate.encode("utf-8")) > _INDEX_MAX_BYTES:
                break
            lines.append(line)
        text = "\n".join(lines).rstrip() + "\n"
        _atomic_write(self.index_path, text)
        return text

    def index_text(self) -> str:
        if not self.index_path.exists():
            return self.rebuild_index()
        text = self.index_path.read_text(encoding="utf-8")
        if len(text.splitlines()) > _INDEX_MAX_LINES or len(text.encode("utf-8")) > _INDEX_MAX_BYTES:
            return self.rebuild_index()
        return text

    def validate(self, *, repair: bool = False) -> ValidationReport:
        report = ValidationReport()
        seen: set[str] = set()
        for path in self._paths(include_archived=True):
            report.scanned += 1
            try:
                document = _parse_document(path)
                self._validate_document(document, writing=False)
            except Exception as exc:  # noqa: BLE001 - validation reports every malformed file
                report.errors.append(f"{path}: {type(exc).__name__}: {exc}")
                continue
            if document.id in seen:
                report.errors.append(f"{path}: duplicate id {document.id}")
            seen.add(document.id)
        if self.index_path.exists():
            text = self.index_path.read_text(encoding="utf-8")
            if len(text.splitlines()) > _INDEX_MAX_LINES or len(text.encode("utf-8")) > _INDEX_MAX_BYTES:
                report.errors.append("MEMORY.md exceeds its 200-line/25KB budget")
        else:
            report.warnings.append("MEMORY.md is missing")
        if repair:
            self.rebuild_index()
            report.index_rebuilt = True
        return report

    def migrate_jsonl(self, source: str | Path) -> MigrationReport:
        source_path = Path(source)
        raw = source_path.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        report = MigrationReport(str(source_path), checksum)
        self._ensure_root()
        marker = self.root / ".legacy-migration.json"
        with FileLock(self.lock_path):
            if marker.exists():
                try:
                    old = json.loads(marker.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    old = {}
                if old.get("checksum") == checksum and old.get("complete") is True:
                    report.total = int(old.get("total", 0))
                    report.imported = int(old.get("imported", 0))
                    report.skipped = int(old.get("skipped", 0))
                    report.corrupt_lines = list(old.get("corrupt_lines", []))
                    report.already_complete = True
                    return report
            staging = self.root / f".migration-staging-{checksum[:12]}"
            staging.mkdir(parents=True, exist_ok=True)
            staged: list[tuple[Path, Path]] = []
            for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                report.total += 1
                try:
                    data = json.loads(line)
                    if not isinstance(data, dict) or not str(data.get("content", "")).strip():
                        raise ValueError("record must be an object with content")
                    memory_id = str(data.get("id") or hashlib.sha256(line.encode()).hexdigest()[:12])
                    self._validate_id(memory_id)
                    destination = self.root / f"legacy-{memory_id}.md"
                    if destination.exists():
                        report.skipped += 1
                        continue
                    created = str(data.get("created_at", ""))
                    try:
                        created_iso = datetime.fromtimestamp(float(created), UTC).isoformat().replace("+00:00", "Z")
                    except (TypeError, ValueError, OSError):
                        created_iso = utc_now()
                    document = MemoryDocument(
                        id=memory_id,
                        name=f"Legacy memory {memory_id}",
                        description="Imported without loss from the previous JSONL memory store.",
                        type="legacy",
                        content=str(data["content"]),
                        created_at=created_iso,
                        updated_at=created_iso,
                        confidence=max(0.0, min(1.0, float(data.get("importance", 0.5)))),
                        tags=[str(value) for value in data.get("tags", [])],
                        sources=[str(data["source_run_id"])] if data.get("source_run_id") else [],
                        explicit=False,
                        legacy={
                            "kind": data.get("kind", "fact"),
                            "importance": data.get("importance", 0.5),
                            "last_accessed_at": data.get("last_accessed_at"),
                            "access_count": data.get("access_count", 0),
                            "source_run_id": data.get("source_run_id"),
                        },
                    )
                    stage_path = staging / destination.name
                    _atomic_write(stage_path, _render_document(document))
                    staged.append((stage_path, destination))
                except (json.JSONDecodeError, TypeError, ValueError):
                    report.corrupt_lines.append(line_number)
            for stage_path, destination in staged:
                os.replace(stage_path, destination)
                report.imported += 1
            shutil.rmtree(staging, ignore_errors=True)
            self._rebuild_index_locked()
            _atomic_write(
                marker,
                json.dumps(
                    {
                        "schema_version": 1,
                        "source": str(source_path.resolve()),
                        "checksum": checksum,
                        "complete": True,
                        "total": report.total,
                        "imported": report.imported,
                        "skipped": report.skipped,
                        "corrupt_lines": report.corrupt_lines,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            )
        return report

    def _find_path(self, memory_id: str) -> Path | None:
        for path in self._paths(include_archived=True):
            try:
                if _parse_document(path, headers_only=True).id == memory_id:
                    return path
            except (OSError, UnicodeError, ValueError, yaml.YAMLError):
                continue
        return None

    def _contained(self, path: Path) -> Path:
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValueError("memory file escapes repository root") from exc
        return resolved

    @staticmethod
    def _validate_id(memory_id: str) -> None:
        if not _SAFE_ID.fullmatch(memory_id):
            raise ValueError(f"unsafe memory id: {memory_id!r}")

    def _validate_document(self, document: MemoryDocument, *, writing: bool) -> None:
        self._validate_id(document.id)
        if document.schema_version != 1:
            raise ValueError(f"unsupported schema_version: {document.schema_version}")
        if document.type not in {*MEMORY_TYPES, "legacy"}:
            raise ValueError(f"unsupported memory type: {document.type}")
        if writing and document.type == "legacy" and not document.id:
            raise ValueError("legacy documents can only be created during migration")
        if not document.name.strip() or not document.description.strip():
            raise ValueError("name and description are required")
        if len(document.content.encode("utf-8")) > _CONTENT_MAX_BYTES:
            raise ValueError("memory content exceeds 64KB")
        require_secret_free(document.name, document.description, document.content, *document.tags)
