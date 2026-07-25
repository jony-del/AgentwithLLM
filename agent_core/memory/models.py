from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

# A memory's ``kind`` is a free-form label, but these are the values the rest of the
# subsystem produces and reasons about. ``insight`` is reserved for the higher-level
# memories synthesised during dreaming; everything else is captured during extraction.
MEMORY_KINDS = ("fact", "preference", "episode", "insight", "summary")
MEMORY_TYPES = ("user", "feedback", "project", "reference")
MemoryType = Literal["user", "feedback", "project", "reference", "legacy"]
MemoryScope = Literal["private", "team", "user", "project", "local"]
MemoryOperation = Literal["create", "update", "archive", "forget"]


@dataclass(slots=True)
class MemorySearchRequest:
    """Public, serializable contract for one bounded memory search."""

    query: str
    scope: str = "private"
    limit: int = 5
    filters: dict[str, list[str]] = field(default_factory=dict)
    include_content: bool = False
    explain: bool = False

    def __post_init__(self) -> None:
        self.query = str(self.query)[:65_536]
        self.scope = str(self.scope)
        self.limit = max(0, min(20, int(self.limit)))
        normalized: dict[str, list[str]] = {}
        for key, raw in self.filters.items():
            values = [raw] if isinstance(raw, str) else list(raw)
            normalized[str(key)] = [
                str(value)[:1024] for value in values[:50] if str(value).strip()
            ]
        self.filters = normalized

    @classmethod
    def from_values(
        cls,
        query: str,
        *,
        scope: str = "private",
        limit: int = 5,
        filters: Mapping[str, str | Sequence[str]] | None = None,
        include_content: bool = False,
        explain: bool = False,
    ) -> "MemorySearchRequest":
        return cls(
            query=str(query),
            scope=str(scope),
            limit=int(limit),
            filters=dict(filters or {}),  # type: ignore[arg-type]
            include_content=bool(include_content),
            explain=bool(explain),
        )


@dataclass(slots=True)
class MemoryPassage:
    chunk_id: str
    content: str
    heading: str = ""
    ordinal: int = 0
    score: float = 0.0
    start_char: int = 0
    end_char: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "heading": self.heading,
            "ordinal": self.ordinal,
            "score": self.score,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "content": self.content,
        }


@dataclass(slots=True)
class RetrievalTrace:
    """Privacy-safe observability for retrieval; it never contains query or content."""

    mode: str = "hybrid"
    candidate_counts: dict[str, int] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    index_coverage: float = 0.0
    index_fingerprint: str = ""
    embedding_fingerprint: str = ""
    reranker_fingerprint: str = ""
    degraded_reasons: list[str] = field(default_factory=list)
    final_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "candidate_counts": dict(self.candidate_counts),
            "timings_ms": dict(self.timings_ms),
            "index_coverage": round(float(self.index_coverage), 6),
            "index_fingerprint": self.index_fingerprint,
            "embedding_fingerprint": self.embedding_fingerprint,
            "reranker_fingerprint": self.reranker_fingerprint,
            "degraded_reasons": list(self.degraded_reasons),
            "final_ids": list(self.final_ids),
        }


@dataclass(slots=True)
class MemorySearchHit:
    id: str
    name: str
    description: str
    type: str
    updated_at: str
    passages: list[MemoryPassage] = field(default_factory=list)
    score: float = 0.0
    rrf_score: float = 0.0
    exact: bool = False
    explicit_filter: bool = False
    confidence: float = 0.5
    verified: bool = False
    explicit: bool = False
    tags: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    content: str | None = None
    stale_until_verified: bool = True
    trace: RetrievalTrace | None = None

    def to_dict(self, *, include_trace: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "updated_at": self.updated_at,
            "score": self.score,
            "rrf_score": self.rrf_score,
            "exact": self.exact,
            "explicit_filter": self.explicit_filter,
            "confidence": self.confidence,
            "verified": self.verified,
            "explicit": self.explicit,
            "tags": list(self.tags),
            "sources": list(self.sources),
            "stale_until_verified": self.stale_until_verified,
            "passages": [passage.to_dict() for passage in self.passages],
        }
        if self.content is not None:
            result["content"] = self.content
        if include_trace and self.trace is not None:
            result["trace"] = self.trace.to_dict()
        return result


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Injectable dense encoder. Implementations must return one vector per text."""

    @property
    def fingerprint(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed(
        self,
        texts: Sequence[str],
        *,
        deadline: float | None = None,
    ) -> Sequence[Sequence[float]]: ...


@runtime_checkable
class RerankerBackend(Protocol):
    """Injectable cross-encoder returning probabilities or declared logits."""

    @property
    def fingerprint(self) -> str: ...

    def rerank(
        self,
        query: str,
        passages: Sequence[str],
        *,
        deadline: float | None = None,
    ) -> Sequence[float]: ...


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class MemoryDocument:
    """A human-editable, versioned long-term-memory topic."""

    name: str
    description: str
    content: str
    type: MemoryType = "project"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    schema_version: int = 1
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    confidence: float = 0.5
    tags: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    explicit: bool = False
    archived: bool = False
    verified_at: str | None = None
    legacy: dict[str, Any] = field(default_factory=dict)
    path: str | None = field(default=None, compare=False)

    def header(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "confidence": max(0.0, min(1.0, float(self.confidence))),
            "tags": list(self.tags),
            "sources": list(self.sources),
            "explicit": bool(self.explicit),
            "archived": bool(self.archived),
        }
        if self.verified_at:
            result["verified_at"] = self.verified_at
        if self.legacy:
            result["legacy"] = dict(self.legacy)
        return result

    @classmethod
    def from_parts(cls, header: dict[str, Any], content: str, *, path: str | None = None) -> "MemoryDocument":
        memory_type = str(header.get("type", "project"))
        if memory_type not in {*MEMORY_TYPES, "legacy"}:
            raise ValueError(f"unsupported memory type: {memory_type}")
        now = utc_now()
        return cls(
            schema_version=int(header.get("schema_version", 1)),
            id=str(header.get("id") or uuid.uuid4().hex[:12]),
            name=str(header.get("name") or "untitled"),
            description=str(header.get("description") or ""),
            type=memory_type,  # type: ignore[arg-type]
            created_at=str(header.get("created_at") or now),
            updated_at=str(header.get("updated_at") or header.get("created_at") or now),
            confidence=max(0.0, min(1.0, float(header.get("confidence", 0.5)))),
            tags=[str(value) for value in header.get("tags", [])],
            sources=[str(value) for value in header.get("sources", [])],
            explicit=bool(header.get("explicit", False)),
            archived=bool(header.get("archived", False)),
            verified_at=str(header["verified_at"]) if header.get("verified_at") else None,
            legacy=dict(header.get("legacy") or {}),
            content=content.strip(),
            path=path,
        )


@dataclass(slots=True)
class MemoryChange:
    operation: MemoryOperation
    scope: MemoryScope = "private"
    target_id: str | None = None
    name: str | None = None
    description: str | None = None
    type: MemoryType | None = None
    content: str | None = None
    confidence: float | None = None
    tags: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    explicit: bool = False


@dataclass(slots=True)
class MemoryRecord:
    """A single durable thing the agent has chosen to remember.

    ``importance`` is a 0..1 lifecycle salience score used by dreaming and
    forgetting. It is deliberately not mixed into retrieval relevance.
    """

    content: str
    kind: str = "fact"
    importance: float = 0.5
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    source_run_id: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "kind": self.kind,
            "importance": self.importance,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "source_run_id": self.source_run_id,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        # Tolerant of older/partial records: every field falls back to its default
        # so a hand-edited or schema-evolved memory file still loads.
        now = time.time()
        return cls(
            content=str(data.get("content", "")),
            kind=str(data.get("kind", "fact")),
            importance=float(data.get("importance", 0.5)),
            id=str(data.get("id") or uuid.uuid4().hex[:12]),
            created_at=float(data.get("created_at", now)),
            last_accessed_at=float(data.get("last_accessed_at", data.get("created_at", now))),
            access_count=int(data.get("access_count", 0)),
            source_run_id=data.get("source_run_id"),
            tags=list(data.get("tags") or []),
        )


@dataclass(slots=True)
class DreamReport:
    """Summary of one dreaming consolidation pass over the store."""

    scanned: int = 0
    forgotten: int = 0
    merged: int = 0
    insights_added: int = 0
    details: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "forgotten": self.forgotten,
            "merged": self.merged,
            "insights_added": self.insights_added,
            "details": list(self.details),
        }
