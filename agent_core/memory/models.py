from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

# A memory's ``kind`` is a free-form label, but these are the values the rest of the
# subsystem produces and reasons about. ``insight`` is reserved for the higher-level
# memories synthesised during dreaming; everything else is captured during extraction.
MEMORY_KINDS = ("fact", "preference", "episode", "insight", "summary")
MEMORY_TYPES = ("user", "feedback", "project", "reference")
MemoryType = Literal["user", "feedback", "project", "reference", "legacy"]
MemoryScope = Literal["private", "team", "user", "project", "local"]
MemoryOperation = Literal["create", "update", "archive", "forget"]


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

    ``importance`` is a 0..1 salience score used (together with relevance and
    recency) to rank recall and to decide what to forget during dreaming.
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
