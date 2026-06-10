from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# A memory's ``kind`` is a free-form label, but these are the values the rest of the
# subsystem produces and reasons about. ``insight`` is reserved for the higher-level
# memories synthesised during dreaming; everything else is captured during extraction.
MEMORY_KINDS = ("fact", "preference", "episode", "insight", "summary")


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
