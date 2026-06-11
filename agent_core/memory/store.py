from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from agent_core.memory.models import MemoryRecord


class MemoryStore:
    """JSONL-backed persistence for :class:`MemoryRecord`s.

    Mirrors :class:`agent_core.storage.JSONLRunLogger` conventions (utf-8,
    ``ensure_ascii=False``) but, unlike the append-only run log, memories are
    *mutable* (importance decays, access counts grow, dreaming merges them). So the
    store keeps the authoritative state in memory and persists with an atomic
    rewrite via :meth:`flush`; individual mutating calls flush by default.

    Mutating methods are async: the disk rewrite runs on a worker thread (so the
    event loop never blocks on file IO) and an ``asyncio.Lock`` serializes flushes —
    a full-file rewrite is not concurrency-safe, so concurrent mutations are made
    to wait their turn. In-memory reads stay synchronous.
    """

    def __init__(self, path: str | Path = "memory/memory.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, MemoryRecord] = {}
        # Created lazily so the lock binds to the running event loop, not whatever
        # loop (if any) existed at construction.
        self._flush_lock: asyncio.Lock | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = MemoryRecord.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError):
                # A corrupt line must not sink the whole store; skip it.
                continue
            self._records[record.id] = record

    # --- reads ----------------------------------------------------------------

    def all(self) -> list[MemoryRecord]:
        return list(self._records.values())

    def get(self, record_id: str) -> MemoryRecord | None:
        return self._records.get(record_id)

    def __len__(self) -> int:
        return len(self._records)

    # --- writes ---------------------------------------------------------------

    async def add(
        self,
        content: str,
        *,
        kind: str = "fact",
        importance: float = 0.5,
        tags: list[str] | None = None,
        source_run_id: str | None = None,
        flush: bool = True,
    ) -> MemoryRecord:
        record = MemoryRecord(
            content=content.strip(),
            kind=kind,
            importance=max(0.0, min(1.0, importance)),
            tags=list(tags or []),
            source_run_id=source_run_id,
        )
        self._records[record.id] = record
        if flush:
            await self.flush()
        return record

    async def update(self, record: MemoryRecord, *, flush: bool = True) -> None:
        self._records[record.id] = record
        if flush:
            await self.flush()

    async def delete(self, record_id: str, *, flush: bool = True) -> bool:
        existed = self._records.pop(record_id, None) is not None
        if existed and flush:
            await self.flush()
        return existed

    async def touch(self, record_id: str, *, flush: bool = True) -> None:
        """Record that a memory was recalled: bump access count + recency.

        Recall reinforces a memory, which in turn protects it from being forgotten
        during dreaming — the more a memory proves useful, the longer it survives.
        """
        record = self._records.get(record_id)
        if record is None:
            return
        record.access_count += 1
        record.last_accessed_at = time.time()
        if flush:
            await self.flush()

    async def replace_all(self, records: list[MemoryRecord], *, flush: bool = True) -> None:
        """Swap the entire contents (used by dreaming to commit a consolidated set)."""
        self._records = {record.id: record for record in records}
        if flush:
            await self.flush()

    async def flush(self) -> None:
        """Atomically rewrite the backing file off the event loop, one flush at a time."""
        if self._flush_lock is None:
            self._flush_lock = asyncio.Lock()
        async with self._flush_lock:
            # Snapshot on the loop so the worker thread never races a mutation.
            lines = [
                json.dumps(record.to_dict(), ensure_ascii=False, default=str)
                for record in self._records.values()
            ]
            await asyncio.to_thread(self._flush_sync, lines)

    def _flush_sync(self, lines: list[str]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as file:
            for line in lines:
                file.write(line + "\n")
        os.replace(tmp, self.path)
