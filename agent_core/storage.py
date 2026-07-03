from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any


# Schema version stamped on every event record (the "v" field), so replay/analysis
# tooling can detect format changes instead of guessing. Bump on breaking changes to
# the record shape and keep readers tolerant of older versions.
SCHEMA_VERSION = 1


class JSONLRunLogger:
    def __init__(self, run_dir: str | Path = "runs", run_id: str | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        self.path = self.run_dir / f"{self.run_id}.jsonl"
        # Guards interleaved writes across the worker threads ``write`` offloads to.
        # Each child agent gets its own logger/file, so this only serializes one
        # agent's own records — no cross-file contention.
        self._lock = threading.Lock()

    async def write(self, event: str, payload: dict[str, Any]) -> None:
        """Append one event without blocking the event loop.

        The actual locked file append runs on a worker thread; the ``threading.Lock``
        keeps concurrent appends (from overlapping ``to_thread`` workers) atomic.
        """
        record = {"ts": time.time(), "v": SCHEMA_VERSION, "event": event, **payload}
        await asyncio.to_thread(self._write_sync, record)

    def _write_sync(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8", errors="replace") as file:
                file.write(line)
