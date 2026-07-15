from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

from agent_core.permission_audit import sanitize_log_payload


# Schema version stamped on every event record (the "v" field), so replay/analysis
# tooling can detect format changes instead of guessing. Bump on breaking changes to
# the record shape and keep readers tolerant of older versions.
SCHEMA_VERSION = 2


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
        # Held append handle, opened lazily on first write and flushed per line.
        # Measured on the primary dev box (Windows, E7 benchmark 2026-07-03):
        # open-per-event ~2.7k events/s vs held-handle+flush ~52k events/s (~19x),
        # so the handle is kept open for the logger's lifetime. Flushing every line
        # keeps the file readable mid-run (tests, tail -f, replay) and loses nothing
        # on a hard exit.
        self._file: TextIO | None = None

    async def write(self, event: str, payload: dict[str, Any]) -> None:
        """Append one event without blocking the event loop.

        The actual locked file append runs on a worker thread; the ``threading.Lock``
        keeps concurrent appends (from overlapping ``to_thread`` workers) atomic.
        """
        await asyncio.to_thread(self.write_nowait, event, payload)

    def write_nowait(self, event: str, payload: dict[str, Any]) -> None:
        """Append one event synchronously from a synchronous control-path callback.

        Runtime permission-mode changes originate inside prompt_toolkit key handlers,
        where awaiting is not possible.  They are tiny, infrequent control events; the
        same lock/flush path keeps them ordered with ordinary asynchronous records.
        """
        record = {
            "ts": time.time(),
            "v": SCHEMA_VERSION,
            "event": event,
            **sanitize_log_payload(payload),
        }
        self._write_sync(record)

    def _write_sync(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            if self._file is None or self._file.closed:
                self._file = self.path.open("a", encoding="utf-8", errors="replace")
            self._file.write(line)
            self._file.flush()

    def close(self) -> None:
        """Release the held file handle. Idempotent; a later write reopens it."""
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                finally:
                    self._file = None

    def __del__(self) -> None:  # best-effort: flush-per-line means nothing is lost
        try:
            self.close()
        except Exception:
            pass


def read_events(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield the event records of a ``runs/*.jsonl`` file, oldest first.

    The reading half of the logger (used by ``polaris replay``). Tolerant by
    design: records without a ``"v"`` field (pre-v1) pass through unchanged, and a
    line that fails to parse is surfaced as a synthetic ``{"event": "_unparseable"}``
    record instead of being silently dropped or crashing the replay.
    """
    with Path(path).open("r", encoding="utf-8", errors="replace") as file:
        for lineno, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                yield {"event": "_unparseable", "line": lineno, "raw": line[:200]}
                continue
            if isinstance(record, dict):
                yield record
            else:
                yield {"event": "_unparseable", "line": lineno, "raw": line[:200]}
