from __future__ import annotations

from collections import OrderedDict
import threading


class CodeCache:
    """Content-addressed line cache; every consumer still verifies the file hash."""

    def __init__(self, max_entries: int = 128, max_bytes: int = 8 * 1024 * 1024) -> None:
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._size = 0
        self._items: OrderedDict[str, tuple[list[str], int]] = OrderedDict()
        self._lock = threading.Lock()

    def lines(self, digest: str, data: bytes) -> list[str]:
        with self._lock:
            existing = self._items.pop(digest, None)
            if existing is not None:
                self._items[digest] = existing
                return existing[0]
            lines = data.decode("utf-8").splitlines()
            # Conservative allowance for Python strings/list entries, including short lines.
            size = len(data) * 4 + len(lines) * 80
            if size <= self.max_bytes:
                self._items[digest] = (lines, size)
                self._size += size
                while len(self._items) > self.max_entries or self._size > self.max_bytes:
                    _, (_, removed) = self._items.popitem(last=False)
                    self._size -= removed
            return lines
