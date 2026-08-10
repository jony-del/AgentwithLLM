"""Secret-safe append-only audit trail for capability lifecycle decisions."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping

from agent_core.file_lock import FileLock


_SECRET_KEY = re.compile(r"(?i)(?:secret|token|password|authorization|api[_-]?key|cookie)")


def _redact(value: Any, key: str = "") -> Any:
    if _SECRET_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(item_key): _redact(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, str) and value.casefold().startswith(("bearer ", "keychain://")):
        return "[REDACTED]"
    return value


class CapabilityAuditLog:
    def __init__(self, root: str | Path) -> None:
        self.path = Path(root) / "audit" / "capabilities.jsonl"
        self.lock_path = self.path.with_suffix(".lock")

    def write(self, event: str, detail: Mapping[str, Any] | None = None) -> None:
        record = {
            "schema_version": 1,
            "timestamp": time.time(),
            "event": str(event)[:100],
            "detail": _redact(dict(detail or {})),
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(self.lock_path):
                # Open in append mode only while holding the cross-process lock.  fsync
                # is intentional, but an unavailable user home must not sink a session.
                with self.path.open("a", encoding="utf-8", newline="") as stream:
                    stream.write(line)
                    stream.flush()
                    os.fsync(stream.fileno())
        except OSError:
            return
