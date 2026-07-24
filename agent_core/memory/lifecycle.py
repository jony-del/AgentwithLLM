from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from agent_core.file_lock import FileLock


@dataclass(frozen=True, slots=True)
class DreamEligibility:
    due: bool
    elapsed_hours: float
    new_sessions: int


class MemoryLifecycle:
    """Cross-process session counters for the 24-hour/5-session dream gate."""

    def __init__(self, root: str | Path, *, min_hours: float = 24.0, min_sessions: int = 5) -> None:
        self.root = Path(root)
        self.state_path = self.root / ".lifecycle.json"
        self.lock_path = self.root / ".lifecycle.lock"
        self.min_hours = min_hours
        self.min_sessions = min_sessions

    def _read(self) -> dict[str, float | int]:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            value = {}
        return {
            "last_dream_at": float(value.get("last_dream_at", time.time())),
            "new_sessions": int(value.get("new_sessions", 0)),
        }

    def _write(self, state: dict[str, float | int]) -> None:
        from agent_core.memory.repository import _atomic_write

        _atomic_write(self.state_path, json.dumps(state, indent=2) + "\n")

    def record_session(self, *, now: float | None = None) -> DreamEligibility:
        current = time.time() if now is None else now
        self.root.mkdir(parents=True, exist_ok=True)
        with FileLock(self.lock_path):
            state = self._read()
            state["new_sessions"] = int(state["new_sessions"]) + 1
            self._write(state)
            return self._eligibility(state, current)

    def eligibility(self, *, now: float | None = None) -> DreamEligibility:
        current = time.time() if now is None else now
        with FileLock(self.lock_path):
            return self._eligibility(self._read(), current)

    def mark_dream_succeeded(self, *, now: float | None = None) -> None:
        current = time.time() if now is None else now
        with FileLock(self.lock_path):
            self._write({"last_dream_at": current, "new_sessions": 0})

    def _eligibility(self, state: dict[str, float | int], now: float) -> DreamEligibility:
        elapsed = max(0.0, (now - float(state["last_dream_at"])) / 3600.0)
        sessions = int(state["new_sessions"])
        return DreamEligibility(
            due=elapsed >= self.min_hours and sessions >= self.min_sessions,
            elapsed_hours=elapsed,
            new_sessions=sessions,
        )
