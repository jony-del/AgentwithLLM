from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any


class JSONLRunLogger:
    def __init__(self, run_dir: str | Path = "runs", run_id: str | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        self.path = self.run_dir / f"{self.run_id}.jsonl"
        # Guards interleaved writes when ordinary tools run in worker threads while the
        # async loop also logs. Each child agent gets its own logger/file, so this only
        # serializes one agent's own records — no cross-file contention.
        self._lock = threading.Lock()

    def write(self, event: str, payload: dict[str, Any]) -> None:
        record = {"ts": time.time(), "event": event, **payload}
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line)

