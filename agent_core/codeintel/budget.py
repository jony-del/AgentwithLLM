from __future__ import annotations

import time
import threading
from typing import Any

from agent_core.codeintel.config import CodeIntelConfig


class BudgetExceeded(RuntimeError):
    pass


class QueryBudget:
    """One consumable budget for every stage, including fallback and verification."""

    def __init__(self, config: CodeIntelConfig, scope: Any = None) -> None:
        self.config = config
        self.scope = scope
        self.started = time.monotonic()
        self.deadline = self.started + config.query_seconds
        if scope is not None and scope.deadline is not None:
            self.deadline = min(self.deadline, scope.deadline)
        self.files = 0
        self.bytes = 0
        self.candidates = 0
        self.edges = 0
        self.output_bytes = 0
        self.cancelled = threading.Event()

    def check(self) -> None:
        if self.cancelled.is_set():
            raise BudgetExceeded("cancelled")
        if self.scope is not None:
            self.scope.raise_if_cancelled()
        if time.monotonic() >= self.deadline:
            raise BudgetExceeded("deadline")

    def consume(self, *, files: int = 0, bytes: int = 0, candidates: int = 0, edges: int = 0) -> None:
        self.check()
        for key, amount, limit in (
            ("files", files, self.config.max_files), ("bytes", bytes, self.config.max_bytes),
            ("candidates", candidates, self.config.max_candidates), ("edges", edges, self.config.max_edges),
        ):
            if getattr(self, key) + amount > limit:
                raise BudgetExceeded(f"max_{key}")
        self.files += files
        self.bytes += bytes
        self.candidates += candidates
        self.edges += edges

    def output(self, text: str) -> None:
        # UTF-8 bytes provide a deliberately conservative token upper bound.
        size = len(text.encode("utf-8"))
        if self.output_bytes + size > self.config.max_context_tokens:
            raise BudgetExceeded("context_tokens")
        self.output_bytes += size

    def usage(self) -> dict[str, int | float]:
        return {"files": self.files, "bytes": self.bytes, "candidates": self.candidates,
                "edges": self.edges, "output_bytes": self.output_bytes,
                "elapsed_ms": round((time.monotonic() - self.started) * 1000, 2)}
