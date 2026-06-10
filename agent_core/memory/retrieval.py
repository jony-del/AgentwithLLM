from __future__ import annotations

import math
import time

from agent_core.memory.config import MemoryConfig
from agent_core.memory.models import MemoryRecord
from agent_core.memory.store import MemoryStore
from agent_core.memory.text import lexical_relevance, tokenize


class MemoryRetriever:
    """Selects the memories most worth surfacing for a given query.

    Scoring follows the generative-agents recipe, adapted to be dependency-free:
    a blend of *relevance* (lexical overlap with the query), *importance* (the
    stored salience), and *recency* (exponential decay since last access).
    """

    def __init__(self, store: MemoryStore, config: MemoryConfig | None = None) -> None:
        self.store = store
        self.config = config or MemoryConfig()

    def score(self, record: MemoryRecord, query_tokens: set[str], now: float) -> float:
        relevance = lexical_relevance(query_tokens, tokenize(record.content))
        age_hours = max(0.0, (now - record.last_accessed_at) / 3600.0)
        recency = math.exp(-self.config.recency_decay_per_hour * age_hours)
        return (
            self.config.w_relevance * relevance
            + self.config.w_importance * record.importance
            + self.config.w_recency * recency
        )

    def recall(self, query: str, k: int | None = None, *, touch: bool = True) -> list[MemoryRecord]:
        """Return up to ``k`` memories most salient to ``query``, best first.

        Memories with zero lexical relevance are excluded entirely — importance and
        recency only *rank* among things that are actually about the query, they
        don't drag in unrelated memories.
        """
        k = self.config.recall_k if k is None else k
        if k <= 0 or len(self.store) == 0:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        now = time.time()

        scored: list[tuple[float, MemoryRecord]] = []
        for record in self.store.all():
            if lexical_relevance(query_tokens, tokenize(record.content)) == 0.0:
                continue
            scored.append((self.score(record, query_tokens, now), record))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = [record for _, record in scored[:k]]
        if touch:
            for record in top:
                self.store.touch(record.id, flush=False)
            if top:
                self.store.flush()
        return top

    @staticmethod
    def format_block(records: list[MemoryRecord]) -> str:
        """Render recalled memories as a system-prompt block."""
        lines = [
            "Relevant memories from earlier conversations (most salient first). "
            "Use them only if they help with the current task:",
        ]
        for record in records:
            lines.append(f"- [{record.kind}] {record.content}")
        return "\n".join(lines)
