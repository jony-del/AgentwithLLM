from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from agent_core.memory.config import MemoryConfig
from agent_core.memory.extraction import MEMORY_EXTRACTION_MARKER, parse_memory_items
from agent_core.memory.models import MemoryRecord, DreamReport
from agent_core.memory.store import MemoryStore
from agent_core.memory.text import lexical_relevance, tokenize
from agent_core.models import Message
from agent_core.providers.base import LLMProvider

_SECONDS_PER_DAY = 86400.0

_INSIGHT_SYSTEM_PROMPT = f"""{MEMORY_EXTRACTION_MARKER}
You are consolidating an agent's long-term memory during an offline "dreaming" pass.
Given a list of existing memories, infer at most two NEW higher-level insights that
generalise or connect them — patterns the individual memories don't state outright.

Respond with ONLY a JSON array (no prose). Each item:
  {{"content": str, "kind": "insight", "importance": 0.0-1.0, "tags": [str]}}
Return [] if no genuinely new insight emerges."""


class Dreamer:
    """Offline memory consolidation: decay/forget, merge duplicates, synthesise insight.

    Loosely inspired by how sleep consolidates memory — weak traces fade, similar
    traces fuse, and new abstractions form across them. Pure-stdlib for the first two
    stages; the optional third stage uses the LLM.
    """

    def __init__(
        self,
        store: MemoryStore,
        config: MemoryConfig | None = None,
        provider: LLMProvider | None = None,
        provider_config: dict[str, Any] | None = None,
    ) -> None:
        self.store = store
        self.config = config or MemoryConfig()
        self.provider = provider
        self.provider_config = provider_config or {}

    def dream(self, *, commit: bool = True) -> DreamReport:
        """Run the consolidation pass and return a report.

        With ``commit=False`` (dry run) the report is computed exactly as normal but
        nothing is persisted — the store is left untouched, so callers can preview
        what dreaming *would* do. Note ``recall``-style access bookkeeping is not
        involved here, so a dry run has no side effects on the store at all.
        """
        report = DreamReport(scanned=len(self.store))
        survivors = self._consolidate(report)
        insights = self._synthesize_insights(survivors, report)
        if commit:
            self.store.replace_all([*survivors, *insights])
        return report

    async def adream(self, *, commit: bool = True) -> DreamReport:
        """Async counterpart to :meth:`dream`.

        The decay/merge stages are pure and reused as-is; only insight synthesis differs,
        going through ``acomplete`` so an in-loop caller shares the provider gate and does
        not block the event loop. The offline ``polaris dream`` CLI stays on :meth:`dream`.
        """
        report = DreamReport(scanned=len(self.store))
        survivors = self._consolidate(report)
        insights = await self._asynthesize_insights(survivors, report)
        if commit:
            self.store.replace_all([*survivors, *insights])
        return report

    def _consolidate(self, report: DreamReport) -> list[MemoryRecord]:
        """Run the two pure (no-LLM) stages: forgetting curve, then duplicate merge."""
        now = time.time()
        # Work on copies so the stages' in-place mutation never touches the stored
        # objects until (and unless) we commit — keeping dry runs side-effect-free.
        snapshot = [replace(record, tags=list(record.tags)) for record in self.store.all()]
        survivors = self._decay_and_forget(snapshot, now, report)
        return self._merge_duplicates(survivors, report)

    # --- stage 1: forgetting curve --------------------------------------------

    def _decay_and_forget(self, records: list[MemoryRecord], now: float, report: DreamReport) -> list[MemoryRecord]:
        half_life = max(0.1, self.config.importance_half_life_days)
        survivors: list[MemoryRecord] = []
        for record in records:
            age_days = max(0.0, (now - record.last_accessed_at) / _SECONDS_PER_DAY)
            decay = 0.5 ** (age_days / half_life)
            decayed = record.importance * decay
            # A memory is forgotten only if it has both faded below threshold AND
            # not been recalled enough to prove its worth.
            if decayed < self.config.forget_threshold and record.access_count < self.config.forget_min_access:
                report.forgotten += 1
                report.details.append(f"forgot [{record.kind}] {self._preview(record.content)}")
                continue
            record.importance = round(decayed, 4)
            survivors.append(record)
        return survivors

    # --- stage 2: merge near-duplicates ---------------------------------------

    def _merge_duplicates(self, records: list[MemoryRecord], report: DreamReport) -> list[MemoryRecord]:
        merged: list[MemoryRecord] = []
        consumed: set[str] = set()
        for record in records:
            if record.id in consumed:
                continue
            cluster = [record]
            tokens = tokenize(record.content)
            for other in records:
                if other.id == record.id or other.id in consumed:
                    continue
                if lexical_relevance(tokens, tokenize(other.content)) >= self.config.merge_threshold:
                    cluster.append(other)
                    consumed.add(other.id)
            if len(cluster) == 1:
                merged.append(record)
                continue
            merged.append(self._fuse(cluster))
            report.merged += len(cluster) - 1
            report.details.append(
                f"merged {len(cluster)} memories -> {self._preview(cluster[0].content)}"
            )
        return merged

    @staticmethod
    def _fuse(cluster: list[MemoryRecord]) -> MemoryRecord:
        # Keep the highest-importance memory as canonical; absorb the rest's
        # salience, tags, and access history so the survivor is strictly stronger.
        canonical = max(cluster, key=lambda record: record.importance)
        tags: list[str] = []
        for record in cluster:
            for tag in record.tags:
                if tag not in tags:
                    tags.append(tag)
        canonical.tags = tags
        canonical.access_count = sum(record.access_count for record in cluster)
        canonical.created_at = min(record.created_at for record in cluster)
        canonical.last_accessed_at = max(record.last_accessed_at for record in cluster)
        # Reinforce: fusing several traces into one makes the survivor more important.
        canonical.importance = min(1.0, canonical.importance + 0.05 * (len(cluster) - 1))
        return canonical

    # --- stage 3: insight synthesis (the "dream") -----------------------------

    def _synthesize_insights(self, survivors: list[MemoryRecord], report: DreamReport) -> list[MemoryRecord]:
        request = self._build_insight_request(survivors)
        if request is None:
            return []
        try:
            result = self.provider.complete(request, [], self.provider_config)
        except Exception as exc:  # noqa: BLE001 - dreaming is best-effort; never fail the pass
            report.details.append(f"insight synthesis skipped: {type(exc).__name__}")
            return []
        return self._records_from_insight_text(result.content, survivors, report)

    async def _asynthesize_insights(self, survivors: list[MemoryRecord], report: DreamReport) -> list[MemoryRecord]:
        request = self._build_insight_request(survivors)
        if request is None:
            return []
        try:
            result = await self.provider.acomplete(request, [], self.provider_config)
        except Exception as exc:  # noqa: BLE001 - dreaming is best-effort; never fail the pass
            report.details.append(f"insight synthesis skipped: {type(exc).__name__}")
            return []
        return self._records_from_insight_text(result.content, survivors, report)

    def _build_insight_request(self, survivors: list[MemoryRecord]) -> list[Message] | None:
        if not self.config.synthesize_insights or self.provider is None or len(survivors) < 2:
            return None
        listing = "\n".join(f"- [{record.kind}] {record.content}" for record in survivors)
        return [
            Message("system", _INSIGHT_SYSTEM_PROMPT),
            Message("user", f"Existing memories:\n{listing}"),
        ]

    def _records_from_insight_text(
        self, text: str, survivors: list[MemoryRecord], report: DreamReport
    ) -> list[MemoryRecord]:
        insights: list[MemoryRecord] = []
        existing = survivors
        for item in parse_memory_items(text):
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            tokens = tokenize(content)
            if any(
                lexical_relevance(tokens, tokenize(record.content)) >= self.config.dedup_threshold
                for record in existing + insights
            ):
                continue
            insights.append(
                MemoryRecord(
                    content=content,
                    kind="insight",
                    importance=self._clamp(item.get("importance"), default=0.6),
                    tags=[str(tag) for tag in (item.get("tags") or []) if str(tag).strip()],
                )
            )
        report.insights_added = len(insights)
        for insight in insights:
            report.details.append(f"insight: {self._preview(insight.content)}")
        return insights

    @staticmethod
    def _clamp(value: Any, *, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _preview(content: str, limit: int = 60) -> str:
        return content if len(content) <= limit else content[:limit] + "…"
