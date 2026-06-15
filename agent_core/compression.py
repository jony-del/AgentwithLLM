from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from agent_core.models import Message

# Called after each compaction stage so a UI can drive a progress bar:
# (completed_stages, total_stages, event).
StageReporter = Callable[[int, int, "CompressionEvent"], None]

# Injected async callback that folds an old message prefix into a single summary
# string (Track A). It lives here — with the pipeline that consumes it — so the
# seam module (compression_summary) can depend on the pipeline without a cycle. The
# pipeline never imports a provider; it only awaits this opaque callback.
Summarizer = Callable[[list[Message]], Awaitable[str]]

# Markers stamped on a folded summary block (either track). A prior summary is itself
# foldable on the next pass, so repeated compactions re-summarize rather than pile up.
_COLLAPSE_MARKERS = frozenset({"context_collapse", "llm_summary"})


@dataclass(slots=True)
class CompressionEvent:
    stage: str
    before_chars: int
    after_chars: int
    detail: str = ""


@dataclass(slots=True)
class CompressionConfig:
    max_context_chars: int = 24000
    auto_threshold_ratio: float = 0.8
    max_message_chars: int = 6000
    collapsed_keep_recent: int = 8
    # Track A (LLM summary) knobs. When a real summarizer is injected and
    # use_llm_summary is on, _context_collapse folds the old prefix into a model
    # summary instead of the deterministic Track B join; any failure degrades to
    # Track B. With no summarizer (FakeProvider / no key) these are inert.
    use_llm_summary: bool = True
    summary_max_tokens: int = 2048
    summary_keep_recent: int = 8
    summary_input_max_chars: int = 16000


@dataclass
class CompressionPipeline:
    config: CompressionConfig = field(default_factory=CompressionConfig)

    async def auto_compact(
        self,
        messages: list[Message],
        *,
        summarizer: Summarizer | None = None,
        on_stage: StageReporter | None = None,
    ) -> tuple[list[Message], list[CompressionEvent]]:
        """Proactive compaction, gated on the history crossing the auto threshold.

        Public execution API is ``async`` (CLAUDE.md): the loop awaits it once per
        turn. The cheap shrink stages (snip/microcompact) are synchronous string
        operations run inline; the prefix-collapse step awaits ``summarizer`` when one
        is injected (Track A) and otherwise folds deterministically (Track B), which is
        why the entry point is a coroutine.
        """
        if self._char_count(messages) < self.config.max_context_chars * self.config.auto_threshold_ratio:
            return messages, []
        return await self._run_stages(messages, aggressive=False, summarizer=summarizer, on_stage=on_stage)

    async def reactive_compact(
        self,
        messages: list[Message],
        *,
        summarizer: Summarizer | None = None,
        on_stage: StageReporter | None = None,
    ) -> tuple[list[Message], list[CompressionEvent]]:
        """Aggressive compaction run after a context-overflow error; always compacts."""
        return await self._run_stages(messages, aggressive=True, summarizer=summarizer, on_stage=on_stage)

    async def _run_stages(
        self,
        messages: list[Message],
        *,
        aggressive: bool,
        summarizer: Summarizer | None,
        on_stage: StageReporter | None,
    ) -> tuple[list[Message], list[CompressionEvent]]:
        total = 3
        events: list[CompressionEvent] = []
        current = messages

        def report(done: int, event: CompressionEvent) -> None:
            events.append(event)
            if on_stage is not None:
                on_stage(done, total, event)

        # Stages 1–2 are deterministic and cheap; stage 3 (prefix collapse) may await
        # the injected summarizer (Track A) and so is the only async one.
        current, event = self._snip(current, aggressive)
        report(1, event)
        current, event = self._microcompact(current, aggressive)
        report(2, event)
        current, event = await self._context_collapse(current, aggressive, summarizer)
        report(3, event)
        return current, [event for event in events if event.before_chars != event.after_chars]

    def _snip(self, messages: list[Message], aggressive: bool) -> tuple[list[Message], CompressionEvent]:
        before = self._char_count(messages)
        limit = self.config.max_message_chars // (2 if aggressive else 1)
        snipped: list[Message] = []
        count = 0
        for message in messages:
            if message.role == "tool" and len(message.content) > limit:
                half = max(100, limit // 2)
                content = f"{message.content[:half]}\n[snip]\n{message.content[-half:]}"
                snipped.append(Message(message.role, content, message.name, {**message.metadata, "compressed": "snip"}))
                count += 1
            else:
                snipped.append(message)
        detail = f"snipped {count}" if count else ""
        return snipped, CompressionEvent("snip", before, self._char_count(snipped), detail)

    def _microcompact(self, messages: list[Message], aggressive: bool) -> tuple[list[Message], CompressionEvent]:
        before = self._char_count(messages)
        budget_limit = max(40, self.config.max_context_chars // (6 if aggressive else 4))
        limit = min(self.config.max_message_chars // (3 if aggressive else 2), budget_limit)
        compacted: list[Message] = []
        count = 0
        for message in messages:
            # Pinned blocks (e.g. injected CLAUDE.md) are kept verbatim — they are
            # one-time context that must survive compaction. _context_collapse already
            # preserves system messages, but microcompact truncates by length without
            # regard to role, so it needs its own guard.
            if message.metadata.get("pinned"):
                compacted.append(message)
                continue
            if len(message.content) > limit:
                content = f"{message.content[:limit]} [microcompact: omitted {len(message.content) - limit} chars]"
                compacted.append(Message(message.role, content, message.name, {**message.metadata, "compressed": "microcompact"}))
                count += 1
            else:
                compacted.append(message)
        detail = f"microcompacted {count}" if count else ""
        return compacted, CompressionEvent("microcompact", before, self._char_count(compacted), detail)

    async def _context_collapse(
        self, messages: list[Message], aggressive: bool, summarizer: Summarizer | None
    ) -> tuple[list[Message], CompressionEvent]:
        before = self._char_count(messages)
        use_llm = summarizer is not None and self.config.use_llm_summary
        # Track A keeps its own (configurable) recent window; Track B keeps the legacy
        # collapsed_keep_recent so a no-LLM history collapses byte-for-byte as before.
        base_keep = self.config.summary_keep_recent if use_llm else self.config.collapsed_keep_recent
        keep = max(4, base_keep // (2 if aggressive else 1))
        # A prior summary block (either track) is itself foldable, so repeated
        # compactions re-summarize it rather than accumulating stale summaries. Pure
        # Track-B histories never carry an "llm_summary" marker, so broadening the set
        # leaves their behavior unchanged.
        system_messages = [
            message
            for message in messages
            if message.role == "system" and message.metadata.get("compressed") not in _COLLAPSE_MARKERS
        ]
        conversation_messages = [
            message
            for message in messages
            if message.role != "system" or message.metadata.get("compressed") in _COLLAPSE_MARKERS
        ]
        if len(conversation_messages) <= keep + 1:
            return messages, CompressionEvent("context_collapse", before, before)
        prefix = conversation_messages[:-keep]
        recent = conversation_messages[-keep:]
        block, note = await self._collapse_prefix(prefix, summarizer if use_llm else None)
        collapsed = [*system_messages, block, *recent]
        detail = f"collapsed {len(prefix)} msgs ({note})"
        return collapsed, CompressionEvent("context_collapse", before, self._char_count(collapsed), detail)

    async def _collapse_prefix(
        self, prefix: list[Message], summarizer: Summarizer | None
    ) -> tuple[Message, str]:
        """Fold the old prefix into one block, returning it plus a track note.

        Track A (``summarizer`` present): ask the model for a structured summary.
        Any failure — exception, timeout, or an empty/blank result — degrades to the
        deterministic Track B join so the compaction always succeeds and the run never
        breaks. The note (``llm_summary`` / ``summary_fallback: ...`` / ``track_b``)
        flows into the compression log event for observability.
        """
        if summarizer is None:
            return self._collapse_prefix_naive(prefix), "track_b"
        try:
            summary = await summarizer(prefix)
            if not summary or not summary.strip():
                raise ValueError("empty summary")
        except Exception as exc:  # noqa: BLE001 - any summarizer failure must degrade, not crash
            return self._collapse_prefix_naive(prefix), f"summary_fallback: {type(exc).__name__}"
        block = Message(
            "system",
            f"Earlier conversation summary: {summary.strip()}",
            metadata={"compressed": "llm_summary", "messages_collapsed": len(prefix)},
        )
        return block, "llm_summary"

    @staticmethod
    def _collapse_prefix_naive(prefix: list[Message]) -> Message:
        """Deterministic Track B fold: join the old prefix into one summary block.

        Pure string manipulation, no model call — the offline/no-key default and the
        fallback when an LLM summary (Track A) is unavailable or fails."""
        summary = " | ".join(f"{m.role}: {m.content[:160]}" for m in prefix)
        return Message(
            "system",
            f"Earlier conversation summary: {summary}",
            metadata={"compressed": "context_collapse", "messages_collapsed": len(prefix)},
        )

    @staticmethod
    def _char_count(messages: list[Message]) -> int:
        return sum(len(message.content) for message in messages)
