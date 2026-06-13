from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from agent_core.models import Message

# Called after each compaction stage so a UI can drive a progress bar:
# (completed_stages, total_stages, event).
StageReporter = Callable[[int, int, "CompressionEvent"], None]


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


@dataclass
class CompressionPipeline:
    config: CompressionConfig = field(default_factory=CompressionConfig)

    def maybe_auto_compact(
        self, messages: list[Message], on_stage: StageReporter | None = None
    ) -> tuple[list[Message], list[CompressionEvent]]:
        if self._char_count(messages) < self.config.max_context_chars * self.config.auto_threshold_ratio:
            return messages, []
        return self.compact(messages, aggressive=False, on_stage=on_stage)

    def reactive_compact(
        self, messages: list[Message], on_stage: StageReporter | None = None
    ) -> tuple[list[Message], list[CompressionEvent]]:
        return self.compact(messages, aggressive=True, on_stage=on_stage)

    def compact(
        self, messages: list[Message], aggressive: bool = False, on_stage: StageReporter | None = None
    ) -> tuple[list[Message], list[CompressionEvent]]:
        stages = (self._snip, self._microcompact, self._context_collapse)
        events: list[CompressionEvent] = []
        current = messages
        for index, stage in enumerate(stages):
            current, event = stage(current, aggressive)
            events.append(event)
            if on_stage is not None:
                on_stage(index + 1, len(stages), event)
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

    def _context_collapse(self, messages: list[Message], aggressive: bool) -> tuple[list[Message], CompressionEvent]:
        before = self._char_count(messages)
        keep = max(4, self.config.collapsed_keep_recent // (2 if aggressive else 1))
        system_messages = [
            message
            for message in messages
            if message.role == "system" and message.metadata.get("compressed") != "context_collapse"
        ]
        conversation_messages = [
            message
            for message in messages
            if message.role != "system" or message.metadata.get("compressed") == "context_collapse"
        ]
        if len(conversation_messages) <= keep + 1:
            return messages, CompressionEvent("context_collapse", before, before)
        prefix = conversation_messages[:-keep]
        recent = conversation_messages[-keep:]
        summary = " | ".join(f"{m.role}: {m.content[:160]}" for m in prefix)
        collapsed = [
            *system_messages,
            Message(
                "system",
                f"Earlier conversation summary: {summary}",
                metadata={"compressed": "context_collapse", "messages_collapsed": len(prefix)},
            ),
            *recent,
        ]
        detail = f"collapsed {len(prefix)} msgs"
        return collapsed, CompressionEvent("context_collapse", before, self._char_count(collapsed), detail)

    @staticmethod
    def _char_count(messages: list[Message]) -> int:
        return sum(len(message.content) for message in messages)
