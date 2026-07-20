"""In-memory queue for messages entered while an agent run is active.

The queue deliberately stores each submission as its own object.  That preserves a
stable UUID for transcript identity even when several ordinary prompts are delivered
to the provider as one between-turn batch.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum

from agent_core.models import Message


class QueuePriority(IntEnum):
    """Lower values are drained first; FIFO is preserved inside each priority."""

    NOW = 0
    NEXT = 1
    LATER = 2

    @classmethod
    def parse(cls, value: str | "QueuePriority") -> "QueuePriority":
        if isinstance(value, cls):
            return value
        return cls[str(value).strip().upper()]


@dataclass(slots=True)
class QueuedPrompt:
    content: str
    priority: QueuePriority = QueuePriority.NEXT
    mode: str = "prompt"
    origin: str = "interactive"
    editable: bool = True
    uuid: str = field(default_factory=lambda: uuid.uuid4().hex)
    enqueued_at: float = field(default_factory=time.monotonic)

    @property
    def is_slash_command(self) -> bool:
        stripped = self.content.lstrip()
        return stripped.startswith("/") and not stripped.startswith("//")

    def to_message(self, *, delivery: str) -> Message:
        return Message(
            "user",
            self.content,
            metadata={
                "queued_command": True,
                "queue_priority": self.priority.name.lower(),
                "queue_mode": self.mode,
                "queue_origin": self.origin,
                "queue_delivery": delivery,
            },
            uuid=self.uuid,
        )


class PromptQueue:
    """Unlimited, priority-aware FIFO queue owned by one interactive chat."""

    def __init__(self) -> None:
        self._items: list[QueuedPrompt] = []

    def __len__(self) -> int:
        return len(self._items)

    def enqueue(
        self,
        content: str,
        *,
        priority: QueuePriority | str = QueuePriority.NEXT,
        mode: str = "prompt",
        origin: str = "interactive",
        editable: bool = True,
    ) -> QueuedPrompt:
        item = QueuedPrompt(
            content=content,
            priority=QueuePriority.parse(priority),
            mode=mode,
            origin=origin,
            editable=editable,
        )
        self._items.append(item)
        return item

    def snapshot(self) -> tuple[QueuedPrompt, ...]:
        return tuple(self._items)

    def clear(self) -> list[QueuedPrompt]:
        items, self._items = self._items, []
        return items

    def recall_editable(self) -> str:
        """Remove and return every editable item, joined for composer editing."""

        recalled = [item for item in self._items if item.editable]
        if not recalled:
            return ""
        ids = {item.uuid for item in recalled}
        self._items = [item for item in self._items if item.uuid not in ids]
        return "\n".join(item.content for item in recalled)

    def drain_midturn(self) -> list[Message]:
        """Drain ordinary prompts after a tool batch.

        Slash commands stay queued for between-turn dispatch.  NOW/NEXT are eligible
        for current-turn delivery; LATER intentionally waits for the next run.
        """

        ordered = sorted(self._items, key=lambda item: (item.priority, item.enqueued_at))
        eligible: list[QueuedPrompt] = []
        for item in ordered:
            if item.priority is QueuePriority.LATER or item.is_slash_command:
                break
            eligible.append(item)
        if not eligible:
            return []
        eligible.sort(key=lambda item: (item.priority, item.enqueued_at))
        ids = {item.uuid for item in eligible}
        self._items = [item for item in self._items if item.uuid not in ids]
        return [item.to_message(delivery="midturn") for item in eligible]

    def pop_between_turn(self) -> list[QueuedPrompt]:
        """Pop one dispatch unit.

        Slash commands are always individual.  Ordinary prompts with the same current
        priority and mode are batched while retaining their individual identities.
        """

        if not self._items:
            return []
        ordered = sorted(self._items, key=lambda item: (item.priority, item.enqueued_at))
        first = ordered[0]
        if first.is_slash_command:
            selected = [first]
        else:
            selected = []
            for item in ordered:
                if (
                    item.is_slash_command
                    or item.priority is not first.priority
                    or item.mode != first.mode
                ):
                    break
                selected.append(item)
        ids = {item.uuid for item in selected}
        self._items = [item for item in self._items if item.uuid not in ids]
        return selected
