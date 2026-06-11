from __future__ import annotations

import json
from typing import Any

from agent_core.memory.config import MemoryConfig
from agent_core.memory.models import MEMORY_KINDS, MemoryRecord
from agent_core.memory.store import MemoryStore
from agent_core.memory.text import lexical_relevance, tokenize
from agent_core.models import Message
from agent_core.providers.base import LLMProvider

# Embedded verbatim in the extraction/dreaming system prompts. Providers can detect
# it to behave deterministically (FakeProvider does), and it documents intent inline.
MEMORY_EXTRACTION_MARKER = "<<MEMORY_EXTRACTION>>"

_EXTRACTION_SYSTEM_PROMPT = f"""{MEMORY_EXTRACTION_MARKER}
You distil durable, reusable memories from a conversation. Capture only things worth
remembering for *future, separate* conversations: stable user preferences, facts about
the user or their projects, and decisions — not transient task chatter, not greetings,
not anything already obvious.

Respond with ONLY a JSON array (no prose). Each item:
  {{"content": str, "kind": one of {list(MEMORY_KINDS)}, "importance": 0.0-1.0, "tags": [str]}}
Return [] if nothing is worth remembering."""


def parse_memory_items(text: str) -> list[dict[str, Any]]:
    """Tolerantly pull a JSON array of memory items out of an LLM response.

    Models often wrap JSON in prose or code fences, so we slice from the first ``[``
    to the last ``]`` before parsing. Any malformed response yields ``[]`` rather
    than raising — a bad extraction must never break the run that triggered it.
    """
    if not text:
        return []
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


class MemoryExtractor:
    """Turns a finished conversation into stored :class:`MemoryRecord`s via the LLM."""

    def __init__(
        self,
        provider: LLMProvider,
        store: MemoryStore,
        config: MemoryConfig | None = None,
        provider_config: dict[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.store = store
        self.config = config or MemoryConfig()
        self.provider_config = provider_config or {}

    async def extract(self, messages: list[Message], source_run_id: str | None = None) -> list[MemoryRecord]:
        """Distil and store durable memories from a finished conversation.

        The LLM call goes through the provider's async ``complete``: when invoked from the
        agent's loop it flows through the shared ``GatedProvider`` (concurrency cap +
        rate limit). ``should_cancel`` is intentionally not forwarded: only
        ``GatedProvider`` accepts it, and post-run extraction is best-effort regardless.
        """
        request = self._build_request(messages)
        if request is None:
            return []
        result = await self.provider.complete(request, [], self.provider_config)
        return await self._store_items(parse_memory_items(result.content), source_run_id)

    def _build_request(self, messages: list[Message]) -> list[Message] | None:
        transcript = self._transcript(messages)
        if not transcript:
            return None
        return [
            Message("system", _EXTRACTION_SYSTEM_PROMPT),
            Message("user", f"Conversation transcript:\n{transcript}"),
        ]

    async def _store_items(self, items: list[dict[str, Any]], source_run_id: str | None) -> list[MemoryRecord]:
        stored: list[MemoryRecord] = []
        for item in items:
            content = str(item.get("content", "")).strip()
            if not content or self._is_duplicate(content):
                continue
            kind = str(item.get("kind", "fact"))
            record = await self.store.add(
                content,
                kind=kind if kind in MEMORY_KINDS else "fact",
                importance=self._clamp_importance(item.get("importance")),
                tags=[str(tag) for tag in (item.get("tags") or []) if str(tag).strip()],
                source_run_id=source_run_id,
                flush=False,
            )
            stored.append(record)
        if stored:
            await self.store.flush()
        return stored

    def _is_duplicate(self, content: str) -> bool:
        tokens = tokenize(content)
        return any(
            lexical_relevance(tokens, tokenize(existing.content)) >= self.config.dedup_threshold
            for existing in self.store.all()
        )

    @staticmethod
    def _clamp_importance(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.5

    @staticmethod
    def _transcript(messages: list[Message]) -> str:
        # Only user/assistant turns carry rememberable signal; skip system prompts,
        # tool observations, and anything compression already rewrote.
        lines = [
            f"{message.role}: {message.content}"
            for message in messages
            if message.role in {"user", "assistant"}
            and message.content.strip()
            and message.metadata.get("compressed") is None
            and message.metadata.get("memory") is None
        ]
        return "\n".join(lines)
