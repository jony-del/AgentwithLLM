from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Protocol

from agent_core.memory.config import MemoryConfig
from agent_core.memory.models import MEMORY_KINDS, MemoryRecord
from agent_core.memory.security import require_secret_free
from agent_core.memory.text import lexical_relevance, tokenize
from agent_core.models import Message
from agent_core.providers.base import LLMProvider, ProviderConfig

# Embedded verbatim in the extraction/dreaming system prompts. Providers can detect
# it to behave deterministically (FakeProvider does), and it documents intent inline.
MEMORY_EXTRACTION_MARKER = "<<MEMORY_EXTRACTION>>"

_EXTRACTION_SYSTEM_PROMPT = f"""{MEMORY_EXTRACTION_MARKER}
You distil durable, reusable memories from a conversation. Capture only things worth
remembering for *future, separate* conversations: stable user preferences, facts about
the user or their projects, and decisions — not transient task chatter, not greetings,
not anything already obvious.

Respond with ONLY a JSON array (no prose). Each item is a change:
  {{"operation": "create"|"update"|"archive"|"forget", "target_id": str|null,
    "content": str, "kind": one of {list(MEMORY_KINDS)}, "importance": 0.0-1.0,
    "tags": [str]}}
Use update/archive/forget only when a valid target id was supplied in the transcript.
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


class ExtractionStore(Protocol):
    def all(self) -> list[MemoryRecord]: ...
    async def add(
        self, content: str, *, kind: str = "fact", importance: float = 0.5,
        tags: list[str] | None = None, source_run_id: str | None = None,
        flush: bool = True,
    ) -> MemoryRecord: ...
    async def flush(self) -> None: ...


class MemoryExtractor:
    """Turns a finished conversation into stored :class:`MemoryRecord`s via the LLM."""

    def __init__(
        self,
        provider: LLMProvider,
        store: ExtractionStore,
        config: MemoryConfig | None = None,
        provider_config: ProviderConfig | None = None,
    ) -> None:
        self.provider = provider
        self.store = store
        self.config = config or MemoryConfig()
        self.provider_config = provider_config or ProviderConfig()
        self._cursor_uuid: str | None = None
        self._cursor_by_key: dict[str, str] = {}
        repository = getattr(store, "repository", None)
        self._cursor_path: Path | None = (
            repository.root / ".extraction-cursors.json" if repository is not None else None
        )
        if self._cursor_path is not None and self._cursor_path.exists():
            try:
                loaded = json.loads(self._cursor_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._cursor_by_key = {
                        str(key): str(value)
                        for key, value in loaded.items()
                        if isinstance(value, str)
                    }
            except (OSError, json.JSONDecodeError):
                pass
        self._extract_lock: asyncio.Lock | None = None
        self._direct_write_since_extract = False

    def mark_direct_write(self) -> None:
        """Prevent automatic extraction from duplicating an explicit memory tool write."""
        self._direct_write_since_extract = True

    async def extract(
        self,
        messages: list[Message],
        source_run_id: str | None = None,
        *,
        cursor_key: str | None = None,
    ) -> list[MemoryRecord]:
        """Distil and store durable memories from a finished conversation.

        The LLM call goes through the provider's async ``complete``: when invoked from the
        agent's loop it flows through the shared ``GatedProvider`` (concurrency cap +
        rate limit). ``should_cancel`` is intentionally not forwarded: only
        ``GatedProvider`` accepts it, and post-run extraction is best-effort regardless.
        """
        if self._extract_lock is None:
            self._extract_lock = asyncio.Lock()
        async with self._extract_lock:
            key = cursor_key or source_run_id or "__default__"
            self._cursor_uuid = self._cursor_by_key.get(key)
            eligible = self._messages_after_cursor(messages)
            if not eligible:
                return []
            final_uuid = eligible[-1].uuid
            if self._direct_write_since_extract:
                self._direct_write_since_extract = False
                self._cursor_uuid = final_uuid
                await self._commit_cursor(key, final_uuid)
                return []
            request = self._build_request(eligible)
            if request is None:
                self._cursor_uuid = final_uuid
                await self._commit_cursor(key, final_uuid)
                return []
            result = await self.provider.complete(request, [], self.provider_config)
            stored = await self._store_items(parse_memory_items(result.content), source_run_id)
            # The cursor advances only after provider parsing and repository commit succeed.
            self._cursor_uuid = final_uuid
            await self._commit_cursor(key, final_uuid)
            return stored

    async def _commit_cursor(self, key: str, message_uuid: str) -> None:
        self._cursor_by_key[key] = message_uuid
        if self._cursor_path is None:
            return
        cursor_path = self._cursor_path

        def write() -> None:
            from agent_core.file_lock import FileLock
            from agent_core.memory.repository import _atomic_write

            lock_path = cursor_path.with_suffix(".lock")
            with FileLock(lock_path):
                try:
                    current = json.loads(cursor_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    current = {}
                if not isinstance(current, dict):
                    current = {}
                current[key] = message_uuid
                # Bound old session metadata; insertion order keeps the newest entries.
                bounded = dict(list(current.items())[-1000:])
                _atomic_write(
                    cursor_path,
                    json.dumps(bounded, ensure_ascii=False, indent=2) + "\n",
                )

        await asyncio.to_thread(write)

    def _messages_after_cursor(self, messages: list[Message]) -> list[Message]:
        eligible = [
            message
            for message in messages
            if message.role in {"user", "assistant"}
            and message.content.strip()
            and message.metadata.get("compressed") is None
            and message.metadata.get("memory") is None
            and message.metadata.get("pinned") is None
            and message.metadata.get("subagent") is None
        ]
        if self._cursor_uuid is None:
            return eligible
        for index, message in enumerate(eligible):
            if message.uuid == self._cursor_uuid:
                return eligible[index + 1 :]
        # A resumed/compacted chain may no longer contain the cursor. Processing the
        # visible chain is safer than silently skipping new durable information.
        return eligible

    def _build_request(self, messages: list[Message]) -> list[Message] | None:
        transcript = self._transcript(messages)
        if not transcript:
            return None
        return [
            Message("system", _EXTRACTION_SYSTEM_PROMPT),
            Message(
                "user",
                "Existing memory targets (id and short preview):\n"
                + "\n".join(
                    f"- {record.id} [{record.kind}] {' '.join(record.content.split())[:120]}"
                    for record in self.store.all()[:200]
                )
                + f"\n\nConversation transcript:\n{transcript}",
            ),
        ]

    async def _store_items(self, items: list[dict[str, Any]], source_run_id: str | None) -> list[MemoryRecord]:
        stored: list[MemoryRecord] = []
        for item in items:
            operation = str(item.get("operation", "create"))
            if operation != "create":
                changed = await self._apply_non_create(item)
                if changed is not None:
                    stored.append(changed)
                continue
            content = str(item.get("content", "")).strip()
            if not content or self._is_duplicate(content):
                continue
            require_secret_free(content, *[str(tag) for tag in (item.get("tags") or [])])
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

    async def _apply_non_create(self, item: dict[str, Any]) -> MemoryRecord | None:
        repository = getattr(self.store, "repository", None)
        target_id = str(item.get("target_id") or "")
        if repository is None or not target_id:
            return None
        operation = str(item.get("operation"))
        if operation == "forget":
            await asyncio.to_thread(repository.forget, target_id)
            return None
        if operation == "archive":
            document = await asyncio.to_thread(repository.archive, target_id)
            converter = getattr(self.store, "_record", None)
            return converter(document) if converter is not None else None
        if operation != "update":
            return None
        content = str(item.get("content", "")).strip()
        if not content:
            return None
        require_secret_free(content)
        document = await asyncio.to_thread(
            repository.update,
            target_id,
            content=content,
            confidence=self._clamp_importance(item.get("importance")),
            tags=[str(tag) for tag in (item.get("tags") or []) if str(tag).strip()],
        )
        converter = getattr(self.store, "_record", None)
        return converter(document) if converter is not None else None

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
        # tool observations, anything compression already rewrote, and injected meta
        # context (the pinned <system-reminder> userContext message is a *user* message
        # but carries no conversational signal — exclude it like the system prompt).
        lines = [
            f"{message.role}: {message.content}"
            for message in messages
            if message.role in {"user", "assistant"}
            and message.content.strip()
            and message.metadata.get("compressed") is None
            and message.metadata.get("memory") is None
            and message.metadata.get("pinned") is None
        ]
        return "\n".join(lines)
