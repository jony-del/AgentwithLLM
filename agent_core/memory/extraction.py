from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Protocol

from agent_core.memory.config import MemoryConfig
from agent_core.memory.models import MEMORY_KINDS, MemoryRecord
from agent_core.memory.security import SecretDetectedError, require_secret_free
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


class PermanentExtractionItemError(ValueError):
    pass


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
        self._dead_letters: list[dict[str, Any]] = []
        repository = getattr(store, "repository", None)
        self._state_path: Path | None = (
            repository.root / ".extraction-state.json" if repository is not None else None
        )
        self._legacy_cursor_path: Path | None = (
            repository.root / ".extraction-cursors.json" if repository is not None else None
        )
        self._load_state()
        self.last_report: dict[str, int] = {
            "accepted": 0,
            "dead_lettered": 0,
            "skipped": 0,
        }
        self._extract_lock: asyncio.Lock | None = None
        self._direct_write_since_extract = False

    def _load_state(self) -> None:
        loaded: object = None
        if self._state_path is not None and self._state_path.exists():
            try:
                loaded = json.loads(self._state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        if isinstance(loaded, dict) and loaded.get("v") == 1:
            cursors = loaded.get("cursors")
            dead_letters = loaded.get("dead_letters")
            if isinstance(cursors, dict):
                self._cursor_by_key = {
                    str(key): value for key, value in cursors.items() if isinstance(value, str)
                }
            if isinstance(dead_letters, list):
                self._dead_letters = [item for item in dead_letters if isinstance(item, dict)]
            return
        # Compatibility import.  The legacy file is intentionally left untouched so a
        # failed migration can never destroy the only durable cursor copy.
        legacy = self._legacy_cursor_path
        if legacy is not None and legacy.exists():
            try:
                cursors = json.loads(legacy.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cursors = {}
            if isinstance(cursors, dict):
                self._cursor_by_key = {
                    str(key): value for key, value in cursors.items() if isinstance(value, str)
                }

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
            self.last_report = {"accepted": 0, "dead_lettered": 0, "skipped": 0}
            key = cursor_key or source_run_id or "__default__"
            self._cursor_uuid = self._cursor_by_key.get(key)
            eligible = self._messages_after_cursor(messages)
            if not eligible:
                return []
            final_uuid = eligible[-1].uuid
            if self._direct_write_since_extract:
                self._direct_write_since_extract = False
                await self._commit_cursor(key, final_uuid)
                self._cursor_uuid = final_uuid
                return []
            request = self._build_request(eligible)
            if request is None:
                await self._commit_cursor(key, final_uuid)
                self._cursor_uuid = final_uuid
                return []
            result = await self.provider.complete(request, [], self.provider_config)
            stored, dead_letters, skipped = await self._store_items(
                parse_memory_items(result.content), source_run_id, key, final_uuid
            )
            # Cursor and poison diagnostics commit together. Repository/IO failures still
            # abort before this point and deliberately leave the cursor unchanged.
            await self._commit_state(key, final_uuid, dead_letters)
            self._cursor_uuid = final_uuid
            self.last_report = {
                "accepted": len(stored),
                "dead_lettered": len(dead_letters),
                "skipped": skipped,
            }
            return stored

    async def _commit_cursor(self, key: str, message_uuid: str) -> None:
        await self._commit_state(key, message_uuid, [])

    async def _commit_state(
        self, key: str, message_uuid: str, dead_letters: list[dict[str, Any]]
    ) -> None:
        if self._state_path is None:
            self._cursor_by_key[key] = message_uuid
            self._dead_letters = (
                self._dead_letters + dead_letters
            )[-self.config.extraction_dead_letter_limit :]
            return
        state_path = self._state_path

        def write() -> None:
            from agent_core.file_lock import FileLock
            from agent_core.memory.repository import _atomic_write

            lock_path = state_path.with_suffix(".lock")
            with FileLock(lock_path):
                try:
                    current = json.loads(state_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    current = {}
                current_cursors = current.get("cursors", {}) if isinstance(current, dict) else {}
                current_dead = current.get("dead_letters", []) if isinstance(current, dict) else []
                if not isinstance(current_cursors, dict):
                    current_cursors = {}
                if not isinstance(current_dead, list):
                    current_dead = []
                current_cursors[key] = message_uuid
                bounded_cursors = dict(list(current_cursors.items())[-1000:])
                bounded_dead = [
                    item for item in [*current_dead, *dead_letters] if isinstance(item, dict)
                ][-self.config.extraction_dead_letter_limit :]
                state = {
                    "v": 1,
                    "cursors": bounded_cursors,
                    "dead_letters": bounded_dead,
                    "legacy_cursor_imported": bool(self._legacy_cursor_path),
                }
                _atomic_write(
                    state_path,
                    json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                )
                self._cursor_by_key = {
                    str(item_key): str(value)
                    for item_key, value in bounded_cursors.items()
                    if isinstance(value, str)
                }
                self._dead_letters = bounded_dead

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

    async def _store_items(
        self,
        items: list[dict[str, Any]],
        source_run_id: str | None,
        cursor_key: str,
        message_uuid: str,
    ) -> tuple[list[MemoryRecord], list[dict[str, Any]], int]:
        stored: list[MemoryRecord] = []
        dead_letters: list[dict[str, Any]] = []
        skipped = 0
        for ordinal, item in enumerate(items):
            operation = str(item.get("operation", "create"))
            try:
                if operation not in {"create", "update", "archive", "forget"}:
                    raise PermanentExtractionItemError("unsupported_operation")
                if operation != "create":
                    if not str(item.get("target_id") or ""):
                        raise PermanentExtractionItemError("missing_target_id")
                    changed = await self._apply_non_create(item)
                    if changed is not None:
                        stored.append(changed)
                    continue
                content = str(item.get("content", "")).strip()
                if not content:
                    raise PermanentExtractionItemError("empty_content")
                if self._is_duplicate(content):
                    skipped += 1
                    continue
                raw_tags = item.get("tags") or []
                if not isinstance(raw_tags, list):
                    raise PermanentExtractionItemError("invalid_tags")
                tags = [str(tag) for tag in raw_tags if str(tag).strip()]
                require_secret_free(content, *tags)
                kind = str(item.get("kind", "fact"))
                if kind not in MEMORY_KINDS:
                    raise PermanentExtractionItemError("unsupported_kind")
                record = await self.store.add(
                    content,
                    kind=kind,
                    importance=self._clamp_importance(item.get("importance")),
                    tags=tags,
                    source_run_id=source_run_id,
                    flush=False,
                )
                stored.append(record)
            except SecretDetectedError as exc:
                dead_letters.append(
                    self._dead_letter(
                        item, cursor_key, source_run_id, message_uuid, ordinal,
                        operation, "secret_detected", exc.rules,
                    )
                )
            except PermanentExtractionItemError as exc:
                dead_letters.append(
                    self._dead_letter(
                        item, cursor_key, source_run_id, message_uuid, ordinal,
                        operation, str(exc), [],
                    )
                )
        if items:
            await self.store.flush()
        return stored, dead_letters, skipped

    @staticmethod
    def _dead_letter(
        item: dict[str, Any],
        cursor_key: str,
        source_run_id: str | None,
        message_uuid: str,
        ordinal: int,
        operation: str,
        reason: str,
        rules: list[str],
    ) -> dict[str, Any]:
        canonical = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        return {
            "id": hashlib.sha256(
                f"{cursor_key}:{message_uuid}:{ordinal}:{canonical}".encode("utf-8")
            ).hexdigest()[:24],
            "ts": time.time(),
            "cursor_key": cursor_key,
            "source_run_id": source_run_id,
            "message_uuid": message_uuid,
            "item_ordinal": ordinal,
            "operation": operation,
            "reason": reason,
            "rules": sorted(set(rules)),
            "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        }

    async def _apply_non_create(self, item: dict[str, Any]) -> MemoryRecord | None:
        repository = getattr(self.store, "repository", None)
        target_id = str(item.get("target_id") or "")
        if repository is None or not target_id:
            return None
        operation = str(item.get("operation"))
        if repository.get(target_id) is None:
            raise PermanentExtractionItemError("unknown_target_id")
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
