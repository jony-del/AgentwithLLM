import asyncio
import json
from pathlib import Path
from typing import Any

from agent_core.memory.config import MemoryConfig
from agent_core.memory.extraction import MemoryExtractor, parse_memory_items
from agent_core.memory.store import MemoryStore
from agent_core.models import LLMResult, Message
from agent_core.providers import FakeProvider
from agent_core.providers.base import gated_provider


class StubProvider:
    """Returns a fixed completion content regardless of input."""

    def __init__(self, content: str) -> None:
        self.content = content

    def complete(self, messages, tools, config) -> LLMResult:
        return LLMResult(content=self.content)


class PathRecordingProvider:
    """Returns fixed content and records whether the sync or async path was used."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.sync_calls = 0
        self.async_calls = 0

    def complete(self, messages, tools, config, stream=None) -> LLMResult:
        self.sync_calls += 1
        return LLMResult(content=self.content)

    async def acomplete(self, messages, tools, config, stream=None) -> LLMResult:
        self.async_calls += 1
        return LLMResult(content=self.content)


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.jsonl")


def test_parse_memory_items_is_tolerant() -> None:
    text = 'Sure! Here you go:\n```json\n[{"content": "x", "kind": "fact"}]\n```'
    assert parse_memory_items(text) == [{"content": "x", "kind": "fact"}]
    assert parse_memory_items("no array here") == []
    assert parse_memory_items("[not, valid, json]") == []


def test_extract_with_fake_provider_creates_record(tmp_path: Path) -> None:
    store = _store(tmp_path)
    extractor = MemoryExtractor(FakeProvider(), store)
    messages = [
        Message("user", "I prefer dark mode and Python 3.11"),
        Message("assistant", "Noted."),
    ]
    stored = extractor.extract(messages, source_run_id="run-1")

    assert len(stored) == 1
    assert "dark mode" in stored[0].content
    assert stored[0].source_run_id == "run-1"
    assert len(store) == 1


def test_extract_dedups_against_existing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add("I prefer dark mode and Python 3.11", kind="preference")
    extractor = MemoryExtractor(FakeProvider(), store)
    messages = [Message("user", "I prefer dark mode and Python 3.11")]

    stored = extractor.extract(messages)
    assert stored == []
    assert len(store) == 1  # no near-duplicate added


def test_extract_returns_empty_on_non_json(tmp_path: Path) -> None:
    store = _store(tmp_path)
    extractor = MemoryExtractor(StubProvider("I could not find anything to remember."), store)
    messages = [Message("user", "hello there")]
    assert extractor.extract(messages) == []
    assert len(store) == 0


def test_extract_ignores_empty_transcript(tmp_path: Path) -> None:
    store = _store(tmp_path)
    # Only system + tool messages → nothing rememberable, no provider call needed.
    extractor = MemoryExtractor(StubProvider("[]"), store)
    messages = [Message("system", "prompt"), Message("tool", "obs", metadata={"ok": True})]
    assert extractor.extract(messages) == []


# --- async path (aextract) ---------------------------------------------------


def test_aextract_matches_extract(tmp_path: Path) -> None:
    """aextract stores the same record sync extract would, for identical input."""
    store = _store(tmp_path)
    extractor = MemoryExtractor(FakeProvider(), store)
    messages = [
        Message("user", "I prefer dark mode and Python 3.11"),
        Message("assistant", "Noted."),
    ]
    stored = asyncio.run(extractor.aextract(messages, source_run_id="run-1"))

    assert len(stored) == 1
    assert "dark mode" in stored[0].content
    assert stored[0].source_run_id == "run-1"
    assert len(store) == 1


def test_aextract_uses_async_provider_path(tmp_path: Path) -> None:
    """aextract calls acomplete (async), not the blocking sync complete."""
    store = _store(tmp_path)
    content = '[{"content": "user prefers dark mode", "kind": "preference", "importance": 0.7}]'
    provider = PathRecordingProvider(content)
    extractor = MemoryExtractor(provider, store)
    messages = [Message("user", "remember I like dark mode")]

    stored = asyncio.run(extractor.aextract(messages))

    assert provider.async_calls == 1
    assert provider.sync_calls == 0
    assert len(stored) == 1


def test_aextract_flows_through_gate(tmp_path: Path) -> None:
    """When the provider is gated, aextract's API call goes through the gate."""
    store = _store(tmp_path)
    content = '[{"content": "user prefers dark mode", "kind": "preference", "importance": 0.7}]'
    inner = PathRecordingProvider(content)
    gated = gated_provider(inner, max_concurrency=2)
    extractor = MemoryExtractor(gated, store)
    messages = [Message("user", "remember I like dark mode")]

    stored = asyncio.run(extractor.aextract(messages))

    assert inner.async_calls == 1  # gate delegated to the inner async path
    assert len(stored) == 1


def test_aextract_ignores_empty_transcript(tmp_path: Path) -> None:
    store = _store(tmp_path)
    provider = PathRecordingProvider("[]")
    extractor = MemoryExtractor(provider, store)
    messages = [Message("system", "prompt"), Message("tool", "obs", metadata={"ok": True})]

    assert asyncio.run(extractor.aextract(messages)) == []
    assert provider.async_calls == 0  # no transcript → no provider call at all
