from pathlib import Path

from agent_core.memory.extraction import MemoryExtractor, parse_memory_items
from agent_core.memory.store import MemoryStore
from agent_core.models import LLMResult, Message
from agent_core.providers import FakeProvider
from agent_core.providers.base import gated_provider


class StubProvider:
    """Returns a fixed completion content regardless of input."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    async def complete(self, messages, tools, config, stream=None, should_cancel=None) -> LLMResult:
        self.calls += 1
        return LLMResult(content=self.content)


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.jsonl")


def test_parse_memory_items_is_tolerant() -> None:
    text = 'Sure! Here you go:\n```json\n[{"content": "x", "kind": "fact"}]\n```'
    assert parse_memory_items(text) == [{"content": "x", "kind": "fact"}]
    assert parse_memory_items("no array here") == []
    assert parse_memory_items("[not, valid, json]") == []


async def test_extract_with_fake_provider_creates_record(tmp_path: Path) -> None:
    store = _store(tmp_path)
    extractor = MemoryExtractor(FakeProvider(), store)
    messages = [
        Message("user", "I prefer dark mode and Python 3.11"),
        Message("assistant", "Noted."),
    ]
    stored = await extractor.extract(messages, source_run_id="run-1")

    assert len(stored) == 1
    assert "dark mode" in stored[0].content
    assert stored[0].source_run_id == "run-1"
    assert len(store) == 1


async def test_extract_dedups_against_existing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.add("I prefer dark mode and Python 3.11", kind="preference")
    extractor = MemoryExtractor(FakeProvider(), store)
    messages = [Message("user", "I prefer dark mode and Python 3.11")]

    stored = await extractor.extract(messages)
    assert stored == []
    assert len(store) == 1  # no near-duplicate added


async def test_extract_returns_empty_on_non_json(tmp_path: Path) -> None:
    store = _store(tmp_path)
    extractor = MemoryExtractor(StubProvider("I could not find anything to remember."), store)
    messages = [Message("user", "hello there")]
    assert await extractor.extract(messages) == []
    assert len(store) == 0


async def test_extract_ignores_empty_transcript(tmp_path: Path) -> None:
    store = _store(tmp_path)
    # Only system + tool messages → nothing rememberable, no provider call needed.
    provider = StubProvider("[]")
    extractor = MemoryExtractor(provider, store)
    messages = [Message("system", "prompt"), Message("tool", "obs", metadata={"ok": True})]
    assert await extractor.extract(messages) == []
    assert provider.calls == 0  # no transcript → no provider call at all


async def test_extract_flows_through_gate(tmp_path: Path) -> None:
    """When the provider is gated, extract's API call goes through the gate."""
    store = _store(tmp_path)
    content = '[{"content": "user prefers dark mode", "kind": "preference", "importance": 0.7}]'
    inner = StubProvider(content)
    gated = gated_provider(inner, max_concurrency=2)
    extractor = MemoryExtractor(gated, store)
    messages = [Message("user", "remember I like dark mode")]

    stored = await extractor.extract(messages)

    assert inner.calls == 1  # gate delegated to the inner async path
    assert len(stored) == 1
