import json
from pathlib import Path
from typing import Any

from agent_core.memory.config import MemoryConfig
from agent_core.memory.extraction import MemoryExtractor, parse_memory_items
from agent_core.memory.store import MemoryStore
from agent_core.models import LLMResult, Message
from agent_core.providers import FakeProvider


class StubProvider:
    """Returns a fixed completion content regardless of input."""

    def __init__(self, content: str) -> None:
        self.content = content

    def complete(self, messages, tools, config) -> LLMResult:
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
