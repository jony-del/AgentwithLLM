import time
from pathlib import Path

from agent_core.memory.config import MemoryConfig
from agent_core.memory.retrieval import MemoryRetriever
from agent_core.memory.store import MemoryStore


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.jsonl")


async def test_relevant_memory_ranks_above_irrelevant(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.add("the user prefers dark mode in the editor", kind="preference")
    await store.add("the user lives in a coastal city", kind="fact")

    recalled = await MemoryRetriever(store).recall("what theme does the editor use?")
    assert recalled
    assert "dark mode" in recalled[0].content


async def test_irrelevant_query_recalls_nothing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.add("the user prefers dark mode")
    assert await MemoryRetriever(store).recall("quantum chromodynamics lattice") == []


async def test_importance_breaks_ties(tmp_path: Path) -> None:
    store = _store(tmp_path)
    # Same words → equal relevance & recency; importance decides the order.
    low = await store.add("python tooling notes", importance=0.2)
    high = await store.add("python tooling notes", importance=0.9)

    recalled = await MemoryRetriever(store).recall("python tooling", k=2)
    assert [r.id for r in recalled] == [high.id, low.id]


async def test_recall_touches_returned_records(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = await store.add("user uses VS Code")
    assert record.access_count == 0

    await MemoryRetriever(store).recall("which editor does the user use")
    assert store.get(record.id).access_count == 1


async def test_recency_boosts_recent_memory(tmp_path: Path) -> None:
    store = _store(tmp_path)
    old = await store.add("alpha beta gamma", importance=0.5)
    new = await store.add("alpha beta gamma", importance=0.5)
    # Make `old` look stale; recency should then favour `new`.
    old.last_accessed_at = time.time() - 3600 * 24 * 30
    await store.update(old)

    config = MemoryConfig(w_relevance=0.0, w_importance=0.0, w_recency=1.0)
    recalled = await MemoryRetriever(store, config).recall("alpha beta", k=2)
    assert recalled[0].id == new.id
