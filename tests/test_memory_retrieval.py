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


async def test_relevance_is_the_primary_ranking_signal(tmp_path: Path) -> None:
    store = _store(tmp_path)
    weak = await store.add("python unrelated", importance=1.0)
    strong = await store.add("python tooling notes", importance=0.0)

    recalled = await MemoryRetriever(store).recall("python tooling", k=2)
    assert [record.id for record in recalled] == [strong.id, weak.id]


async def test_recall_touches_returned_records(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = await store.add("user uses VS Code")
    assert record.access_count == 0

    await MemoryRetriever(store).recall("which editor does the user use")
    assert store.get(record.id).access_count == 1


async def test_importance_and_recency_do_not_override_relevance(tmp_path: Path) -> None:
    store = _store(tmp_path)
    relevant = await store.add("alpha beta gamma", importance=0.0)
    irrelevant = await store.add("totally different topic", importance=1.0)
    irrelevant.last_accessed_at = relevant.last_accessed_at + 1_000_000
    await store.update(irrelevant)

    recalled = await MemoryRetriever(store, MemoryConfig()).recall("alpha beta", k=2)
    assert [item.id for item in recalled] == [relevant.id]
