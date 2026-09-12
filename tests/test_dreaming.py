import time
from pathlib import Path

from agent_core.memory.config import MemoryConfig
from agent_core.memory.dreaming import Dreamer
from agent_core.memory.store import MemoryStore
from agent_core.providers import FakeProvider
from agent_core.providers.base import gated_provider


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.jsonl")


async def test_forgets_weak_unaccessed_memories(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.add("trivial passing remark", importance=0.1)          # weak, never recalled
    kept_important = await store.add("strong durable fact", importance=0.5)
    weak_but_used = await store.add("weak but recalled note", importance=0.1)
    await store.touch(weak_but_used.id)  # access_count >= forget_min_access protects it

    report = await Dreamer(store, MemoryConfig(), provider=None).dream()

    assert report.scanned == 3
    assert report.forgotten == 1
    ids = {r.id for r in store.all()}
    assert kept_important.id in ids
    assert weak_but_used.id in ids


async def test_merges_near_duplicates(tmp_path: Path) -> None:
    store = _store(tmp_path)
    a = await store.add("python tooling tips", importance=0.8, tags=["py"])
    await store.add("python tooling tricks", importance=0.6, tags=["dev"])

    report = await Dreamer(store, MemoryConfig(), provider=None).dream()

    assert report.merged == 1
    survivors = store.all()
    assert len(survivors) == 1
    survivor = survivors[0]
    assert survivor.id == a.id  # higher-importance memory is canonical
    assert set(survivor.tags) == {"py", "dev"}  # tags unioned
    assert survivor.importance >= 0.8  # fusing reinforces


async def test_synthesizes_insight_with_provider(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.add("user lives in Tokyo", importance=0.5)
    await store.add("user codes primarily in Rust", importance=0.5)

    report = await Dreamer(store, MemoryConfig(), provider=FakeProvider()).dream()

    assert report.merged == 0
    assert report.forgotten == 0
    assert report.insights_added == 1
    insights = [r for r in store.all() if r.kind == "insight"]
    assert len(insights) == 1


async def test_insight_synthesis_flows_through_gate(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.add("user lives in Tokyo", importance=0.5)
    await store.add("user codes primarily in Rust", importance=0.5)

    report = await Dreamer(store, MemoryConfig(), provider=gated_provider(FakeProvider())).dream()

    assert report.insights_added == 1
    assert len([r for r in store.all() if r.kind == "insight"]) == 1


async def test_insight_synthesis_skipped_without_provider(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.add("fact one about apples", importance=0.5)
    await store.add("fact two about oranges", importance=0.5)

    report = await Dreamer(store, MemoryConfig(), provider=None).dream()
    assert report.insights_added == 0


async def test_dry_run_reports_without_mutating(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.add("python tooling tips", importance=0.8)
    await store.add("python tooling tricks", importance=0.6)

    report = await Dreamer(store, MemoryConfig(), provider=FakeProvider()).dream(commit=False)

    # The report reflects what *would* happen ...
    assert report.merged == 1
    # ... but the store is untouched: no merge, no decay, no insight written.
    assert len(store) == 2
    assert {round(r.importance, 2) for r in store.all()} == {0.8, 0.6}
    assert all(r.kind != "insight" for r in store.all())


async def test_decayed_importance_is_persisted(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = await store.add("a memory that should fade a little", importance=0.9)
    record.last_accessed_at = time.time() - 86400 * 7  # one week old
    await store.update(record)

    await Dreamer(store, MemoryConfig(importance_half_life_days=14.0), provider=None).dream()
    # 7 days at a 14-day half-life ~= 0.5**0.5 ≈ 0.707 of original.
    faded = store.get(record.id)
    assert faded is not None
    assert 0.6 < faded.importance < 0.9


async def test_insight_with_secret_is_quarantined(tmp_path: Path) -> None:
    import json

    from agent_core.models import LLMResult

    store = _store(tmp_path)
    await store.add("user lives in Tokyo", importance=0.5)
    await store.add("user codes primarily in Rust", importance=0.5)

    class _InsightProvider:
        async def complete(self, *args, **kwargs) -> LLMResult:
            return LLMResult(json.dumps([
                {"content": "their key is sk-abcdefghijklmnopqrstuvwxyz123456"},
                {"content": "enjoys hiking on weekends"},
            ]))

    report = await Dreamer(store, MemoryConfig(), provider=_InsightProvider()).dream()

    assert report.quarantined == 1
    assert report.insights_added == 1
    contents = [r.content for r in store.all()]
    assert any("hiking" in content for content in contents)
    assert not any("sk-" in content for content in contents)


async def test_dream_quarantines_poisoned_existing_record(tmp_path: Path, monkeypatch) -> None:
    from agent_core.memory.repository import MemoryRepository
    from agent_core.memory.store import RepositoryMemoryStore

    repository = MemoryRepository(tmp_path / "memory")
    # Plant a poisoned record the way a pre-quarantine legacy import could have.
    monkeypatch.setattr(
        "agent_core.memory.repository.require_secret_free", lambda *values: None
    )
    repository.write(
        name="legacy import",
        description="imported without scanning",
        type="project",
        content="token sk-abcdefghijklmnopqrstuvwxyz123456",
    )
    monkeypatch.undo()
    repository.write(
        name="clean", description="durable fact", type="project", content="prefers tea"
    )
    store = RepositoryMemoryStore(repository)
    assert len(store.all()) == 2

    # Before the fix this raised SecretDetectedError out of replace_records on every
    # pass, so mark_dream_succeeded never ran and consolidation stalled forever.
    report = await Dreamer(store, MemoryConfig(), provider=None).dream()

    assert report.quarantined == 1
    remaining = store.all()
    assert [record.content for record in remaining] == ["prefers tea"]
    # The poisoned record was archived out of the active set: the next pass is clean.
    again = await Dreamer(store, MemoryConfig(), provider=None).dream()
    assert again.quarantined == 0
