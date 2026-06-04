from pathlib import Path

from agent_core.memory.store import MemoryStore


def test_add_get_all_and_persist(tmp_path: Path) -> None:
    path = tmp_path / "memory.jsonl"
    store = MemoryStore(path)
    record = store.add("user prefers dark mode", kind="preference", importance=0.8, tags=["ui"])

    assert store.get(record.id) is record
    assert len(store) == 1
    assert path.exists()

    # Reopening reads the same record back (round-trips through JSONL).
    reopened = MemoryStore(path)
    loaded = reopened.get(record.id)
    assert loaded is not None
    assert loaded.content == "user prefers dark mode"
    assert loaded.kind == "preference"
    assert loaded.importance == 0.8
    assert loaded.tags == ["ui"]


def test_update_delete_and_touch(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.jsonl")
    record = store.add("python 3.11", importance=0.5)

    record.importance = 0.9
    store.update(record)
    assert MemoryStore(tmp_path / "memory.jsonl").get(record.id).importance == 0.9

    before = record.access_count
    store.touch(record.id)
    assert store.get(record.id).access_count == before + 1

    assert store.delete(record.id) is True
    assert store.delete(record.id) is False
    assert len(store) == 0


def test_replace_all_rewrites_file(tmp_path: Path) -> None:
    path = tmp_path / "memory.jsonl"
    store = MemoryStore(path)
    store.add("one")
    store.add("two")
    keep = store.add("three")

    store.replace_all([keep])
    assert len(store) == 1
    assert len(MemoryStore(path).all()) == 1


def test_corrupt_line_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "memory.jsonl"
    path.write_text('{"id":"a","content":"good"}\nnot json\n', encoding="utf-8")
    store = MemoryStore(path)
    assert len(store) == 1
    assert store.get("a").content == "good"
