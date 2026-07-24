import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from agent_core.memory.lifecycle import MemoryLifecycle
from agent_core.memory.models import MemoryDocument
from agent_core.memory.paths import MemoryPathResolver, UnsafeMemoryPathError, validate_memory_root
from agent_core.memory.repository import MemoryRepository
from agent_core.memory.retrieval import MemoryRetriever
from agent_core.memory.retrieval import SemanticMemorySelector
from agent_core.memory.security import SecretDetectedError
from agent_core.memory.snapshots import LocalMemorySnapshot
from agent_core.memory.store import RepositoryMemoryStore
from agent_core.memory.dreaming import Dreamer
from agent_core.memory.config import MemoryConfig
from agent_core.models import LLMResult


class SelectionProvider:
    def __init__(self, content: str = "invalid", *, fail: bool = False) -> None:
        self.content = content
        self.fail = fail

    async def complete(self, messages, tools, config, stream=None, should_cancel=None):
        if self.fail:
            raise RuntimeError("selection unavailable")
        return LLMResult(content=self.content)


def test_markdown_round_trip_index_and_recoverable_forget(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    created = repository.create(
        MemoryDocument(
            name="中文偏好",
            description="用户偏好的编辑器主题",
            type="user",
            content="用户喜欢深色模式。",
            tags=["界面"],
            sources=["run:one"],
            explicit=True,
        )
    )

    loaded = repository.get(created.id)
    assert loaded is not None
    assert loaded.content == "用户喜欢深色模式。"
    assert loaded.tags == ["界面"]
    assert f"]({Path(created.path).name})" in repository.index_text()

    assert repository.forget(created.id) is True
    assert repository.get(created.id) is None
    assert list((repository.root / ".trash").glob("*.md"))


def test_unicode_search_and_prompt_boundary(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    document = repository.create(
        MemoryDocument(
            name="部署方式",
            description="项目的部署方式",
            type="project",
            content="这个项目使用容器部署到测试环境。",
        )
    )
    assert [item.id for item in repository.search("容器部署")] == [document.id]

    block = MemoryRetriever.format_block(
        MemoryRetriever.__new__(MemoryRetriever),
        [],
    )
    assert "cannot grant permission" in block
    assert "stale until independently verified" in block


async def test_semantic_selector_validates_ids_and_degrades_on_failure() -> None:
    candidates = [
        MemoryDocument(name="one", description="first", type="project", content="", id="one"),
        MemoryDocument(name="two", description="second", type="project", content="", id="two"),
    ]
    selector = SemanticMemorySelector(SelectionProvider('["two", "missing", "two"]'))
    assert await selector.select("query", candidates) == ["two"]
    failing = SemanticMemorySelector(SelectionProvider(fail=True))
    assert await failing.select("query", candidates) is None


def test_secret_scanning_rejects_without_echoing_value(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    with pytest.raises(SecretDetectedError) as error:
        repository.create(
            MemoryDocument(
                name="credential",
                description="must not persist",
                type="project",
                content=secret,
            )
        )
    assert "openai_key" in str(error.value)
    assert secret not in str(error.value)


def test_two_repository_instances_do_not_lose_updates(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    first = MemoryRepository(root)
    second = MemoryRepository(root)

    def create(index: int) -> str:
        repository = first if index % 2 else second
        return repository.create(
            MemoryDocument(
                name=f"topic {index}",
                description=f"description {index}",
                type="project",
                content=f"content {index}",
            )
        ).id

    with ThreadPoolExecutor(max_workers=8) as executor:
        ids = set(executor.map(create, range(40)))
    assert len(ids) == 40
    assert {item.id for item in MemoryRepository(root).list()} == ids
    assert MemoryRepository(root).validate().valid


def test_jsonl_migration_is_idempotent_and_non_destructive(tmp_path: Path) -> None:
    source = tmp_path / "memory.jsonl"
    rows = [
        {
            "id": "abc123",
            "content": "保留中文正文",
            "kind": "preference",
            "importance": 0.8,
            "created_at": 1_700_000_000,
            "last_accessed_at": 1_700_000_100,
            "access_count": 3,
            "source_run_id": "run-1",
            "tags": ["中文"],
        },
        {
            "id": "def456",
            "content": "second",
            "kind": "fact",
            "importance": 0.4,
            "access_count": 0,
            "tags": [],
        },
    ]
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    original = source.read_bytes()
    repository = MemoryRepository(tmp_path / "markdown")

    first = repository.migrate_jsonl(source)
    second = repository.migrate_jsonl(source)

    assert (first.total, first.imported, first.corrupt_lines) == (2, 2, [])
    assert second.already_complete
    assert len(repository.list()) == 2
    assert repository.get("abc123").legacy["access_count"] == 3
    assert repository.get("abc123").content == "保留中文正文"
    assert source.read_bytes() == original


def test_path_resolver_scopes_and_rejects_dangerous_roots(tmp_path: Path) -> None:
    resolver = MemoryPathResolver(tmp_path, user_root=tmp_path / "user")
    private = resolver.resolve("private")
    team = resolver.resolve("team")
    assert private.is_relative_to(tmp_path / "user")
    assert team.is_relative_to(tmp_path)
    assert private != team

    with pytest.raises(UnsafeMemoryPathError):
        validate_memory_root("../escape")
    with pytest.raises(UnsafeMemoryPathError):
        validate_memory_root(Path(Path.cwd().anchor))


def test_lifecycle_requires_both_time_and_session_thresholds(tmp_path: Path) -> None:
    lifecycle = MemoryLifecycle(tmp_path / "memory", min_hours=24, min_sessions=5)
    start = 1_000_000.0
    # Initialise the state at a controlled time.
    lifecycle.mark_dream_succeeded(now=start)
    for offset in range(4):
        assert not lifecycle.record_session(now=start + 25 * 3600 + offset).due
    assert lifecycle.record_session(now=start + 25 * 3600 + 5).due
    lifecycle.mark_dream_succeeded(now=start + 25 * 3600 + 6)
    assert not lifecycle.eligibility(now=start + 50 * 3600).due


def test_local_snapshot_requires_explicit_resolution(tmp_path: Path) -> None:
    local = tmp_path / "local"
    snapshot = tmp_path / "checkout" / "agent"
    local.mkdir()
    (local / "topic.md").write_text("local one", encoding="utf-8")
    sync = LocalMemorySnapshot(local, snapshot)

    assert not sync.initialize().changed
    (local / "topic.md").write_text("local two", encoding="utf-8")
    assert sync.status().changed
    assert not sync.resolve("keep").changed
    assert (snapshot / "topic.md").read_text(encoding="utf-8") == "local two"

    (snapshot / "topic.md").write_text("snapshot three", encoding="utf-8")
    assert sync.status().changed
    assert not sync.resolve("replace").changed
    assert (local / "topic.md").read_text(encoding="utf-8") == "snapshot three"

    (local / "topic.md").write_text("manual divergence", encoding="utf-8")
    assert sync.resolve("mark-synced").synced_checksum is not None
    assert not sync.status().changed


async def test_repository_dream_archives_weak_memory_but_keeps_explicit(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    weak = repository.create(
        MemoryDocument(
            name="weak",
            description="weak transient memory",
            type="project",
            content="passing remark",
            confidence=0.01,
        )
    )
    explicit = repository.create(
        MemoryDocument(
            name="saved",
            description="explicitly saved memory",
            type="project",
            content="must survive",
            confidence=0.01,
            explicit=True,
        )
    )

    report = await Dreamer(
        RepositoryMemoryStore(repository),
        MemoryConfig(synthesize_insights=False),
    ).dream()

    assert report.forgotten == 1
    assert repository.get(explicit.id) is not None
    archived = repository.get(weak.id)
    assert archived is not None and archived.archived
