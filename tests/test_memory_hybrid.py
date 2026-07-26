from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent_core.memory.chunking import chunk_markdown
from agent_core.memory.config import MemoryConfig, MemoryRetrievalConfig
from agent_core.memory.indexing import INDEX_SCHEMA_VERSION
from agent_core.memory.models import (
    MemoryDocument,
    MemoryPassage,
    MemoryRecord,
    MemorySearchHit,
    MemorySearchRequest,
)
from agent_core.memory.repository import MemoryRepository
from agent_core.memory.retrieval import HybridMemoryRetriever, RepositoryMemoryRetriever
from agent_core.memory.text import exact_atoms, fts_query, lexical_tokens, normalize_text


def _config(**changes) -> MemoryConfig:
    retrieval = MemoryRetrievalConfig(**changes)
    return MemoryConfig(retrieval=retrieval)


def _document(
    repository: MemoryRepository,
    *,
    memory_id: str,
    name: str,
    content: str,
    description: str = "historical project context",
    memory_type: str = "project",
    tags: list[str] | None = None,
    sources: list[str] | None = None,
) -> MemoryDocument:
    return repository.create(
        MemoryDocument(
            id=memory_id,
            name=name,
            description=description,
            type=memory_type,  # type: ignore[arg-type]
            content=content,
            tags=tags or [],
            sources=sources or [],
        )
    )


def test_normalization_cjk_code_and_exact_atoms() -> None:
    assert normalize_text("Ａ\\B") == "a/b"
    tokens = lexical_tokens("修复身份验证 parseAuthToken user_id")
    assert "身份" in tokens
    assert "验证" in tokens
    assert "parse" in tokens and "auth" in tokens and "token" in tokens
    atoms = dict(exact_atoms('open "Exact Phrase" at src\\auth.py with ERR401 v1.2.3'))
    assert atoms["phrase"] == "exact phrase"
    assert atoms["path"] == "src/auth.py"
    assert atoms["version"] == "v1.2.3"
    assert atoms["error"] == "err401"


def test_chunking_is_bounded_and_overlaps_without_invalid_truncation() -> None:
    text = "# Heading\n\n" + " ".join(f"token{index}" for index in range(1000))
    chunks = chunk_markdown("topic", text, chunk_tokens=384, overlap_tokens=64)
    assert len(chunks) >= 3
    assert all(chunk.token_count <= 384 for chunk in chunks)
    first_tail = chunks[0].content.split()[-64:]
    assert chunks[1].content.split()[:64] == first_tail
    assert all(chunk.content.encode("utf-8").decode("utf-8") == chunk.content for chunk in chunks)


def test_more_than_200_topics_are_enumerated_and_searchable(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    records = [
        MemoryRecord(
            id=f"topic-{index:03d}",
            content=(
                "unique-overflow-marker"
                if index == 205
                else f"ordinary content {index}"
            ),
        )
        for index in range(206)
    ]
    repository.replace_records(records)
    target = repository.get("topic-205")
    assert target is not None
    assert len(repository.list()) == 206
    assert len(repository.index_text().splitlines()) <= 200
    engine = HybridMemoryRetriever(repository, index_base=tmp_path / "indexes")
    hits = engine.search(MemorySearchRequest.from_values("unique-overflow-marker"))
    assert [hit.id for hit in hits] == [target.id]
    assert engine.index.status().documents == 206


def test_exact_structured_filters_bm25_weight_and_fts_safety(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    exact = _document(
        repository,
        memory_id="auth-topic",
        name="Authentication",
        description="HTTP handling",
        content="The parseAuthToken symbol in src/auth.py raises ERR401.",
        tags=["backend", "security"],
        sources=["run:one"],
    )
    name_match = _document(
        repository,
        memory_id="name-match",
        name="Quasar deployment",
        content="ordinary notes",
        tags=["ops"],
    )
    _document(
        repository,
        memory_id="content-match",
        name="Deployment notes",
        content="quasar",
        tags=["ops"],
    )
    engine = HybridMemoryRetriever(repository, index_base=tmp_path / "indexes")

    hits = engine.search(
        MemorySearchRequest.from_values("src/auth.py ERR401", filters={"tag": "security"})
    )
    assert hits and hits[0].id == exact.id and hits[0].exact
    phrase_hits = engine.search(MemorySearchRequest.from_values('"HTTP handling"'))
    assert phrase_hits and phrase_hits[0].id == exact.id and phrase_hits[0].exact

    filtered = engine.search(
        MemorySearchRequest.from_values(
            "deployment",
            filters={"tag": ["backend", "ops"], "type": "project", "source": "run:one"},
        )
    )
    assert [hit.id for hit in filtered] == [exact.id]

    weighted = engine.index.bm25_candidates("quasar", {}, limit=5)
    assert weighted[0].document_id == name_match.id

    hostile = 'quasar" OR * NOT (content:secret) -foo'
    assert all(character not in fts_query(hostile) for character in ("*", "(", ")", ":"))
    assert engine.search(MemorySearchRequest.from_values(hostile))


def test_reranker_threshold_and_exact_hard_priority(tmp_path: Path) -> None:
    class LowReranker:
        fingerprint = "fake-reranker-v1"

        def rerank(self, query, passages, *, deadline=None):
            return [0.1] * len(passages)

    repository = MemoryRepository(tmp_path / "memory")
    exact = _document(
        repository,
        memory_id="symbol-topic",
        name="Symbol",
        content="Use parseAuthToken in src/auth.py.",
    )
    _document(
        repository,
        memory_id="fuzzy-topic",
        name="Orchard",
        content="orchard keyword",
    )
    engine = HybridMemoryRetriever(
        repository,
        _config(min_rerank_score=0.5),
        reranker_backend=LowReranker(),
        index_base=tmp_path / "indexes",
    )
    assert engine.search(MemorySearchRequest.from_values("orchard keyword")) == []
    hits = engine.search(MemorySearchRequest.from_values("parseAuthToken"))
    assert hits and hits[0].id == exact.id and hits[0].exact


def test_candidate_stages_cap_each_topic_before_fusion(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    dominant = _document(
        repository,
        memory_id="dominant-topic",
        name="Dominant",
        content="\n\n".join(
            [("sharedterm src/shared.py " * 100).strip() for _ in range(10)]
        ),
    )
    other = _document(
        repository,
        memory_id="other-topic",
        name="Other",
        content="sharedterm src/shared.py appears here too.",
    )
    engine = HybridMemoryRetriever(repository, index_base=tmp_path / "indexes")
    engine.index.ensure_current()

    for candidates in (
        engine.index.exact_candidates("src/shared.py", {}, limit=4),
        engine.index.bm25_candidates("sharedterm", {}, limit=4),
    ):
        ids = [candidate.document_id for candidate in candidates]
        assert ids.count(dominant.id) <= 3
        assert other.id in ids


def test_dense_backend_is_incremental_and_resumable(tmp_path: Path) -> None:
    pytest.importorskip("numpy")

    class TinyEmbedding:
        fingerprint = "tiny-embedding-v1"
        dimension = 2

        def embed(self, texts, *, deadline=None):
            return [
                [1.0, 0.0] if ("cat" in text.casefold() or "feline" in text.casefold()) else [0.0, 1.0]
                for text in texts
            ]

    repository = MemoryRepository(tmp_path / "memory")
    target = _document(
        repository,
        memory_id="cat-topic",
        name="Animals",
        content="A cat sleeps on the mat.",
    )
    _document(
        repository,
        memory_id="build-topic",
        name="Build",
        content="Compile the binary.",
    )
    engine = HybridMemoryRetriever(
        repository,
        _config(),
        embedding_backend=TinyEmbedding(),
        index_base=tmp_path / "indexes",
        background_embeddings=False,
    )
    hits = engine.search(MemorySearchRequest.from_values("feline"))
    assert hits and hits[0].id == target.id
    assert engine.index.status().coverage == 1.0


def test_ann_generation_exact_rescore_and_delta_visibility(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("usearch")

    class TinyEmbedding:
        fingerprint = "tiny-ann-embedding-v1"
        dimension = 2

        def embed(self, texts, *, deadline=None):
            vectors = []
            for text in texts:
                lowered = text.casefold()
                if "cat" in lowered or "feline" in lowered:
                    vectors.append([1.0, 0.0])
                elif "dog" in lowered or "puppy" in lowered:
                    vectors.append([0.0, 1.0])
                else:
                    vectors.append([0.7, 0.7])
            return vectors

    repository = MemoryRepository(tmp_path / "memory")
    cat = _document(
        repository,
        memory_id="cat-ann-topic",
        name="Animals",
        content="A cat sleeps here.",
    )
    _document(
        repository,
        memory_id="other-ann-topic",
        name="Other",
        content="A neutral note.",
    )
    engine = HybridMemoryRetriever(
        repository,
        _config(
            dense_strategy="ann",
            ann_min_vectors=1,
            ann_candidate_multiplier=4,
            ann_expansion_search=32,
        ),
        embedding_backend=TinyEmbedding(),
        index_base=tmp_path / "indexes",
        background_embeddings=False,
    )
    engine.index.ensure_current()
    engine.index.populate_embeddings(TinyEmbedding())
    status = engine.ann_index.build()
    assert status.state == "ready"

    hits = engine.search(MemorySearchRequest.from_values("feline"))
    assert hits and hits[0].id == cat.id
    assert engine.last_trace.dense_strategy == "ann_exact_rescore"
    assert engine.last_trace.candidate_counts["dense_exact_rescored"] >= 1

    dog = _document(
        repository,
        memory_id="dog-ann-topic",
        name="Canines",
        content="A dog waits here.",
    )
    delta_hits = engine.search(MemorySearchRequest.from_values("puppy"))
    assert delta_hits and delta_hits[0].id == dog.id
    assert engine.last_trace.candidate_counts["dense_delta"] >= 1


def test_ann_failure_falls_back_to_bounded_exact_dense_search(tmp_path: Path) -> None:
    pytest.importorskip("numpy")

    class TinyEmbedding:
        fingerprint = "tiny-fallback-embedding-v1"
        dimension = 2

        def embed(self, texts, *, deadline=None):
            return [
                [1.0, 0.0] if "target" in text.casefold() else [0.0, 1.0]
                for text in texts
            ]

    class MissingAnn:
        def search(self, query_vector, *, count):
            from agent_core.memory.ann import AnnStatus

            return [], AnnStatus(state="backend_unavailable")

        def schedule_build(self):
            return None

    repository = MemoryRepository(tmp_path / "memory")
    target = _document(
        repository,
        memory_id="fallback-topic",
        name="Fallback",
        content="semantic target",
    )
    engine = HybridMemoryRetriever(
        repository,
        _config(dense_strategy="ann", ann_min_vectors=1),
        embedding_backend=TinyEmbedding(),
        index_base=tmp_path / "indexes",
        background_embeddings=False,
        ann_index=MissingAnn(),
    )
    hits = engine.search(MemorySearchRequest.from_values("target"))
    assert hits and hits[0].id == target.id
    assert engine.last_trace.dense_strategy == "exact"
    assert "ann_backend_unavailable" in engine.last_trace.fallback_reasons


def test_auto_strategy_uses_exact_search_for_selective_filters(tmp_path: Path) -> None:
    pytest.importorskip("numpy")

    class TinyEmbedding:
        fingerprint = "tiny-filter-embedding-v1"
        dimension = 2

        def embed(self, texts, *, deadline=None):
            return [[1.0, 0.0] for _ in texts]

    repository = MemoryRepository(tmp_path / "memory")
    selected = _document(
        repository,
        memory_id="selected-filter-topic",
        name="Selected",
        content="semantic marker",
        tags=["selected"],
    )
    for index in range(3):
        _document(
            repository,
            memory_id=f"other-filter-topic-{index}",
            name="Other",
            content=f"other content {index}",
            tags=["other"],
        )
    engine = HybridMemoryRetriever(
        repository,
        _config(dense_strategy="auto", ann_min_vectors=2),
        embedding_backend=TinyEmbedding(),
        index_base=tmp_path / "indexes",
        background_embeddings=False,
    )
    hits = engine.search(
        MemorySearchRequest.from_values(
            "semantic",
            filters={"tag": "selected"},
        )
    )
    assert hits and hits[0].id == selected.id
    assert engine.last_trace.dense_strategy == "exact"


def test_manual_edit_incremental_sync_and_corrupt_index_recovery(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    document = _document(
        repository,
        memory_id="manual-topic",
        name="Manual",
        content="old-marker",
    )
    engine = HybridMemoryRetriever(repository, index_base=tmp_path / "indexes")
    assert engine.search(MemorySearchRequest.from_values("old-marker"))

    path = Path(document.path)
    path.write_text(path.read_text(encoding="utf-8").replace("old-marker", "new-marker"), encoding="utf-8")
    assert engine.search(MemorySearchRequest.from_values("new-marker"))[0].id == document.id

    engine.index.path.write_bytes(b"not a sqlite database")
    recovered = engine.search(MemorySearchRequest.from_values("new-marker"))
    assert recovered and recovered[0].id == document.id
    assert engine.index.status().schema_version == INDEX_SCHEMA_VERSION


def test_warm_incremental_sync_does_not_reparse_unchanged_topics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    document = _document(
        repository,
        memory_id="warm-topic",
        name="Warm",
        content="warm-marker",
    )
    engine = HybridMemoryRetriever(repository, index_base=tmp_path / "indexes")
    assert engine.search(MemorySearchRequest.from_values("warm-marker"))

    original = repository.load_document_path
    parsed: list[Path] = []

    def counted(path, *, headers_only=False):
        parsed.append(Path(path))
        return original(path, headers_only=headers_only)

    monkeypatch.setattr(repository, "load_document_path", counted)
    assert engine.search(MemorySearchRequest.from_values("warm-marker"))
    assert parsed == []

    path = Path(document.path)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert engine.search(MemorySearchRequest.from_values("warm-marker"))
    assert parsed == [path]


def test_passage_budget_never_slices_utf8_or_injects_whole_index(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    retriever = RepositoryMemoryRetriever(
        object(),  # type: ignore[arg-type]
        repository,
        MemoryConfig(content_budget_bytes=1024),
        index_base=tmp_path / "indexes",
    )
    hit = MemorySearchHit(
        id="topic",
        name="Topic",
        description="description",
        type="project",
        updated_at="2026-01-01T00:00:00Z",
        passages=[
            MemoryPassage("large", "界" * 2000),
            MemoryPassage("small", "完整片段"),
        ],
    )
    block = retriever.format_block([hit])
    assert "完整片段" in block
    assert "界" * 10 not in block
    assert "Memory topic index" not in block
    block.encode("utf-8").decode("utf-8")


def test_trace_is_privacy_safe(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    _document(
        repository,
        memory_id="private-topic",
        name="Private",
        content="sensitive-query-marker",
    )
    engine = HybridMemoryRetriever(repository, index_base=tmp_path / "indexes")
    engine.search(MemorySearchRequest.from_values("sensitive-query-marker", explain=True))
    serialized = json.dumps(engine.last_trace.to_dict())
    assert "sensitive-query-marker" not in serialized
    assert "candidate_counts" in serialized and "final_ids" in serialized
