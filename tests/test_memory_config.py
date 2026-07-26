from pathlib import Path

import pytest

from agent_core.config import resolve_memory_config
from agent_core.memory.config import (
    MemoryConfig,
    MemoryRetrievalConfig,
    RemovedMemoryConfigWarning,
)


def test_builtin_default_is_off() -> None:
    # CLAUDE.md invariant: memory is off by built-in default; only config opts in.
    assert MemoryConfig().enabled is False


def test_from_dict_applies_known_fields_and_coerces() -> None:
    config = MemoryConfig.from_dict(
        {"enabled": "true", "recall_k": "9", "merge_threshold": "0.4", "unknown": "x"}
    )
    assert config.enabled is True
    assert config.recall_k == 9
    assert config.merge_threshold == 0.4
    # Unknown keys are ignored; absent fields keep their defaults.
    assert config.dir == MemoryConfig().dir


def test_from_dict_handles_none_and_empty() -> None:
    assert MemoryConfig.from_dict(None) == MemoryConfig()
    assert MemoryConfig.from_dict({}) == MemoryConfig()


def test_hybrid_retrieval_defaults_are_stable() -> None:
    assert MemoryConfig().retrieval == MemoryRetrievalConfig(
        mode="hybrid",
        exact_k=32,
        bm25_k=64,
        dense_k=64,
        rrf_k=60,
        rerank_k=24,
        chunk_tokens=384,
        chunk_overlap_tokens=64,
        min_rerank_score=0.5,
        dense_fallback_min=0.45,
        timeout_seconds=10,
        model_threads=4,
        dense_strategy="auto",
        ann_min_vectors=10_000,
        ann_candidate_multiplier=4,
        ann_expansion_search=256,
    )


def test_removed_retrieval_keys_warn_and_are_not_mapped() -> None:
    with pytest.warns(RemovedMemoryConfigWarning, match="semantic_selection"):
        config = MemoryConfig.from_dict(
            {
                "semantic_selection": True,
                "retrieval": {"mode": "lexical", "bm25_k": 7},
            }
        )
    assert config.retrieval.mode == "lexical"
    assert config.retrieval.bm25_k == 7
    assert not hasattr(config, "semantic_selection")


def test_ann_retrieval_config_is_validated_and_bounded() -> None:
    retrieval = MemoryRetrievalConfig.from_dict(
        {
            "dense_strategy": "unsupported",
            "ann_min_vectors": "0",
            "ann_candidate_multiplier": "100",
            "ann_expansion_search": "2",
        }
    )
    assert retrieval.dense_strategy == "auto"
    assert retrieval.ann_min_vectors == 1
    assert retrieval.ann_candidate_multiplier == 32
    assert retrieval.ann_expansion_search == 16


def _write_toml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "agent.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_resolve_reads_memory_table(tmp_path: Path) -> None:
    toml = _write_toml(tmp_path, "[memory]\nenabled = true\nrecall_k = 7\n")
    config = resolve_memory_config(None, toml)
    assert config.enabled is True
    assert config.recall_k == 7


def test_resolve_env_overrides_toml(tmp_path: Path, monkeypatch) -> None:
    toml = _write_toml(tmp_path, "[memory]\nenabled = true\n")
    monkeypatch.setenv("AGENT_MEMORY", "0")
    config = resolve_memory_config(None, toml)
    assert config.enabled is False


def test_resolve_cli_overrides_env(tmp_path: Path, monkeypatch) -> None:
    toml = _write_toml(tmp_path, "[memory]\nenabled = false\n")
    monkeypatch.setenv("AGENT_MEMORY", "0")
    config = resolve_memory_config(True, toml)  # explicit --memory wins
    assert config.enabled is True


def test_resolve_without_table_is_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AGENT_MEMORY", raising=False)
    toml = _write_toml(tmp_path, "model = \"x\"\n")  # no [memory] table
    assert resolve_memory_config(None, toml) == MemoryConfig()


def test_repository_config_cannot_trust_private_memory_redirect(tmp_path: Path) -> None:
    external = tmp_path.parent / "sensitive"
    toml = _write_toml(
        tmp_path,
        f'[memory]\nenabled = true\ndir = "{external.as_posix()}"\n',
    )
    config = resolve_memory_config(None, toml)
    assert config.dir == external.as_posix()
    assert config.dir_trusted is False
