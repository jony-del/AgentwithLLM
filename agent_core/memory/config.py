from __future__ import annotations

import warnings
from dataclasses import dataclass, fields
from typing import Any


class RemovedMemoryConfigWarning(UserWarning):
    """An old retrieval setting was ignored because its semantics were removed."""


REMOVED_RETRIEVAL_KEYS = frozenset(
    {
        "semantic_selection",
        "memory_model",
        "w_relevance",
        "w_importance",
        "w_recency",
        "recency_decay_per_hour",
    }
)


@dataclass(slots=True)
class MemoryRetrievalConfig:
    """Stable hybrid-retrieval defaults.

    Model paths are deliberately not configuration knobs here. Installed model
    manifests select exact, checksummed artifacts and contribute their fingerprints
    to the derived-index path.
    """

    mode: str = "hybrid"
    exact_k: int = 32
    bm25_k: int = 64
    dense_k: int = 64
    rrf_k: int = 60
    rerank_k: int = 24
    chunk_tokens: int = 384
    chunk_overlap_tokens: int = 64
    min_rerank_score: float = 0.5
    dense_fallback_min: float = 0.45
    timeout_seconds: float = 10.0
    model_threads: int = 4
    dense_strategy: str = "auto"
    ann_min_vectors: int = 10_000
    ann_candidate_multiplier: int = 4
    ann_expansion_search: int = 256

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MemoryRetrievalConfig":
        from agent_core.config import coerce_to_type

        config = cls()
        if data:
            valid = {item.name: item.type for item in fields(config)}
            for key, value in data.items():
                if key in valid:
                    setattr(config, key, coerce_to_type(valid[key], value))
        config.mode = config.mode if config.mode in {"hybrid", "lexical"} else "hybrid"
        config.dense_strategy = (
            config.dense_strategy
            if config.dense_strategy in {"auto", "exact", "ann"}
            else "auto"
        )
        for name in ("exact_k", "bm25_k", "dense_k", "rerank_k"):
            setattr(config, name, max(0, int(getattr(config, name))))
        config.rrf_k = max(1, int(config.rrf_k))
        config.chunk_tokens = max(32, int(config.chunk_tokens))
        config.chunk_overlap_tokens = max(
            0, min(int(config.chunk_overlap_tokens), config.chunk_tokens - 1)
        )
        config.min_rerank_score = max(0.0, min(1.0, float(config.min_rerank_score)))
        config.dense_fallback_min = max(-1.0, min(1.0, float(config.dense_fallback_min)))
        config.timeout_seconds = max(0.1, float(config.timeout_seconds))
        config.model_threads = max(1, int(config.model_threads))
        config.ann_min_vectors = max(1, int(config.ann_min_vectors))
        config.ann_candidate_multiplier = max(
            1, min(32, int(config.ann_candidate_multiplier))
        )
        config.ann_expansion_search = max(
            16, min(8192, int(config.ann_expansion_search))
        )
        return config


@dataclass(slots=True)
class MemoryConfig:
    """Configuration for authoritative Markdown memory and derived retrieval."""

    enabled: bool = False
    dir: str = "memory"
    dir_trusted: bool = True
    scope: str = "private"

    recall_k: int = 5
    content_budget_bytes: int = 64 * 1024
    retrieval: MemoryRetrievalConfig | None = None

    auto_extract: bool = True
    team_auto_extract: bool = False
    dedup_threshold: float = 0.85

    forget_threshold: float = 0.15
    forget_min_access: int = 1
    importance_half_life_days: float = 14.0
    merge_threshold: float = 0.6
    synthesize_insights: bool = True
    auto_dream: bool = True
    dream_min_hours: float = 24.0
    dream_min_sessions: int = 5

    def __post_init__(self) -> None:
        if self.retrieval is None:
            self.retrieval = MemoryRetrievalConfig()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MemoryConfig":
        """Build config while rejecting removed ranking/model-selection semantics."""
        from agent_core.config import coerce_to_type

        config = cls()
        if not data:
            return config
        for key in sorted(REMOVED_RETRIEVAL_KEYS.intersection(data)):
            warnings.warn(
                f"[memory].{key} was removed and is ignored; configure [memory.retrieval] instead",
                RemovedMemoryConfigWarning,
                stacklevel=2,
            )
        valid = {item.name: item.type for item in fields(config)}
        for key, value in data.items():
            if key in REMOVED_RETRIEVAL_KEYS or key == "retrieval":
                continue
            if key in valid:
                setattr(config, key, coerce_to_type(valid[key], value))
        nested = data.get("retrieval")
        config.retrieval = MemoryRetrievalConfig.from_dict(
            nested if isinstance(nested, dict) else None
        )
        config.recall_k = max(0, int(config.recall_k))
        config.content_budget_bytes = max(1024, int(config.content_budget_bytes))
        return config
