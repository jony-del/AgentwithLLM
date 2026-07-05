from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MemoryConfig:
    """Tunables for the memory subsystem.

    Disabled by default: an agent built without opting in behaves exactly as before
    (no recall injection, no extraction calls, no files written). This repo's own
    ``agent.toml`` opts in via ``[memory] enabled = true``; the built-in default
    staying ``False`` is a documented invariant (CLAUDE.md "Memory is off by
    built-in default").
    """

    enabled: bool = False
    dir: str = "memory"

    # --- Recall ---------------------------------------------------------------
    recall_k: int = 5
    # Weights for the blended recall score (need not sum to 1; ranking is relative).
    w_relevance: float = 1.0
    w_importance: float = 0.5
    w_recency: float = 0.3
    # Exponential recency decay per hour since a memory was last accessed.
    recency_decay_per_hour: float = 0.01

    # --- Extraction -----------------------------------------------------------
    auto_extract: bool = True
    # Skip storing a freshly extracted memory if it overlaps an existing one above
    # this lexical-relevance threshold (avoids slow-growing near-duplicates).
    dedup_threshold: float = 0.85

    # --- Dreaming -------------------------------------------------------------
    # Drop a memory whose time-decayed importance falls below this, unless it has
    # been recalled enough to have earned its keep (see Dreamer.forget logic).
    forget_threshold: float = 0.15
    forget_min_access: int = 1
    # Half-life (in days) of importance for the forgetting curve during dreaming.
    importance_half_life_days: float = 14.0
    # Merge two memories during dreaming when their lexical relevance exceeds this.
    merge_threshold: float = 0.6
    # Whether dreaming asks the LLM to synthesise higher-level insight memories.
    synthesize_insights: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MemoryConfig":
        """Build a config from a mapping (e.g. an ``[memory]`` toml table).

        Unknown keys are ignored and every absent field keeps its default, so a
        partial or forward-compatible table still loads cleanly. Values are coerced
        to each field's declared type to tolerate toml/string inputs.
        """
        from agent_core.config import overlay_dataclass

        return overlay_dataclass(cls(), data)
