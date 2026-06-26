"""Selectable model catalog for the interactive ``/model`` picker.

Dependency-light: it only pulls the per-model effort capability from the (import-light)
Claude provider so the picker and the provider can never disagree about which effort
levels a model accepts. The model list here is the user-facing menu — full model ids
paired with a short human label.
"""
from __future__ import annotations

from agent_core.providers.claude import ALL_EFFORT_LEVELS, available_efforts

__all__ = ["SELECTABLE_MODELS", "ALL_EFFORT_LEVELS", "available_efforts"]

# (model id, short label). Order strongest/most-capable → fastest, which is the order
# the picker lists them top → bottom.
SELECTABLE_MODELS: tuple[tuple[str, str], ...] = (
    ("claude-opus-4-8", "Opus 4.8 — 1M context, most capable"),
    ("claude-opus-4-7", "Opus 4.7 — prior Opus"),
    ("claude-sonnet-4-6", "Sonnet 4.6 — balanced"),
    ("claude-fable-5", "Fable 5 — 1M context"),
    ("claude-haiku-4-5-20251001", "Haiku 4.5 — fast / debug"),
)
