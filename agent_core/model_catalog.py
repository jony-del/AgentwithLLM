"""Selectable model catalog for the interactive ``/model`` picker.

Dependency-light: it pulls provider-local effort capability helpers so the picker and
providers never disagree about which effort levels a model accepts. The model lists here
are user-facing menus — full model ids paired with short human labels — selected by the
explicit provider, never inferred from a model id.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agent_core.model_validation import CLAUDE_PROVIDER, OPENAI_PROVIDER
from agent_core.providers.claude import ALL_EFFORT_LEVELS, available_efforts
from agent_core.providers.openai_capabilities import OPENAI_MODEL_EFFORT_OPTIONS, capabilities_for_responses_model

__all__ = [
    "ALL_EFFORT_LEVELS",
    "OPENAI_SELECTABLE_MODELS",
    "PickerSpec",
    "SELECTABLE_MODELS",
    "available_efforts",
    "openai_available_efforts",
    "picker_spec_for_provider",
]


@dataclass(frozen=True, slots=True)
class PickerSpec:
    """Provider-specific model picker data, normalized for terminal/UI code."""

    title: str
    help_text: str
    models: tuple[tuple[str, str], ...]
    efforts_fn: Callable[[str], tuple[str, ...]]
    known_families: str


# (model id, short label). Order strongest/most-capable → fastest, which is the order
# the picker lists them top → bottom.
SELECTABLE_MODELS: tuple[tuple[str, str], ...] = (
    ("claude-opus-4-8", "Opus 4.8 — 1M context, most capable"),
    ("claude-opus-4-7", "Opus 4.7 — prior Opus"),
    ("claude-sonnet-4-6", "Sonnet 4.6 — balanced"),
    ("claude-fable-5", "Fable 5 — 1M context"),
    ("claude-haiku-4-5-20251001", "Haiku 4.5 — fast / debug"),
)


def openai_available_efforts(model: str) -> tuple[str, ...]:
    """Effort levels the OpenAI Responses provider actually accepts for ``model``."""

    return capabilities_for_responses_model(model).reasoning_efforts


def _openai_label(model: str, efforts: tuple[str, ...]) -> str:
    if not efforts:
        return f"{model} — no reasoning effort"
    return f"{model} — efforts: {', '.join(efforts)}"


OPENAI_SELECTABLE_MODELS: tuple[tuple[str, str], ...] = tuple(
    (model, _openai_label(model, efforts)) for model, efforts in OPENAI_MODEL_EFFORT_OPTIONS.items()
)

_CLAUDE_PICKER_SPEC = PickerSpec(
    title="Select a Claude model and reasoning effort",
    help_text="↑/↓ model · ←/→ effort · Enter confirm · Esc cancel",
    models=SELECTABLE_MODELS,
    efforts_fn=available_efforts,
    known_families="claude-opus-4-8, claude-sonnet-4-6, claude-haiku-4-5-*, claude-fable-5",
)

_OPENAI_PICKER_SPEC = PickerSpec(
    title="Select an OpenAI Responses model and reasoning effort",
    help_text="↑/↓ model · ←/→ effort · Enter confirm · Esc cancel",
    models=OPENAI_SELECTABLE_MODELS,
    efforts_fn=openai_available_efforts,
    known_families=", ".join(model for model, _ in OPENAI_SELECTABLE_MODELS[:8]) + ", ...",
)


def picker_spec_for_provider(provider: str) -> PickerSpec | None:
    """Return picker data for an explicitly selected provider, if supported."""

    if provider == CLAUDE_PROVIDER:
        return _CLAUDE_PICKER_SPEC
    if provider == OPENAI_PROVIDER:
        return _OPENAI_PICKER_SPEC
    return None
