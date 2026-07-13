from __future__ import annotations

from dataclasses import dataclass
from typing import Any


OPENAI_REASONING_EFFORT_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

MODEL_EFFORTS: dict[str, tuple[str, ...]] = {
    # GPT-5.6
    "gpt-5.6": ("none", "low", "medium", "high", "xhigh", "max"),
    "gpt-5.6-sol": ("none", "low", "medium", "high", "xhigh", "max"),
    "gpt-5.6-terra": ("none", "low", "medium", "high", "xhigh", "max"),
    "gpt-5.6-luna": ("none", "low", "medium", "high", "xhigh", "max"),
    # GPT-5.5
    "gpt-5.5": ("none", "low", "medium", "high", "xhigh"),
    "gpt-5.5-pro": ("medium", "high", "xhigh"),
    # GPT-5.4
    "gpt-5.4": ("none", "low", "medium", "high", "xhigh"),
    "gpt-5.4-pro": ("medium", "high", "xhigh"),
    "gpt-5.4-mini": ("none", "low", "medium", "high", "xhigh"),
    "gpt-5.4-nano": ("none", "low", "medium", "high", "xhigh"),
    # Codex
    "gpt-5.3-codex": ("low", "medium", "high", "xhigh"),
    # GPT-5.2
    "gpt-5.2": ("none", "low", "medium", "high", "xhigh"),
    "gpt-5.2-pro": ("medium", "high", "xhigh"),
    # GPT-5.1
    "gpt-5.1": ("none", "low", "medium", "high"),
    # Original GPT-5 family
    "gpt-5": ("minimal", "low", "medium", "high"),
    "gpt-5-mini": ("minimal", "low", "medium", "high"),
    "gpt-5-nano": ("minimal", "low", "medium", "high"),
    "gpt-5-pro": ("high",),
    # Current non-deprecated o-series
    "o3": ("low", "medium", "high"),
}


@dataclass(frozen=True, slots=True)
class OpenAIResponsesCapabilities:
    """Provider-local capability profile for the official OpenAI Responses API.

    Unknown models intentionally receive the conservative default. This layer only
    controls OpenAI request shape; it is not a provider-neutral model validator.
    """

    supports_reasoning: bool = False
    reasoning_efforts: tuple[str, ...] = ()
    include_encrypted_reasoning: bool = False
    supports_reasoning_summary: bool = False


_DEFAULT_RESPONSES_CAPABILITIES = OpenAIResponsesCapabilities()


def _normalize_model_id(model: str | None) -> str:
    return (model or "").strip().lower()


def capabilities_for_responses_model(model: str | None) -> OpenAIResponsesCapabilities:
    """Return the conservative Responses API capability profile for ``model``."""

    efforts = MODEL_EFFORTS.get(_normalize_model_id(model))
    if efforts is None:
        return _DEFAULT_RESPONSES_CAPABILITIES
    return OpenAIResponsesCapabilities(
        supports_reasoning=True,
        reasoning_efforts=efforts,
        include_encrypted_reasoning=True,
        supports_reasoning_summary=True,
    )


def reasoning_effort_for_model(model: str | None, level: Any) -> str | None:
    """Resolve a Responses reasoning effort, or ``None`` when unsupported."""

    if not isinstance(level, str):
        return None
    requested = level.strip().lower()
    capabilities = capabilities_for_responses_model(model)
    return requested if requested in capabilities.reasoning_efforts else None


def supports_reasoning_replay(model: str | None) -> bool:
    """Whether encrypted reasoning output items may be replayed to ``model``."""

    return capabilities_for_responses_model(model).supports_reasoning
