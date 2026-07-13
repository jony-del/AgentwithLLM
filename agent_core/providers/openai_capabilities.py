from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_BASE_REASONING_EFFORTS = ("low", "medium", "high")


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
_REASONING_RESPONSES_CAPABILITIES = OpenAIResponsesCapabilities(
    supports_reasoning=True,
    reasoning_efforts=_BASE_REASONING_EFFORTS,
    include_encrypted_reasoning=True,
    supports_reasoning_summary=True,
)

# Anchored family patterns only: avoid accidental matches such as
# "local-gpt-5-compatible" while still accepting version/snapshot suffixes.
_REASONING_MODEL_PATTERNS = (
    re.compile(r"^gpt-5(?:[.-].*)?$"),
    re.compile(r"^o(?:1|3|4)(?:[.-].*)?$"),
)


def _normalize_model_id(model: str | None) -> str:
    return (model or "").strip().lower()


def capabilities_for_responses_model(model: str | None) -> OpenAIResponsesCapabilities:
    """Return the conservative Responses API capability profile for ``model``."""

    model_id = _normalize_model_id(model)
    if any(pattern.match(model_id) for pattern in _REASONING_MODEL_PATTERNS):
        return _REASONING_RESPONSES_CAPABILITIES
    return _DEFAULT_RESPONSES_CAPABILITIES


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
