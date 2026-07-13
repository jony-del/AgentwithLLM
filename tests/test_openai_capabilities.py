from __future__ import annotations

import pytest

from agent_core.providers.openai_capabilities import (
    OpenAIResponsesCapabilities,
    capabilities_for_responses_model,
    reasoning_effort_for_model,
    supports_reasoning_replay,
)


@pytest.mark.parametrize("model", ["gpt-5", "gpt-5.6", "gpt-5.6-sol", "o1", "o1-mini", "o3", "o4-mini"])
def test_reasoning_models_use_reasoning_profile(model: str) -> None:
    capabilities = capabilities_for_responses_model(model)

    assert capabilities == OpenAIResponsesCapabilities(
        supports_reasoning=True,
        reasoning_efforts=("low", "medium", "high"),
        include_encrypted_reasoning=True,
        supports_reasoning_summary=True,
    )
    assert supports_reasoning_replay(model) is True


@pytest.mark.parametrize("model", ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini", "unknown-model", ""])
def test_non_reasoning_and_unknown_models_are_conservative(model: str) -> None:
    capabilities = capabilities_for_responses_model(model)

    assert capabilities == OpenAIResponsesCapabilities()
    assert supports_reasoning_replay(model) is False


@pytest.mark.parametrize("model", ["not-gpt-5", "local-gpt-5-compatible", "my-o3-router", "gpt-50"])
def test_reasoning_detection_avoids_false_positive_substrings(model: str) -> None:
    assert capabilities_for_responses_model(model) == OpenAIResponsesCapabilities()


@pytest.mark.parametrize("effort", ["low", "medium", "high", " HIGH "])
def test_reasoning_effort_is_gated_and_normalized(effort: str) -> None:
    assert reasoning_effort_for_model("gpt-5.6", effort) == effort.strip().lower()


@pytest.mark.parametrize("effort", ["xhigh", "max", "minimal", 123, None, True])
def test_unsupported_effort_is_omitted(effort) -> None:
    assert reasoning_effort_for_model("gpt-5.6", effort) is None
    assert reasoning_effort_for_model("gpt-4.1-nano", effort) is None
