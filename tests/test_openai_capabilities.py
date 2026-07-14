from __future__ import annotations

import pytest

from agent_core.providers.openai_capabilities import (
    MODEL_EFFORTS,
    NON_REASONING_MODELS,
    OPENAI_MODEL_EFFORT_OPTIONS,
    OPENAI_REASONING_EFFORT_LEVELS,
    REASONING_MODEL_EFFORTS,
    OpenAIResponsesCapabilities,
    capabilities_for_responses_model,
    reasoning_effort_for_model,
    supports_reasoning_replay,
)


@pytest.mark.parametrize("model,efforts", REASONING_MODEL_EFFORTS.items())
def test_reasoning_models_use_model_specific_reasoning_profile(
    model: str, efforts: tuple[str, ...]
) -> None:
    capabilities = capabilities_for_responses_model(model)

    assert capabilities == OpenAIResponsesCapabilities(
        supports_reasoning=True,
        reasoning_efforts=efforts,
        include_encrypted_reasoning=True,
        supports_reasoning_summary=True,
    )
    assert supports_reasoning_replay(model) is True
    assert set(efforts).issubset(OPENAI_REASONING_EFFORT_LEVELS)


def test_picker_model_effort_options_include_reasoning_and_non_reasoning_models() -> None:
    assert MODEL_EFFORTS is OPENAI_MODEL_EFFORT_OPTIONS
    for model, efforts in REASONING_MODEL_EFFORTS.items():
        assert OPENAI_MODEL_EFFORT_OPTIONS[model] == efforts
    for model in NON_REASONING_MODELS:
        assert OPENAI_MODEL_EFFORT_OPTIONS[model] == ()


@pytest.mark.parametrize(
    ("model", "efforts"),
    [
        ("gpt-5.6", ("none", "low", "medium", "high", "xhigh", "max")),
        ("gpt-5.5-pro", ("medium", "high", "xhigh")),
        ("gpt-5", ("minimal", "low", "medium", "high")),
        ("gpt-5-pro", ("high",)),
        ("o3", ("low", "medium", "high")),
    ],
)
def test_selected_models_expose_expected_efforts(model: str, efforts: tuple[str, ...]) -> None:
    assert capabilities_for_responses_model(model).reasoning_efforts == efforts


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4o-mini",
        "o1",
        "o1-mini",
        "o4-mini",
        "unknown-model",
        "",
    ],
)
def test_non_reasoning_and_unknown_models_are_conservative(model: str) -> None:
    capabilities = capabilities_for_responses_model(model)

    assert capabilities == OpenAIResponsesCapabilities()
    assert supports_reasoning_replay(model) is False


@pytest.mark.parametrize("model", ["not-gpt-5", "local-gpt-5-compatible", "my-o3-router", "gpt-50"])
def test_reasoning_detection_avoids_false_positive_substrings(model: str) -> None:
    assert capabilities_for_responses_model(model) == OpenAIResponsesCapabilities()


@pytest.mark.parametrize(
    ("model", "effort", "expected"),
    [
        ("gpt-5.6", " NONE ", "none"),
        ("gpt-5.6", "MAX", "max"),
        ("gpt-5.6", "xhigh", "xhigh"),
        ("gpt-5", "minimal", "minimal"),
        ("gpt-5-pro", "high", "high"),
        ("gpt-5.5-pro", "medium", "medium"),
        ("o3", "low", "low"),
    ],
)
def test_reasoning_effort_is_gated_and_normalized(model: str, effort: str, expected: str) -> None:
    assert reasoning_effort_for_model(model, effort) == expected


@pytest.mark.parametrize(
    ("model", "effort"),
    [
        ("gpt-5.6", "minimal"),
        ("gpt-5.5", "max"),
        ("gpt-5.5-pro", "none"),
        ("gpt-5", "none"),
        ("gpt-5-pro", "medium"),
        ("o3", "xhigh"),
        ("gpt-4.1-nano", "high"),
        ("unknown-model", "high"),
        ("gpt-5.6", 123),
        ("gpt-5.6", None),
        ("gpt-5.6", True),
    ],
)
def test_unsupported_effort_is_omitted(model: str, effort) -> None:
    assert reasoning_effort_for_model(model, effort) is None
