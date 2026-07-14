"""Tests for per-model effort capability (``available_efforts``)."""

from __future__ import annotations

from agent_core.model_catalog import openai_available_efforts, picker_spec_for_provider
from agent_core.providers.claude import (
    ALL_EFFORT_LEVELS,
    _effort_for_model,
    available_efforts,
)


def test_haiku_has_no_effort() -> None:
    assert available_efforts("claude-haiku-4-5-20251001") == ()


def test_sonnet_46_supports_up_to_max_but_not_xhigh() -> None:
    assert available_efforts("claude-sonnet-4-6") == ("low", "medium", "high", "max")


def test_opus_45_base_levels_only() -> None:
    assert available_efforts("claude-opus-4-5") == ("low", "medium", "high")


def test_opus_46_adds_max() -> None:
    assert available_efforts("claude-opus-4-6") == ("low", "medium", "high", "max")


def test_opus_48_and_fable_support_all_five() -> None:
    assert available_efforts("claude-opus-4-8") == ALL_EFFORT_LEVELS
    assert available_efforts("claude-fable-5") == ALL_EFFORT_LEVELS


def test_unknown_model_has_no_effort() -> None:
    assert available_efforts("gpt-4") == ()


def test_available_efforts_never_drifts_from_effort_gating() -> None:
    # Single source of truth: a level is "available" iff the provider would actually send
    # it. Guard against the two implementations diverging.
    for model in (
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-6",
        "claude-opus-4-5",
        "claude-opus-4-6",
        "claude-opus-4-8",
        "claude-fable-5",
    ):
        expected = tuple(
            level for level in ALL_EFFORT_LEVELS if _effort_for_model(model, level) == level
        )
        assert available_efforts(model) == expected


def test_openai_picker_efforts_use_responses_capabilities() -> None:
    spec = picker_spec_for_provider("openai")

    assert spec is not None
    assert spec.efforts_fn is openai_available_efforts
    assert spec.efforts_fn("gpt-5.6") == ("none", "low", "medium", "high", "xhigh", "max")
    assert spec.efforts_fn("gpt-4.1-nano") == ()
    assert spec.efforts_fn("custom-model") == ()


def test_picker_specs_are_selected_only_by_explicit_provider() -> None:
    assert picker_spec_for_provider("claude") is not None
    assert picker_spec_for_provider("openai") is not None
    assert picker_spec_for_provider("openai-compat") is None
    assert picker_spec_for_provider("fake") is None
