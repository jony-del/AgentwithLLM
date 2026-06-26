"""Tests for the pure ModelPicker navigation state machine."""

from __future__ import annotations

import pytest

from agent_core.terminal.model_picker import ModelPicker

# A fixed catalog + effort map so the tests don't depend on the real model list.
_MODELS = (
    ("claude-opus-4-8", "Opus 4.8"),
    ("claude-sonnet-4-6", "Sonnet 4.6"),
    ("claude-haiku-4-5", "Haiku 4.5"),
)
_EFFORTS = {
    "claude-opus-4-8": ("low", "medium", "high", "xhigh", "max"),
    "claude-sonnet-4-6": ("low", "medium", "high", "max"),
    "claude-haiku-4-5": (),
}


def _picker(model="claude-opus-4-8", effort="high") -> ModelPicker:
    return ModelPicker(_MODELS, lambda m: _EFFORTS[m], model, effort)


def test_initial_state_reflects_current_model_and_effort() -> None:
    p = _picker("claude-sonnet-4-6", "max")
    assert p.current_model_id == "claude-sonnet-4-6"
    assert p.selection() == ("claude-sonnet-4-6", "max")


def test_initial_model_matches_by_substring() -> None:
    # A dated/suffixed id still resolves to the catalog family.
    p = ModelPicker(_MODELS, lambda m: _EFFORTS[m], "claude-haiku-4-5-20251001", None)
    assert p.current_model_id == "claude-haiku-4-5"


def test_up_down_clamp_at_edges() -> None:
    p = _picker()
    p.up()  # already at top
    assert p.model_idx == 0
    p.down(); p.down(); p.down()  # past the bottom
    assert p.model_idx == len(_MODELS) - 1


def test_left_right_move_within_efforts_and_clamp() -> None:
    p = _picker("claude-opus-4-8", "high")  # idx 2 of 5
    p.right()
    assert p.selection() == ("claude-opus-4-8", "xhigh")
    p.right()
    assert p.selection() == ("claude-opus-4-8", "max")
    p.right()  # clamp at the strongest
    assert p.selection() == ("claude-opus-4-8", "max")
    p.left(); p.left(); p.left(); p.left()
    assert p.selection() == ("claude-opus-4-8", "low")
    p.left()  # clamp at the weakest
    assert p.selection() == ("claude-opus-4-8", "low")


def test_effort_label_preserved_across_models_when_available() -> None:
    p = _picker("claude-opus-4-8", "max")  # max exists on opus
    p.down()  # → sonnet, which also has max
    assert p.selection() == ("claude-sonnet-4-6", "max")


def test_effort_clamps_when_label_unavailable_then_restores() -> None:
    p = _picker("claude-opus-4-8", "xhigh")  # xhigh only on opus
    p.down()  # → sonnet has no xhigh → clamp
    model, effort = p.selection()
    assert model == "claude-sonnet-4-6"
    assert effort in _EFFORTS["claude-sonnet-4-6"]  # a valid sonnet level
    p.up()  # back to opus → xhigh restored
    assert p.selection() == ("claude-opus-4-8", "xhigh")


def test_haiku_has_no_effort_selection_is_none() -> None:
    p = _picker("claude-opus-4-8", "high")
    p.down(); p.down()  # → haiku
    assert p.current_model_id == "claude-haiku-4-5"
    assert p.selection() == ("claude-haiku-4-5", None)
    # left/right are no-ops on a model with no efforts
    p.left(); p.right()
    assert p.selection() == ("claude-haiku-4-5", None)


def test_rows_marks_selected_model_and_effort() -> None:
    p = _picker("claude-opus-4-8", "high")
    rows = p.rows()
    selected = [r for r in rows if r.selected]
    assert len(selected) == 1 and selected[0].model_id == "claude-opus-4-8"
    # the high effort is the marked one on the selected row
    marked = [lvl for lvl, sel in selected[0].efforts if sel]
    assert marked == ["high"]
    # haiku row exposes no effort levels
    haiku = [r for r in rows if r.model_id == "claude-haiku-4-5"][0]
    assert haiku.efforts is None


def test_empty_models_rejected() -> None:
    with pytest.raises(ValueError):
        ModelPicker((), lambda m: (), None, None)
