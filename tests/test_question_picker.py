from __future__ import annotations

import asyncio

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from agent_core.terminal.question_picker import (
    QuestionOption,
    QuestionPicker,
    QuestionSpec,
    run_question_picker,
)


def _question(*, multi_select: bool = False) -> QuestionSpec:
    return QuestionSpec(
        "scope",
        "Which scope?",
        (
            QuestionOption("API", "Change the public API"),
            QuestionOption("CLI", "Change the terminal interface"),
        ),
        multi_select,
    )


def test_picker_appends_two_ui_owned_rows_with_dim_copy() -> None:
    rows = QuestionPicker(_question()).rows()
    assert [(row.kind, row.label, row.description) for row in rows[-2:]] == [
        ("custom", "Something else", "Type something"),
        ("discussion", "Discuss first", "Chat about this"),
    ]


def test_single_select_submits_the_focused_preset() -> None:
    picker = QuestionPicker(_question())
    picker.down()
    assert picker.activate() == "submit"
    response = picker.answer()
    assert response.kind == "answer"
    assert response.selected_options == ("CLI",)
    assert response.answer == "CLI"


def test_multi_select_combines_presets_in_source_order() -> None:
    picker = QuestionPicker(_question(multi_select=True))
    picker.toggle()
    picker.down()
    picker.toggle()
    assert picker.activate() == "submit"
    assert picker.answer().selected_options == ("API", "CLI")


def test_custom_text_is_additive_for_multi_select() -> None:
    picker = QuestionPicker(_question(multi_select=True))
    picker.toggle()
    picker.down()
    picker.down()
    assert picker.activate() == "custom"
    response = picker.answer("Keep backward compatibility")
    assert response.selected_options == ("API",)
    assert response.custom_text == "Keep backward compatibility"
    assert response.answer == "API; Keep backward compatibility"


def test_discussion_is_not_a_decision_and_omits_provisional_selections() -> None:
    picker = QuestionPicker(_question(multi_select=True))
    picker.toggle()
    picker.down()
    picker.down()
    picker.down()
    assert picker.activate() == "discussion"
    response = picker.discussion("Can we compare the migration risk first?")
    assert response.kind == "discussion"
    assert response.selected_options == ()
    assert response.discussion == "Can we compare the migration risk first?"


def test_navigation_wraps_across_business_and_system_rows() -> None:
    picker = QuestionPicker(_question())
    picker.up()
    assert picker.rows()[-1].focused
    picker.down()
    assert picker.rows()[0].focused


async def _interact(keys: str, *, multi_select: bool = False):
    with create_pipe_input() as inp:
        inp.send_text(keys)
        with create_app_session(input=inp, output=DummyOutput()):
            return await asyncio.wait_for(
                run_question_picker(_question(multi_select=multi_select)), timeout=5
            )


async def test_terminal_picker_submits_multiple_checked_options() -> None:
    response = await _interact(" \x1b[B \r", multi_select=True)
    assert response.selected_options == ("API", "CLI")


async def test_terminal_picker_collects_custom_text() -> None:
    response = await _interact("\x1b[B\x1b[B\rKeep Windows support\r")
    assert response.kind == "answer"
    assert response.custom_text == "Keep Windows support"


async def test_terminal_picker_collects_discussion_text() -> None:
    response = await _interact(
        "\x1b[B\x1b[B\x1b[B\rCompare the tradeoffs first\r"
    )
    assert response.kind == "discussion"
    assert response.discussion == "Compare the tradeoffs first"
