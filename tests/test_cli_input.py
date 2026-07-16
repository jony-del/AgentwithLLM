"""Regression test for the TTY branch of ``agent_core.cli._async_input``.

The real prompt_toolkit imports (``PromptSession``, ``CompleteStyle``, ``HTML``,
``create_keybindings``) live inside the ``isatty()``-guarded branch, so the
non-TTY test suite normally takes the threaded-input fallback and never executes
them. A wrong import there (e.g. ``CompleteStyle`` from the wrong module) would
only crash a real terminal session. This test fakes a TTY + ``PromptSession`` so
those imports run under pytest and any breakage fails here instead.
"""

from __future__ import annotations

import prompt_toolkit
from prompt_toolkit.shortcuts import CompleteStyle

from agent_core import cli


class _FakeStdin:
    def isatty(self) -> bool:
        return True


class _FakePromptSession:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    async def prompt_async(self, message) -> str:
        return "typed line"


async def test_async_input_tty_branch_constructs_session(monkeypatch) -> None:
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin())
    monkeypatch.setattr(prompt_toolkit, "PromptSession", _FakePromptSession)
    # Force the singleton to be (re)built so the constructor path runs; clean up after
    # so the fake session never leaks into another test.
    if hasattr(cli._async_input, "_session"):
        delattr(cli._async_input, "_session")
    try:
        result = await cli._async_input("›", ui=None, completer=None)
        assert result == "typed line"
        session = cli._async_input._session
        assert isinstance(session, _FakePromptSession)
        # The completer wiring + CompleteStyle import must have executed.
        assert session.kwargs["multiline"] is True
        assert session.kwargs["complete_while_typing"] is True
        assert session.kwargs["complete_style"] == CompleteStyle.COLUMN
        toolbar_style = session.kwargs["style"].get_attrs_for_style_str(
            "class:bottom-toolbar"
        )
        assert toolbar_style.bgcolor == "default"
        assert toolbar_style.color == "ansibrightblack"
        assert toolbar_style.reverse is False
    finally:
        if hasattr(cli._async_input, "_session"):
            delattr(cli._async_input, "_session")
