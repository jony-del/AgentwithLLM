"""Interactive permission-prompt tests.

``TerminalRenderer.ask_permission_async`` is exercised headlessly with a
prompt_toolkit pipe input + ``DummyOutput`` inside a ``create_app_session`` (so
the ``PromptSession`` it builds reads the piped keys). The ``ConsoleUI``
worker-thread → main-loop bridge is tested separately with a stubbed prompt,
because the app-session context var does not propagate across
``run_coroutine_threadsafe``.
"""
import asyncio

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from agent_core.terminal.app import TerminalRenderer
from agent_core.ui import ConsoleUI


async def _ask(keys: str) -> str:
    r = TerminalRenderer(color=False)
    with create_pipe_input() as inp:
        inp.send_text(keys)
        with create_app_session(input=inp, output=DummyOutput()):
            return await asyncio.wait_for(
                r.ask_permission_async("echo", "write", {"text": "hi"}), timeout=5
            )


async def test_y_grants_once() -> None:
    assert await _ask("y") == "once"


async def test_a_grants_always() -> None:
    assert await _ask("a") == "always"


async def test_n_denies() -> None:
    assert await _ask("n") == "deny"


async def test_bare_enter_denies() -> None:
    # A bare Enter is not a deliberate decision → fail closed.
    assert await _ask("\r") == "deny"


async def test_ctrl_d_denies() -> None:
    assert await _ask("\x04") == "deny"


async def test_uppercase_is_accepted() -> None:
    assert await _ask("A") == "always"


async def test_permission_panel_is_printed(capsys) -> None:
    r = TerminalRenderer(color=False)
    with create_pipe_input() as inp:
        inp.send_text("y")
        with create_app_session(input=inp, output=DummyOutput()):
            await asyncio.wait_for(
                r.ask_permission_async("write_text_file", "write", {"path": "a.txt"}), timeout=5
            )
    out = capsys.readouterr().out
    assert "permission required" in out
    assert "write_text_file" in out


# --- the worker-thread → main-loop bridge ------------------------------------


async def test_confirm_tool_bridges_to_bound_loop(monkeypatch) -> None:
    ui = ConsoleUI(color=False)
    ui.bind_event_loop(asyncio.get_running_loop())

    async def fake_prompt(tool_name, risk, arguments):
        assert (tool_name, risk) == ("echo", "write")
        return "always"

    monkeypatch.setattr(ui._renderer, "ask_permission_async", fake_prompt)
    # confirm_tool blocks a worker thread and bridges the prompt onto the loop.
    result = await asyncio.to_thread(ui.confirm_tool, "echo", "write", {"x": 1})
    assert result == "always"


def test_confirm_tool_without_bound_loop_denies() -> None:
    ui = ConsoleUI(color=False)  # no bind_event_loop → no one to prompt
    assert ui.confirm_tool("echo", "write", {"x": 1}) == "deny"


async def test_confirm_tool_swallows_prompt_failure(monkeypatch) -> None:
    ui = ConsoleUI(color=False)
    ui.bind_event_loop(asyncio.get_running_loop())

    async def boom(*a, **k):
        raise RuntimeError("prompt blew up")

    monkeypatch.setattr(ui._renderer, "ask_permission_async", boom)
    result = await asyncio.to_thread(ui.confirm_tool, "echo", "write", {"x": 1})
    assert result == "deny"
