from prompt_toolkit.keys import Keys

from agent_core.terminal import keybindings
from agent_core.terminal.keybindings import create_keybindings


class _Buffer:
    """Stand-in for prompt_toolkit's Buffer: tracks text, reset() and delete() calls."""

    def __init__(self, text: str = "") -> None:
        self.text = text
        self.reset_called = 0
        self.delete_called = 0

    def reset(self) -> None:
        self.reset_called += 1
        self.text = ""

    def delete(self) -> None:
        self.delete_called += 1


class _App:
    def __init__(self) -> None:
        self.exit_exception: type[BaseException] | None = None

    def exit(self, *, exception=None) -> None:
        self.exit_exception = exception


class _Event:
    def __init__(self, buffer: _Buffer) -> None:
        self.current_buffer = buffer
        self.app = _App()


def _ctrl_c(kb):
    for binding in kb.bindings:
        if binding.keys == (Keys.ControlC,):
            return binding.handler
    raise AssertionError("no Ctrl+C binding registered")


def _ctrl_d(kb):
    for binding in kb.bindings:
        if binding.keys == (Keys.ControlD,):
            return binding.handler
    raise AssertionError("no Ctrl+D binding registered")


def test_ctrl_d_exits_on_empty_buffer() -> None:
    handler = _ctrl_d(create_keybindings())

    buffer = _Buffer("")
    event = _Event(buffer)
    handler(event)

    assert event.app.exit_exception is EOFError
    assert buffer.delete_called == 0  # exited, did not edit


def test_ctrl_d_deletes_char_on_nonempty_buffer() -> None:
    handler = _ctrl_d(create_keybindings())

    buffer = _Buffer("text")
    event = _Event(buffer)
    handler(event)

    assert event.app.exit_exception is None  # did NOT exit
    assert buffer.delete_called == 1


def test_single_ctrl_c_clears_buffer_in_place(monkeypatch) -> None:
    monkeypatch.setattr(keybindings.time, "monotonic", lambda: 100.0)
    handler = _ctrl_c(create_keybindings())

    event = _Event(_Buffer("some typed text"))
    handler(event)  # must not raise

    assert event.current_buffer.reset_called == 1
    assert event.app.exit_exception is None  # cleared, did NOT exit


def test_double_ctrl_c_exits(monkeypatch) -> None:
    clock = {"now": 100.0}
    monkeypatch.setattr(keybindings.time, "monotonic", lambda: clock["now"])
    handler = _ctrl_c(create_keybindings())

    buffer = _Buffer("draft")
    event = _Event(buffer)

    handler(event)  # first press: clears, arms exit
    assert buffer.reset_called == 1
    assert event.app.exit_exception is None

    clock["now"] += 0.5  # second press within the window, buffer now empty
    handler(event)
    assert event.app.exit_exception is EOFError
    assert buffer.reset_called == 1  # no further reset on exit


def test_typing_between_presses_clears_instead_of_exiting(monkeypatch) -> None:
    clock = {"now": 100.0}
    monkeypatch.setattr(keybindings.time, "monotonic", lambda: clock["now"])
    handler = _ctrl_c(create_keybindings())

    buffer = _Buffer("draft")
    event = _Event(buffer)

    handler(event)  # clears + arms
    clock["now"] += 0.3
    buffer.text = "typed again"  # user refilled the buffer

    handler(event)  # within window, but buffer non-empty → clear, do not exit
    assert event.app.exit_exception is None
    assert buffer.reset_called == 2


def test_slow_second_press_does_not_exit(monkeypatch) -> None:
    clock = {"now": 100.0}
    monkeypatch.setattr(keybindings.time, "monotonic", lambda: clock["now"])
    handler = _ctrl_c(create_keybindings())

    buffer = _Buffer("draft")
    event = _Event(buffer)

    handler(event)  # clears + arms
    clock["now"] += keybindings._DOUBLE_CTRL_C_SECONDS + 0.5  # outside the window

    handler(event)  # empty buffer but too slow → clear/re-arm, do not exit
    assert event.app.exit_exception is None
    assert buffer.reset_called == 2  # took the clear path again, not the exit path
