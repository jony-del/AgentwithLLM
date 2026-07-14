import asyncio
from types import SimpleNamespace

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
        self.tasks = []
        self.invalidated = 0

    def exit(self, *, exception=None) -> None:
        self.exit_exception = exception

    def create_background_task(self, coroutine) -> None:
        self.tasks.append(asyncio.create_task(coroutine))

    def invalidate(self) -> None:
        self.invalidated += 1


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


def _enter(kb):
    # Enter is historically Ctrl+M; it is the binding registered via kb.add("enter").
    for binding in kb.bindings:
        if binding.keys == (Keys.ControlM,):
            return binding.handler
    raise AssertionError("no Enter binding registered")


def _by_key(kb, key):
    for binding in kb.bindings:
        if binding.keys == (key,):
            return binding.handler
    raise AssertionError(f"no binding for {key}")


class _CompletionBuffer(_Buffer):
    """Buffer stand-in tracking submit / accept / completion-restart / cursor moves."""

    def __init__(self, text: str = "", current_completion=None, has_state: bool = True) -> None:
        super().__init__(text)
        self.complete_state = (
            SimpleNamespace(current_completion=current_completion) if has_state else None
        )
        self.submitted = 0
        self.applied = []
        self.started = []  # select_first args passed to start_completion
        self.cursor_right_called = 0

    @property
    def document(self):
        return SimpleNamespace(text_before_cursor=self.text)

    def validate_and_handle(self) -> None:
        self.submitted += 1

    def apply_completion(self, completion) -> None:
        self.applied.append(completion)

    def start_completion(self, select_first: bool = False) -> None:
        self.started.append(select_first)

    def cursor_right(self) -> None:
        self.cursor_right_called += 1

    def delete_before_cursor(self, count: int = 1) -> None:
        if self.text:
            self.text = self.text[:-count]


# --- Enter: accept AND run -----------------------------------------------------


def test_enter_submits_when_no_completion_open() -> None:
    handler = _enter(create_keybindings())
    buffer = _CompletionBuffer("a message", current_completion=None)
    handler(_Event(buffer))
    assert buffer.submitted == 1 and buffer.applied == []


def test_enter_accepts_and_runs_highlighted_completion() -> None:
    handler = _enter(create_keybindings())
    sentinel = object()
    buffer = _CompletionBuffer("/res", current_completion=sentinel)
    handler(_Event(buffer))
    # one keypress both accepts the completion and submits — no second Enter needed
    assert buffer.applied == [sentinel] and buffer.submitted == 1


def test_enter_submits_when_complete_state_absent() -> None:
    handler = _enter(create_keybindings())
    buffer = _CompletionBuffer("hi", has_state=False)
    handler(_Event(buffer))
    assert buffer.submitted == 1 and buffer.applied == []


# --- Tab / Right: accept WITHOUT running ---------------------------------------


def test_tab_accepts_highlighted_without_running() -> None:
    handler = _by_key(create_keybindings(), Keys.ControlI)
    sentinel = object()
    buffer = _CompletionBuffer("/res", current_completion=sentinel)
    handler(_Event(buffer))
    assert buffer.applied == [sentinel] and buffer.submitted == 0


def test_tab_opens_completion_when_nothing_highlighted() -> None:
    handler = _by_key(create_keybindings(), Keys.ControlI)
    buffer = _CompletionBuffer("/re", current_completion=None)
    handler(_Event(buffer))
    assert buffer.applied == [] and buffer.started == [True]


async def test_shift_tab_cycles_permission_without_touching_buffer(monkeypatch) -> None:
    calls = []

    async def immediate(callback):
        callback()

    monkeypatch.setattr(keybindings, "run_in_terminal", immediate)
    handler = _by_key(create_keybindings(on_cycle_permission=lambda: calls.append("cycle")), Keys.BackTab)
    buffer = _CompletionBuffer("unfinished input", current_completion=None)
    event = _Event(buffer)

    handler(event)
    await asyncio.gather(*event.app.tasks)

    assert calls == ["cycle"]
    assert buffer.text == "unfinished input"
    assert event.app.invalidated == 1


def test_right_accepts_highlighted_without_running() -> None:
    handler = _by_key(create_keybindings(), Keys.Right)
    sentinel = object()
    buffer = _CompletionBuffer("/res", current_completion=sentinel)
    handler(_Event(buffer))
    assert buffer.applied == [sentinel] and buffer.submitted == 0


def test_right_moves_cursor_when_nothing_highlighted() -> None:
    handler = _by_key(create_keybindings(), Keys.Right)
    buffer = _CompletionBuffer("hello", current_completion=None)
    handler(_Event(buffer))
    assert buffer.cursor_right_called == 1 and buffer.applied == []


# --- Backspace: re-filter the menu in a slash context --------------------------


def test_backspace_restarts_completion_in_slash_context() -> None:
    handler = _by_key(create_keybindings(), Keys.ControlH)
    buffer = _CompletionBuffer("/res")
    handler(_Event(buffer))
    assert buffer.text == "/re"  # deleted one char
    assert buffer.started == [False]  # menu re-opened to re-filter


def test_backspace_does_not_complete_on_prose() -> None:
    handler = _by_key(create_keybindings(), Keys.ControlH)
    buffer = _CompletionBuffer("hello")
    handler(_Event(buffer))
    assert buffer.text == "hell" and buffer.started == []


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
