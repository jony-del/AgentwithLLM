import sys
from types import SimpleNamespace

from agent_core.interrupt import KeyInterrupt


def test_confirmed_esc_sets_interrupt() -> None:
    interrupt = KeyInterrupt(confirm=True, input_func=lambda prompt: "y")

    interrupt._request_interrupt()

    assert interrupt.is_set()


def test_declined_esc_does_not_set_interrupt() -> None:
    interrupt = KeyInterrupt(confirm=True, input_func=lambda prompt: "n")

    interrupt._request_interrupt()

    assert not interrupt.is_set()


def test_declined_esc_can_be_requested_again() -> None:
    answers = iter(["n", "y"])
    interrupt = KeyInterrupt(confirm=True, input_func=lambda prompt: next(answers))

    interrupt._request_interrupt()
    assert not interrupt.is_set()

    interrupt._request_interrupt()
    assert interrupt.is_set()


def test_empty_confirmation_does_not_set_interrupt() -> None:
    interrupt = KeyInterrupt(confirm=True, input_func=lambda prompt: "")

    interrupt._request_interrupt()

    assert not interrupt.is_set()


def test_ctrl_b_background_request_is_consumed_once() -> None:
    interrupt = KeyInterrupt()
    interrupt._background.set()

    assert interrupt.consume_background()
    assert not interrupt.consume_background()


def test_windows_watcher_ignores_non_interrupt_key(monkeypatch) -> None:
    interrupt = KeyInterrupt(confirm=True, input_func=lambda prompt: "y")
    keys = ["q"]

    def kbhit() -> bool:
        if keys:
            return True
        interrupt._stop.set()
        return False

    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(kbhit=kbhit, getwch=lambda: keys.pop(0)))

    interrupt._watch_windows()

    assert not interrupt.is_set()


def test_windows_watcher_ignores_extended_key(monkeypatch) -> None:
    interrupt = KeyInterrupt(confirm=True, input_func=lambda prompt: "y")
    keys = ["\xe0", "K"]

    def kbhit() -> bool:
        if keys:
            return True
        interrupt._stop.set()
        return False

    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(kbhit=kbhit, getwch=lambda: keys.pop(0)))

    interrupt._watch_windows()

    assert not interrupt.is_set()


def test_posix_reader_keeps_lone_escape_as_interrupt_key(monkeypatch) -> None:
    class Input:
        def read(self, size: int) -> str:
            return "\x1b"

    monkeypatch.setattr(sys, "stdin", Input())
    interrupt = KeyInterrupt()

    key = interrupt._read_posix_key(lambda read, write, error, timeout: ([], [], []))

    assert key == "\x1b"


def test_posix_reader_treats_escape_sequence_as_other_key(monkeypatch) -> None:
    class Input:
        def __init__(self) -> None:
            self.keys = ["\x1b", "[", "A"]

        def read(self, size: int) -> str:
            return self.keys.pop(0)

    stdin = Input()
    monkeypatch.setattr(sys, "stdin", stdin)
    interrupt = KeyInterrupt()

    def select_func(read, write, error, timeout):
        return ([sys.stdin], [], []) if stdin.keys else ([], [], [])

    key = interrupt._read_posix_key(select_func)

    assert key == "\x1b[A"
