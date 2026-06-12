from __future__ import annotations

import sys
import threading
from collections.abc import Callable


class KeyInterrupt:
    """Watch the terminal for the Esc key on a background thread and expose a
    cooperative cancellation flag the agent loop can poll.

    This mirrors Claude Code's "press Esc to interrupt": the keypress does not
    kill anything, it just sets a flag. The loop checks the flag at safe points
    (turn boundaries, between tool calls) and unwinds gracefully. In confirmation
    mode, Esc only marks an interrupt request; the next poll asks the user before
    setting the cancellation flag.

    Falls back to a no-op when stdin is not an interactive TTY (piped input,
    tests, CI) — there the flag simply never fires.
    """

    ESC = "\x1b"
    _WINDOWS_EXTENDED_PREFIXES = {"\x00", "\xe0"}

    def __init__(
        self,
        key: str = ESC,
        *,
        confirm: bool = False,
        confirm_prompt: str = "Interrupt current agent run? [y/N] ",
        input_func: Callable[[str], str] | None = None,
    ) -> None:
        self._key = key
        self._confirm = confirm
        self._confirm_prompt = confirm_prompt
        self._input = input_func or input
        self._event = threading.Event()
        self._pending = threading.Event()
        self._stop = threading.Event()
        self._confirm_lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def is_set(self) -> bool:
        if self._event.is_set():
            return True
        if self._confirm and self._pending.is_set():
            self._confirm_pending()
        return self._event.is_set()

    def __enter__(self) -> "KeyInterrupt":
        if not self._stdin_is_tty():
            return self  # nothing to watch; is_set() stays False
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            # The watch loops poll _stop every 50ms and never block on a read, so
            # this returns promptly and lets POSIX restore the terminal mode before
            # the caller reads input again.
            self._thread.join(timeout=0.5)

    @staticmethod
    def _stdin_is_tty() -> bool:
        try:
            return bool(sys.stdin) and sys.stdin.isatty()
        except (ValueError, OSError):
            return False

    def _watch(self) -> None:
        try:
            if sys.platform == "win32":
                self._watch_windows()
            else:
                self._watch_posix()
        except Exception:
            # Never let the watcher crash the program; just stop watching.
            pass

    def _watch_windows(self) -> None:
        import msvcrt  # type: ignore[import-not-found]

        while not self._stop.is_set() and not self._event.is_set():
            if self._pending.is_set():
                self._stop.wait(0.05)
                continue
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                if key in self._WINDOWS_EXTENDED_PREFIXES and msvcrt.kbhit():
                    msvcrt.getwch()
                    continue
                if key == self._key:
                    self._request_interrupt()
                    if not self._confirm:
                        return
            else:
                self._stop.wait(0.05)

    def _watch_posix(self) -> None:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop.is_set() and not self._event.is_set():
                if self._pending.is_set():
                    self._stop.wait(0.05)
                    continue
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if ready and self._read_posix_key(select.select) == self._key:
                    self._request_interrupt()
                    if not self._confirm:
                        return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _read_posix_key(self, select_func: Callable) -> str:
        key = sys.stdin.read(1)
        if key != self._key:
            return key

        sequence = [key]
        while True:
            ready, _, _ = select_func([sys.stdin], [], [], 0)
            if not ready:
                break
            sequence.append(sys.stdin.read(1))
        return key if len(sequence) == 1 else "".join(sequence)

    def _request_interrupt(self) -> None:
        if self._confirm:
            self._pending.set()
        else:
            self._event.set()

    def _confirm_pending(self) -> None:
        with self._confirm_lock:
            if self._event.is_set() or not self._pending.is_set():
                return
            try:
                answer = self._input(self._confirm_prompt).strip().lower()
            except EOFError:
                answer = ""
            self._pending.clear()
            if answer in {"y", "yes"}:
                self._event.set()
