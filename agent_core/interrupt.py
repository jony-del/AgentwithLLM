from __future__ import annotations

import sys
import threading


class KeyInterrupt:
    """Watch the terminal for the Esc key on a background thread and expose a
    cooperative cancellation flag the agent loop can poll.

    This mirrors Claude Code's "press Esc to interrupt": the keypress does not
    kill anything, it just sets a flag. The loop checks the flag at safe points
    (turn boundaries, between tool calls) and unwinds gracefully.

    Falls back to a no-op when stdin is not an interactive TTY (piped input,
    tests, CI) — there the flag simply never fires.
    """

    ESC = "\x1b"

    def __init__(self, key: str = ESC) -> None:
        self._key = key
        self._event = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def is_set(self) -> bool:
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

        while not self._stop.is_set():
            if msvcrt.kbhit():
                if msvcrt.getwch() == self._key:
                    self._event.set()
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
            while not self._stop.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if ready and sys.stdin.read(1) == self._key:
                    self._event.set()
                    return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
