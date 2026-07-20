"""Key bindings for the interactive chat input (prompt_toolkit).

- ``Enter``                      submit the message.
- ``Shift+Enter``                insert a newline (modern terminals only — see below).
- ``Alt+Enter`` / ``Ctrl+J``     insert a newline (compose a multi-line message).
- ``Ctrl+C``                     clear the current input in place; press again within
                                  a short window (on an empty buffer) to exit the chat.
- ``Ctrl+D``                     exit the chat on an empty buffer (EOF); on a non-empty
                                  buffer it deletes the character under the cursor.
- ``Esc``                        cooperatively cancel the active agent run.
- ``Ctrl+B``                     move the foreground shell process to background.
- ``Ctrl+O``                     show the transcript (verbose fallback outside chat).
- ``Ctrl+T``                     show tasks and queued input.
- ``Ctrl+R``                     search input history.
- ``Ctrl+L``                     redraw the terminal.
- ``Shift+Tab``                  cycle default/acceptedits/plan/auto modes.

``Shift+Enter`` requires a terminal that emits a distinct escape sequence for the
shifted key.  Two common encodings are supported:

- **kitty / CSI-u protocol**: ``ESC[13;2u``  (Windows Terminal ≥ 1.20, WezTerm,
  kitty, foot, Alacritty ≥ 0.14)
- **xterm modifyOtherKeys=1**: ``ESC[27;2;13~``  (xterm, gnome-terminal, VTE)

On terminals that cannot distinguish ``Shift+Enter`` from plain ``Enter``
(legacy conhost, PuTTY, older mintty) the keystroke behaves as ``Enter``
(submit) — there is no breakage.  ``Alt+Enter`` / ``Ctrl+J`` remain as
reliable cross-terminal alternatives.
"""
from __future__ import annotations

import time
from collections.abc import Callable

from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

# A second Ctrl+C within this window (on an already-empty buffer) exits the chat.
_DOUBLE_CTRL_C_SECONDS: float = 1.0

# ---------------------------------------------------------------------------
# Teach prompt_toolkit to recognise Shift+Enter.
#
# prompt_toolkit 3.x has no built-in ``ShiftEnter`` key; the VT100 parser maps
# the two most common Shift+Enter escape sequences to ``c-m`` (plain Enter).
# We override those entries so they instead map to an obscure but valid key
# (``F24``), which we then bind to "insert newline".
#
# The patch only *adds / overwrites* entries in the static dict and is safe to
# execute at import time.  It is idempotent if this module is re-imported.
# ---------------------------------------------------------------------------

_SHIFT_ENTER_KEY: Keys = Keys.F24  # sentinel: F24 is never produced by a real keypress

_SHIFT_ENTER_SEQUENCES: dict[str, Keys] = {
    "\x1b[13;2u":     _SHIFT_ENTER_KEY,  # kitty / CSI-u protocol
    "\x1b[27;2;13~":  _SHIFT_ENTER_KEY,  # xterm modifyOtherKeys
}


def _patch_ansi_sequences() -> None:
    """Register Shift+Enter escape sequences in prompt_toolkit's VT100 table."""
    try:
        from prompt_toolkit.input.vt100_parser import ANSI_SEQUENCES
    except ImportError:  # defensive — prompt_toolkit internal API change
        return
    for seq, key in _SHIFT_ENTER_SEQUENCES.items():
        ANSI_SEQUENCES[seq] = key


_patch_ansi_sequences()


def create_keybindings(
    on_toggle_verbose: Callable[[], None] | None = None,
    on_cycle_permission: Callable[[], None] | None = None,
    *,
    is_running: Callable[[], bool] | None = None,
    on_interrupt: Callable[[], None] | None = None,
    on_background: Callable[[], None] | None = None,
    on_transcript: Callable[[], None] | None = None,
    on_tasks: Callable[[], None] | None = None,
    on_history_search: Callable[[], None] | None = None,
    on_redraw: Callable[[], None] | None = None,
    on_recall_queue: Callable[[], str] | None = None,
) -> KeyBindings:
    kb = KeyBindings()

    @kb.add("enter")
    def _(event) -> None:
        # When the completion dropdown is open with an item highlighted (↑/↓), Enter
        # accepts that completion AND submits in one keypress — pick a command and run
        # it without a second Enter. With nothing highlighted it just submits the typed
        # line, so bare `/resume` (no selection) still runs as typed.
        buffer = event.current_buffer
        state = buffer.complete_state
        if state is not None and state.current_completion is not None:
            buffer.apply_completion(state.current_completion)
        buffer.validate_and_handle()

    @kb.add("tab")
    def _(event) -> None:
        # Tab accepts the highlighted completion WITHOUT running it (keep editing); with
        # nothing highlighted it opens the menu / selects the first match.
        buffer = event.current_buffer
        state = buffer.complete_state
        if state is not None and state.current_completion is not None:
            buffer.apply_completion(state.current_completion)
        else:
            buffer.start_completion(select_first=True)

    @kb.add("s-tab")
    @kb.add("escape", "Z")
    @kb.add("escape", "m")
    def _(event) -> None:
        if on_cycle_permission is None:
            return

        # The callback can show the one-time unsandboxed warning via input(). Suspend
        # prompt_toolkit while it runs so the confirmation and the current edit buffer
        # never compete for terminal input, then redraw the live bottom toolbar.
        async def cycle() -> None:
            await run_in_terminal(on_cycle_permission)
            event.app.invalidate()

        event.app.create_background_task(cycle())

    @kb.add("right")
    def _(event) -> None:
        # Right-arrow accepts a highlighted completion inline (no run); otherwise it is
        # an ordinary cursor move.
        buffer = event.current_buffer
        state = buffer.complete_state
        if state is not None and state.current_completion is not None:
            buffer.apply_completion(state.current_completion)
        else:
            buffer.cursor_right()

    @kb.add("backspace")
    def _(event) -> None:
        # Delete one char, then re-open completion while still in a single-line slash
        # context so the menu re-filters as the user erases letters (complete_while_typing
        # only auto-triggers on insertion, not deletion). Non-slash prose is untouched.
        buffer = event.current_buffer
        buffer.delete_before_cursor(1)
        text = buffer.document.text_before_cursor
        if "\n" not in text and text.startswith("/"):
            buffer.start_completion(select_first=False)

    @kb.add(_SHIFT_ENTER_KEY)       # Shift+Enter — modern terminals (via patched VT100 table)
    @kb.add("escape", "enter")      # Alt/Meta+Enter
    @kb.add("c-j")                   # Ctrl+J — reliable across all terminals
    def _(event) -> None:
        event.current_buffer.insert_text("\n")

    last_ctrl_c = {"at": 0.0}  # monotonic timestamp of the previous Ctrl+C, for double-press exit

    @kb.add("c-c")
    def _(event) -> None:
        # First Ctrl+C clears the current input in place; a second Ctrl+C within
        # _DOUBLE_CTRL_C_SECONDS on an already-empty buffer exits the chat (raises
        # EOFError in prompt_async, mirroring Ctrl-D). Typing refills the buffer, so
        # the next Ctrl+C clears again instead of exiting. Overriding c-c suppresses
        # prompt_toolkit's default abort (grey-out + new line + KeyboardInterrupt).
        buffer = event.current_buffer
        now = time.monotonic()
        if not buffer.text and (now - last_ctrl_c["at"]) <= _DOUBLE_CTRL_C_SECONDS:
            event.app.exit(exception=EOFError)
            return
        buffer.reset()
        last_ctrl_c["at"] = now

    @kb.add("c-d")
    def _(event) -> None:
        # On an empty buffer Ctrl+D exits the chat (EOFError in prompt_async,
        # mirroring Ctrl-D's POSIX EOF semantics and our double-Ctrl+C exit).
        # On a non-empty buffer it falls back to the default delete-forward-char
        # so editing still works. We bind it explicitly because our custom
        # KeyBindings + multiline mode + Windows terminals make prompt_toolkit's
        # default c-d handling unreliable (it often never fires the exit).
        buffer = event.current_buffer
        if not buffer.text:
            event.app.exit(exception=EOFError)
            return
        buffer.delete()

    @kb.add("c-o")
    def _(event) -> None:
        if on_transcript is not None:
            on_transcript()
        elif on_toggle_verbose is not None:
            on_toggle_verbose()

    @kb.add("escape")
    def _(event) -> None:
        if is_running is not None and is_running() and on_interrupt is not None:
            on_interrupt()

    @kb.add("c-b")
    def _(event) -> None:
        if is_running is not None and is_running() and on_background is not None:
            on_background()

    @kb.add("c-t")
    def _(event) -> None:
        if on_tasks is not None:
            on_tasks()

    @kb.add("c-r")
    def _(event) -> None:
        if on_history_search is not None:
            on_history_search()
        else:
            event.app.current_buffer.start_history_lines_completion()

    @kb.add("c-l")
    def _(event) -> None:
        if on_redraw is not None:
            on_redraw()
        event.app.renderer.clear()
        event.app.invalidate()

    if on_recall_queue is not None:
        @kb.add("up")
        def _(event) -> None:
            buffer = event.current_buffer
            if buffer.text:
                buffer.history_backward()
                return
            recalled = on_recall_queue()
            if recalled:
                buffer.insert_text(recalled)
            else:
                buffer.history_backward()

    return kb
