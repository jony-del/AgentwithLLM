"""Key bindings for the interactive chat input (prompt_toolkit).

- ``Enter``                      submit the message.
- ``Shift+Enter``                insert a newline (modern terminals only — see below).
- ``Alt+Enter`` / ``Ctrl+J``     insert a newline (compose a multi-line message).
- ``Ctrl+O``                     toggle verbose detail for subsequent turns.

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

from collections.abc import Callable

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

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


def create_keybindings(on_toggle_verbose: Callable[[], None] | None = None) -> KeyBindings:
    kb = KeyBindings()

    @kb.add("enter")
    def _(event) -> None:
        event.current_buffer.validate_and_handle()

    @kb.add(_SHIFT_ENTER_KEY)       # Shift+Enter — modern terminals (via patched VT100 table)
    @kb.add("escape", "enter")      # Alt/Meta+Enter
    @kb.add("c-j")                   # Ctrl+J — reliable across all terminals
    def _(event) -> None:
        event.current_buffer.insert_text("\n")

    @kb.add("c-o")
    def _(event) -> None:
        if on_toggle_verbose is not None:
            on_toggle_verbose()

    return kb
