"""Key bindings for the interactive chat input (prompt_toolkit).

- ``Enter``                submit the message.
- ``Alt+Enter`` / ``Ctrl+J``  insert a newline (compose a multi-line message).
- ``Ctrl+O``               toggle verbose detail for subsequent turns.

NOTE: ``Shift+Enter`` is intentionally not relied on — most terminals send it
identically to ``Enter``, so a dedicated newline chord (Alt+Enter / Ctrl+J) is
used instead.
"""
from collections.abc import Callable

from prompt_toolkit.key_binding import KeyBindings


def create_keybindings(on_toggle_verbose: Callable[[], None] | None = None) -> KeyBindings:
    kb = KeyBindings()

    @kb.add("enter")
    def _(event) -> None:
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")  # Alt/Meta+Enter
    @kb.add("c-j")              # Ctrl+J — reliable across terminals
    def _(event) -> None:
        event.current_buffer.insert_text("\n")

    @kb.add("c-o")
    def _(event) -> None:
        if on_toggle_verbose is not None:
            on_toggle_verbose()

    return kb
