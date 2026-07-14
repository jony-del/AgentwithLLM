"""Interactive six-mode permission picker for ``/permissions``."""

from __future__ import annotations

from dataclasses import dataclass

from agent_core.permissions import PermissionMode, permission_mode_label

PERMISSION_MODE_DESCRIPTIONS: dict[PermissionMode, str] = {
    PermissionMode.DEFAULT: "Ask before edits and actions",
    PermissionMode.ACCEPTEDITS: "Automatically accept workspace file edits",
    PermissionMode.PLAN: "Read-only investigation and planning",
    PermissionMode.AUTO: "Use an AI safety classifier for actions",
    PermissionMode.DONTASK: "Deny actions that would need a prompt",
    PermissionMode.BYPASS: "Allow unmatched actions; safety rules still apply",
}


@dataclass(frozen=True, slots=True)
class PermissionPickerRow:
    mode: PermissionMode
    label: str
    description: str
    selected: bool


class PermissionPicker:
    def __init__(self, current: PermissionMode | str) -> None:
        self.modes = list(PermissionMode)
        resolved = PermissionMode(current)
        self.index = self.modes.index(resolved)

    def up(self) -> None:
        self.index = (self.index - 1) % len(self.modes)

    def down(self) -> None:
        self.index = (self.index + 1) % len(self.modes)

    def selection(self) -> PermissionMode:
        return self.modes[self.index]

    def rows(self) -> list[PermissionPickerRow]:
        return [
            PermissionPickerRow(
                mode,
                permission_mode_label(mode),
                PERMISSION_MODE_DESCRIPTIONS[mode],
                index == self.index,
            )
            for index, mode in enumerate(self.modes)
        ]


async def run_permission_picker(current: PermissionMode | str) -> PermissionMode | None:
    """Return the selected mode, or ``None`` outside a TTY/on cancellation."""
    import sys

    if not (sys.stdin and sys.stdin.isatty()):
        return None

    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    picker = PermissionPicker(current)

    def fragments() -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = [
            ("bold", "Select a permission mode\n"),
            ("fg:#808080", "↑/↓ select · Enter confirm · Esc cancel\n\n"),
        ]
        for row in picker.rows():
            marker = "❯ " if row.selected else "  "
            style = "fg:#5fafff bold" if row.selected else ""
            out.append((style, f"{marker}{row.mode.value:<12} {row.label}\n"))
            if row.selected:
                out.append(("fg:#808080", f"    {row.description}\n"))
        return out

    kb = KeyBindings()

    @kb.add("up")
    def _(event) -> None:
        picker.up()

    @kb.add("down")
    def _(event) -> None:
        picker.down()

    @kb.add("enter")
    def _(event) -> None:
        event.app.exit(result=picker.selection())

    @kb.add("escape")
    @kb.add("c-c")
    @kb.add("q")
    def _(event) -> None:
        event.app.exit(result=None)

    app: Application = Application(
        layout=Layout(HSplit([Window(FormattedTextControl(fragments), wrap_lines=True)])),
        key_bindings=kb,
        mouse_support=False,
        full_screen=False,
    )
    try:
        return await app.run_async()
    except (EOFError, KeyboardInterrupt):
        return None
