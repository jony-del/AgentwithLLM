"""Interactive ``/model`` picker: choose a model (↑/↓) and a reasoning effort (←/→).

Split into a pure state machine (:class:`ModelPicker`, fully unit-testable without a
terminal) and a thin prompt_toolkit ``Application`` runner (:func:`run_model_picker`).
Each provider supplies its own model list and effort function so the picker can only
ever offer what that provider will actually send.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from agent_core.model_catalog import SELECTABLE_MODELS, available_efforts


@dataclass(slots=True)
class PickerRow:
    """One rendered row: a model plus (for the highlighted model) its effort options."""

    model_id: str
    label: str
    selected: bool
    #: ``None`` → the model has no effort levels; else (level, is_selected) per level.
    efforts: list[tuple[str, bool]] | None


class ModelPicker:
    """Pure navigation state for the model/effort picker.

    ``↑/↓`` move between models (preserving the chosen effort *label* across models when
    that level is available, otherwise clamping); ``←/→`` move within the current model's
    effort levels. :meth:`selection` returns ``(model_id, effort | None)``.
    """

    def __init__(
        self,
        models: Sequence[tuple[str, str]],
        efforts_fn: Callable[[str], tuple[str, ...]],
        current_model: str | None = None,
        current_effort: str | None = None,
    ) -> None:
        if not models:
            raise ValueError("ModelPicker needs at least one model")
        self.models = list(models)
        self._efforts_fn = efforts_fn
        self.model_idx = self._index_of_model(current_model)
        # The last explicitly-chosen effort label, restored when moving to a model that
        # supports it. Seeded from the current effort.
        self._desired_effort = (current_effort or "").lower() or None
        self.effort_idx = 0
        self._resync_effort()

    # -- queries --------------------------------------------------------------

    @property
    def current_model_id(self) -> str:
        return self.models[self.model_idx][0]

    @property
    def current_efforts(self) -> tuple[str, ...]:
        return self._efforts_fn(self.current_model_id)

    def selection(self) -> tuple[str, str | None]:
        efforts = self.current_efforts
        if not efforts:
            return self.current_model_id, None
        return self.current_model_id, efforts[self.effort_idx]

    def rows(self) -> list[PickerRow]:
        out: list[PickerRow] = []
        for i, (mid, label) in enumerate(self.models):
            selected = i == self.model_idx
            levels = self._efforts_fn(mid)
            if not levels:
                efforts: list[tuple[str, bool]] | None = None
            else:
                sel = self.effort_idx if selected else -1
                efforts = [(lvl, j == sel) for j, lvl in enumerate(levels)]
            out.append(PickerRow(mid, label, selected, efforts))
        return out

    # -- navigation -----------------------------------------------------------

    def up(self) -> None:
        if self.model_idx > 0:
            self.model_idx -= 1
            self._resync_effort()

    def down(self) -> None:
        if self.model_idx < len(self.models) - 1:
            self.model_idx += 1
            self._resync_effort()

    def left(self) -> None:
        if self.current_efforts and self.effort_idx > 0:
            self.effort_idx -= 1
            self._desired_effort = self.current_efforts[self.effort_idx]

    def right(self) -> None:
        efforts = self.current_efforts
        if efforts and self.effort_idx < len(efforts) - 1:
            self.effort_idx += 1
            self._desired_effort = efforts[self.effort_idx]

    # -- internals ------------------------------------------------------------

    def _index_of_model(self, current_model: str | None) -> int:
        if current_model:
            name = current_model.lower()
            for i, (mid, _) in enumerate(self.models):
                low = mid.lower()
                if low == name or low in name or name in low:
                    return i
        return 0

    def _resync_effort(self) -> None:
        """Re-aim ``effort_idx`` at the desired level for the now-current model."""
        efforts = self.current_efforts
        if not efforts:
            self.effort_idx = 0
            return
        if self._desired_effort in efforts:
            self.effort_idx = efforts.index(self._desired_effort)
        else:
            self.effort_idx = min(self.effort_idx, len(efforts) - 1)


async def run_model_picker(
    current_model: str | None,
    current_effort: str | None,
    *,
    models: Sequence[tuple[str, str]] = SELECTABLE_MODELS,
    efforts_fn: Callable[[str], tuple[str, ...]] = available_efforts,
    title: str = "Select a model and reasoning effort",
    help_text: str = "↑/↓ model · ←/→ effort · Enter confirm · Esc cancel",
) -> tuple[str, str | None] | None:
    """Drive the picker in a prompt_toolkit ``Application``; ``None`` if cancelled.

    Returns ``None`` (no change) when stdin is not a TTY or the user presses Esc/Ctrl-C/q.
    On confirm (Enter) returns ``(model_id, effort | None)``.
    """
    import sys

    if not (sys.stdin and sys.stdin.isatty()):
        return None

    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    picker = ModelPicker(models, efforts_fn, current_model, current_effort)

    def fragments() -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = [
            ("bold", f"{title}\n"),
            ("fg:#808080", f"{help_text}\n\n"),
        ]
        for row in picker.rows():
            marker = "❯ " if row.selected else "  "
            row_style = "fg:#5fafff bold" if row.selected else ""
            out.append((row_style, f"{marker}{row.label}\n"))
            if not row.selected:
                continue
            if row.efforts is None:
                out.append(("fg:#808080", "      (no effort levels)\n"))
            else:
                out.append(("", "      effort: "))
                for level, sel in row.efforts:
                    out.append(
                        ("fg:#5fafff bold", f"[{level}] ") if sel else ("fg:#808080", f"{level} ")
                    )
                out.append(("", "\n"))
        return out

    kb = KeyBindings()

    @kb.add("up")
    def _(event) -> None:
        picker.up()

    @kb.add("down")
    def _(event) -> None:
        picker.down()

    @kb.add("left")
    def _(event) -> None:
        picker.left()

    @kb.add("right")
    def _(event) -> None:
        picker.right()

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
