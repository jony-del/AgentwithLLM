"""Structured question picker used by ``ask_user_question``.

The state machine is deliberately independent from prompt_toolkit so navigation,
single/multiple selection, and the two UI-owned escape hatches are cheap to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


QuestionRowKind = Literal["option", "custom", "discussion"]
QuestionAction = Literal["submit", "custom", "discussion", "cancel"]


@dataclass(frozen=True, slots=True)
class QuestionOption:
    label: str
    description: str


@dataclass(frozen=True, slots=True)
class QuestionSpec:
    id: str
    question: str
    options: tuple[QuestionOption, ...]
    multi_select: bool = False

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "QuestionSpec":
        options = tuple(
            QuestionOption(str(item["label"]), str(item["description"]))
            for item in raw.get("options", [])
            if isinstance(item, dict) and "label" in item and "description" in item
        )
        return cls(
            id=str(raw.get("id", "")),
            question=str(raw.get("question", "Question")),
            options=options,
            multi_select=raw.get("multi_select") is True,
        )


@dataclass(frozen=True, slots=True)
class QuestionRow:
    kind: QuestionRowKind
    label: str
    description: str
    focused: bool
    checked: bool = False
    option_index: int | None = None


@dataclass(frozen=True, slots=True)
class QuestionResponse:
    id: str
    kind: Literal["answer", "discussion", "cancelled"]
    answer: str = ""
    selected_options: tuple[str, ...] = ()
    custom_text: str = ""
    discussion: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"id": self.id, "kind": self.kind, "answer": self.answer}
        if self.kind == "answer":
            payload["selected_options"] = list(self.selected_options)
            payload["custom_text"] = self.custom_text
        elif self.kind == "discussion":
            payload["discussion"] = self.discussion
        return payload


class QuestionPicker:
    """Pure navigation and selection state for one structured question."""

    def __init__(self, question: QuestionSpec) -> None:
        if len(question.options) < 2:
            raise ValueError("QuestionPicker needs at least two options")
        self.question = question
        self.cursor = 0
        self._selected: set[int] = set()

    @property
    def row_count(self) -> int:
        return len(self.question.options) + 2

    @property
    def selected_labels(self) -> tuple[str, ...]:
        return tuple(
            option.label
            for index, option in enumerate(self.question.options)
            if index in self._selected
        )

    def up(self) -> None:
        self.cursor = (self.cursor - 1) % self.row_count

    def down(self) -> None:
        self.cursor = (self.cursor + 1) % self.row_count

    def toggle(self) -> None:
        if not self.question.multi_select or self.cursor >= len(self.question.options):
            return
        if self.cursor in self._selected:
            self._selected.remove(self.cursor)
        else:
            self._selected.add(self.cursor)

    def activate(self) -> QuestionAction | None:
        option_count = len(self.question.options)
        if self.cursor == option_count:
            return "custom"
        if self.cursor == option_count + 1:
            return "discussion"
        if self.question.multi_select:
            return "submit" if self._selected else None
        self._selected = {self.cursor}
        return "submit"

    def answer(self, custom_text: str = "") -> QuestionResponse:
        custom = custom_text.strip()
        labels = self.selected_labels
        parts = list(labels)
        if custom:
            parts.append(custom)
        return QuestionResponse(
            self.question.id,
            "answer",
            answer="; ".join(parts),
            selected_options=labels,
            custom_text=custom,
        )

    def discussion(self, text: str) -> QuestionResponse:
        message = text.strip()
        return QuestionResponse(
            self.question.id,
            "discussion",
            answer=message,
            discussion=message,
        )

    def rows(self) -> list[QuestionRow]:
        rows = [
            QuestionRow(
                "option",
                option.label,
                option.description,
                focused=index == self.cursor,
                checked=index in self._selected,
                option_index=index,
            )
            for index, option in enumerate(self.question.options)
        ]
        rows.extend(
            (
                QuestionRow(
                    "custom",
                    "Something else",
                    "Type something",
                    focused=self.cursor == len(self.question.options),
                ),
                QuestionRow(
                    "discussion",
                    "Discuss first",
                    "Chat about this",
                    focused=self.cursor == len(self.question.options) + 1,
                ),
            )
        )
        return rows


async def _run_picker(picker: QuestionPicker) -> QuestionAction:
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.utils import get_cwidth

    def fragments() -> list[tuple[str, str]]:
        help_text = "Up/Down select · Enter confirm · Esc cancel"
        if picker.question.multi_select:
            help_text = "Up/Down select · Space toggle · Enter confirm · Esc cancel"
        rows = picker.rows()
        label_width = min(30, max(get_cwidth(row.label) for row in rows) + 2)
        out: list[tuple[str, str]] = [
            ("bold", f"{picker.question.question}\n"),
            ("fg:#808080", f"{help_text}\n\n"),
        ]
        for row in rows:
            focus = "> " if row.focused else "  "
            if row.kind == "option" and picker.question.multi_select:
                marker = "[x] " if row.checked else "[ ] "
            elif row.kind == "option":
                marker = "(*) " if row.checked else "( ) "
            else:
                marker = "    "
            label_style = "fg:#5fafff bold" if row.focused else ""
            padding = " " * max(2, label_width - get_cwidth(row.label))
            out.append((label_style, f"{focus}{marker}{row.label}{padding}"))
            out.append(("fg:#808080", f"{row.description}\n"))
        return out

    keys = KeyBindings()

    @keys.add("up")
    def _(event) -> None:
        picker.up()

    @keys.add("down")
    def _(event) -> None:
        picker.down()

    @keys.add(" ")
    def _(event) -> None:
        picker.toggle()

    @keys.add("enter")
    def _(event) -> None:
        action = picker.activate()
        if action is not None:
            event.app.exit(result=action)

    @keys.add("escape")
    @keys.add("c-c")
    @keys.add("c-d")
    def _(event) -> None:
        event.app.exit(result="cancel")

    app: Application[QuestionAction] = Application(
        layout=Layout(HSplit([Window(FormattedTextControl(fragments), wrap_lines=True)])),
        key_bindings=keys,
        mouse_support=False,
        full_screen=False,
    )
    try:
        return await app.run_async()
    except (EOFError, KeyboardInterrupt):
        return "cancel"


async def _read_text(prompt: str) -> str | None:
    """Read a non-empty, optionally multiline answer; ``None`` returns to the picker."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings

    # Importing the chat keybindings registers the portable Shift+Enter escape
    # sequences with prompt_toolkit. F24 is the project's internal Shift+Enter sentinel.
    from agent_core.terminal.keybindings import _SHIFT_ENTER_KEY

    keys = KeyBindings()

    @keys.add("enter")
    def _(event) -> None:
        event.current_buffer.validate_and_handle()

    @keys.add(_SHIFT_ENTER_KEY)
    @keys.add("escape", "enter")
    @keys.add("c-j")
    def _(event) -> None:
        event.current_buffer.insert_text("\n")

    @keys.add("escape")
    def _(event) -> None:
        event.app.exit(result=None)

    @keys.add("c-c")
    @keys.add("c-d")
    def _(event) -> None:
        event.app.exit(exception=EOFError)

    try:
        value: str = await PromptSession(key_bindings=keys, multiline=True).prompt_async(prompt)
    except (EOFError, KeyboardInterrupt):
        return None
    value = value.strip()
    return value or None


async def run_question_picker(question: QuestionSpec) -> QuestionResponse:
    """Ask one question, returning an explicit answer/discussion/cancellation outcome."""
    picker = QuestionPicker(question)
    while True:
        action = await _run_picker(picker)
        if action == "submit":
            return picker.answer()
        if action == "cancel":
            return QuestionResponse(question.id, "cancelled")
        if action == "custom":
            text = await _read_text("Type your answer: ")
            if text is not None:
                return picker.answer(text)
        if action == "discussion":
            text = await _read_text("What would you like to discuss? ")
            if text is not None:
                return picker.discussion(text)
