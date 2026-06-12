from __future__ import annotations

import json
import sys
from typing import Any, Literal

from agent_core.models import ToolResult

# What confirm_tool may return: run the tool once, allow it for the rest of the
# session (never ask again for this tool name), or deny it.
PermissionChoice = Literal["once", "always", "deny"]


class AgentUI:
    """Event sink the agent loop emits to so a run can be made visible.

    The base class is a no-op (see ``NullUI``): every hook does nothing and the
    permission prompt denies. ``ReActAgent`` defaults to ``NullUI`` so calling
    ``run()`` directly (tests, library use, piped input) behaves exactly as it
    did before this layer existed. ``ConsoleUI`` is the interactive renderer the
    CLI installs when attached to a real terminal.
    """

    #: Whether this UI actually shows a live trace. The CLI uses it to decide
    #: whether the final answer was already displayed (so it doesn't print twice)
    #: and ``ReActAgent`` uses it to decide whether to wire an interactive prompter.
    is_live: bool = False

    def on_turn_start(self) -> None:
        """A new LLM turn is about to begin — reset any per-turn streaming state."""

    def on_text_delta(self, text: str) -> None:
        """A chunk of assistant answer text arrived (streaming)."""

    def on_thinking_delta(self, text: str) -> None:
        """A chunk of extended-thinking text arrived (streaming)."""

    def on_tool_args_delta(self, tool_name: str, partial_json: str) -> None:
        """A chunk of a tool call's argument JSON arrived (streaming)."""

    def on_thinking(self, text: str) -> None:
        """Extended-thinking block content emitted by the model this turn."""

    def on_reasoning(self, text: str) -> None:
        """Assistant text on a turn that also calls tools (the 'why' before acting)."""

    def on_tool_call(self, tool_name: str, risk: str, arguments: dict[str, Any]) -> None:
        """A tool is about to run (after permission, before/around execution)."""

    def on_tool_result(self, result: ToolResult) -> None:
        """The observation a tool produced."""

    def on_final(self, answer: str) -> None:
        """The model returned no tool calls — this is the final answer."""

    def on_todos(self, todos: list[Any]) -> None:
        """The task-planning tool (``update_todos``) rewrote the to-do list."""

    def on_stopped(self, reason: str, human: str) -> None:
        """A safety-net guard (cancel / max_steps / deadline) ended the run."""

    def confirm_tool(self, tool_name: str, risk: str, arguments: dict[str, Any]) -> PermissionChoice:
        """Ask the user whether to run a tool. Base/non-interactive answer: deny."""
        return "deny"


class NullUI(AgentUI):
    """The default: a silent sink. Present so the loop can always emit events."""

    is_live = False


# Minimal ANSI styling. Kept tiny and optional so output stays readable even when
# a terminal ignores escape codes; the CLI only installs ConsoleUI on a real TTY.
_DIM = "\x1b[2m"
_CYAN = "\x1b[36m"
_GREEN = "\x1b[32m"
_RED = "\x1b[31m"
_YELLOW = "\x1b[33m"
_RESET = "\x1b[0m"

# Risk levels that warrant a louder color in the permission panel.
_RISK_COLOR = {"read": _GREEN, "write": _YELLOW, "dangerous": _RED}


class ConsoleUI(AgentUI):
    """Render a Claude-Code-style live trace and interactive permission prompts.

    Supports two modes that coexist cleanly:

    - **Streaming**: the provider pushes ``on_thinking_delta`` / ``on_text_delta`` /
      ``on_tool_args_delta`` as tokens arrive and they are written incrementally.
      The per-turn ``on_thinking`` / ``on_reasoning`` / ``on_final`` calls that
      follow then act as *finalizers* — they only close the streamed block with a
      newline, so the text is never printed twice.
    - **Per-turn**: when nothing was streamed this turn (e.g. the FakeProvider with
      streaming off, or a tool-only turn), those same hooks print the full section,
      exactly as before streaming existed.
    """

    is_live = True

    def __init__(self, color: bool = True, preview_chars: int = 240) -> None:
        self._color = color
        self._preview_chars = preview_chars
        self._streamed_text = False
        self._streamed_thinking = False
        self._open_block = False  # a streamed line is open and needs a trailing newline

    # --- styling helpers ---------------------------------------------------

    def _style(self, text: str, code: str) -> str:
        return f"{code}{text}{_RESET}" if self._color else text

    def _emit(self, line: str) -> None:
        print(line, flush=True)

    def _write(self, text: str) -> None:
        """Write streamed text with no newline (the block is closed on finalize)."""
        print(text, end="", flush=True)

    def _close_block(self) -> None:
        if self._open_block:
            print("", flush=True)
            self._open_block = False

    def _preview(self, text: str) -> str:
        text = text.strip()
        if len(text) <= self._preview_chars:
            return text
        return text[: self._preview_chars] + " […]"

    # --- streaming deltas --------------------------------------------------

    def on_turn_start(self) -> None:
        self._streamed_text = False
        self._streamed_thinking = False
        self._open_block = False

    def on_thinking_delta(self, text: str) -> None:
        if not text:
            return
        if not self._streamed_thinking:
            self._emit(self._style("\N{THOUGHT BALLOON} thinking", _DIM))
            self._streamed_thinking = True
            self._open_block = True
        self._write(self._style(text, _DIM))

    def on_text_delta(self, text: str) -> None:
        if not text:
            return
        if not self._streamed_text:
            self._close_block()  # close a preceding thinking block first
            self._emit(self._style("● answer", _GREEN))
            self._streamed_text = True
            self._open_block = True
        self._write(text)

    def on_tool_args_delta(self, tool_name: str, partial_json: str) -> None:
        if partial_json:
            self._write(self._style(partial_json, _DIM))
            self._open_block = True

    # --- event hooks / finalizers -----------------------------------------

    def on_thinking(self, text: str) -> None:
        if self._streamed_thinking:
            self._close_block()
            return
        text = text.strip()
        if not text:
            return
        self._emit(self._style("\N{THOUGHT BALLOON} thinking", _DIM))
        self._emit(self._style(text, _DIM))

    def on_reasoning(self, text: str) -> None:
        # On a streamed turn the answer text is already on screen; just close it.
        if self._streamed_text:
            self._close_block()
            return
        text = text.strip()
        if not text:
            return
        self._emit(self._style("· reasoning", _CYAN))
        self._emit(text)

    def on_tool_call(self, tool_name: str, risk: str, arguments: dict[str, Any]) -> None:
        self._close_block()
        risk_tag = self._style(f"[{risk}]", _RISK_COLOR.get(risk, _CYAN))
        self._emit(f"{self._style('→', _CYAN)} {tool_name}{self._format_args(arguments)} {risk_tag}")

    def on_tool_result(self, result: ToolResult) -> None:
        self._close_block()
        marker = self._style("← ok", _GREEN) if result.ok else self._style("← err", _RED)
        self._emit(f"{marker}: {self._preview(result.content)}")

    def on_final(self, answer: str) -> None:
        if self._streamed_text:
            self._close_block()
            return
        self._emit(self._style("● answer", _GREEN))
        self._emit(answer)

    def on_todos(self, todos: list[Any]) -> None:
        self._close_block()
        if not todos:
            return
        self._emit(self._style("☰ plan", _CYAN))
        marks = {"pending": "○", "in_progress": "◐", "completed": "●"}
        colors = {"pending": _DIM, "in_progress": _YELLOW, "completed": _GREEN}
        for todo in todos:
            status = getattr(todo, "status", "pending")
            content = getattr(todo, "content", str(todo))
            self._emit(self._style(f"  {marks.get(status, '○')} {content}", colors.get(status, _DIM)))

    def on_stopped(self, reason: str, human: str) -> None:
        self._close_block()
        self._emit(self._style(f"■ stopped: {human}", _YELLOW))

    # --- interactive permission prompt ------------------------------------

    def confirm_tool(self, tool_name: str, risk: str, arguments: dict[str, Any]) -> PermissionChoice:
        risk_tag = self._style(f"{risk}", _RISK_COLOR.get(risk, _CYAN))
        self._emit(self._style("⚠ permission required", _YELLOW))
        self._emit(f"  tool: {tool_name}  ({risk_tag})")
        self._emit(f"  args: {self._format_args(arguments, full=True)}")
        # Every outcome must be a deliberate keystroke: an unrecognised key *and* a
        # bare Enter both re-prompt, so a stray/empty input can never silently deny a
        # tool. Only EOF (closed/piped stdin) denies without asking — there is no one
        # to answer, so a non-interactive caller fails closed instead of hanging.
        while True:
            try:
                answer = input("Allow? [y/once · a/always · n/deny] ").strip().lower()
            except EOFError:
                return "deny"
            if answer in {"a", "always"}:
                return "always"
            if answer in {"y", "yes", "once"}:
                return "once"
            if answer in {"n", "no", "deny"}:
                return "deny"
            self._emit(self._style("  ? didn't catch that — type y, a, or n", _YELLOW))

    @staticmethod
    def _format_args(arguments: dict[str, Any], full: bool = False) -> str:
        if not arguments:
            return "()"
        try:
            rendered = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            rendered = str(arguments)
        if not full and len(rendered) > 80:
            rendered = rendered[:80] + " […]"
        return f"({rendered})"
