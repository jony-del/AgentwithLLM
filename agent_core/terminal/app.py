"""The terminal renderer: a single Rich ``Console`` that draws a flat, compact,
Claude-Code-style trace (``●`` tool headers, ``⎿`` result branches), markup-safe
streaming, syntax-highlighted diffs, folding for read/search bursts, and an
interactive (prompt_toolkit) permission prompt.

Design rules:
- ONE writer. Everything goes through ``self.console``; carriage-return/erase
  control sequences are emitted only on a real terminal (``_control``), so piped
  output stays clean (no stray ``\\r``/``\\x1b[K``).
- Model-derived text is printed as ``Text``/``markup=False`` so brackets in the
  model's output can never be mistaken for Rich markup.
"""
from __future__ import annotations

import json
from typing import Any, Literal

from agent_core.permission_types import (
    PermissionBehavior,
    PermissionDestination,
    PermissionRequest,
    PermissionResponse,
    PermissionUpdate,
)

from rich.console import Console
from rich.padding import Padding
from rich.text import Text

from .theme import claude_theme, risk_style, SYMBOLS
from .components import DiffBlock

PermissionChoice = Literal["once", "always", "deny"]

# Tools whose bursts are folded into a single summary line unless --verbose.
_FOLDABLE = {
    "view_file": "read",
    "read_file": "read",
    "search_text": "search",
    "glob": "search",
    "grep": "search",
    "list_dir": "list",
    "ls": "list",
}
_FOLD_VERB = {"read": "Read", "search": "Searched", "list": "Listed"}
_FOLD_UNIT = {"read": "files", "search": "patterns", "list": "dirs"}


def _human_count(n: int) -> str:
    """Compact human count: ``45200 -> '45.2k'``, ``1_000_000 -> '1.0M'``, else plain."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(int(n))


class TerminalRenderer:
    def __init__(self, color: bool = True, preview_chars: int = 240, verbose: bool = False):
        # no_color honors --color; force_terminal keeps ANSI when the CLI wired a
        # live UI onto a real TTY. color=False pins force_terminal=False (not None):
        # the contract is ZERO escape sequences, and auto-detection would let env
        # vars like FORCE_COLOR (which Rich honors even under no_color, re-enabling
        # bold/dim SGR) leak ANSI into piped/test output.
        # legacy_windows=False emits plain ANSI to the (UTF-8 reconfigured) stream
        # instead of Rich's Win32 console path — matching the old print()-based
        # renderer and avoiding narrow-codec (GBK) write failures on zh-CN Windows.
        self.console = Console(
            theme=claude_theme,
            no_color=not color,
            force_terminal=True if color else False,
            legacy_windows=False,
            highlight=False,
            soft_wrap=False,
        )
        self.preview_chars = preview_chars
        self.verbose = verbose

        # Streaming state: an open line awaits a trailing newline before the next block.
        self._open_block = False
        self._streamed_text = False
        self._streamed_thinking = False
        self._compaction_reactive = False

        # Folding state for consecutive read/search/list calls.
        self._folding = False
        self._folded: dict[str, int] = {"read": 0, "search": 0, "list": 0}

    # --- low-level writers -------------------------------------------------

    def emit(self, renderable: Any) -> None:
        self.console.print(renderable)

    def _control(self, sequence: str) -> None:
        """Emit a raw terminal control sequence, but only on a real terminal."""
        if self.console.is_terminal:
            self.console.file.write(sequence)
            self.console.file.flush()

    def _write_delta(self, text: str, style: str | None) -> None:
        """Stream a chunk with no trailing newline; markup-safe."""
        self.console.print(Text(text, style=style or ""), end="", markup=False, highlight=False)
        self.console.file.flush()

    def close_block(self) -> None:
        self._flush_fold()
        if self._open_block:
            self.console.print("")
            self._open_block = False

    # --- previews ----------------------------------------------------------

    def format_args(self, arguments: dict[str, Any], full: bool = False) -> str:
        if not arguments:
            return ""
        try:
            rendered = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            rendered = str(arguments)
        limit = 2000 if full else 80
        if len(rendered) > limit:
            rendered = rendered[:limit] + " […]"
        return rendered

    def preview(self, text: str) -> str:
        text = text.strip()
        if len(text) <= self.preview_chars:
            return text
        return text[: self.preview_chars] + " […]"

    # --- streaming ---------------------------------------------------------

    def reset_stream_state(self) -> None:
        self._flush_fold()
        self._streamed_text = False
        self._streamed_thinking = False
        self._open_block = False

    def _start_thinking(self) -> None:
        if not self._streamed_thinking:
            self.emit(Text(f"{SYMBOLS['thinking']} Thinking…", style="thinking"))
            self._streamed_thinking = True
            self._open_block = True

    def _start_answer(self) -> None:
        if not self._streamed_text:
            self.close_block()
            self.console.print(Text(f"{SYMBOLS['answer']} ", style="answer"), end="", markup=False)
            self._streamed_text = True
            self._open_block = True

    def write_thinking_delta(self, text: str) -> None:
        if text:
            self._start_thinking()
            self._write_delta(text, "thinking")

    def write_text_delta(self, text: str) -> None:
        if text:
            self._start_answer()
            self._write_delta(text, None)

    def write_tool_args_delta(self, partial_json: str) -> None:
        # The compact ``● tool(args)`` header (printed once permission resolves)
        # is clearer than a raw streamed argument blob, so streamed args are dropped.
        return

    # --- per-turn finalizers ----------------------------------------------

    def print_thinking(self, text: str) -> None:
        if self._streamed_thinking:
            self.close_block()
            return
        text = text.strip()
        if text:
            self.emit(Text(f"{SYMBOLS['thinking']} Thinking…", style="thinking"))
            self.emit(Text(text, style="thinking"))

    def print_reasoning(self, text: str) -> None:
        if self._streamed_text:
            self.close_block()
            return
        text = text.strip()
        if text:
            self.emit(Text(text, style="answer"))

    def print_final(self, answer: str) -> None:
        if self._streamed_text:
            self.close_block()
            return
        self.close_block()
        line = Text()
        line.append(f"{SYMBOLS['answer']} ", style="answer")
        line.append(answer)
        self.emit(line)

    # --- tool stream -------------------------------------------------------

    def print_tool_call(self, tool_name: str, risk: str, arguments: dict[str, Any], label: str | None = None) -> None:
        if not self.verbose and tool_name in _FOLDABLE:
            if not self._folding:
                self.close_block()
                self._folding = True
            self._folded[_FOLDABLE[tool_name]] += 1
            self._render_fold_live()
            return

        self.close_block()
        summary = label if label is not None else self.format_args(arguments)
        header = Text()
        header.append(f"{SYMBOLS['tool_call']} ", style="success")
        header.append(tool_name, style="tool_name")
        header.append(f"({summary})")
        header.append(f"  [{risk}]", style=risk_style(risk))
        self.emit(header)

    def print_tool_result(self, ok: bool, content: str, diff: str | None = None) -> None:
        if self._folding:
            # Folded reads/searches don't print their own result; surface only errors.
            if ok:
                return
            self._flush_fold()

        self.close_block()
        if diff:
            branch = Text(f"  {SYMBOLS['branch']}  ", style="branch")
            branch.append("diff", style="dim")
            self.emit(branch)
            self.emit(Padding(DiffBlock(diff), (0, 0, 0, 5)))
            return

        line = Text(f"  {SYMBOLS['branch']}  ", style="branch")
        if ok:
            line.append(self.preview(content), style="dim")
        else:
            line.append(f"{SYMBOLS['fail']} {self.preview(content)}", style="danger")
        self.emit(line)

    # --- folding -----------------------------------------------------------

    def _fold_summary(self) -> str:
        parts = [
            f"{_FOLD_VERB[k]} {v} {_FOLD_UNIT[k]}"
            for k, v in self._folded.items()
            if v > 0
        ]
        return ", ".join(parts)

    def _render_fold_live(self) -> None:
        summary = self._fold_summary()
        if not summary:
            return
        # Overwrite the same line on a TTY; on a pipe stay silent until flush.
        self._control("\r\x1b[K")
        if self.console.is_terminal:
            self.console.print(Text(f"{SYMBOLS['tool_call']} {summary}…", style="dim"), end="")

    def _flush_fold(self) -> None:
        if not self._folding:
            return
        summary = self._fold_summary()
        self._control("\r\x1b[K")
        if summary:
            self.emit(Text(f"{SYMBOLS['tool_call']} {summary}", style="dim"))
        self._folding = False
        self._folded = {"read": 0, "search": 0, "list": 0}

    # --- planning / progress ----------------------------------------------

    def print_todos(self, todos: list[Any]) -> None:
        self.close_block()
        if not todos:
            return
        self.emit(Text(f"{SYMBOLS['plan']} plan", style="info"))
        marks = {
            "pending": SYMBOLS["plan_pending"],
            "in_progress": SYMBOLS["plan_progress"],
            "completed": SYMBOLS["plan_completed"],
        }
        colors = {"pending": "dim", "in_progress": "warning", "completed": "success"}
        for todo in todos:
            status = getattr(todo, "status", "pending")
            content = getattr(todo, "content", str(todo))
            mark = marks.get(status, SYMBOLS["plan_pending"])
            self.emit(Text(f"  {mark} {content}", style=colors.get(status, "dim")))

    def print_tool_use_summary(self, label: str) -> None:
        self.close_block()
        self.emit(Text(f"{SYMBOLS['compacting']} {label}", style="dim"))

    def print_stopped(self, reason: str, human: str) -> None:
        self.close_block()
        self.emit(Text(f"{SYMBOLS['stopped']} stopped: {human}", style="warning"))

    def print_token_usage(self, usage: dict[str, Any]) -> None:
        """Dim one-line running token figure, emitted under each model turn."""
        self.close_block()
        context = int(usage.get("context_tokens", 0) or 0)
        window = int(usage.get("window", 0) or 0)
        in_tok = int(usage.get("input_tokens", 0) or 0)
        out_tok = int(usage.get("output_tokens", 0) or 0)
        if context <= 0 and in_tok <= 0 and out_tok <= 0:
            return
        ctx_part = f"ctx {_human_count(context)}"
        if window > 0:
            pct = context / window * 100
            ctx_part += f"/{_human_count(window)} ({pct:.1f}%)"
        # Split into conversation vs. the fixed per-run baseline (system prompt + CLAUDE.md
        # + tool schemas) when the loop reports it, so a fresh/cleared session reads ~0 chat.
        if "conversation_tokens" in usage:
            conv = max(0, min(int(usage.get("conversation_tokens", 0) or 0), context))
            base = max(0, context - conv)
            ctx_part += f" · chat {_human_count(conv)} / base {_human_count(base)}"
        line = Text(f"{SYMBOLS['compacting']} {ctx_part} · {_human_count(in_tok)} in / {_human_count(out_tok)} out", style="dim")
        self.emit(line)

    def print_run_completed(self, stats: dict[str, Any]) -> None:
        self.close_block()
        duration = stats.get("duration", 0)
        minutes, seconds = divmod(int(duration), 60)
        dur_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
        reason = stats.get("reason", "completed")
        steps = stats.get("steps", 0)
        tool_counts = stats.get("tool_counts", {})

        line = Text()
        if reason == "completed":
            line.append(f"{SYMBOLS['ok']} ", style="success")
            line.append(f"Done in {dur_str} · {steps} steps", style="dim")
        else:
            line.append(f"{SYMBOLS['fail']} ", style="danger")
            line.append(f"Stopped ({reason}) after {dur_str} · {steps} steps", style="warning")
        details = [f"{c} {t}" for t, c in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)]
        if details:
            line.append("  ·  " + ", ".join(details), style="dim")
        in_tok = int(stats.get("input_tokens", 0) or 0)
        out_tok = int(stats.get("output_tokens", 0) or 0)
        if in_tok > 0 or out_tok > 0:
            total = in_tok + out_tok
            line.append(f"  ·  {_human_count(total)} tok ({_human_count(in_tok)} in / {_human_count(out_tok)} out)", style="dim")
        self.emit(line)

    # --- compaction --------------------------------------------------------

    def start_compaction(self, reactive: bool) -> None:
        self.close_block()
        self._compaction_reactive = reactive

    def update_compaction(self, fraction: float, stage: str) -> None:
        fraction = max(0.0, min(1.0, fraction))
        width = 14
        filled = round(fraction * width)
        bar = "█" * filled + "░" * (width - filled)
        style = "warning" if self._compaction_reactive else "dim"
        line = Text(
            f"{SYMBOLS['compacting']} compacting ▕{bar}▏ {int(fraction * 100)}%  {stage}",
            style=style,
        )
        self._control("\r")
        self.console.print(line, end="")
        self._control("\x1b[K")

    def end_compaction(self, before_chars: int, after_chars: int, detail: str, reactive: bool) -> None:
        def _human(n: int) -> str:
            return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

        sizes = f"{_human(before_chars)}→{_human(after_chars)} chars"
        suffix = f" · {detail}" if detail else ""
        self._control("\r\x1b[K")
        if reactive:
            self.emit(Text(f"{SYMBOLS['warning']} context overflowed, compacted {sizes}{suffix}", style="warning"))
        else:
            self.emit(Text(f"{SYMBOLS['compacting']} compacted {sizes}{suffix}", style="dim"))

    # --- interactive permission prompt ------------------------------------

    def _print_permission_panel(self, tool_name: str, risk: str, arguments: dict[str, Any]) -> None:
        from rich.panel import Panel
        from rich.table import Table

        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim")
        table.add_column()
        table.add_row("tool", Text(f"{tool_name} [{risk}]", style=risk_style(risk)))
        table.add_row("args", Text(self.format_args(arguments, full=True), style="dim"))
        self.emit(
            Panel(
                table,
                title=f"{SYMBOLS['warning']} permission required",
                title_align="left",
                border_style="warning",
                padding=(0, 1),
                expand=False,
            )
        )

    async def ask_permission_async(
        self, tool_name: str, risk: str, arguments: dict[str, Any]
    ) -> PermissionChoice:
        """Render the permission panel and resolve a single y/a/n keypress.

        Runs on the main event loop (bridged from the executor's worker thread by
        ``ConsoleUI.confirm_tool``). EOF / Ctrl-C deny — a closed stdin or an
        explicit interrupt fails closed rather than hanging.
        """
        from prompt_toolkit import PromptSession
        from prompt_toolkit.key_binding import KeyBindings

        self._print_permission_panel(tool_name, risk, arguments)

        kb = KeyBindings()

        def _resolve(value: PermissionChoice):
            def handler(event) -> None:
                event.app.exit(result=value)
            return handler

        for keys, value in (("yY", "once"), ("aA", "always"), ("nN", "deny")):
            for key in keys:
                kb.add(key)(_resolve(value))

        @kb.add("enter")
        def _(event) -> None:  # bare Enter is not a decision → deny rather than guess
            event.app.exit(result="deny")

        @kb.add("c-c")
        @kb.add("c-d")
        def _(event) -> None:
            event.app.exit(result="deny")

        session: PromptSession = PromptSession(key_bindings=kb)
        try:
            answer = await session.prompt_async("Allow? [y/once · a/always · n/deny] ")
        except (EOFError, KeyboardInterrupt):
            return "deny"
        return answer if answer in ("once", "always", "deny") else "deny"

    async def ask_permission_request_async(self, request: PermissionRequest) -> PermissionResponse:
        """Prompt for once/session/persistent least-privilege grants."""
        from prompt_toolkit import PromptSession
        from prompt_toolkit.key_binding import KeyBindings

        self._print_permission_panel(request.tool_name, request.risk, dict(request.arguments))
        if request.tool_name == "exit_plan" and request.arguments.get("requested_permissions"):
            return await self._ask_plan_permission_bundle_async(request)
        if request.suggestions:
            self.emit("Suggested scope: " + ", ".join(item.rule for item in request.suggestions))
        kb = KeyBindings()

        def bind(key: str, value: str) -> None:
            @kb.add(key)
            def _(event) -> None:
                event.app.exit(result=value)

        bind("y", "once")
        if not request.session_grants_disabled:
            bind("s", "session")
        if not request.persistent_grants_disabled:
            bind("l", "local")
            bind("p", "project")
            bind("u", "user")
        bind("n", "deny")

        @kb.add("enter")
        def _(event) -> None:
            event.app.exit(result="once")

        @kb.add("c-c")
        @kb.add("c-d")
        def _(event) -> None:
            event.app.exit(result="deny")

        suffix = "y/once · n/deny"
        if not request.session_grants_disabled:
            suffix += " · s/session"
        if not request.persistent_grants_disabled:
            suffix += " · l/local · p/project · u/user"
        try:
            choice = await PromptSession(key_bindings=kb).prompt_async(f"Allow? [{suffix}] ")
        except (EOFError, KeyboardInterrupt):
            return PermissionResponse(False, reason="permission prompt cancelled")
        if choice == "deny" or choice not in {"once", "session", "local", "project", "user"}:
            return PermissionResponse(False, reason="user rejected")
        if choice == "once":
            return PermissionResponse(True, reason="user confirmed once")
        destination = PermissionDestination(choice)
        if destination is not PermissionDestination.SESSION:
            confirmed = await self._confirm_persistent_permission_async(destination, len(request.suggestions))
            if not confirmed:
                return PermissionResponse(False, reason="persistent grant confirmation rejected")
        updates = tuple(
            PermissionUpdate(PermissionBehavior.ALLOW, item.rule, destination)
            for item in request.suggestions
        )
        return PermissionResponse(True, updates, f"user allowed suggested scope in {choice}")

    async def _ask_plan_permission_bundle_async(
        self, request: PermissionRequest
    ) -> PermissionResponse:
        raw_items = request.arguments.get("requested_permissions", [])
        selected: list[dict[str, Any]] = []
        self.emit("Review requested session permissions individually:")
        for item in raw_items if isinstance(raw_items, list) else []:
            if not isinstance(item, dict):
                continue
            rule = str(item.get("rule", ""))
            reason = str(item.get("reason", ""))
            if await self._confirm_yes_no_async(f"Allow {rule} ({reason})? [y/N] "):
                selected.append(dict(item))
        if not await self._confirm_yes_no_async("Approve this plan and leave plan mode? [y/N] "):
            return PermissionResponse(False, reason="plan approval rejected")
        updated = dict(request.arguments)
        updated["requested_permissions"] = selected
        return PermissionResponse(
            True,
            reason=f"plan approved with {len(selected)} scoped session grant(s)",
            updated_arguments=updated,
        )

    async def _confirm_yes_no_async(self, prompt: str) -> bool:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.key_binding import KeyBindings

        kb = KeyBindings()
        for key, value in (("y", True), ("n", False)):
            @kb.add(key)
            def _(event, resolved=value) -> None:
                event.app.exit(result=resolved)
        @kb.add("enter")
        @kb.add("c-c")
        @kb.add("c-d")
        def _(event) -> None:
            event.app.exit(result=False)
        try:
            answer = await PromptSession(key_bindings=kb).prompt_async(prompt)
        except (EOFError, KeyboardInterrupt):
            return False
        return answer is True

    async def _confirm_persistent_permission_async(
        self, destination: PermissionDestination, count: int
    ) -> bool:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.key_binding import KeyBindings

        kb = KeyBindings()
        for key, value in (("y", True), ("n", False)):
            @kb.add(key)
            def _(event, resolved=value) -> None:
                event.app.exit(result=resolved)

        @kb.add("enter")
        @kb.add("c-c")
        @kb.add("c-d")
        def _(event) -> None:
            event.app.exit(result=False)
        try:
            answer = await PromptSession(key_bindings=kb).prompt_async(
                f"Persist {count} allow rule(s) to {destination.value}? [y/N] "
            )
        except (EOFError, KeyboardInterrupt):
            return False
        return answer is True
