from __future__ import annotations

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

    def on_tool_call(
        self, tool_name: str, risk: str, arguments: dict[str, Any], label: str | None = None
    ) -> None:
        """A tool is about to run (after permission, before/around execution).

        ``label`` is the tool's optional compact one-line argument summary
        (``Tool.render_args``); ``None`` means the UI should derive its own.
        """

    def on_tool_result(self, result: ToolResult, diff: str | None = None) -> None:
        """The observation a tool produced.

        ``diff`` is an optional unified-diff string (``Tool.render_result``) a
        write/edit tool produced so the UI can render it specially.
        """

    def on_final(self, answer: str) -> None:
        """The model returned no tool calls — this is the final answer."""

    def on_todos(self, todos: list[Any]) -> None:
        """The task-planning tool (``update_todos``) rewrote the to-do list."""

    def on_tool_use_summary(self, label: str, tool_names: list[str]) -> None:
        """A one-line progress label for the tool batch that just ran (UI-only).

        Generated asynchronously by a cheap model; never sent to the API or stored in the
        transcript. The base sink ignores it — only a live UI renders it.
        """

    def on_stopped(self, reason: str, human: str) -> None:
        """A safety-net guard (cancel / max_steps / deadline) ended the run."""

    def on_run_completed(self, stats: dict[str, Any]) -> None:
        """The run has finished. ``stats`` contains duration, steps, tool_counts, and reason."""

    def on_compaction_start(self, reactive: bool) -> None:
        """Context compaction began (``reactive`` = emergency after an overflow)."""

    def on_compaction_progress(self, fraction: float, stage: str) -> None:
        """A compaction stage finished; ``fraction`` advances 0.0 → 1.0."""

    def on_compaction_end(self, before_chars: int, after_chars: int, detail: str, reactive: bool) -> None:
        """Compaction finished; report the net size change (never the content)."""

    def bind_event_loop(self, loop: Any) -> None:
        """Capture the main event loop so a worker-thread prompt can bridge back to it.

        No-op on the base/silent sink; a live UI stores it (see ``ConsoleUI``).
        """

    def confirm_tool(self, tool_name: str, risk: str, arguments: dict[str, Any]) -> PermissionChoice:
        """Ask the user whether to run a tool. Base/non-interactive answer: deny."""
        return "deny"

    async def pick_model(
        self, current_model: str, current_effort: str | None
    ) -> tuple[str, str | None] | None:
        """Interactively choose a model + reasoning effort.

        Returns ``(model_id, effort | None)`` on confirm, or ``None`` to leave the
        current selection unchanged. The base/silent sink is non-interactive → ``None``,
        so ``/model`` falls back to its text listing outside a live terminal.
        """
        return None


class NullUI(AgentUI):
    """The default: a silent sink. Present so the loop can always emit events."""

    is_live = False


import asyncio

from agent_core.terminal.app import TerminalRenderer


class ConsoleUI(AgentUI):
    """Render a Claude-Code-style live trace and interactive permission prompts."""
    is_live = True

    def __init__(self, color: bool = True, preview_chars: int = 240, verbose: bool = False) -> None:
        self._renderer = TerminalRenderer(color=color, preview_chars=preview_chars, verbose=verbose)
        self._loop: Any = None

    def bind_event_loop(self, loop: Any) -> None:
        self._loop = loop

    def toggle_verbose(self) -> None:
        """Flip read/search folding off/on for subsequent turns (Ctrl+O)."""
        self._renderer.verbose = not self._renderer.verbose

    def on_turn_start(self) -> None:
        self._renderer.reset_stream_state()

    def on_thinking_delta(self, text: str) -> None:
        self._renderer.write_thinking_delta(text)

    def on_text_delta(self, text: str) -> None:
        self._renderer.write_text_delta(text)

    def on_tool_args_delta(self, tool_name: str, partial_json: str) -> None:
        self._renderer.write_tool_args_delta(partial_json)

    def on_thinking(self, text: str) -> None:
        self._renderer.print_thinking(text)

    def on_reasoning(self, text: str) -> None:
        self._renderer.print_reasoning(text)

    def on_tool_call(
        self, tool_name: str, risk: str, arguments: dict[str, Any], label: str | None = None
    ) -> None:
        self._renderer.print_tool_call(tool_name, risk, arguments, label=label)

    def on_tool_result(self, result: ToolResult, diff: str | None = None) -> None:
        self._renderer.print_tool_result(result.ok, result.content, diff=diff)

    def on_final(self, answer: str) -> None:
        self._renderer.print_final(answer)

    def on_todos(self, todos: list[Any]) -> None:
        self._renderer.print_todos(todos)

    def on_tool_use_summary(self, label: str, tool_names: list[str]) -> None:
        self._renderer.print_tool_use_summary(label)

    def on_stopped(self, reason: str, human: str) -> None:
        self._renderer.print_stopped(reason, human)

    def on_run_completed(self, stats: dict[str, Any]) -> None:
        self._renderer.print_run_completed(stats)

    def on_compaction_start(self, reactive: bool) -> None:
        self._renderer.start_compaction(reactive)

    def on_compaction_progress(self, fraction: float, stage: str) -> None:
        self._renderer.update_compaction(fraction, stage)

    def on_compaction_end(self, before_chars: int, after_chars: int, detail: str, reactive: bool) -> None:
        self._renderer.end_compaction(before_chars, after_chars, detail, reactive)

    async def pick_model(
        self, current_model: str, current_effort: str | None
    ) -> tuple[str, str | None] | None:
        # Runs on the main event loop (called directly from the chat dispatch), so no
        # thread bridging is needed. Non-TTY is handled inside run_model_picker → None.
        from agent_core.terminal.model_picker import run_model_picker

        return await run_model_picker(current_model, current_effort)

    def confirm_tool(self, tool_name: str, risk: str, arguments: dict[str, Any]) -> PermissionChoice:
        # confirm_tool is invoked on the executor's worker thread (the permission
        # step runs under asyncio.to_thread). prompt_toolkit needs the main thread,
        # so we bridge the async prompt back onto the bound event loop and block
        # this worker on the result. No loop bound (degenerate) → fail closed.
        loop = self._loop
        if loop is None:
            return "deny"
        future = asyncio.run_coroutine_threadsafe(
            self._renderer.ask_permission_async(tool_name, risk, arguments), loop
        )
        try:
            return future.result()
        except Exception:
            return "deny"
