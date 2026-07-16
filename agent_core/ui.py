from __future__ import annotations

import asyncio
from typing import Any, Literal

from agent_core.models import ToolResult
from agent_core.permission_types import (
    PermissionBehavior,
    PermissionDestination,
    PermissionRequest,
    PermissionResponse,
    PermissionUpdate,
)

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

    def on_token_usage(self, usage: dict[str, Any]) -> None:
        """Running token usage after a model turn (UI-only; emitted once per response).

        ``usage`` carries ``context_tokens`` (running prompt size), ``window`` (the
        model's context window), cumulative ``input_tokens``/``output_tokens`` for the
        run, and ``conversation_tokens`` (the estimated conversation-only slice of
        ``context_tokens``; the remainder is the fixed per-run baseline). The base sink
        ignores it — only a live UI renders the figure.
        """

    def on_run_completed(self, stats: dict[str, Any]) -> None:
        """The run has finished. ``stats`` carries duration, steps, tool_counts, reason,
        and the run's input_tokens/output_tokens/context_tokens."""

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

    def request_permission(self, request: PermissionRequest) -> PermissionResponse:
        """Structured permission prompt; legacy UIs are adapted through confirm_tool."""
        choice = self.confirm_tool(request.tool_name, request.risk, dict(request.arguments))
        if choice == "once":
            return PermissionResponse(True, reason="user confirmed once")
        if choice == "always" and request.suggestions:
            updates = tuple(
                PermissionUpdate(PermissionBehavior.ALLOW, item.rule, PermissionDestination.SESSION)
                for item in request.suggestions
            )
            return PermissionResponse(True, updates, "user allowed suggested scope for session")
        return PermissionResponse(False, reason="user rejected")

    def confirm_action(self, message: str) -> bool:
        """A one-off yes/no safety confirmation (e.g. "continue without a sandbox?").

        Called at agent construction time, before any event loop exists, so
        implementations must not rely on ``bind_event_loop``. Base/non-interactive
        answer: **no** — every caller treats False as the fail-closed path.
        """
        return False

    async def pick_model(
        self, current_model: str, current_effort: str | None, spec: Any
    ) -> tuple[str, str | None] | None:
        """Interactively choose a model + reasoning effort.

        Returns ``(model_id, effort | None)`` on confirm, or ``None`` to leave the
        current selection unchanged. The base/silent sink is non-interactive → ``None``,
        so ``/model`` falls back to its text listing outside a live terminal.
        """
        return None

    async def pick_permission_mode(
        self, current_mode: str, forbidden_modes: tuple[str, ...] = ()
    ) -> str | None:
        """Choose one of the six permission modes; silent UIs cannot pick."""
        return None


class NullUI(AgentUI):
    """The default: a silent sink. Present so the loop can always emit events."""

    is_live = False


class ConsoleUI(AgentUI):
    """Render a Claude-Code-style live trace and interactive permission prompts."""
    is_live = True

    def __init__(self, color: bool = True, preview_chars: int = 240, verbose: bool = False) -> None:
        # Lazy import: the terminal stack (rich + prompt_toolkit) is the optional
        # [terminal] extra. NullUI/library embedding must import without it; only
        # actually constructing the live console requires it.
        try:
            from agent_core.terminal.app import TerminalRenderer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "the interactive console needs the [terminal] extra — "
                "pip install 'agent-with-llm[terminal]' (or [all])"
            ) from exc
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

    def on_token_usage(self, usage: dict[str, Any]) -> None:
        self._renderer.print_token_usage(usage)

    def on_run_completed(self, stats: dict[str, Any]) -> None:
        self._renderer.print_run_completed(stats)

    def on_compaction_start(self, reactive: bool) -> None:
        self._renderer.start_compaction(reactive)

    def on_compaction_progress(self, fraction: float, stage: str) -> None:
        self._renderer.update_compaction(fraction, stage)

    def on_compaction_end(self, before_chars: int, after_chars: int, detail: str, reactive: bool) -> None:
        self._renderer.end_compaction(before_chars, after_chars, detail, reactive)

    async def pick_model(
        self, current_model: str, current_effort: str | None, spec: Any
    ) -> tuple[str, str | None] | None:
        # Runs on the main event loop (called directly from the chat dispatch), so no
        # thread bridging is needed. Non-TTY is handled inside run_model_picker → None.
        from agent_core.terminal.model_picker import run_model_picker

        return await run_model_picker(
            current_model,
            current_effort,
            models=spec.models,
            efforts_fn=spec.efforts_fn,
            title=spec.title,
            help_text=spec.help_text,
        )

    async def pick_permission_mode(
        self, current_mode: str, forbidden_modes: tuple[str, ...] = ()
    ) -> str | None:
        from agent_core.terminal.permission_picker import run_permission_picker

        selected = await run_permission_picker(current_mode, forbidden_modes=forbidden_modes)
        return selected.value if selected is not None else None

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

    def request_permission(self, request: PermissionRequest) -> PermissionResponse:
        loop = self._loop
        if loop is None:
            return PermissionResponse(False, reason="interactive event loop unavailable")
        future = asyncio.run_coroutine_threadsafe(
            self._renderer.ask_permission_request_async(request), loop
        )
        try:
            return future.result()
        except Exception:
            return PermissionResponse(False, reason="permission prompt failed")

    def confirm_action(self, message: str) -> bool:
        # Construction-time confirmation: no event loop is bound yet, so this is a
        # plain blocking prompt on the CLI's own terminal. Anything but an explicit
        # yes stays the fail-closed answer.
        try:
            self._renderer.emit(message)
            answer = input("Continue? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt, OSError):
            return False
        return answer in {"y", "yes"}
