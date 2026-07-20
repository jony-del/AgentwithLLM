from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from agent_core.config import (
    resolve_compression_config,
    resolve_concurrency_config,
    resolve_config,
    resolve_context_config,
    resolve_hooks_config,
    resolve_limits_config,
    resolve_mcp_config,
    resolve_memory_config,
    resolve_output_config,
    resolve_permission_rules,
    resolve_persist_compaction_boundary,
    resolve_sandbox_config,
    resolve_session_dir,
    resolve_skills_config,
    resolve_tool_use_summary_config,
    resolve_tool_suite_config,
    resolve_web_config,
)
from agent_core.permission_rules import RuleSet
from agent_core.permissions import (
    PermissionMode,
    next_shift_tab_permission_mode,
    permission_mode_label,
)
from agent_core.interrupt import KeyInterrupt
from agent_core.memory import Dreamer, MemoryConfig, MemoryStore
from agent_core.model_validation import PROVIDERS
from agent_core.models import LLMTransientError, Message
from agent_core.providers import (
    ClaudeProvider,
    FakeProvider,
    OpenAICompatProvider,
    OpenAIResponsesProvider,
    ProviderConfig,
)
from agent_core.chat_commands import (
    dispatch as dispatch_chat_command,
    is_immediate_command,
)
from agent_core.react import ReActAgent, ReActConfig
from agent_core.sandbox import SandboxRequiredError
from agent_core.tools.registry import ToolRegistry
from agent_core.transcript import (
    build_chain,
    find_session,
    fork_chain,
    latest_session,
    list_sessions,
    load_transcript,
    new_session_id,
    project_dir,
    session_label,
)
from agent_core.ui import AgentUI, ConsoleUI, NullUI

if TYPE_CHECKING:
    from agent_core.mcp import MCPClientManager
    from prompt_toolkit.completion import Completer


def _config_file(args: argparse.Namespace) -> str:
    """The toml file every resolver reads: ``--config PATH`` or the in-repo default.

    Only the default relative ``agent.toml`` is treated as repo-controlled input and
    trust-filtered (D2); an explicit ``--config`` path is user-chosen and honored as
    user-level config (see ``config.load_agent_toml``).
    """
    return getattr(args, "config", None) or "agent.toml"


def _resolve(args: argparse.Namespace) -> dict:
    return resolve_config(
        {
            "model": args.model,
            "permission": args.permission,
            "provider": args.provider,
            "effort": getattr(args, "effort", None),
        },
        config_file=_config_file(args),
    )


def _memory_config(args: argparse.Namespace) -> MemoryConfig:
    # Numeric tunables come from the [memory] toml table; enabled is overridable
    # by AGENT_MEMORY / --memory. (resolve_config above already loaded the .env.)
    return resolve_memory_config(getattr(args, "memory", None), _config_file(args))


def _permission_rules(args: argparse.Namespace) -> RuleSet:
    """Fine-grained rules from ``[permissions]`` toml, with CLI ``--allow/--deny/--ask``
    session rules layered on top (they append, deny still wins in the decision pipeline)."""
    rules = resolve_permission_rules(_config_file(args))
    from agent_core.permission_types import PermissionRuleSource

    cli_values = {
        "allow": getattr(args, "allow", None) or [],
        "deny": getattr(args, "deny", None) or [],
        "ask": getattr(args, "ask", None) or [],
    }
    legacy = [value for values in cli_values.values() for value in values if "run_command" in value]
    if legacy:
        raise ValueError(
            f"legacy run_command CLI rules are unsupported: {legacy!r}. "
            "Split each rule into bash(...) and/or powershell(...)."
        )
    cli = RuleSet.from_lists(
        allow=cli_values["allow"],
        deny=cli_values["deny"],
        ask=cli_values["ask"],
        source=PermissionRuleSource.CLI,
    )
    return rules.merge(cli)


def _sandbox_config(args: argparse.Namespace):
    """Sandbox config from ``[sandbox]`` toml/env, with the ``--sandbox/--no-sandbox``
    CLI flag layered on ``enabled`` (None = leave the resolved value untouched)."""
    config = resolve_sandbox_config(_config_file(args))
    cli_sandbox = getattr(args, "sandbox", None)
    if cli_sandbox is not None:
        config.enabled = bool(cli_sandbox)
    cli_backend = getattr(args, "sandbox_backend", None)
    if cli_backend is not None:
        config.backend = cli_backend
    return config


def _make_provider(values: dict):
    provider = values["provider"]
    if provider == "claude":
        return ClaudeProvider()
    if provider == "openai":
        return OpenAIResponsesProvider()
    if provider == "openai-compat":
        return OpenAICompatProvider()
    if provider == "fake":
        return FakeProvider()
    raise RuntimeError(
        f"unknown provider {provider!r}; choose one of: {', '.join(PROVIDERS)}"
    )


def _make_ui(args: argparse.Namespace) -> AgentUI:
    """A live console trace when attached to a real terminal; silent otherwise.

    Gated on both stdin and stdout being TTYs (so the permission prompt can read a
    reply and the trace isn't dumped into a pipe) and on the user not opting out
    with --quiet. Mirrors the TTY-gating that KeyInterrupt uses for Esc handling.
    """
    if getattr(args, "quiet", False):
        return NullUI()
    try:
        interactive = bool(sys.stdin) and sys.stdin.isatty() and bool(sys.stdout) and sys.stdout.isatty()
    except (ValueError, OSError):
        interactive = False
    return ConsoleUI(verbose=getattr(args, "verbose", False)) if interactive else NullUI()


def _describe_mcp_error(exc: BaseException) -> str:
    """Flatten an exception into a readable one-liner.

    anyio wraps a server's transport/handshake failure in an ``ExceptionGroup`` (often
    nested), so ``str(exc)`` is just "unhandled errors in a TaskGroup". Walk down to the
    leaf causes — e.g. ``McpError: Connection closed`` when a stdio server process exits
    immediately (a bad command/args, or the server program isn't installed).
    """
    leaves: list[str] = []

    def walk(error: BaseException) -> None:
        nested = getattr(error, "exceptions", None)
        if nested:
            for sub in nested:
                walk(sub)
        else:
            leaves.append(f"{type(error).__name__}: {error}")

    walk(exc)
    # dict.fromkeys de-dups while preserving order.
    return "; ".join(dict.fromkeys(leaves)) or f"{type(exc).__name__}: {exc}"


def _connect_mcp(mcp_config):
    """Start a manager for the configured servers, raising a clean RuntimeError on failure.

    A connect/handshake failure (an anyio ``ExceptionGroup``) becomes a readable
    ``RuntimeError`` so callers can report it without leaking a raw traceback.
    """
    from agent_core.mcp import MCPClientManager

    manager = MCPClientManager(mcp_config)
    try:
        manager.start()
    except Exception as exc:  # noqa: BLE001 - anyio ExceptionGroup et al. → one clean message
        raise RuntimeError(
            f"could not connect MCP servers: {_describe_mcp_error(exc)} "
            "(check each server's command/args and that the server program is installed)"
        ) from exc
    return manager


def _start_mcp(
    registry: ToolRegistry, config_file: str = "agent.toml"
) -> "MCPClientManager | None":
    """Connect any configured MCP servers and register their tools, or return ``None``.

    Only connects when ``[mcp.servers.*]`` is non-empty. The caller owns the returned
    manager and must ``close()`` it.
    """
    mcp_config = resolve_mcp_config(config_file)
    if not any(server.enabled for server in mcp_config.servers):
        return None
    from agent_core.mcp import MCPAdapter

    manager = _connect_mcp(mcp_config)
    registry.register_adapter(MCPAdapter(manager))
    return manager


def build_agent(args: argparse.Namespace) -> "BuiltAgent":
    values = _resolve(args)
    config_file = _config_file(args)
    tool_suite = resolve_tool_suite_config(config_file)
    if tool_suite.shell.enabled and tool_suite.shell.bash.enabled:
        from agent_core.process_supervisor import resolve_bash_executable

        resolve_bash_executable(
            tool_suite.shell.bash.executable or os.getenv("POLARIS_BASH_PATH")
        )
    provider = _make_provider(values)
    ui = _make_ui(args)
    concurrency = resolve_concurrency_config(config_file)
    cli_api_concurrency = getattr(args, "max_api_concurrency", None)
    max_api_concurrency = (
        max(1, int(cli_api_concurrency)) if cli_api_concurrency is not None
        else int(concurrency["max_api_concurrency"])
    )
    # Run-level safety limits: [limits]/env resolved here, CLI flags layered on top.
    # A CLI value of 0 disables the cap (None), mirroring the toml/env convention.
    limits = resolve_limits_config(config_file)
    cli_wall = getattr(args, "max_wall_seconds", None)
    max_wall_seconds = (
        (None if cli_wall <= 0 else float(cli_wall)) if cli_wall is not None
        else limits["max_wall_seconds"]
    )
    cli_steps = getattr(args, "max_steps", None)
    max_steps = (
        (None if cli_steps <= 0 else int(cli_steps)) if cli_steps is not None
        else limits["max_steps"]
    )
    context = resolve_context_config(config_file)
    config = ReActConfig(
        provider=values["provider"],
        model=values["model"],
        permission=values["permission"],
        memory=_memory_config(args),
        output=resolve_output_config(config_file),
        compression=resolve_compression_config(config_file),
        tool_use_summary=resolve_tool_use_summary_config(config_file),
        project_instructions=bool(context["project_instructions"]),
        git_context=bool(context["git_context"]),
        claudemd_max_chars=int(context["claudemd_max_chars"]),
        thinking_budget=getattr(args, "thinking_budget", None),
        effort=values["effort"],
        stream=not getattr(args, "no_stream", False),
        parallel_tools=bool(concurrency["parallel_tools"]),
        max_tool_workers=int(concurrency["max_tool_workers"]),
        max_api_concurrency=max_api_concurrency,
        api_rate_limit_per_min=int(concurrency["api_rate_limit_per_min"]),
        max_wall_seconds=max_wall_seconds,
        max_steps=max_steps,
        soft_deadline_fraction=float(limits["soft_deadline_fraction"]),
        session_dir=_session_dir(args),
        persist_compaction_boundary=resolve_persist_compaction_boundary(config_file),
        skills=resolve_skills_config(config_file),
        hooks=resolve_hooks_config(config_file),
        sandbox=_sandbox_config(args),
        permission_rules=_permission_rules(args),
        web=resolve_web_config(config_file),
        tools=tool_suite,
    )
    registry = ReActAgent.default_registry()
    manager = _start_mcp(registry, config_file)
    # Resolve which session this run writes to (new / resumed / continued / forked) and
    # load any prior conversation to seed it. ``seed`` is the fork's cloned chain that
    # must be written into the fresh transcript before the run; for plain resume it is
    # empty because the history already lives on disk.
    session_id, history, seed = _resolve_session(args, config.session_dir)
    agent = ReActAgent(
        provider=provider, config=config, tools=registry, ui=ui, session_id=session_id,
        mcp_manager=manager,
    )
    agent._sandbox_cli_locked = (
        getattr(args, "sandbox", None) is not None
        or getattr(args, "sandbox_backend", None) is not None
    )
    return BuiltAgent(agent, ui, manager, history, seed)


@dataclass(slots=True)
class BuiltAgent:
    agent: ReActAgent
    ui: AgentUI
    mcp: "MCPClientManager | None"
    history: list[Message]
    seed: list[Message]


def _session_dir(args: argparse.Namespace) -> str:
    """Transcript root: config/env resolution, then ``--session-dir`` /
    ``--no-session-persistence`` CLI overrides."""
    if getattr(args, "no_session_persistence", False):
        return ""
    cli = getattr(args, "session_dir", None)
    return cli if cli else resolve_session_dir(_config_file(args))


def _resolve_session(
    args: argparse.Namespace, session_dir: str
) -> tuple[str, list[Message], list[Message]]:
    """Pick the session id and seed history from ``--resume``/``--continue``/``--fork-session``.

    Returns ``(session_id, history, seed)``: ``history`` is fed to ``run(history=...)``;
    ``seed`` is the (cloned) chain that still needs writing to a fresh transcript (fork),
    empty when the history already exists on disk.
    """
    fork = getattr(args, "fork_session", False)
    explicit = getattr(args, "session_id", None)
    resume_id = getattr(args, "resume", None)
    cont = getattr(args, "continue_", False)
    cwd = Path.cwd().resolve()

    path = None
    if resume_id:
        if not session_dir:
            raise RuntimeError("--resume needs session persistence (it is disabled)")
        path = find_session(session_dir, cwd, resume_id)
        if path is None:
            raise RuntimeError(f"no session found with id {resume_id!r}")
    elif cont:
        if not session_dir:
            raise RuntimeError("--continue needs session persistence (it is disabled)")
        info = latest_session(project_dir(session_dir, cwd))
        if info is None:
            raise RuntimeError("no prior session to continue in this project")
        path = info.path

    if path is None:
        return explicit or new_session_id(), [], []

    loaded = load_transcript(path)
    if fork:
        new_id, cloned = fork_chain(loaded)
        return explicit or new_id, cloned, list(cloned)
    return loaded.session_id, build_chain(loaded), []


async def _async_input(
    prompt: str,
    ui: "AgentUI | None" = None,
    completer: "Completer | None" = None,
    bottom_toolbar: "Callable[[], Any] | None" = None,
    on_cycle_permission: "Callable[[], None] | None" = None,
    *,
    is_running: "Callable[[], bool] | None" = None,
    on_interrupt: "Callable[[], None] | None" = None,
    on_background: "Callable[[], None] | None" = None,
    on_transcript: "Callable[[], None] | None" = None,
    on_tasks: "Callable[[], None] | None" = None,
    on_history_search: "Callable[[], None] | None" = None,
    on_redraw: "Callable[[], None] | None" = None,
    on_recall_queue: "Callable[[], str] | None" = None,
) -> str | None:
    """Read one chat message without blocking the loop; ``None`` on EOF (exit).

    On a real terminal this is a multi-line ``prompt_toolkit`` session (Enter
    sends, Shift+Enter/Alt+Enter/Ctrl+J inserts a newline, Ctrl+O toggles
    verbose, Ctrl-C clears the current input in place via our keybinding). When a
    ``completer`` is supplied, typing ``/`` pops a styled dropdown of slash-commands
    / skills (and session candidates for ``/resume``). ``bottom_toolbar`` (when
    given) renders a persistent status line under the prompt. The
    ``KeyboardInterrupt`` branch below is a fallback for the rare terminal/race
    where the default abort still fires. When stdin is not a TTY (piped/CI)
    ``prompt_toolkit`` can't drive the terminal, so we fall back to a
    daemon-thread ``input()`` whose Ctrl-C stays an immediate exit.
    """
    if not (sys.stdin and sys.stdin.isatty()):
        return await _threaded_input(prompt)

    from prompt_toolkit import PromptSession
    from prompt_toolkit.shortcuts import CompleteStyle
    from prompt_toolkit.formatted_text import HTML
    from agent_core.terminal.keybindings import create_keybindings
    from agent_core.terminal.theme import completion_menu_style

    # PromptSession is cached for the life of the process. Keep the callback in a
    # mutable function attribute so a later chat session cannot retain an old agent.
    input_state = cast(Any, _async_input)
    input_state._on_cycle_permission = on_cycle_permission
    input_state._is_running = is_running
    input_state._on_interrupt = on_interrupt
    input_state._on_background = on_background
    input_state._on_transcript = on_transcript
    input_state._on_tasks = on_tasks
    input_state._on_history_search = on_history_search
    input_state._on_redraw = on_redraw
    input_state._on_recall_queue = on_recall_queue

    def cycle_permission() -> None:
        callback = getattr(_async_input, "_on_cycle_permission", None)
        if callback is not None:
            callback()

    def state_call(name: str, default: Any = None) -> Any:
        callback = getattr(_async_input, name, None)
        return callback() if callback is not None else default

    session = getattr(input_state, "_session", None)
    if session is None:
        toggle = getattr(ui, "toggle_verbose", None)
        session = PromptSession(
            key_bindings=create_keybindings(
                toggle,
                cycle_permission,
                is_running=lambda: bool(state_call("_is_running", False)),
                on_interrupt=lambda: state_call("_on_interrupt"),
                on_background=lambda: state_call("_on_background"),
                on_transcript=(
                    (lambda: state_call("_on_transcript"))
                    if on_transcript is not None
                    else None
                ),
                on_tasks=(
                    (lambda: state_call("_on_tasks"))
                    if on_tasks is not None
                    else None
                ),
                on_history_search=(
                    (lambda: state_call("_on_history_search"))
                    if on_history_search is not None
                    else None
                ),
                on_redraw=lambda: state_call("_on_redraw"),
                on_recall_queue=lambda: str(state_call("_on_recall_queue", "") or ""),
            ),
            multiline=True,
            completer=completer,
            complete_while_typing=True,  # menu pops the moment '/' is typed
            complete_style=CompleteStyle.COLUMN,  # single column shows the description meta
            style=completion_menu_style(),
            bottom_toolbar=bottom_toolbar,
        )
        input_state._session = session

    try:
        message = HTML(f"<ansicyan>{prompt}</ansicyan> ")
        line = await session.prompt_async(message)
        return line
    except EOFError:
        return None  # Ctrl-D / closed stdin → leave the chat loop
    except KeyboardInterrupt:
        return ""  # Ctrl-C clears the current line and re-prompts


def _clean_surrogates(text: str) -> str:
    """Collapse lone surrogateescape code points (U+DC80..U+DCFF) to valid text.

    Non-TTY stdin (a Windows pipe) decodes undecodable bytes into lone
    surrogates; those cannot be re-encoded to UTF-8 downstream (JSONL log,
    transcript, API request). Map them back to bytes and re-decode UTF-8 with
    replacement so only clean text ever enters the conversation.
    """
    return text.encode("utf-8", "surrogateescape").decode("utf-8", "replace")


async def _threaded_input(prompt: str) -> str | None:
    """Non-TTY fallback: read one stdin line on a daemon thread; ``None`` on EOF.

    A daemon thread resolving the future via ``call_soon_threadsafe`` keeps Ctrl-C
    an immediate exit instead of leaving a worker stuck in ``input()``.
    """
    import threading

    loop = asyncio.get_running_loop()
    future: asyncio.Future[str | None] = loop.create_future()

    def read() -> None:
        try:
            line: str | None = _clean_surrogates(input(prompt))
        except EOFError:
            line = None

        def resolve_future() -> None:
            if not future.done():
                future.set_result(line)

        try:
            loop.call_soon_threadsafe(resolve_future)
        except RuntimeError:
            pass  # loop already closed (e.g. Ctrl-C tore the session down)

    threading.Thread(target=read, daemon=True, name="chat-input").start()
    return await future


async def _seed_transcript(built: "BuiltAgent") -> None:
    """Write a fork's cloned chain into its fresh transcript before the first turn."""
    if built.seed and built.agent.transcript is not None:
        for message in built.seed:
            await built.agent.transcript.append_message(message)


def run_task(args: argparse.Namespace) -> int:
    try:
        built = build_agent(args)
    except (RuntimeError, ValueError) as exc:
        # E.g. an MCP server failed to connect, or a bad --resume id.
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    agent, ui, mcp = built.agent, built.ui, built.mcp

    async def run_once():
        await _seed_transcript(built)
        try:
            await agent.scheduler_heartbeat()
            with KeyInterrupt(confirm=True) as interrupt:
                agent.session.should_background = interrupt.consume_background
                try:
                    result = await agent.run(
                        args.task, should_cancel=interrupt.is_set, history=built.history or None
                    )
                finally:
                    agent.session.should_background = None
            _history, scheduled = await agent.drain_scheduler_deliveries(result.messages)
            return result, scheduled
        finally:
            # SessionEnd is host-driven: a one-shot run IS the whole session.
            await agent.fire_session_end("run_exit")

    try:
        result, scheduled = asyncio.run(run_once())
    except RuntimeError as exc:
        # Covers LLMTransientError (network exhausted retries) and API errors.
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    finally:
        agent.sandbox.teardown()
        for retired in getattr(agent, "_retired_sandboxes", []):
            retired.teardown()
        agent.logger.close()
        plugin_mcp = getattr(agent, "_plugin_mcp_manager", None)
        if plugin_mcp is not None:
            plugin_mcp.close()
        if mcp is not None:
            mcp.close()
    # A live UI already streamed the answer via on_final; only print it ourselves
    # when the run was silent (piped/--quiet) so we don't echo it twice.
    if not ui.is_live:
        print(result.answer)
        for scheduled_result in scheduled:
            print(scheduled_result.answer)
    print(f"\nRun log: runs/{result.run_id}.jsonl")
    if agent.transcript is not None:
        print(f"Session: {agent.session_id}  (resume with --resume {agent.session_id})")
    return 0


def chat_command(args: argparse.Namespace) -> int:
    try:
        built = build_agent(args)
    except (RuntimeError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    agent, ui, mcp = built.agent, built.ui, built.mcp
    if agent.transcript is not None:
        print(f"Session {agent.session_id} (resume later with --resume {agent.session_id})")

    async def session() -> None:
        # One event loop for the whole chat: every turn shares the same provider
        # gate, httpx pool, and asyncio primitives instead of rebinding per turn.
        # ``history`` carries the conversation across turns (and seeds from --resume),
        # so the agent finally has cross-turn memory within a session.
        await _seed_transcript(built)
        history: list[Message] = list(built.history)
        from agent_core.terminal.completion import SlashCompleter

        # A chat owns one persistent PromptSession. Do not retain a toolbar/completer
        # closure from an earlier embedded chat invocation in the same process.
        if hasattr(_async_input, "_session"):
            delattr(_async_input, "_session")
        completer = SlashCompleter(agent)

        async def scheduler_heartbeats() -> None:
            while True:
                try:
                    await agent.scheduler_heartbeat()
                except Exception as exc:  # noqa: BLE001 - scheduler health is observational
                    await agent.logger.write(
                        "scheduler_delivery", {"state": "heartbeat_error", "error": str(exc)}
                    )
                await asyncio.sleep(45)

        heartbeat_task = asyncio.create_task(scheduler_heartbeats())

        def _toolbar() -> str:
            # Persistent status line: the active model + effort, so the current config
            # (and the effect of the /model picker) is always visible. Read per render.
            effort = agent.config.effort or "—"
            mode = permission_mode_label(agent.config.permission)
            running = "running" if run_task is not None else "ready"
            queued = f" · queued: {len(prompt_queue)}" if len(prompt_queue) else ""
            title = getattr(agent, "session_title", None)
            titled = f" · {title}" if title else ""
            fast = " · fast" if getattr(agent, "fast_mode", False) else ""
            return (
                f" {mode}  ·  {running}{queued}{titled}  ·  model: {agent.config.model}{fast}  "
                f"·  effort: {effort}  ·  /help for commands "
            )

        def _cycle_permission() -> None:
            current = PermissionMode(agent.config.permission)
            target = next_shift_tab_permission_mode(current)
            try:
                agent.set_permission_mode(target, source="shift_tab")
            except SandboxRequiredError as exc:
                # Most commonly the user declined the no-sandbox confirmation. The
                # mode remains unchanged; surface the actionable gate message once.
                print(f"[permission] {exc}")

        from agent_core.terminal.prompt_queue import PromptQueue

        prompt_queue = PromptQueue()
        run_task: asyncio.Task[Any] | None = None
        cancel_requested = threading.Event()
        background_requested = threading.Event()

        def _consume_background() -> bool:
            requested = background_requested.is_set()
            if requested:
                background_requested.clear()
            return requested

        def _show_transcript() -> None:
            visible = getattr(agent, "_active_messages", None) or history
            print("\nTranscript:")
            for message in visible[-20:]:
                label = message.name or message.role
                text = " ".join(message.content.strip().split())
                print(f"  {label:<9} {text[:160]}")
            if not visible:
                print("  (empty)")

        def _show_tasks() -> None:
            print("\nTasks:")
            print(agent.session.todos.render())
            queued = prompt_queue.snapshot()
            print(f"Queued input ({len(queued)}):")
            for item in queued:
                preview = " ".join(item.content.split())[:120]
                print(f"  [{item.priority.name.lower()}] {preview}")
            if not queued:
                print("  (none)")

        def _input_kwargs() -> dict[str, Any]:
            return {
                "is_running": lambda: run_task is not None,
                "on_interrupt": cancel_requested.set,
                "on_background": background_requested.set,
                "on_transcript": _show_transcript,
                "on_tasks": _show_tasks,
                "on_redraw": lambda: None,
                "on_recall_queue": prompt_queue.recall_editable,
            }

        async def _legacy_loop() -> None:
            nonlocal history
            while True:
                task = await _async_input("›", ui, completer, _toolbar, _cycle_permission)
                if task is None:
                    return
                task = task.strip()
                if not task:
                    continue
                turn = await dispatch_chat_command(task, agent, ui, history)
                if turn.quit:
                    return
                if turn.history is not None:
                    history = turn.history
                if turn.prompt is None:
                    continue
                try:
                    with KeyInterrupt(confirm=True) as interrupt:
                        agent.session.should_background = interrupt.consume_background
                        try:
                            result = await agent.run(
                                turn.prompt,
                                should_cancel=interrupt.is_set,
                                history=history or None,
                            )
                        finally:
                            agent.session.should_background = None
                    history = result.messages
                    history, scheduled = await agent.drain_scheduler_deliveries(history)
                except LLMTransientError as exc:
                    print(f"[network] {exc}", file=sys.stderr)
                    print(
                        "The session is still alive — please send your message again.",
                        file=sys.stderr,
                    )
                    continue
                except RuntimeError as exc:
                    print(f"[error] {exc}", file=sys.stderr)
                    continue
                if not ui.is_live:
                    print(result.answer)
                    for scheduled_result in scheduled:
                        print(scheduled_result.answer)

        async def _persistent_loop() -> None:
            nonlocal history, run_task
            exit_requested = False
            input_task: asyncio.Task[str | None] | None = None

            def start_input() -> asyncio.Task[str | None]:
                return asyncio.create_task(
                    _async_input(
                        "›",
                        ui,
                        completer,
                        _toolbar,
                        _cycle_permission,
                        **_input_kwargs(),
                    )
                )

            def start_prompt(prompt: str) -> None:
                nonlocal run_task
                cancel_requested.clear()
                background_requested.clear()
                agent.session.should_background = _consume_background
                run_task = asyncio.create_task(
                    agent.run(
                        prompt,
                        should_cancel=cancel_requested.is_set,
                        history=history or None,
                        midturn_drain=prompt_queue.drain_midturn,
                    )
                )

            def start_batch(messages: list[Message]) -> None:
                nonlocal run_task
                cancel_requested.clear()
                background_requested.clear()
                agent.session.should_background = _consume_background
                run_task = asyncio.create_task(
                    agent.run_messages(
                        messages,
                        should_cancel=cancel_requested.is_set,
                        history=history or None,
                        midturn_drain=prompt_queue.drain_midturn,
                    )
                )

            async def apply_turn(task: str) -> bool:
                """Dispatch one idle/immediate command; return True to exit."""

                nonlocal history, exit_requested
                command_history = (
                    getattr(agent, "_active_messages", history)
                    if run_task is not None
                    else history
                )
                turn = await dispatch_chat_command(task, agent, ui, command_history)
                if turn.quit:
                    if run_task is not None:
                        exit_requested = True
                        cancel_requested.set()
                    return True
                if turn.history is not None:
                    history = turn.history
                if turn.prompt is not None:
                    if run_task is None:
                        start_prompt(turn.prompt)
                    else:
                        prompt_queue.enqueue(turn.prompt)
                        print(f"[queued] {len(prompt_queue)} input(s) waiting")
                return False

            async def start_next_queued() -> bool:
                """Dispatch queue units until one starts a run or the queue is empty."""

                while run_task is None and len(prompt_queue):
                    batch = prompt_queue.pop_between_turn()
                    if not batch:
                        return False
                    first = batch[0]
                    if first.is_slash_command:
                        if await apply_turn(first.content):
                            return True
                        continue
                    start_batch([item.to_message(delivery="between_turn") for item in batch])
                return False

            input_task = start_input()
            try:
                while True:
                    assert input_task is not None
                    wait_for: set[asyncio.Task[Any]] = {input_task}
                    if run_task is not None:
                        wait_for.add(run_task)
                    done, _ = await asyncio.wait(wait_for, return_when=asyncio.FIRST_COMPLETED)

                    # Treat input completed in the same event-loop tick as the run as
                    # having been submitted while running, preserving queue ordering.
                    if input_task in done:
                        line = await input_task
                        input_task = None
                        if line is None:
                            if run_task is not None:
                                exit_requested = True
                                cancel_requested.set()
                            else:
                                break
                        else:
                            task = line.strip()
                            if task:
                                if run_task is not None and not is_immediate_command(task):
                                    prompt_queue.enqueue(task)
                                    print(f"[queued] {len(prompt_queue)} input(s) waiting")
                                else:
                                    wants_exit = await apply_turn(task)
                                    if wants_exit and run_task is None:
                                        break

                    if run_task is not None and run_task in done:
                        completed = run_task
                        run_task = None
                        agent.session.should_background = None
                        try:
                            result = await completed
                            history = result.messages
                            history, scheduled = await agent.drain_scheduler_deliveries(history)
                            if not ui.is_live:
                                print(result.answer)
                                for scheduled_result in scheduled:
                                    print(scheduled_result.answer)
                        except LLMTransientError as exc:
                            print(f"[network] {exc}", file=sys.stderr)
                            print(
                                "The session is still alive — queued input was kept.",
                                file=sys.stderr,
                            )
                        except RuntimeError as exc:
                            print(f"[error] {exc}", file=sys.stderr)
                        if exit_requested:
                            dropped = prompt_queue.clear()
                            if dropped:
                                print(f"[exit] dropped {len(dropped)} queued input(s)")
                            break
                        if await start_next_queued():
                            if run_task is None:
                                break

                    if input_task is None and not exit_requested:
                        input_task = start_input()
            finally:
                if input_task is not None and not input_task.done():
                    input_task.cancel()
                    await asyncio.gather(input_task, return_exceptions=True)
                if run_task is not None and not run_task.done():
                    cancel_requested.set()
                    await asyncio.gather(run_task, return_exceptions=True)
                agent.session.should_background = None

        try:
            if sys.stdin and sys.stdin.isatty():
                from prompt_toolkit.patch_stdout import patch_stdout

                with patch_stdout(raw=True):
                    await _persistent_loop()
            else:
                await _legacy_loop()
        finally:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
            # SessionEnd is host-driven: leaving the chat loop closes the session.
            await agent.fire_session_end("chat_exit")

    try:
        asyncio.run(session())
    finally:
        agent.sandbox.teardown()
        for retired in getattr(agent, "_retired_sandboxes", []):
            retired.teardown()
        agent.logger.close()
        plugin_mcp = getattr(agent, "_plugin_mcp_manager", None)
        if plugin_mcp is not None:
            plugin_mcp.close()
        if mcp is not None:
            mcp.close()
    return 0


def sessions_command(args: argparse.Namespace) -> int:
    """List resumable sessions saved for the current project, newest first."""
    root = getattr(args, "session_dir", None) or resolve_session_dir(_config_file(args))
    if not root:
        print("Session persistence is disabled (empty session dir).")
        return 0
    cwd = Path.cwd().resolve()
    infos = list_sessions(project_dir(root, cwd))
    if not infos:
        print(f"No saved sessions for {cwd}")
        return 0
    import datetime as _dt

    print(f"Sessions for {cwd}:\n")
    for info in infos:
        when = _dt.datetime.fromtimestamp(info.modified).strftime("%Y-%m-%d %H:%M")
        label = session_label(info)
        branch = f" [{info.git_branch}]" if info.git_branch else ""
        print(f"  {info.session_id}  {when}  ({info.message_count} msgs){branch}")
        print(f"      {label}")
    print("\nResume with: polaris run <task> --resume <id>   (or --continue for the newest)")
    return 0


def _open_store(config: MemoryConfig) -> MemoryStore:
    return MemoryStore(Path(config.dir) / "memory.jsonl")


def dream_command(args: argparse.Namespace) -> int:
    """Run an offline dreaming pass: decay/forget, merge, and synthesise insights."""
    values = _resolve(args)
    config = _memory_config(args)
    store = _open_store(config)
    dreamer = Dreamer(store, config, _make_provider(values), ProviderConfig(model=values["model"]))
    try:
        report = asyncio.run(dreamer.dream(commit=not args.dry_run))
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    prefix = "Dreaming (dry run, nothing written)" if args.dry_run else "Dreaming done"
    print(
        f"{prefix}: scanned={report.scanned} forgotten={report.forgotten} "
        f"merged={report.merged} insights+={report.insights_added}"
    )
    for detail in report.details:
        print(f"  - {detail}")
    return 0


def memory_command(args: argparse.Namespace) -> int:
    """Inspect or curate stored memories: list / add / forget."""
    store = _open_store(_memory_config(args))
    if args.action == "list":
        records = sorted(store.all(), key=lambda r: r.importance, reverse=True)
        if not records:
            print("(no memories)")
            return 0
        for record in records:
            print(f"{record.id}  [{record.kind}] imp={record.importance:.2f}  {record.content}")
        return 0
    if args.action == "add":
        if not args.value:
            print("[error] `memory add` needs text", file=sys.stderr)
            return 1
        record = asyncio.run(store.add(args.value, kind="fact", importance=0.6))
        print(f"Added {record.id}")
        return 0
    if args.action == "forget":
        if not args.value:
            print("[error] `memory forget` needs an id", file=sys.stderr)
            return 1
        if asyncio.run(store.delete(args.value)):
            print(f"Forgot {args.value}")
            return 0
        print(f"[error] no memory {args.value}", file=sys.stderr)
        return 1
    return 0


def _short(value: object, limit: int = 160) -> str:
    text = str(value).replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _render_replay_event(record: dict) -> str:
    """One human-readable timeline line per JSONL record (unknown events included)."""
    import datetime as _dt

    ts = record.get("ts")
    try:
        when = _dt.datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S") if ts else "--:--:--"
    except (ValueError, OSError, OverflowError):
        when = "--:--:--"
    event = str(record.get("event", "?"))

    if event == "user":
        detail = _short(record.get("content", ""), 200)
    elif event == "permission":
        decision = record.get("decision") or {}
        detail = (
            f"{record.get('tool')} -> "
            f"{'allowed' if decision.get('allowed') else 'denied'}"
            f" ({_short(decision.get('reason', ''), 80)})"
        )
    elif event == "tool_pre":
        call = record.get("tool_call") or {}
        detail = f"{call.get('name')} args={_short(call.get('arguments', {}), 120)}"
    elif event == "tool_result":
        call = record.get("tool_call") or {}
        result = record.get("result") or {}
        status = "ok" if result.get("ok", True) else "FAILED"
        detail = f"{call.get('name')} [{status}] {_short(result.get('content', ''), 140)}"
    elif event == "compression":
        detail = ", ".join(
            f"{key}={_short(value, 40)}" for key, value in record.items()
            if key not in {"ts", "v", "event"}
        )
    elif event == "final":
        stopped = record.get("stopped")
        suffix = f" [stopped: {stopped}]" if stopped else ""
        detail = _short(record.get("answer", ""), 300) + suffix
    elif event == "_unparseable":
        detail = f"line {record.get('line')}: {_short(record.get('raw', ''), 120)}"
    else:
        # Generic (and forward-compatible) rendering for every other/unknown event.
        detail = ", ".join(
            f"{key}={_short(value, 60)}" for key, value in record.items()
            if key not in {"ts", "v", "event"}
        )
    return f"{when}  {event:<16} {detail}"


def replay_command(args: argparse.Namespace) -> int:
    """Re-render a recorded run's JSONL event log as a readable timeline.

    Post-hoc debugging only: reads ``runs/<run_id>.jsonl`` (exact id or unique
    prefix), never constructs an agent and never issues an API call.
    """
    from agent_core.storage import read_events

    run_dir = Path(getattr(args, "run_dir", None) or "runs")
    if not run_dir.is_dir():
        print(f"[error] no run directory at {run_dir}", file=sys.stderr)
        return 1
    path = run_dir / f"{args.run_id}.jsonl"
    if not path.exists():
        matches = [p for p in sorted(run_dir.glob("*.jsonl")) if p.stem.startswith(args.run_id)]
        if not matches:
            recent = [p.stem for p in sorted(run_dir.glob("*.jsonl"))[-5:]]
            print(
                f"[error] no run matching {args.run_id!r} in {run_dir}"
                + (f"; most recent: {', '.join(recent)}" if recent else ""),
                file=sys.stderr,
            )
            return 1
        if len(matches) > 1:
            print(
                f"[error] {args.run_id!r} is ambiguous: {', '.join(p.stem for p in matches)}",
                file=sys.stderr,
            )
            return 1
        path = matches[0]

    print(f"Replay of {path.stem}  ({path})\n")
    count = 0
    for record in read_events(path):
        print(_render_replay_event(record))
        count += 1
    print(f"\n{count} event(s).")
    return 0


def health_command(args: argparse.Namespace) -> int:
    """Aggregate application and installation checks without failing early."""
    from agent_core.health import HealthCheck, HealthReport, collect_dependency_checks, render_human

    checks: list[HealthCheck] = []
    tool_suite = None
    try:
        resolve_config({}, config_file=_config_file(args))
        tool_suite = resolve_tool_suite_config(_config_file(args))
        checks.append(HealthCheck("configuration", True, "ok", detail="loaded successfully"))
    except Exception as e:
        checks.append(HealthCheck("configuration", True, "error", detail=str(e)))

    try:
        provider = _make_provider(_resolve(args))
        checks.append(
            HealthCheck("provider", True, "ok", version=type(provider).__name__)
        )
    except Exception as e:
        checks.append(HealthCheck("provider", True, "error", detail=str(e)))

    try:
        from agent_core.tools import default_tools

        tool_count = len(default_tools(Path.cwd()))
        checks.append(
            HealthCheck("tool-registry", True, "ok", detail=f"{tool_count} tools available")
        )
    except Exception as e:
        checks.append(HealthCheck("tool-registry", True, "error", detail=str(e)))

    try:
        memory_config = _memory_config(args)
        if memory_config.enabled:
            store = _open_store(memory_config)
            memory_count = len(store.all())
            checks.append(
                HealthCheck("memory", True, "ok", detail=f"{memory_count} memories stored")
            )
        else:
            checks.append(HealthCheck("memory", False, "ok", detail="disabled"))
    except Exception as e:
        checks.append(HealthCheck("memory", True, "error", detail=str(e)))

    bash_executable = os.getenv("POLARIS_BASH_PATH")
    powershell_executable = None
    if tool_suite is not None:
        bash_executable = tool_suite.shell.bash.executable or bash_executable
        powershell_executable = tool_suite.shell.powershell.executable
    checks.extend(
        collect_dependency_checks(
            args.profile,
            bash_executable=bash_executable,
            powershell_executable=powershell_executable,
        )
    )
    report = HealthReport(args.profile, tuple(checks))
    print(report.to_json() if args.json else render_human(report))
    return 0 if report.status != "error" else 1


def uninstall_command(args: argparse.Namespace) -> int:
    """Hand self-removal to a stdlib-only worker outside the active environment."""

    from agent_core.uninstall import uninstall_from_cli

    return uninstall_from_cli(args)


def scheduler_service_command(args: argparse.Namespace) -> int:
    """Install, inspect, or remove the least-privilege scheduler user service."""
    from agent_core.scheduler_service import (
        default_receipt_path,
        install_user_service,
        uninstall_user_service,
    )

    receipt_path = default_receipt_path()
    try:
        if args.action == "install":
            config = resolve_tool_suite_config(_config_file(args)).scheduler
            receipt = install_user_service(
                executable=sys.executable, database=config.database_path(),
                receipt_path=receipt_path,
            )
            print(json.dumps(receipt, ensure_ascii=False, indent=2))
            return 0
        if args.action == "uninstall":
            uninstall_user_service(
                expected_executable=sys.executable, receipt_path=receipt_path,
                purge_data=bool(args.purge_data),
            )
            print("Scheduler user service removed.")
            return 0
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


def mcp_command(args: argparse.Namespace) -> int:
    """List the tools exposed by the configured MCP servers (a verification aid)."""
    mcp_config = resolve_mcp_config(_config_file(args))
    if not any(server.enabled for server in mcp_config.servers):
        print("(no MCP servers configured in agent.toml — see [mcp.servers.*])")
        return 0
    from agent_core.mcp import MCPAdapter

    try:
        manager = _connect_mcp(mcp_config)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    try:
        tools = MCPAdapter(manager).list_tools()
        if not tools:
            print("(servers connected but exposed no tools)")
            return 0
        for tool in sorted(tools, key=lambda t: t.name):
            summary = tool.description.splitlines()[0] if tool.description else ""
            print(f"{tool.name}  [{tool.risk.value}]  {summary}")
        return 0
    finally:
        manager.close()


def _force_utf8_output() -> None:
    """Ensure stdout/stderr use UTF-8 so model output (emoji, CJK) prints on
    consoles whose default codec is narrow (e.g. GBK on zh-CN Windows)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()
    parser = argparse.ArgumentParser(prog="polaris")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config_flag(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--config",
            metavar="PATH",
            default=None,
            help="Read settings from this toml file instead of ./agent.toml. An explicit "
            "path is user-chosen config: the repo-config trust filter (TOFU) does not apply.",
        )

    def add_common(subparser: argparse.ArgumentParser) -> None:
        add_config_flag(subparser)
        subparser.add_argument("--model", default=None)
        subparser.add_argument(
            "--permission",
            type=lambda value: PermissionMode(value).value,
            metavar="MODE",
            default=None,
        )
        subparser.add_argument(
            "--sandbox",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Run dangerous commands under the OS sandbox. Overrides [sandbox].enabled.",
        )
        subparser.add_argument(
            "--sandbox-backend",
            choices=["auto", "native", "container", "vm"],
            default=None,
            help="Isolation tier: native (bwrap/sandbox-exec), container (podman/docker), "
            "vm (Hyper-V/Kata/Lima), or auto (container→native→noop). "
            "Overrides [sandbox].backend.",
        )
        subparser.add_argument(
            "--allow",
            action="append",
            metavar="RULE",
            help="Add an allow rule, e.g. --allow 'bash(git *)'. Repeatable.",
        )
        subparser.add_argument(
            "--deny",
            action="append",
            metavar="RULE",
            help="Add a deny rule, e.g. --deny 'bash(rm *)'. Repeatable; deny wins.",
        )
        subparser.add_argument(
            "--ask",
            action="append",
            metavar="RULE",
            help="Add an ask rule (force confirmation), e.g. --ask 'bash'. Repeatable.",
        )
        subparser.add_argument("--provider", choices=list(PROVIDERS), default=None)
        subparser.add_argument(
            "--memory",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable cross-conversation memory (recall + extraction).",
        )
        subparser.add_argument(
            "--quiet",
            action="store_true",
            help="Suppress the live thinking/tool trace (only print the final answer).",
        )
        subparser.add_argument(
            "--no-stream",
            action="store_true",
            help="Disable token-by-token streaming; render each turn after it completes.",
        )
        subparser.add_argument(
            "--verbose",
            action="store_true",
            help="Show every read/search tool call instead of folding bursts into one line.",
        )
        subparser.add_argument(
            "--thinking-budget",
            type=int,
            default=None,
            metavar="TOKENS",
            help="Enable Claude extended thinking with this token budget (claude provider).",
        )
        subparser.add_argument(
            "--effort",
            choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
            default=None,
            help="Reasoning/effort depth level; providers gate levels by model, "
            "OpenAI Responses uses model-specific none/minimal/low/medium/high/xhigh/max support "
            "and drops unsupported levels.",
        )
        subparser.add_argument(
            "--max-api-concurrency",
            type=int,
            default=None,
            metavar="N",
            help="Cap simultaneous in-flight LLM API calls across the multi-agent fan-out.",
        )
        subparser.add_argument(
            "--max-wall-seconds",
            type=float,
            default=None,
            metavar="SECONDS",
            help="Wall-clock budget for the whole run (shared by sub-agents); 0 disables it.",
        )
        subparser.add_argument(
            "--max-steps",
            type=int,
            default=None,
            metavar="N",
            help="Hard ceiling on tool turns; 0 (or omitted) means no cap.",
        )

    def add_session_flags(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--resume",
            metavar="SESSION_ID",
            default=None,
            help="Resume a saved session by id (searches this project, then all projects).",
        )
        subparser.add_argument(
            "-c",
            "--continue",
            dest="continue_",
            action="store_true",
            help="Resume the most recent session in the current project.",
        )
        subparser.add_argument(
            "--fork-session",
            action="store_true",
            help="With --resume/--continue: branch into a NEW session, leaving the source intact.",
        )
        subparser.add_argument(
            "--session-id",
            metavar="UUID",
            default=None,
            help="Use this id for the (new or forked) session instead of a generated one.",
        )
        subparser.add_argument(
            "--session-dir",
            metavar="PATH",
            default=None,
            help="Root for resumable transcripts (overrides config/env; ~ is expanded).",
        )
        subparser.add_argument(
            "--no-session-persistence",
            action="store_true",
            help="Do not write a resumable transcript for this run.",
        )

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("task")
    add_common(run_parser)
    add_session_flags(run_parser)
    run_parser.set_defaults(func=run_task)

    chat_parser = subparsers.add_parser("chat")
    add_common(chat_parser)
    add_session_flags(chat_parser)
    chat_parser.set_defaults(func=chat_command)

    sessions_parser = subparsers.add_parser(
        "sessions", help="List resumable sessions saved for the current project."
    )
    sessions_parser.add_argument(
        "action", nargs="?", choices=["list"], default="list", help="list: show saved sessions."
    )
    sessions_parser.add_argument("--session-dir", metavar="PATH", default=None)
    add_config_flag(sessions_parser)
    sessions_parser.set_defaults(func=sessions_command)

    dream_parser = subparsers.add_parser("dream", help="Consolidate memory (decay, merge, insights).")
    add_common(dream_parser)
    dream_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the consolidation without writing any changes.",
    )
    dream_parser.set_defaults(func=dream_command)

    memory_parser = subparsers.add_parser("memory", help="Inspect or curate stored memories.")
    memory_parser.add_argument("action", choices=["list", "add", "forget"])
    memory_parser.add_argument("value", nargs="?", default=None, help="Text for add, id for forget.")
    add_config_flag(memory_parser)
    memory_parser.set_defaults(func=memory_command)

    mcp_parser = subparsers.add_parser("mcp", help="Inspect configured MCP servers and their tools.")
    mcp_parser.add_argument("action", choices=["list"], help="list: show tools from configured servers.")
    add_config_flag(mcp_parser)
    mcp_parser.set_defaults(func=mcp_command)

    replay_parser = subparsers.add_parser(
        "replay", help="Re-render a recorded run's JSONL event log as a readable timeline."
    )
    replay_parser.add_argument("run_id", help="Run id (or unique prefix) of a runs/*.jsonl log.")
    replay_parser.add_argument(
        "--run-dir", metavar="PATH", default="runs", help="Directory holding the run logs."
    )
    replay_parser.set_defaults(func=replay_command)

    health_parser = subparsers.add_parser("health", help="Check the health status of the agent system.")
    add_common(health_parser)
    health_parser.add_argument(
        "--profile",
        choices=["runtime", "dev"],
        default="runtime",
        help="Dependency profile to validate (default: runtime).",
    )
    health_parser.add_argument(
        "--json", action="store_true", help="Emit a machine-readable health report."
    )
    health_parser.set_defaults(func=health_command)

    scheduler_parser = subparsers.add_parser(
        "scheduler-service", help="Manage the scheduler's current-user service."
    )
    scheduler_parser.add_argument("action", choices=["install", "status", "uninstall"])
    scheduler_parser.add_argument(
        "--purge-data", action="store_true", help="Delete the scheduler database on uninstall."
    )
    add_config_flag(scheduler_parser)
    scheduler_parser.set_defaults(func=scheduler_service_command)

    uninstall_parser = subparsers.add_parser(
        "uninstall",
        help="Remove an installer-owned Polaris CLI and its private dependencies.",
    )
    uninstall_parser.add_argument(
        "--purge-data",
        action="store_true",
        help="Also remove user-level ~/.polaris data and the installer state.",
    )
    uninstall_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the exact removal plan without changing files or settings.",
    )
    uninstall_parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the displayed removal plan without prompting.",
    )
    uninstall_parser.set_defaults(func=uninstall_command)

    # Default to `chat` when invoked with no subcommand, so a bare `polaris`
    # (like `claude`/`codex`) drops straight into an interactive session. This also
    # applies when only flags are given (e.g. `polaris --provider fake`), since the
    # leading token is then a flag rather than a command. `-h`/`--help` still shows the
    # top-level help, and a non-flag, non-command token falls through to argparse's
    # usual "invalid choice" error.
    if argv is None:
        argv = sys.argv[1:]
    if not argv or (argv[0].startswith("-") and argv[0] not in {"-h", "--help"}):
        argv = ["chat", *argv]

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

