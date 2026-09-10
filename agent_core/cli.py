from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from agent_core.config import (
    resolve_capabilities_config,
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
    resolve_session_retention_config,
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
from agent_core.memory import (
    Dreamer,
    HybridMemoryRetriever,
    MemoryConfig,
    MemoryPathResolver,
    MemoryRepository,
    MemorySearchRequest,
    RepositoryMemoryStore,
)
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
from agent_core.session import SessionDescriptor, SessionSelection
from agent_core.sandbox import SandboxManager, SandboxRequiredError, get_shared_manager
from agent_core.tools.base import ExecutionScope
from agent_core.tools.registry import ToolRegistry
from agent_core.tools.transaction import JournalStorage, TurnExecutionJournal
from agent_core.transcript import (
    TranscriptStore,
    build_chain,
    fork_chain,
    latest_session,
    list_sessions,
    load_transcript,
    locate_session,
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
    config = resolve_memory_config(getattr(args, "memory", None), _config_file(args))
    override = getattr(args, "memory_dir", None)
    if override:
        config.dir = override
        config.dir_trusted = True
    return config


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
    registry: ToolRegistry,
    config_file: str = "agent.toml",
    *,
    sandbox: SandboxManager | None = None,
    workspace: str | Path | None = None,
) -> "MCPClientManager | None":
    """Connect any configured MCP servers and register their tools, or return ``None``.

    Only connects when ``[mcp.servers.*]`` is non-empty. The caller owns the returned
    manager and must ``close()`` it.
    """
    mcp_config = resolve_mcp_config(config_file)
    if not any(server.enabled for server in mcp_config.servers):
        return None
    if sandbox is not None:
        mcp_config = _sandbox_mcp_config(mcp_config, sandbox, workspace)
    from agent_core.mcp import MCPAdapter

    manager = _connect_mcp(mcp_config)
    # MCP descriptors are discoverable immediately, but their full JSON schemas are
    # exposed only after tool_search/capability_activate selects a matching tool.
    for tool in MCPAdapter(manager).list_tools():
        if getattr(tool, "_always_load", False):
            registry.register(tool)
        else:
            registry.register_deferred(
                tool.name,
                tool.description,
                _constant_factory(tool),
                metadata={
                    "kind": "mcp",
                    "server": str(getattr(tool, "_server", "")),
                    "remote": str(getattr(tool, "_remote", "")),
                },
            )
    return manager


def _sandbox_mcp_config(mcp_config, sandbox: SandboxManager, workspace=None):
    """Translate configured stdio MCP servers into prepared Linux-guest processes."""

    if not sandbox.config.enabled:
        return mcp_config
    from agent_core.mcp import MCPConfig
    from agent_core.lsp import guest_path_to_uri
    from agent_core.plugins import sandbox_runtime_environment, sandboxed_guest_invocation

    root = Path(workspace or Path.cwd()).resolve()
    runtime_env = sandbox_runtime_environment()
    prepared = []
    for server in mcp_config.servers:
        if not server.enabled:
            prepared.append(server)
            continue
        if (server.transport or "stdio").casefold() != "stdio":
            raise RuntimeError(
                f"MCP server {server.name!r} uses a remote transport, but the "
                "container sandbox supports network=deny only"
            )
        if server.env:
            raise RuntimeError(
                f"MCP server {server.name!r} cannot inject host environment into a guest"
            )
        if server.cwd:
            cwd = (
                (root / server.cwd).resolve()
                if not Path(server.cwd).is_absolute()
                else Path(server.cwd).resolve()
            )
            if cwd != root and root not in cwd.parents:
                raise RuntimeError(
                    f"MCP server {server.name!r} cwd is outside the mounted workspace"
                )
        guest_roots: list[str] = []
        for raw in server.roots:
            path = (root / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
            if path != root and root not in path.parents:
                raise RuntimeError(
                    f"MCP server {server.name!r} root is outside the mounted workspace"
                )
            guest_roots.append(guest_path_to_uri(sandbox.translate_path(path)))
        scope = ExecutionScope.for_workspace(root, network="deny")
        invocation = sandboxed_guest_invocation(
            sandbox,
            [server.command, *server.args],
            mounted_roots=(root,),
            scope=scope,
        )
        wrapped, shell = sandbox.wrap_invocation(invocation)
        if shell or not isinstance(wrapped, list) or not wrapped:
            raise RuntimeError(f"sandbox could not wrap MCP server {server.name!r}")
        prepared.append(
            replace(
                server,
                command=str(wrapped[0]),
                args=[str(item) for item in wrapped[1:]],
                cwd="",
                env=runtime_env,
                roots=guest_roots,
            )
        )
    return MCPConfig(prepared)


def _constant_factory(value: Any) -> Callable[[], Any]:
    """Return a typed zero-argument loader for one deferred runtime object."""

    def load() -> Any:
        return value

    return load


def build_agent(args: argparse.Namespace) -> "BuiltAgent":
    # Check the selected session before provider, sandbox, MCP, hooks or scheduler startup.
    session_dir = _session_dir(args)
    selection = _resolve_session(args, session_dir)
    TurnExecutionJournal.require_recovered(
        JournalStorage.user_state(selection.descriptor.workspace, selection.descriptor.session_id, "startup-check"),
        history_path=(
            project_dir(session_dir, selection.descriptor.workspace) / f"{selection.descriptor.session_id}.jsonl"
            if session_dir else None
        ),
    )
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
        streaming_tool_execution=bool(concurrency["streaming_tool_execution"]),
        max_api_concurrency=max_api_concurrency,
        api_rate_limit_per_min=int(concurrency["api_rate_limit_per_min"]),
        max_wall_seconds=max_wall_seconds,
        max_steps=max_steps,
        soft_deadline_fraction=float(limits["soft_deadline_fraction"]),
        session_dir=session_dir,
        persist_compaction_boundary=resolve_persist_compaction_boundary(config_file),
        session_retention=resolve_session_retention_config(config_file),
        skills=resolve_skills_config(config_file),
        capabilities=resolve_capabilities_config(config_file),
        hooks=resolve_hooks_config(config_file),
        sandbox=_sandbox_config(args),
        permission_rules=_permission_rules(args),
        web=resolve_web_config(config_file),
        tools=tool_suite,
    )
    registry = ReActAgent.default_registry()
    sandbox = get_shared_manager(config.sandbox, workspace=Path.cwd())
    sandbox.prepare()
    manager = _start_mcp(
        registry, config_file, sandbox=sandbox, workspace=Path.cwd()
    )
    # Resolve which session this run writes to (new / resumed / continued / forked) and
    # load any prior conversation to seed it. ``seed`` is the fork's cloned chain that
    # must be written into the fresh transcript before the run; for plain resume it is
    # empty because the history already lives on disk.
    agent = ReActAgent(
        provider=provider,
        config=config,
        tools=registry,
        ui=ui,
        session_id=selection.descriptor.session_id,
        mcp_manager=manager, sandbox=sandbox,
    )
    agent._sandbox_cli_locked = (
        getattr(args, "sandbox", None) is not None
        or getattr(args, "sandbox_backend", None) is not None
    )
    return BuiltAgent(agent, ui, manager, list(selection.history), list(selection.seed))


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
) -> SessionSelection:
    """Pick the session id and seed history from ``--resume``/``--continue``/``--fork-session``.

    Returns ``(session_id, history, seed)``: ``history`` is fed to ``run(history=...)``;
    ``seed`` is the (cloned) chain that still needs writing to a fresh transcript (fork),
    empty when the history already exists on disk.

    Frozen product contract:
    - ``--resume`` continues an existing session in the CURRENT project only;
      a session found in another project is rejected with guidance (never
      silently continued or partially imported).
    - ``--continue`` picks the newest session of the current project only.
    - ``--fork-session`` is the ONLY cross-project channel: it clones the
      message chain with fresh uuids/re-linked parents and carries nothing
      else — no permissions, todos, plans, or any other runtime state.
    """
    fork = getattr(args, "fork_session", False)
    explicit = getattr(args, "session_id", None)
    resume_id = getattr(args, "resume", None)
    cont = getattr(args, "continue_", False)
    cwd = Path.cwd().resolve()

    location = None
    if resume_id:
        if not session_dir:
            raise RuntimeError("--resume needs session persistence (it is disabled)")
        location = locate_session(session_dir, cwd, resume_id)
        if location is None:
            raise RuntimeError(f"no session found with id {resume_id!r}")
    elif cont:
        if not session_dir:
            raise RuntimeError("--continue needs session persistence (it is disabled)")
        info = latest_session(project_dir(session_dir, cwd))
        if info is None:
            raise RuntimeError("no prior session to continue in this project")
        source_workspace = info.workspace or cwd
        from agent_core.transcript import SessionLocation

        location = SessionLocation(info.path, source_workspace)

    if location is None:
        session_id = explicit or new_session_id()
        return SessionSelection("new", SessionDescriptor(session_id, cwd))

    loaded = load_transcript(location.path)
    if fork:
        new_id, cloned = fork_chain(loaded)
        session_id = explicit or new_id
        target = SessionDescriptor(session_id, cwd)
        return SessionSelection("fork", target, tuple(cloned), tuple(cloned))
    if os.path.normcase(str(location.workspace.resolve())) != os.path.normcase(str(cwd)):
        raise RuntimeError(
            f"session {loaded.session_id} belongs to project {location.workspace.resolve()}; "
            f"change to that directory and run --resume {loaded.session_id} again, or use "
            "--fork-session to branch into the current project"
        )
    descriptor = SessionDescriptor(loaded.session_id, cwd, location.path)
    return SessionSelection("resume", descriptor, tuple(build_chain(loaded)))


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
    except KeyboardInterrupt:
        # Ctrl-C in a one-shot run: exit quietly with the conventional 130 instead
        # of dumping an asyncio traceback. run_once's finally already fired
        # SessionEnd; the finally below still tears down sandbox/MCP/logger.
        print("[interrupted] run cancelled by user", file=sys.stderr)
        return 130
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


def _recovery_session_id(value: str) -> str:
    if TurnExecutionJournal._safe_relative(value) != value or "/" in value:
        raise argparse.ArgumentTypeError("session id must be a single safe filename component")
    return value


def recovery_command(args: argparse.Namespace) -> int:
    """Inspect/apply recovery without constructing a provider or Agent runtime."""
    try:
        workspace = Path.cwd().resolve()
        storage = JournalStorage.user_state(workspace, args.session_id, "explicit-recovery")
        root = args.session_dir
        history_path = (
            (project_dir(root, workspace) / f"{args.session_id}.jsonl").absolute() if root else None
        )
        report = TurnExecutionJournal.inspect_recovery(storage, history_path=history_path)
        result = report.to_dict()
        result["dry_run"] = not args.apply
        outcomes = report.outcomes
        if args.apply:
            transcript = TranscriptStore(root, workspace, args.session_id) if root else None
            outcomes = TurnExecutionJournal.recover_all(
                storage, dry_run=False,
                history_writer=transcript.recover_tool_round if transcript is not None else None,
                history_path=history_path, authorization_source="explicit_cli",
            )
            result["outcomes"] = outcomes
            result["blocked"] = TurnExecutionJournal.inspect_recovery(storage, history_path=history_path).blocked
        acceptable = {"rolled_back", "history_persisted"} if args.apply else {
            "would_rollback_workspace", "would_recover_history", "would_cleanup_overlay",
        }
        failed = any(item["status"] not in acceptable for item in outcomes)
        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(f"Recovery for session {args.session_id} in {workspace} ({'apply' if args.apply else 'preview'}):")
            for plan in report.plans:
                print(f"  Turn {plan['turn_id']} / run {plan['run_id']}: {plan['action']}")
                for action in plan["actions"]:
                    print(f"    {action['action']}: {action['target']}")
            for outcome in outcomes:
                print(f"  {outcome['turn_id'] or 'state directory'}: {outcome['status']} {outcome.get('reason', '')}".rstrip())
            if not outcomes:
                print("  No unfinished recovery journals.")
            elif not args.apply:
                print("Review these actions, then use --apply to execute. Rejected records are never applied.")
        return 1 if failed or (args.apply and result["blocked"]) else 0
    except (OSError, RuntimeError, ValueError) as exc:
        if args.json:
            print(json.dumps({"schema_version": 1, "status": "rejected", "reason": str(exc)}, ensure_ascii=False))
        else:
            print(f"[recovery] {exc}", file=sys.stderr)
        return 1


def sessions_command(args: argparse.Namespace) -> int:
    """List resumable sessions saved for the current project, newest first."""
    root = getattr(args, "session_dir", None) or resolve_session_dir(_config_file(args))
    if not root:
        print("Session persistence is disabled (empty session dir).")
        return 0
    cwd = Path.cwd().resolve()
    if getattr(args, "action", "list") == "prune":
        from agent_core.retention import prune_sessions

        report = prune_sessions(
            root,
            cwd,
            resolve_session_retention_config(_config_file(args)),
            apply=bool(getattr(args, "apply", False)),
        )
        verb = "Deleted" if report["dry_run"] is False else "Would delete"
        print(
            f"{verb} {report['deleted'] if not report['dry_run'] else len(report['selected'])} "
            f"session(s), {report['bytes']} byte(s); protected={report['protected']}, "
            f"errors={report['errors']}."
        )
        for session_id in report["selected"]:
            print(f"  {session_id}")
        if report["dry_run"]:
            print("Re-run with --apply to execute this retention plan.")
        return 0
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


def _open_repository(config: MemoryConfig, *, scope: str = "private") -> MemoryRepository:
    override = config.dir if config.dir_trusted and scope == "private" else None
    resolver = MemoryPathResolver(Path.cwd(), private_override=override)
    root = resolver.resolve(scope)  # type: ignore[arg-type]
    repository = MemoryRepository(root, scope=scope)
    legacy_root = Path(config.dir) if config.dir_trusted else Path.cwd() / "memory"
    legacy = legacy_root / "memory.jsonl"
    if legacy.exists() and scope == "private":
        repository.migrate_jsonl(legacy)
    return repository


def _open_store(config: MemoryConfig) -> RepositoryMemoryStore:
    return RepositoryMemoryStore(_open_repository(config))


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
    """Inspect, search, validate, migrate, or curate Markdown memories."""
    if args.action == "models":
        from agent_core.memory.models_manager import MemoryModelManager, ModelInstallError

        operation = args.value or "status"
        manager = MemoryModelManager()
        golden_ok = True
        golden_detail = ""
        if operation == "status":
            status = manager.status()
        elif operation == "install":
            try:
                status = manager.install(
                    bundle=getattr(args, "model_bundle", None),
                )
            except ModelInstallError as exc:
                print(f"[error] {exc}", file=sys.stderr)
                return 1
            if status.valid:
                from agent_core.memory.runtime import golden_inference_check

                golden_ok, golden_detail = golden_inference_check(manager=manager)
        else:
            print("[error] `memory models` accepts status or install", file=sys.stderr)
            return 1
        payload = status.to_dict()
        if operation == "install":
            payload["golden_inference"] = {
                "valid": golden_ok,
                "detail": golden_detail,
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if status.valid and golden_ok else 1
    config = _memory_config(args)
    scope = getattr(args, "scope", "private")
    repository = _open_repository(config, scope=scope)
    if args.action == "list":
        documents = sorted(repository.list(), key=lambda item: item.updated_at, reverse=True)
        if not documents:
            print("(no memories)")
            return 0
        for document in documents:
            print(
                f"{document.id}  [{scope}/{document.type}] updated={document.updated_at}  "
                f"{document.name}  {document.path}"
            )
        return 0
    if args.action == "show":
        if not args.value:
            print("[error] `memory show` needs an id", file=sys.stderr)
            return 1
        shown_document = repository.get(args.value)
        if shown_document is None:
            print(f"[error] no memory {args.value}", file=sys.stderr)
            return 1
        print(
            f"{shown_document.name}\n[{scope}/{shown_document.type}] "
            f"updated={shown_document.updated_at}"
        )
        print(f"source={', '.join(shown_document.sources) or '-'}\npath={shown_document.path}\n")
        print(shown_document.content)
        return 0
    if args.action == "search":
        if not args.value:
            print("[error] `memory search` needs a query", file=sys.stderr)
            return 1
        filters = {
            key: list(getattr(args, f"filter_{key}", None) or [])
            for key in ("id", "tag", "source")
        }
        if getattr(args, "memory_type", None):
            filters["type"] = [args.memory_type]
        retriever = HybridMemoryRetriever(repository, config)
        request = MemorySearchRequest.from_values(
            args.value,
            scope=scope,
            limit=getattr(args, "limit", 5),
            filters=filters,
            include_content=bool(getattr(args, "full_content", False)),
            explain=bool(getattr(args, "explain", False)),
        )
        hits = retriever.search(request)
        if request.explain:
            print(
                json.dumps(
                    {
                        "hits": [hit.to_dict() for hit in hits],
                        "trace": retriever.last_trace.to_dict(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        for hit in hits:
            print(f"{hit.id}  [{hit.type}] {hit.name}  {hit.description}")
            if request.include_content and hit.content is not None:
                print(hit.content)
            else:
                for passage in hit.passages:
                    heading = f" ({passage.heading})" if passage.heading else ""
                    print(f"  {passage.chunk_id}{heading}: {passage.content}")
        return 0
    if args.action == "add":
        if not args.value:
            print("[error] `memory add` needs text", file=sys.stderr)
            return 1
        name = getattr(args, "name", None) or " ".join(args.value.split())[:60]
        description = getattr(args, "description", None) or " ".join(args.value.split())[:160]
        document = repository.write(
            name=name,
            description=description,
            type=getattr(args, "memory_type", None) or "project",
            content=args.value,
            confidence=0.6,
            explicit=True,
            sources=["cli"],
        )
        print(f"Added {document.id} at {document.path}")
        return 0
    if args.action == "edit":
        if not args.value or not getattr(args, "text", None):
            print("[error] `memory edit` needs an id and --text", file=sys.stderr)
            return 1
        document = repository.update(args.value, content=args.text, explicit=True)
        print(f"Updated {document.id} at {document.path}")
        return 0
    if args.action == "forget":
        if not args.value:
            print("[error] `memory forget` needs an id", file=sys.stderr)
            return 1
        if repository.forget(args.value):
            print(f"Forgot {args.value}")
            return 0
        print(f"[error] no memory {args.value}", file=sys.stderr)
        return 1
    if args.action == "validate":
        validation = repository.validate(repair=getattr(args, "repair", False))
        for warning in validation.warnings:
            print(f"[warning] {warning}")
        for error in validation.errors:
            print(f"[error] {error}", file=sys.stderr)
        print(
            f"scanned={validation.scanned} valid={validation.valid} "
            f"index_rebuilt={validation.index_rebuilt}"
        )
        return 0 if validation.valid else 1
    if args.action == "migrate":
        source = Path(args.value) if args.value else Path(config.dir) / "memory.jsonl"
        migration = repository.migrate_jsonl(source)
        print(
            f"source={migration.source} total={migration.total} imported={migration.imported} "
            f"skipped={migration.skipped} corrupt={len(migration.corrupt_lines)} "
            f"already_complete={migration.already_complete}"
        )
        return 0 if not migration.corrupt_lines else 1
    if args.action == "index":
        operation = args.value or "status"
        assert config.retrieval is not None
        # Construct the same engine as runtime recall so the active model
        # fingerprint selects the same versioned derived-index directory.
        engine = HybridMemoryRetriever(repository, config)
        index = engine.index
        if operation == "rebuild":
            index.rebuild()
            if engine.embedding_backend is not None:
                index.populate_embeddings(engine.embedding_backend)
                index_status = index.status()
                if (
                    config.retrieval.dense_strategy != "exact"
                    and index_status.coverage >= 1.0
                    and index_status.embedded_chunks
                    >= config.retrieval.ann_min_vectors
                ):
                    engine.ann_index.build()
            index_status = index.status()
        elif operation == "status":
            try:
                index_status = index.ensure_current()
            except Exception:
                index_status = index.status()
        else:
            print("[error] `memory index` accepts status or rebuild", file=sys.stderr)
            return 1
        print(json.dumps(index_status.to_dict(), ensure_ascii=False, indent=2))
        return 0 if index_status.schema_version else 1
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
            import sqlite3

            try:
                with sqlite3.connect(":memory:") as connection:
                    connection.execute("CREATE VIRTUAL TABLE probe_fts USING fts5(content)")
            except sqlite3.DatabaseError as exc:
                checks.append(HealthCheck("memory-fts5", True, "error", detail=str(exc)))
            else:
                checks.append(HealthCheck("memory-fts5", True, "ok", detail="available"))
            repository = store.repository
            assert memory_config.retrieval is not None
            index_status = HybridMemoryRetriever(repository, memory_config).index.ensure_current()
            checks.append(
                HealthCheck(
                    "memory-index",
                    True,
                    "ok" if index_status.schema_version else "error",
                    version=str(index_status.schema_version),
                    detail=(
                        f"documents={index_status.documents} chunks={index_status.chunks} "
                        f"coverage={index_status.coverage:.1%} "
                        f"pending={index_status.pending_embeddings} "
                        f"ann={index_status.ann_state} "
                        f"ann_coverage={index_status.ann_coverage:.1%}"
                    ),
                )
            )
            if index_status.pending_embeddings:
                checks.append(
                    HealthCheck(
                        "memory-index-coverage",
                        False,
                        "degraded",
                        detail=(
                            f"{index_status.pending_embeddings} embedding(s) pending; "
                            "lexical retrieval remains available"
                        ),
                    )
                )
            checks.append(
                HealthCheck(
                    "memory-ann",
                    False,
                    (
                        "ok"
                        if index_status.ann_state
                        in {"ready", "not_needed", "disabled"}
                        else "degraded"
                    ),
                    version=index_status.ann_generation,
                    detail=(
                        f"state={index_status.ann_state} "
                        f"vectors={index_status.ann_vectors} "
                        f"coverage={index_status.ann_coverage:.1%}; "
                        f"dense_backend={index_status.dense_backend}"
                    ),
                )
            )
            from agent_core.memory.models_manager import MemoryModelManager

            manager = MemoryModelManager()
            model_status = manager.status()
            model_required = not manager.explicitly_skipped
            checks.append(
                HealthCheck(
                    "memory-models",
                    model_required,
                    "ok" if model_status.valid else ("missing" if not model_required else "error"),
                    version=model_status.bundle_id,
                    detail=model_status.detail,
                )
            )
            if model_status.valid:
                from agent_core.memory.runtime import golden_inference_check

                golden_ok, golden_detail = golden_inference_check(manager=manager)
                checks.append(
                    HealthCheck(
                        "memory-model-golden",
                        model_required,
                        "ok" if golden_ok else "error",
                        detail=golden_detail,
                    )
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

    sandbox: SandboxManager | None = None
    try:
        sandbox = get_shared_manager(_sandbox_config(args), workspace=Path.cwd())
        sandbox.prepare()
        mcp_config = _sandbox_mcp_config(mcp_config, sandbox, Path.cwd())
        manager = _connect_mcp(mcp_config)
    except RuntimeError as exc:
        if sandbox is not None:
            sandbox.teardown()
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
        if sandbox is not None:
            sandbox.teardown()


def capabilities_command(args: argparse.Namespace) -> int:
    """Inspect the configured local/trusted discovery inputs without constructing an agent."""

    from types import SimpleNamespace

    from agent_core.capabilities import CapabilityManager
    from agent_core.plugins import PluginManager
    from agent_core.skills import SkillRegistry, discover_skill_dirs, load_skills
    from agent_core.tools.registry import ToolRegistry

    config_file = _config_file(args)
    config = resolve_capabilities_config(config_file)
    skills_config = resolve_skills_config(config_file)
    manager = PluginManager(Path.cwd())
    if args.action == "status":
        print(f"mode={config.mode}")
        print(f"trusted_marketplaces={','.join(config.trusted_marketplaces) or '(none)'}")
        print(f"installed_plugins={len(manager.records())}")
        print(f"enabled_plugins={len(manager.enabled_ids())}")
        print(f"reset_required={str(manager.state_status().reset_required).lower()}")
        print(f"mcp_registry={str(config.mcp_registry.enabled).lower()}")
        return 0

    query = str(args.query or "").casefold().strip()
    if not query:
        print("[error] capabilities search requires a query", file=sys.stderr)
        return 2
    skills = SkillRegistry(
        load_skills(discover_skill_dirs(Path.cwd(), skills_config), skills_config.disabled)
    )
    catalog_agent = SimpleNamespace(
        skills=skills,
        registry=ToolRegistry(),
        session=SimpleNamespace(workspace=Path.cwd()),
        _plugin_active_ids=frozenset(),
    )
    result = CapabilityManager(cast(Any, catalog_agent), config).search(query)
    matches = result.get("matches", [])
    for item in matches:
        components = ",".join(item.get("components", []))
        print(
            f"{item['id']}  [{item.get('trust_tier')}; {components or item.get('kind')}]  "
            f"{item.get('description', '')}".rstrip()
        )
    if not matches:
        print("(no matching capabilities)")
    return 0


def plugins_command(args: argparse.Namespace) -> int:
    """Inspect or explicitly clear capability/plugin v1-v2 state."""

    from agent_core.plugins import PluginManager

    manager = PluginManager(Path.cwd())
    if args.action == "status":
        state = manager.state_status()
        print(f"schema_version={state.schema_version}")
        print(f"reset_required={str(state.reset_required).lower()}")
        print(f"installed_plugins={len(manager.records())}")
        print(f"enabled_plugins={len(manager.enabled_ids())}")
        for target in state.targets:
            print(f"legacy_target={target}")
        return 1 if state.reset_required else 0
    if args.action == "errors":
        path = manager.audit.path
        if not path.is_file():
            print("(no capability errors recorded)")
            return 0
        shown = 0
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("event") not in {"activation_failed", "catalog_sync"}:
                continue
            detail = event.get("detail", {})
            if event.get("event") == "catalog_sync" and detail.get("result") != "failed":
                continue
            print(json.dumps(event, ensure_ascii=False))
            shown += 1
            if shown >= 20:
                break
        if not shown:
            print("(no capability errors recorded)")
        return 0
    targets = manager.reset_preview()
    print("Plugin reset targets:")
    for target in targets:
        print(f"  {target}")
    if args.dry_run:
        return 0
    if not args.yes:
        print("[error] reset requires --yes (use --dry-run to preview)", file=sys.stderr)
        return 2
    removed = manager.reset_state()
    print(f"Removed/reset {len(removed)} managed targets; capability schema v3 is ready.")
    return 0


def _force_utf8_output() -> None:
    """Ensure stdout/stderr use UTF-8 so model output (emoji, CJK) prints on
    consoles whose default codec is narrow (e.g. GBK on zh-CN Windows)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def swebench_command(args: argparse.Namespace) -> int:
    try:
        return _swebench_command(args)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"[error] SWE-bench: {exc}", file=sys.stderr)
        return 2


def _swebench_command(args: argparse.Namespace) -> int:
    """Entry point for the isolated SWE-bench Lite runner.

    This command intentionally does not call :func:`build_agent`: that path loads
    the current project's config, MCP servers, memory and one shared workspace.
    Benchmark instances need an independent workspace/runtime and a clean Agent
    configuration for every task.
    """
    from agent_core.benchmarks.swebench.dataset import SWEbenchDataset
    from agent_core.benchmarks.swebench.models import SWEbenchRunConfig
    from agent_core.benchmarks.swebench.runner import SWEbenchRunner, _safe_run_id
    from agent_core.benchmarks.swebench.selection import make_selection

    action = str(getattr(args, "swebench_action", "run"))
    dataset_name = str(args.dataset)
    split = str(args.split)
    if action == "list":
        dataset = SWEbenchDataset.load(
            dataset_name,
            split,
            cache_dir=getattr(args, "cache_dir", None),
            include_gold=False,
        )
        if args.limit is not None and int(args.limit) <= 0:
            raise ValueError("--limit must be positive")
        ids = list(args.instance_id or [])
        if ids:
            rows = dataset.select(ids)
        else:
            rows = tuple(dataset)[: max(1, int(args.limit or 20))]
        for item in rows:
            title = item.problem_statement.strip().splitlines()[0] if item.problem_statement.strip() else ""
            difficulty = str(item.extra.get("difficulty") or "")
            print(f"{item.instance_id}\t{item.repo}\t{difficulty}\t{title[:180]}")
        print(f"Listed {len(rows)} of {len(dataset)} instances ({dataset_name}, {split}).")
        return 0

    smoke = action == "smoke"
    if smoke and args.limit is not None and int(args.limit) > 5:
        raise ValueError("smoke mode accepts at most 5 instances")
    if smoke and not args.instance_id and not args.selection:
        raise ValueError("smoke mode requires explicit --instance-id values or --selection")
    if args.evaluate and args.no_evaluate:
        raise ValueError("--evaluate and --no-evaluate are mutually exclusive")
    if args.resume and not args.run_id:
        raise ValueError("--resume requires --run-id so the prior run directory is unambiguous")
    selection_file = args.selection
    if (
        args.resume
        and not smoke
        and selection_file is None
        and not args.instance_id
        and not args.all
        and args.limit is None
    ):
        # A run writes an ID-only selection manifest.  Reuse it when the caller
        # supplies only ``--resume --run-id``; this keeps retries deterministic
        # without requiring the user to repeat a long list of task IDs.
        prior_selection = (
            Path(args.output_dir).expanduser().resolve()
            / _safe_run_id(str(args.run_id))
            / "selection.yaml"
        )
        if prior_selection.exists():
            selection_file = str(prior_selection)
        else:
            raise ValueError(
                "resume run selection was not found; provide --selection, --instance-id, "
                "--limit, or --all"
            )
    selection = make_selection(
        dataset=dataset_name,
        split=split,
        instance_ids=args.instance_id,
        selection_file=selection_file,
        limit=args.limit,
        all_instances=bool(args.all),
        smoke=smoke,
    )
    provider_name = str(args.provider or os.getenv("AGENT_PROVIDER") or "claude")
    model_name = str(args.model or os.getenv("AGENT_MODEL") or ("claude-opus-4-8" if provider_name == "claude" else ""))
    if provider_name != "fake" and not model_name:
        raise ValueError("--model is required for non-fake SWE-bench providers")
    evaluate = bool(args.evaluate) and not bool(args.no_evaluate)
    run_config = SWEbenchRunConfig(
        dataset=selection.dataset,
        split=selection.split,
        output_dir=args.output_dir,
        run_id=args.run_id or "",
        model=model_name,
        provider=provider_name,
        image=args.image,
        test_command=args.test_command,
        solve_workers=max(1, int(args.solve_workers)),
        evaluation_workers=max(1, int(args.evaluation_workers)),
        max_wall_seconds=float(args.max_wall_seconds),
        max_steps=(None if args.max_steps is None or int(args.max_steps) <= 0 else int(args.max_steps)),
        keep_workspaces=bool(args.keep_workspaces),
        evaluate=evaluate,
        resume=bool(args.resume),
        retry_failed=bool(args.retry_failed),
        live=bool(args.live),
        metadata={
            "runtime": args.runtime,
            "source_dir": args.source_dir,
            "cache_dir": args.cache_dir,
            "no_stream": bool(args.no_stream),
            "max_api_concurrency": int(args.max_api_concurrency),
        },
    )
    values = {"provider": provider_name, "model": model_name, "effort": None}
    provider = _make_provider(values)

    def ui_factory() -> AgentUI:
        if not args.live:
            return NullUI()
        return ConsoleUI(verbose=bool(args.verbose))

    runner = SWEbenchRunner(run_config, provider=provider, ui_factory=ui_factory)
    summary = asyncio.run(runner.run(selection, all_instances=bool(args.all), limit=args.limit))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get("resolved_rate_selected") is not None:
        print(
            "SWE-bench summary: "
            f"resolved={summary['resolved']}/{summary['selected']} "
            f"resolved_rate={summary['resolved_rate_selected']:.3f}"
        )
    return 0 if not any(item.get("state") == "failed" for item in summary.get("instances", [])) else 2


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

    def add_sandbox_flags(subparser: argparse.ArgumentParser) -> None:
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
            "vm (Hyper-V/Kata/Lima), or auto (container→native; fail closed if unavailable). "
            "Overrides [sandbox].backend.",
        )

    def add_common(subparser: argparse.ArgumentParser) -> None:
        add_config_flag(subparser)
        add_sandbox_flags(subparser)
        subparser.add_argument("--model", default=None)
        subparser.add_argument(
            "--permission",
            type=lambda value: PermissionMode(value).value,
            metavar="MODE",
            default=None,
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

    swebench_parser = subparsers.add_parser(
        "swebench", help="Run SWE-bench Lite tasks with isolated workspaces and the official Harness."
    )
    swebench_actions = swebench_parser.add_subparsers(dest="swebench_action", required=True)

    def add_swebench_flags(subparser: argparse.ArgumentParser, *, default_split: str) -> None:
        subparser.add_argument("--dataset", default="SWE-bench/SWE-bench_Lite", help="HuggingFace dataset name or local JSON/JSONL path.")
        subparser.add_argument("--split", default=default_split, help="Dataset split (Lite smoke commonly uses dev; full run uses test).")
        subparser.add_argument("--cache-dir", default=None, metavar="PATH")
        subparser.add_argument("--instance-id", action="append", default=[], metavar="ID", help="Select one instance; repeat for 3-5 task smoke runs.")
        subparser.add_argument("--selection", default=None, metavar="PATH", help="YAML/JSON manifest containing dataset, split and instance_ids.")
        subparser.add_argument("--limit", type=int, default=None, help="Limit selected rows; smoke mode caps this at 5.")
        subparser.add_argument("--all", action="store_true", help="Explicitly allow running every row in the selected split.")
        subparser.add_argument("--output-dir", default="swebench_runs", metavar="PATH")
        subparser.add_argument("--run-id", default=None, metavar="ID")
        subparser.add_argument("--provider", choices=list(PROVIDERS), default=None)
        subparser.add_argument("--model", default=None)
        subparser.add_argument("--runtime", choices=["docker", "local"], default="docker", help="Docker is the production path; local is an explicit development/test fallback.")
        subparser.add_argument("--image", default=None, help="Override the official SWE-bench instance image.")
        subparser.add_argument("--source-dir", default=None, help="Use a local repository source for offline development tests.")
        subparser.add_argument("--test-command", default=None, help="Default command for swebench_test (default: pytest -q).")
        subparser.add_argument("--solve-workers", type=int, default=1)
        subparser.add_argument("--evaluation-workers", type=int, default=1)
        subparser.add_argument("--max-wall-seconds", type=float, default=1800.0)
        subparser.add_argument("--max-steps", type=int, default=80)
        subparser.add_argument("--max-api-concurrency", type=int, default=8)
        subparser.add_argument("--keep-workspaces", action="store_true", help="Retain mutable checkouts for debugging.")
        subparser.add_argument("--resume", action="store_true", help="Resume terminal/patch-ready instance states in this run directory.")
        subparser.add_argument("--retry-failed", action="store_true", help="With --resume, retry failed instances.")
        subparser.add_argument("--evaluate", action="store_true", help="Run the official SWE-bench Harness after solving.")
        subparser.add_argument("--no-evaluate", action="store_true", help="Do not invoke the Harness.")
        subparser.add_argument("--live", action="store_true", help="Show the interactive Agent trace (recommended only for smoke runs).")
        subparser.add_argument("--verbose", action="store_true")
        subparser.add_argument("--no-stream", action="store_true")

    swebench_list = swebench_actions.add_parser("list", help="List safe task metadata without gold patches/tests.")
    add_swebench_flags(swebench_list, default_split="dev")
    swebench_list.set_defaults(func=swebench_command)
    swebench_run = swebench_actions.add_parser("run", help="Solve selected tasks or an explicit full split.")
    add_swebench_flags(swebench_run, default_split="test")
    swebench_run.set_defaults(func=swebench_command)
    swebench_smoke = swebench_actions.add_parser("smoke", help="Run 1-5 manually selected tasks using the full runner path.")
    add_swebench_flags(swebench_smoke, default_split="dev")
    swebench_smoke.set_defaults(func=swebench_command)

    chat_parser = subparsers.add_parser("chat")
    add_common(chat_parser)
    add_session_flags(chat_parser)
    chat_parser.set_defaults(func=chat_command)

    recovery_parser = subparsers.add_parser("recovery", help="Preview or explicitly apply this project's session recovery.")
    recovery_parser.add_argument("--session-id", required=True, type=_recovery_session_id)
    recovery_parser.add_argument(
        "--session-dir", default=os.getenv("AGENT_SESSION_DIR", "~/.polaris/projects"),
        help="Transcript root (default: AGENT_SESSION_DIR or ~/.polaris/projects); pass an empty string to disable history writes.",
    )
    recovery_mode = recovery_parser.add_mutually_exclusive_group()
    recovery_mode.add_argument("--dry-run", action="store_true", help="Preview only (the default).")
    recovery_mode.add_argument("--apply", action="store_true", help="Explicitly execute validated recovery actions.")
    recovery_parser.add_argument("--json", action="store_true", help="Print a structured recovery report.")
    recovery_parser.set_defaults(func=recovery_command)

    sessions_parser = subparsers.add_parser(
        "sessions", help="List resumable sessions saved for the current project."
    )
    sessions_parser.add_argument(
        "action", nargs="?", choices=["list", "prune"], default="list",
        help="list saved sessions, or preview retention pruning."
    )
    sessions_parser.add_argument(
        "--apply", action="store_true", help="Apply a sessions prune plan."
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
    memory_parser.add_argument(
        "action",
        choices=[
            "list", "show", "search", "add", "edit", "forget", "validate",
            "migrate", "index", "models",
        ],
    )
    memory_parser.add_argument("value", nargs="?", default=None, help="Text, query, id, or source path.")
    memory_parser.add_argument("--scope", choices=["private", "team"], default="private")
    memory_parser.add_argument("--memory-dir", metavar="PATH", default=None)
    memory_parser.add_argument("--name", default=None)
    memory_parser.add_argument("--description", default=None)
    memory_parser.add_argument("--type", dest="memory_type", choices=["user", "feedback", "project", "reference"], default=None)
    memory_parser.add_argument("--text", default=None, help="Replacement content for edit.")
    memory_parser.add_argument("--limit", type=int, default=5)
    memory_parser.add_argument("--id", dest="filter_id", action="append", default=[])
    memory_parser.add_argument("--tag", dest="filter_tag", action="append", default=[])
    memory_parser.add_argument("--source", dest="filter_source", action="append", default=[])
    memory_parser.add_argument("--explain", action="store_true")
    memory_parser.add_argument("--full-content", action="store_true")
    memory_parser.add_argument("--model-bundle", metavar="PATH", default=None)
    memory_parser.add_argument("--repair", action="store_true")
    add_config_flag(memory_parser)
    memory_parser.set_defaults(func=memory_command)

    mcp_parser = subparsers.add_parser("mcp", help="Inspect configured MCP servers and their tools.")
    mcp_parser.add_argument("action", choices=["list"], help="list: show tools from configured servers.")
    add_config_flag(mcp_parser)
    add_sandbox_flags(mcp_parser)
    mcp_parser.set_defaults(func=mcp_command)

    capabilities_parser = subparsers.add_parser(
        "capabilities", help="Inspect or search configured capability discovery sources."
    )
    capabilities_parser.add_argument("action", choices=["status", "search"])
    capabilities_parser.add_argument("query", nargs="?", default=None)
    add_config_flag(capabilities_parser)
    capabilities_parser.set_defaults(func=capabilities_command)

    plugins_parser = subparsers.add_parser(
        "plugins", help="Inspect/reset the capability v3 plugin state and audit errors."
    )
    plugins_parser.add_argument("action", choices=["status", "reset", "errors"])
    plugins_parser.add_argument("--dry-run", action="store_true")
    plugins_parser.add_argument("--yes", action="store_true")
    plugins_parser.set_defaults(func=plugins_command)

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

