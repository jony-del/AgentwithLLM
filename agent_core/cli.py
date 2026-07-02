from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

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
)
from agent_core.permission_rules import RuleSet
from agent_core.interrupt import KeyInterrupt
from agent_core.memory import Dreamer, MemoryConfig, MemoryStore
from agent_core.models import LLMTransientError, Message
from agent_core.providers import ClaudeProvider, FakeProvider
from agent_core.chat_commands import dispatch as dispatch_chat_command
from agent_core.react import ReActAgent, ReActConfig
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
    from prompt_toolkit.completion import Completer


def _resolve(args: argparse.Namespace) -> dict:
    return resolve_config(
        {
            "model": args.model,
            "permission": args.permission,
            "provider": args.provider,
            "effort": getattr(args, "effort", None),
        }
    )


def _memory_config(args: argparse.Namespace) -> MemoryConfig:
    # Numeric tunables come from the [memory] toml table; enabled is overridable
    # by AGENT_MEMORY / --memory. (resolve_config above already loaded the .env.)
    return resolve_memory_config(getattr(args, "memory", None))


def _permission_rules(args: argparse.Namespace) -> RuleSet:
    """Fine-grained rules from ``[permissions]`` toml, with CLI ``--allow/--deny/--ask``
    session rules layered on top (they append, deny still wins in the decision pipeline)."""
    rules = resolve_permission_rules()
    cli = RuleSet.from_lists(
        allow=getattr(args, "allow", None) or [],
        deny=getattr(args, "deny", None) or [],
        ask=getattr(args, "ask", None) or [],
    )
    return rules.merge(cli)


def _sandbox_config(args: argparse.Namespace):
    """Sandbox config from ``[sandbox]`` toml/env, with the ``--sandbox/--no-sandbox``
    CLI flag layered on ``enabled`` (None = leave the resolved value untouched)."""
    config = resolve_sandbox_config()
    cli_sandbox = getattr(args, "sandbox", None)
    if cli_sandbox is not None:
        config.enabled = bool(cli_sandbox)
    return config


def _make_provider(values: dict):
    return FakeProvider() if values["provider"] == "fake" else ClaudeProvider()


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


def _start_mcp(registry: ToolRegistry):
    """Connect any configured MCP servers and register their tools, or return ``None``.

    Only connects when ``[mcp.servers.*]`` is non-empty. The caller owns the returned
    manager and must ``close()`` it.
    """
    mcp_config = resolve_mcp_config()
    if not any(server.enabled for server in mcp_config.servers):
        return None
    from agent_core.mcp import MCPAdapter

    manager = _connect_mcp(mcp_config)
    registry.register_adapter(MCPAdapter(manager))
    return manager


def build_agent(args: argparse.Namespace) -> "BuiltAgent":
    values = _resolve(args)
    provider = _make_provider(values)
    ui = _make_ui(args)
    concurrency = resolve_concurrency_config()
    cli_api_concurrency = getattr(args, "max_api_concurrency", None)
    max_api_concurrency = (
        max(1, int(cli_api_concurrency)) if cli_api_concurrency is not None
        else int(concurrency["max_api_concurrency"])
    )
    # Run-level safety limits: [limits]/env resolved here, CLI flags layered on top.
    # A CLI value of 0 disables the cap (None), mirroring the toml/env convention.
    limits = resolve_limits_config()
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
    context = resolve_context_config()
    config = ReActConfig(
        model=values["model"],
        permission=values["permission"],
        memory=_memory_config(args),
        output=resolve_output_config(),
        compression=resolve_compression_config(),
        tool_use_summary=resolve_tool_use_summary_config(),
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
        persist_compaction_boundary=resolve_persist_compaction_boundary(),
        skills=resolve_skills_config(),
        hooks=resolve_hooks_config(),
        sandbox=_sandbox_config(args),
        permission_rules=_permission_rules(args),
    )
    registry = ReActAgent.default_registry()
    manager = _start_mcp(registry)
    # Resolve which session this run writes to (new / resumed / continued / forked) and
    # load any prior conversation to seed it. ``seed`` is the fork's cloned chain that
    # must be written into the fresh transcript before the run; for plain resume it is
    # empty because the history already lives on disk.
    session_id, history, seed = _resolve_session(args, config.session_dir)
    agent = ReActAgent(
        provider=provider, config=config, tools=registry, ui=ui, session_id=session_id
    )
    return BuiltAgent(agent, ui, manager, history, seed)


@dataclass(slots=True)
class BuiltAgent:
    agent: ReActAgent
    ui: AgentUI
    mcp: object | None
    history: list[Message]
    seed: list[Message]


def _session_dir(args: argparse.Namespace) -> str:
    """Transcript root: config/env resolution, then ``--session-dir`` /
    ``--no-session-persistence`` CLI overrides."""
    if getattr(args, "no_session_persistence", False):
        return ""
    cli = getattr(args, "session_dir", None)
    return cli if cli else resolve_session_dir()


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

    session = getattr(_async_input, "_session", None)
    if session is None:
        toggle = getattr(ui, "toggle_verbose", None)
        session = PromptSession(
            key_bindings=create_keybindings(toggle),
            multiline=True,
            completer=completer,
            complete_while_typing=True,  # menu pops the moment '/' is typed
            complete_style=CompleteStyle.COLUMN,  # single column shows the description meta
            style=completion_menu_style(),
            bottom_toolbar=bottom_toolbar,
        )
        _async_input._session = session

    try:
        line = await session.prompt_async(HTML(f"<ansicyan>{prompt}</ansicyan> "))
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
        try:
            loop.call_soon_threadsafe(lambda: future.done() or future.set_result(line))
        except RuntimeError:
            pass  # loop already closed (e.g. Ctrl-C tore the session down)

    threading.Thread(target=read, daemon=True, name="chat-input").start()
    return await future


async def _seed_transcript(built: "BuiltAgent") -> None:
    """Write a fork's cloned chain into its fresh transcript before the first turn."""
    if built.seed and built.agent.transcript is not None:
        for message in built.seed:
            await built.agent.transcript.append_message(message)


def run_command(args: argparse.Namespace) -> int:
    try:
        built = build_agent(args)
    except RuntimeError as exc:
        # E.g. an MCP server failed to connect, or a bad --resume id.
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    agent, ui, mcp = built.agent, built.ui, built.mcp

    async def run_once():
        await _seed_transcript(built)
        with KeyInterrupt(confirm=True) as interrupt:
            return await agent.run(
                args.task, should_cancel=interrupt.is_set, history=built.history or None
            )

    try:
        result = asyncio.run(run_once())
    except RuntimeError as exc:
        # Covers LLMTransientError (network exhausted retries) and API errors.
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    finally:
        if mcp is not None:
            mcp.close()
    # A live UI already streamed the answer via on_final; only print it ourselves
    # when the run was silent (piped/--quiet) so we don't echo it twice.
    if not ui.is_live:
        print(result.answer)
    print(f"\nRun log: runs/{result.run_id}.jsonl")
    if agent.transcript is not None:
        print(f"Session: {agent.session_id}  (resume with --resume {agent.session_id})")
    return 0


def chat_command(args: argparse.Namespace) -> int:
    try:
        built = build_agent(args)
    except RuntimeError as exc:
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

        completer = SlashCompleter(agent)

        def _toolbar() -> str:
            # Persistent status line: the active model + effort, so the current config
            # (and the effect of the /model picker) is always visible. Read per render.
            effort = agent.config.effort or "—"
            return f" model: {agent.config.model}  ·  effort: {effort}  ·  /help for commands "

        while True:
            task = await _async_input("›", ui, completer, _toolbar)
            if task is None:  # EOF
                break
            task = task.strip()
            if not task:
                continue
            # Resolve /commands and skills. A fully-handled command (/help, /clear, a
            # fork skill, …) yields no prompt; a plain message or inline skill yields the
            # prompt to run; /clear and /resume replace the loop's history; /exit quits.
            turn = await dispatch_chat_command(task, agent, ui, history)
            if turn.quit:
                break
            if turn.history is not None:
                history = turn.history
            if turn.prompt is None:
                continue
            try:
                with KeyInterrupt(confirm=True) as interrupt:
                    result = await agent.run(
                        turn.prompt, should_cancel=interrupt.is_set, history=history or None
                    )
                history = result.messages
            except LLMTransientError as exc:
                # A network hiccup must not tear down the whole session: report it and
                # keep the loop (and the accumulated context) alive for a retry.
                print(f"[network] {exc}", file=sys.stderr)
                print("The session is still alive — please send your message again.", file=sys.stderr)
                continue
            except RuntimeError as exc:
                print(f"[error] {exc}", file=sys.stderr)
                continue
            if not ui.is_live:
                print(result.answer)

    try:
        asyncio.run(session())
    finally:
        if mcp is not None:
            mcp.close()
    return 0


def sessions_command(args: argparse.Namespace) -> int:
    """List resumable sessions saved for the current project, newest first."""
    root = getattr(args, "session_dir", None) or resolve_session_dir()
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
    print(f"\nResume with: polaris run <task> --resume <id>   (or --continue for the newest)")
    return 0


def _open_store(config: MemoryConfig) -> MemoryStore:
    return MemoryStore(Path(config.dir) / "memory.jsonl")


def dream_command(args: argparse.Namespace) -> int:
    """Run an offline dreaming pass: decay/forget, merge, and synthesise insights."""
    values = _resolve(args)
    config = _memory_config(args)
    store = _open_store(config)
    dreamer = Dreamer(store, config, _make_provider(values), {"model": values["model"]})
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


def health_command(args: argparse.Namespace) -> int:
    """Check the health status of the agent system."""
    print("Agent System Health Check")
    print("=" * 40)
    
    # Check configuration
    try:
        config = resolve_config()
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return 1
    
    # Check provider
    try:
        provider = _make_provider({"model": args.model, "provider": args.provider})
        print(f"✓ Provider initialized: {type(provider).__name__}")
    except Exception as e:
        print(f"✗ Provider error: {e}")
        return 1
    
    # Check tool registry
    try:
        registry = ToolRegistry()
        tool_count = len(registry.list_tools())
        print(f"✓ Tool registry loaded: {tool_count} tools available")
    except Exception as e:
        print(f"✗ Tool registry error: {e}")
        return 1
    
    # Check memory if enabled
    try:
        memory_config = _memory_config(args)
        if memory_config.enabled:
            store = _open_store(memory_config)
            memory_count = len(store.all())
            print(f"✓ Memory store accessible: {memory_count} memories stored")
        else:
            print("○ Memory disabled")
    except Exception as e:
        print(f"✗ Memory error: {e}")
        return 1
    
    print("=" * 40)
    print("All systems operational!")
    return 0


def mcp_command(args: argparse.Namespace) -> int:
    """List the tools exposed by the configured MCP servers (a verification aid)."""
    mcp_config = resolve_mcp_config()
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

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--model", default=None)
        subparser.add_argument(
            "--permission",
            choices=["default", "acceptedits", "plan", "auto", "dontask", "bypass"],
            default=None,
        )
        subparser.add_argument(
            "--sandbox",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Run dangerous commands under the OS sandbox (bwrap/sandbox-exec on "
            "Linux/macOS; no-op on Windows). Overrides [sandbox].enabled.",
        )
        subparser.add_argument(
            "--allow",
            action="append",
            metavar="RULE",
            help="Add an allow rule, e.g. --allow 'run_command(git *)'. Repeatable.",
        )
        subparser.add_argument(
            "--deny",
            action="append",
            metavar="RULE",
            help="Add a deny rule, e.g. --deny 'run_command(rm *)'. Repeatable; deny wins.",
        )
        subparser.add_argument(
            "--ask",
            action="append",
            metavar="RULE",
            help="Add an ask rule (force confirmation), e.g. --ask 'run_command'. Repeatable.",
        )
        subparser.add_argument("--provider", choices=["claude", "fake"], default=None)
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
            choices=["low", "medium", "high", "xhigh", "max"],
            default=None,
            help="output_config.effort depth/cost level (effort-capable models only; "
            "dropped for models that don't support the level).",
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
    run_parser.set_defaults(func=run_command)

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
    memory_parser.set_defaults(func=memory_command)

    mcp_parser = subparsers.add_parser("mcp", help="Inspect configured MCP servers and their tools.")
    mcp_parser.add_argument("action", choices=["list"], help="list: show tools from configured servers.")
    mcp_parser.set_defaults(func=mcp_command)

    health_parser = subparsers.add_parser("health", help="Check the health status of the agent system.")
    add_common(health_parser)
    health_parser.set_defaults(func=health_command)

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

