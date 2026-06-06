from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_core.config import (
    resolve_config,
    resolve_mcp_config,
    resolve_memory_config,
    resolve_output_config,
)
from agent_core.interrupt import KeyInterrupt
from agent_core.memory import Dreamer, MemoryConfig, MemoryStore
from agent_core.models import LLMTransientError
from agent_core.providers import ClaudeProvider, FakeProvider
from agent_core.react import ReActAgent, ReActConfig
from agent_core.tools.registry import ToolRegistry
from agent_core.ui import AgentUI, ConsoleUI, NullUI


def _resolve(args: argparse.Namespace) -> dict:
    return resolve_config(
        {
            "model": args.model,
            "permission": args.permission,
            "provider": args.provider,
        }
    )


def _memory_config(args: argparse.Namespace) -> MemoryConfig:
    # Numeric tunables come from the [memory] toml table; enabled is overridable
    # by AGENT_MEMORY / --memory. (resolve_config above already loaded the .env.)
    return resolve_memory_config(getattr(args, "memory", None))


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
    return ConsoleUI() if interactive else NullUI()


def _start_mcp(registry: ToolRegistry):
    """Connect any configured MCP servers and register their tools, or return ``None``.

    The ``mcp`` SDK is imported lazily and only when ``[mcp.servers.*]`` is non-empty, so
    the core stays dependency-free. The caller owns the returned manager and must
    ``close()`` it.
    """
    mcp_config = resolve_mcp_config()
    if not any(server.enabled for server in mcp_config.servers):
        return None
    from agent_core.mcp import MCPAdapter, MCPClientManager

    manager = MCPClientManager(mcp_config)
    try:
        manager.start()
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MCP servers are configured in agent.toml but the 'mcp' SDK isn't installed. "
            'Install it with: pip install "mcp>=1.0"  (or  pip install -e ".[mcp]").'
        ) from exc
    registry.register_adapter(MCPAdapter(manager))
    return manager


def build_agent(args: argparse.Namespace) -> tuple[ReActAgent, AgentUI, object | None]:
    values = _resolve(args)
    provider = _make_provider(values)
    ui = _make_ui(args)
    config = ReActConfig(
        model=values["model"],
        permission=values["permission"],
        memory=_memory_config(args),
        output=resolve_output_config(),
        thinking_budget=getattr(args, "thinking_budget", None),
        stream=not getattr(args, "no_stream", False),
    )
    registry = ReActAgent.default_registry()
    manager = _start_mcp(registry)
    agent = ReActAgent(provider=provider, config=config, tools=registry, ui=ui)
    return agent, ui, manager


def run_command(args: argparse.Namespace) -> int:
    try:
        agent, ui, mcp = build_agent(args)
    except RuntimeError as exc:
        # E.g. MCP servers configured without the SDK installed, or a connect failure.
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    try:
        with KeyInterrupt() as interrupt:
            result = agent.run(args.task, should_cancel=interrupt.is_set)
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
    return 0


def chat_command(args: argparse.Namespace) -> int:
    try:
        agent, ui, mcp = build_agent(args)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    print("Agent chat. Type /exit to quit. Press Esc during a turn to interrupt.")
    try:
        while True:
            try:
                task = input("> ").strip()
            except EOFError:
                break
            if task in {"/exit", "/quit"}:
                break
            if not task:
                continue
            try:
                with KeyInterrupt() as interrupt:
                    result = agent.run(task, should_cancel=interrupt.is_set)
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
    finally:
        if mcp is not None:
            mcp.close()
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
        report = dreamer.dream(commit=not args.dry_run)
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
        record = store.add(args.value, kind="fact", importance=0.6)
        print(f"Added {record.id}")
        return 0
    if args.action == "forget":
        if not args.value:
            print("[error] `memory forget` needs an id", file=sys.stderr)
            return 1
        if store.delete(args.value):
            print(f"Forgot {args.value}")
            return 0
        print(f"[error] no memory {args.value}", file=sys.stderr)
        return 1
    return 0


def mcp_command(args: argparse.Namespace) -> int:
    """List the tools exposed by the configured MCP servers (a verification aid)."""
    mcp_config = resolve_mcp_config()
    if not any(server.enabled for server in mcp_config.servers):
        print("(no MCP servers configured in agent.toml — see [mcp.servers.*])")
        return 0
    from agent_core.mcp import MCPAdapter, MCPClientManager

    manager = MCPClientManager(mcp_config)
    try:
        manager.start()
    except ModuleNotFoundError:
        print(
            '[error] the "mcp" SDK isn\'t installed. Install it with: '
            'pip install "mcp>=1.0"  (or  pip install -e ".[mcp]").',
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001 - connect/transport failures are user-facing
        print(f"[error] could not connect MCP servers: {exc}", file=sys.stderr)
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
    parser = argparse.ArgumentParser(prog="agent-core")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--model", default=None)
        subparser.add_argument(
            "--permission",
            choices=["default", "acceptedits", "plan", "auto", "dontask"],
            default=None,
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
            "--thinking-budget",
            type=int,
            default=None,
            metavar="TOKENS",
            help="Enable Claude extended thinking with this token budget (claude provider).",
        )

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("task")
    add_common(run_parser)
    run_parser.set_defaults(func=run_command)

    chat_parser = subparsers.add_parser("chat")
    add_common(chat_parser)
    chat_parser.set_defaults(func=chat_command)

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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

