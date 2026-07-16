"""Built-in chat slash-commands for the interactive ``polaris chat`` loop.

A typed ``/command`` is resolved here into a :class:`ChatTurn` telling the loop what to
do next: run a prompt through the agent, replace the conversation history (``/clear``,
``/resume``), or quit. Skills (markdown + programmatic) are dispatched here too — an
inline skill becomes a prompt, a fork skill runs in a sub-agent and prints its result.

Built-in commands map onto subsystems this project actually has (compaction, token
math, sessions, MCP, memory, model config); the reference's TUI/account/cloud commands
have no analogue in this CLI and are intentionally omitted. Handlers print their own
output and are unit-testable via captured stdout. This module imports only lower-level
pieces (never ``cli``) so there is no import cycle.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent_core import tokens
from agent_core.config import resolve_mcp_config
from agent_core.model_catalog import picker_spec_for_provider
from agent_core.model_validation import CLAUDE_PROVIDER, FAKE_PROVIDER, is_model_allowed
from agent_core.models import Message
from agent_core.permissions import PermissionMode, permission_mode_label
from agent_core.sandbox import SandboxRequiredError
from agent_core.skills import (
    SkillContext,
    SkillPromptContext,
    build_skill_prompt,
    fork_preset,
    looks_like_command,
    parse_slash_command,
)
from agent_core.transcript import (
    TranscriptStore,
    build_chain,
    find_session,
    list_sessions,
    load_transcript,
    project_dir,
    session_label,
)
from agent_core.ui import AgentUI

if TYPE_CHECKING:
    from agent_core.react import ReActAgent


@dataclass(slots=True)
class ChatTurn:
    """What the chat loop should do after a line is handled.

    ``prompt`` non-None → run it through ``agent.run`` (a plain message or an inline
    skill). ``history`` non-None → replace the loop's conversation history. ``quit`` →
    leave the chat. A fully-handled command (``/help``, a fork skill, …) returns the
    default: nothing to run, history unchanged, keep going.
    """

    prompt: str | None = None
    history: list[Message] | None = None
    quit: bool = False


Handler = Callable[["ReActAgent", AgentUI, str, "list[Message]"], "Awaitable[ChatTurn]"]


def _estimate_tokens(agent: "ReActAgent", history: list[Message]) -> int:
    """Best current prompt-size estimate, identical to the auto-compact gate.

    Delegates to the agent's own estimator (anchored real usage + rough delta) so
    ``/context`` reports exactly what the gate thresholds against.
    """
    return agent._estimate_tokens(history)


# --- handlers ----------------------------------------------------------------


async def _cmd_help(agent: "ReActAgent", ui: AgentUI, args: str, history: list[Message]) -> ChatTurn:
    print("Commands:")
    for name, (_, summary) in sorted(_COMMAND_HELP.items()):
        print(f"  /{name:<10} {summary}")
    print("  /exit, /quit  Leave the chat.")
    skills = sorted(agent.skills.user_invocable(), key=lambda skill: skill.name)
    if skills:
        print("\nSkills (run as /<name> [args]):")
        for skill in skills:
            tag = " [fork]" if skill.context is SkillContext.FORK else ""
            print(f"  /{skill.name}{tag} — {skill.description or '(no description)'}")
    print("\nTip: type / to open a menu of commands and skills.")
    print("Anything else is sent to the agent as a normal message.")
    return ChatTurn()


async def _cmd_skills(agent: "ReActAgent", ui: AgentUI, args: str, history: list[Message]) -> ChatTurn:
    skills = sorted(agent.skills.user_invocable(), key=lambda skill: skill.name)
    if not skills:
        print("No skills available. Drop a SKILL.md under ./.polaris/skills or ~/.polaris/skills.")
        return ChatTurn()
    print("Available skills:")
    for skill in skills:
        tag = " [fork]" if skill.context is SkillContext.FORK else ""
        print(f"  /{skill.name}{tag} — {skill.description or '(no description)'}")
    return ChatTurn()


async def _cmd_clear(agent: "ReActAgent", ui: AgentUI, args: str, history: list[Message]) -> ChatTurn:
    print("Conversation history cleared.")
    return ChatTurn(history=[])


async def _cmd_status(agent: "ReActAgent", ui: AgentUI, args: str, history: list[Message]) -> ChatTurn:
    mcp_servers = [s for s in resolve_mcp_config().servers if s.enabled]
    thinking = agent.config.thinking_budget
    thinking_label = f"{thinking:,} tokens" if isinstance(thinking, int) and thinking > 0 else "off"
    print("Status:")
    print(f"  model       {agent.config.model}")
    print(f"  effort      {agent.config.effort or '—'}")
    print(f"  thinking    {thinking_label}")
    mode = PermissionMode(agent.config.permission)
    print(f"  permission  {mode.value} ({permission_mode_label(mode)})")
    print(f"  session     {agent.session_id}")
    print(f"  workspace   {agent.session.workspace}")
    print(f"  skills      {len(agent.skills)}  ({len(agent.skills.model_invocable())} model-invocable)")
    print(f"  tools       {len(agent.registry.list())}")
    print(f"  mcp servers {len(mcp_servers)}")
    return ChatTurn()


async def _cmd_context(agent: "ReActAgent", ui: AgentUI, args: str, history: list[Message]) -> ChatTurn:
    model = agent.config.model
    est = _estimate_tokens(agent, history)
    window = tokens.context_window_for_model(model)
    threshold = tokens.auto_compact_threshold(model)
    pct_window = (est / window * 100) if window else 0.0
    pct_compact = (est / threshold * 100) if threshold else 0.0
    print(f"Context (model {model}):")
    print(f"  estimated prompt tokens  {est:,}")
    print(f"  context window           {window:,}  ({pct_window:.1f}% used)")
    print(f"  auto-compact threshold   {threshold:,}  ({pct_compact:.1f}% of the way there)")
    return ChatTurn()


async def _cmd_cost(agent: "ReActAgent", ui: AgentUI, args: str, history: list[Message]) -> ChatTurn:
    started = getattr(agent, "_session_started", None)
    elapsed = (time.monotonic() - started) if started else 0.0
    inp = getattr(agent, "_session_input_tokens", 0)
    out = getattr(agent, "_session_output_tokens", 0)
    print("Session usage:")
    print(f"  input tokens   {inp:,}")
    print(f"  output tokens  {out:,}")
    print(f"  total tokens   {inp + out:,}")
    print(f"  duration       {elapsed:.0f}s")
    print("  (token usage, not billed cost — no pricing table is configured)")
    return ChatTurn()


async def _cmd_compact(agent: "ReActAgent", ui: AgentUI, args: str, history: list[Message]) -> ChatTurn:
    if not history:
        print("Nothing to compact yet.")
        return ChatTurn()
    compacted, saved = await agent.compact_now(history)
    if saved <= 0:
        print("Nothing to compact (history already compact).")
        return ChatTurn()
    print(f"Compacted: {len(history)} → {len(compacted)} messages, ~{saved:,} chars saved.")
    return ChatTurn(history=compacted)


async def _cmd_model(agent: "ReActAgent", ui: AgentUI, args: str, history: list[Message]) -> ChatTurn:
    target = args.strip()
    if not target:
        spec = picker_spec_for_provider(agent.config.provider)
        if spec is None:
            print(
                f"Current provider/model: {agent.config.provider} / {agent.config.model}  ·  "
                f"effort: {agent.config.effort or '—'}"
            )
            print("Switch with: /model <non-empty model id>; provider stays unchanged.")
            return ChatTurn()
        # Bare /model opens the interactive provider-specific model + effort picker.
        chosen = await ui.pick_model(agent.config.model, agent.config.effort, spec)
        if chosen is None:
            # Non-interactive (or cancelled): fall back to a plain listing.
            print(
                f"Current provider/model: {agent.config.provider} / {agent.config.model}  ·  "
                f"effort: {agent.config.effort or '—'}"
            )
            print(f"Known model families: {spec.known_families}")
            print("Switch with: /model <name>  (or run /model in a terminal to pick interactively)")
            return ChatTurn()
        model, effort = chosen
        agent.config.model = model
        agent.config.effort = effort
        tail = f"  ·  effort: {effort}" if effort is not None else "  ·  (no effort levels)"
        print(f"Model switched to {model}{tail} (takes effect on your next message).")
        return ChatTurn()
    if not is_model_allowed(agent.config.provider, target):
        if agent.config.provider == CLAUDE_PROVIDER:
            print(f"Unsupported model {target!r}. Name a known Claude family (e.g. claude-sonnet-4-6).")
        elif agent.config.provider == FAKE_PROVIDER:
            print(f"Unsupported model {target!r}.")
        else:
            print(f"Unsupported model {target!r} for provider {agent.config.provider!r}.")
        return ChatTurn()
    agent.config.model = target
    print(f"Model switched to {target} (takes effect on your next message).")
    return ChatTurn()


def _print_permission_modes(current: PermissionMode) -> None:
    print(f"Current permission mode: {current.value} ({permission_mode_label(current)})")
    print("Available modes:")
    for mode in PermissionMode:
        marker = "*" if mode == current else " "
        print(f"  {marker} {mode.value:<12} {permission_mode_label(mode)}")
    print("Switch with: /permissions <mode>  (or run /permissions in a terminal to pick)")


async def _cmd_permissions(
    agent: "ReActAgent", ui: AgentUI, args: str, history: list[Message]
) -> ChatTurn:
    current = PermissionMode(agent.config.permission)
    if args.strip().casefold().startswith("rules"):
        await _handle_permission_rules(agent, ui, args.strip()[5:].strip())
        return ChatTurn()
    raw_target = args.strip().lower()
    if raw_target:
        try:
            target = PermissionMode(raw_target)
        except ValueError:
            valid = ", ".join(mode.value for mode in PermissionMode)
            print(f"Unknown permission mode {raw_target!r}. Choose one of: {valid}.")
            return ChatTurn()
    else:
        unavailable = set(agent.permissions.managed_policy.forbidden_modes)
        if (
            agent.permissions.managed_policy.require_sandbox_for_unattended
            and not agent.sandbox.is_enabled()
        ):
            unavailable.update(
                {PermissionMode.AUTO, PermissionMode.DONTASK, PermissionMode.BYPASS}
            )
        forbidden = tuple(mode.value for mode in unavailable)
        try:
            selected = await ui.pick_permission_mode(current.value, forbidden)
        except TypeError:
            # Compatibility for embedded UIs implementing the original one-argument hook.
            selected = await ui.pick_permission_mode(current.value)
        if selected is None:
            _print_permission_modes(current)
            return ChatTurn()
        target = PermissionMode(selected)

    if target == current:
        print(f"Permission mode unchanged: {target.value} ({permission_mode_label(target)}).")
        return ChatTurn()
    try:
        agent.set_permission_mode(target, source="slash")
    except SandboxRequiredError as exc:
        print(f"[permission] {exc}")
        return ChatTurn()
    print(f"Permission mode switched to {target.value} ({permission_mode_label(target)}).")
    return ChatTurn()


async def _handle_permission_rules(agent: "ReActAgent", ui: AgentUI, args: str) -> None:
    """Inspect rules or add an explicit least-privilege allow rule."""
    from agent_core.permission_store import persist_allow_rule
    from agent_core.permission_types import PermissionDestination

    if not args:
        rules = agent.permissions.rules
        deny_keys = {(rule.tool_name, rule.content) for rule in rules.deny}
        ask_keys = {(rule.tool_name, rule.content) for rule in rules.ask}
        print("Effective permission rules (DENY > ASK > ALLOW):")
        for behavior, entries in (("deny", rules.deny), ("ask", rules.ask), ("allow", rules.allow)):
            for rule in entries:
                raw = rule.tool_name + (f"({rule.content})" if rule.content is not None else "")
                shadow = ""
                if behavior == "allow" and (rule.tool_name, rule.content) in deny_keys | ask_keys:
                    shadow = " [shadowed]"
                elif behavior == "ask" and (rule.tool_name, rule.content) in deny_keys:
                    shadow = " [shadowed]"
                print(f"  {behavior:<5} {rule.source.value:<8} {raw}{shadow}")
        if rules.is_empty:
            print("  (none)")
        print("Add: /permissions rules add <session|local|project|user> <allow-rule>")
        return
    parts = args.split(maxsplit=2)
    if len(parts) != 3 or parts[0].casefold() != "add":
        print("Usage: /permissions rules [add <session|local|project|user> <allow-rule>]")
        return
    try:
        destination = PermissionDestination(parts[1].casefold())
    except ValueError:
        print(f"Unknown permission destination: {parts[1]!r}.")
        return
    if agent.permissions.managed_policy.allow_managed_rules_only:
        print("Managed policy ignores non-managed allow rules.")
        return
    if (
        destination is not PermissionDestination.SESSION
        and agent.permissions.managed_policy.disable_persistent_grants
    ):
        print("Managed policy disables persistent permission grants.")
        return
    rule = parts[2].strip()
    try:
        from agent_core.permission_rules import parse_rule

        if parse_rule(rule) is None:
            raise ValueError("malformed permission rule")
        if destination is not PermissionDestination.SESSION:
            confirmed = await asyncio.to_thread(
                ui.confirm_action,
                f"Persist allow rule {rule!r} to {destination.value} configuration?",
            )
            if not confirmed:
                print("Persistent permission rule was not confirmed.")
                return
            persist_allow_rule(rule, destination, agent.session.workspace)
        agent.permissions.add_session_rule(rule)
    except (OSError, ValueError) as exc:
        print(f"Permission rule was not added: {exc}")
        return
    print(f"Added allow rule to {destination.value}: {rule}")


async def _cmd_mcp(agent: "ReActAgent", ui: AgentUI, args: str, history: list[Message]) -> ChatTurn:
    servers = resolve_mcp_config().servers
    if not servers:
        print("No MCP servers configured (see [mcp.servers.*] in agent.toml).")
        return ChatTurn()
    print("MCP servers:")
    for server in servers:
        state = "enabled" if server.enabled else "disabled"
        where = server.command or server.url or "?"
        print(f"  {server.name}  [{state}]  {server.transport}: {where}")
    return ChatTurn()


async def _cmd_memory(agent: "ReActAgent", ui: AgentUI, args: str, history: list[Message]) -> ChatTurn:
    store = getattr(agent, "memory_store", None)
    if store is None:
        print("Memory is disabled (enable it in [memory] or with --memory).")
        return ChatTurn()
    records = store.all()
    if not records:
        print("No memories stored yet.")
        return ChatTurn()
    print(f"Stored memories ({len(records)}):")
    for record in records[-10:]:
        summary = record.content.strip().splitlines()[0] if record.content.strip() else "(empty)"
        print(f"  [{record.kind}] {summary[:100]}")
    return ChatTurn()


async def _cmd_resume(agent: "ReActAgent", ui: AgentUI, args: str, history: list[Message]) -> ChatTurn:
    session_dir = agent.config.session_dir
    if not session_dir:
        print("Session persistence is disabled; cannot resume.")
        return ChatTurn()
    workspace = agent.session.workspace
    target = args.strip()
    if not target:
        infos = list_sessions(project_dir(session_dir, workspace))
        if not infos:
            print(f"No saved sessions for {workspace}.")
            return ChatTurn()
        print("Resumable sessions (newest first):")
        for info in infos[:10]:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(info.modified))
            branch = f" [{info.git_branch}]" if info.git_branch else ""
            print(f"  {session_label(info)[:70]}")
            print(f"      {when} · {info.message_count} msgs{branch} · id {info.session_id[:8]}")
        print("Resume with: /resume <id>  (or type /resume and pick from the menu)")
        return ChatTurn()

    path = find_session(session_dir, workspace, target)
    if path is None:
        print(f"No session found with id {target!r}.")
        return ChatTurn()
    loaded = load_transcript(path)
    resumed = build_chain(loaded)
    # Repoint the agent so subsequent turns persist to the resumed session.
    agent.session_id = loaded.session_id
    agent.session.session_id = loaded.session_id
    agent.transcript = TranscriptStore(session_dir, workspace, loaded.session_id)
    print(f"Resumed session {loaded.session_id} ({len(resumed)} messages).")
    return ChatTurn(history=resumed)


# (handler, one-line summary) — drives both dispatch and /help. Aliases share a handler.
_COMMANDS: dict[str, Handler] = {
    "help": _cmd_help,
    "skills": _cmd_skills,
    "clear": _cmd_clear,
    "reset": _cmd_clear,
    "new": _cmd_clear,
    "status": _cmd_status,
    "context": _cmd_context,
    "cost": _cmd_cost,
    "compact": _cmd_compact,
    "model": _cmd_model,
    "permissions": _cmd_permissions,
    "mcp": _cmd_mcp,
    "memory": _cmd_memory,
    "resume": _cmd_resume,
    "continue": _cmd_resume,
}

# Canonical commands shown by /help (aliases collapsed).
_COMMAND_HELP: dict[str, tuple[Handler, str]] = {
    "help": (_cmd_help, "Show this help."),
    "skills": (_cmd_skills, "List available skills."),
    "clear": (_cmd_clear, "Clear conversation history (aliases /reset /new)."),
    "status": (_cmd_status, "Show model, session, skill/tool counts."),
    "context": (_cmd_context, "Show context-window usage."),
    "cost": (_cmd_cost, "Show session token usage and duration."),
    "compact": (_cmd_compact, "Compact the conversation now."),
    "model": (_cmd_model, "Show or switch the model."),
    "permissions": (_cmd_permissions, "Show or switch the permission mode."),
    "mcp": (_cmd_mcp, "List configured MCP servers."),
    "memory": (_cmd_memory, "List stored memories."),
    "resume": (_cmd_resume, "List or resume a saved session (alias /continue)."),
}


async def dispatch(task: str, agent: "ReActAgent", ui: AgentUI, history: list[Message]) -> ChatTurn:
    """Resolve a chat line into a :class:`ChatTurn`.

    Order: explicit ``/exit``·``/quit`` → not-a-command (verbatim) → built-in command →
    skill (inline returns a prompt; fork runs in a sub-agent) → unknown command.
    """
    if task in {"/exit", "/quit"}:
        return ChatTurn(quit=True)

    parsed = parse_slash_command(task)
    if parsed is None or not looks_like_command(parsed.name):
        return ChatTurn(prompt=task)  # plain message → send verbatim

    handler = _COMMANDS.get(parsed.name.lower())
    if handler is not None:
        return await handler(agent, ui, parsed.args, history)

    skill = agent.skills.get(parsed.name)
    if skill is None or not skill.user_invocable:
        print(f"Unknown command: /{parsed.name}. Try /skills to list skills or /help.")
        return ChatTurn()

    ctx = SkillPromptContext.from_session(agent.session, transcript=agent.transcript)
    prompt = await build_skill_prompt(skill, parsed.args, ctx)
    if skill.context is not SkillContext.FORK:
        return ChatTurn(prompt=prompt)  # inline → run through agent.run (enters history)

    factory = agent.session.subagent_factory
    if factory is None:
        print("[error] fork skills need sub-agents, which are unavailable here.")
        return ChatTurn()
    try:
        answer = await factory(prompt, fork_preset(skill.allowed_tools), skill.model)
    except Exception as exc:  # noqa: BLE001 - a skill failure must not tear down the chat
        print(f"[error] skill /{skill.name} failed: {type(exc).__name__}: {exc}")
        return ChatTurn()
    if not ui.is_live:
        print(answer)
    return ChatTurn()
