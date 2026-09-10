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
import json
import shlex
import time
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from agent_core import tokens
from agent_core.config import resolve_mcp_config
from agent_core.local_config import LocalConfigError, update_local_table
from agent_core.model_catalog import picker_spec_for_provider
from agent_core.model_validation import CLAUDE_PROVIDER, FAKE_PROVIDER, is_model_allowed
from agent_core.models import Message
from agent_core.permissions import PermissionMode, permission_mode_label
from agent_core.sandbox import SandboxRequiredError
from agent_core.skills import (
    Skill,
    SkillContext,
    SkillPromptContext,
    build_skill_prompt,
    fork_preset,
    looks_like_command,
    parse_slash_command,
)
from agent_core.transcript import (
    LoadedTranscript,
    fork_chain,
    list_sessions,
    load_transcript,
    locate_session,
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


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """Execution metadata for an interactive slash command."""

    handler: Handler | None
    summary: str
    immediate: bool = False
    canonical: str | None = None


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
    skills: list[Skill] = sorted(
        agent.skills.user_invocable(), key=lambda skill: skill.name
    )
    if skills:
        print("\nSkills (run as /<name> [args]):")
        for skill in skills:
            tag = " [fork]" if skill.context is SkillContext.FORK else ""
            print(f"  /{skill.name}{tag} — {skill.description or '(no description)'}")
    print("\nTip: type / to open a menu of commands and skills.")
    print("Anything else is sent to the agent as a normal message.")
    return ChatTurn()


async def _cmd_skills(agent: "ReActAgent", ui: AgentUI, args: str, history: list[Message]) -> ChatTurn:
    skills: list[Skill] = sorted(
        agent.skills.user_invocable(), key=lambda skill: skill.name
    )
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
    # Connected servers come from the live managers, not a re-parse of the config
    # file (which may not be the file this agent was started with, e.g. --config).
    mcp_servers: set[str] = set()
    for manager in (getattr(agent.session, "mcp_manager", None), getattr(agent, "_plugin_mcp_manager", None)):
        tools = getattr(manager, "tools", None)
        if callable(tools):
            mcp_servers.update(server.name for server, _tool in tools())
    thinking = agent.config.thinking_budget
    thinking_label = f"{thinking:,} tokens" if isinstance(thinking, int) and thinking > 0 else "off"
    print("Status:")
    print(f"  model       {agent.config.model}")
    print(f"  effort      {agent.config.effort or '—'}")
    print(f"  fast mode   {'on' if getattr(agent, 'fast_mode', False) else 'off'}")
    print(f"  thinking    {thinking_label}")
    mode = PermissionMode(agent.config.permission)
    print(f"  permission  {mode.value} ({permission_mode_label(mode)})")
    policy = agent.permissions.managed_policy
    if (
        policy.policy_digest
        or policy.forbidden_modes
        or policy.require_sandbox_for_unattended
        or policy.allow_managed_rules_only
        or policy.disable_persistent_grants
    ):
        forbidden = ", ".join(sorted(item.value for item in policy.forbidden_modes)) or "—"
        print(f"  managed     digest {policy.policy_digest[:12] or '—'} · forbidden modes: {forbidden}")
    else:
        print("  managed     off")
    print(f"  session     {agent.session_id}")
    print(f"  workspace   {agent.session.workspace}")
    print(f"  skills      {len(agent.skills)}  ({len(agent.skills.model_invocable())} model-invocable)")
    print(f"  tools       {len(agent.registry.list())}")
    print(f"  mcp servers {len(mcp_servers)}")
    return ChatTurn()


async def _cmd_context(agent: "ReActAgent", ui: AgentUI, args: str, history: list[Message]) -> ChatTurn:
    model = agent.config.model
    est = _estimate_tokens(agent, history)
    cfg = agent.config.compression
    window = tokens.context_window_for_model(model)
    if cfg.context_window_tokens is not None and cfg.context_window_tokens > 0:
        window = min(window, cfg.context_window_tokens)
    # Same inputs as the live auto-compact gate (compression.auto_compact), so the
    # displayed threshold is the one the run actually trips on.
    threshold = tokens.auto_compact_threshold(
        model,
        context_window_override=cfg.context_window_tokens,
        buffer_tokens=cfg.autocompact_buffer_tokens,
        reserved_output_tokens=cfg.reserved_output_tokens_for_summary,
        pct_override=cfg.autocompact_pct_override,
    )
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
        if getattr(agent, "fast_mode", False) and model != "claude-opus-4-6":
            agent.fast_mode = False
            print("Fast mode disabled because the selected model is not Claude Opus 4.6.")
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
    if getattr(agent, "fast_mode", False) and target != "claude-opus-4-6":
        agent.fast_mode = False
        print("Fast mode disabled because the selected model is not Claude Opus 4.6.")
    print(f"Model switched to {target} (takes effect on your next message).")
    return ChatTurn()


async def _cmd_capabilities(
    agent: "ReActAgent", ui: AgentUI, args: str, history: list[Message]
) -> ChatTurn:
    import json

    query = args.strip()
    if query:
        print(json.dumps(agent.capability_manager.search(query), ensure_ascii=False, indent=2))
        return ChatTurn()
    records, digest = agent.capability_manager.snapshot()
    counts: dict[str, int] = {}
    for record in records:
        counts[record.kind] = counts.get(record.kind, 0) + 1
    print(
        f"Capability mode: {agent.config.capabilities.mode}; catalog {digest[:12]}; "
        f"skills={counts.get('skill', 0)}, mcp={counts.get('mcp', 0)}, "
        f"plugins={counts.get('plugin', 0)}"
    )
    print("Search with: /capabilities <query>")
    return ChatTurn()


async def _cmd_effort(
    agent: "ReActAgent", ui: AgentUI, args: str, history: list[Message]
) -> ChatTurn:
    spec = picker_spec_for_provider(agent.config.provider)
    available = spec.efforts_fn(agent.config.model) if spec is not None else ()
    target = args.strip().casefold()
    if not target:
        print(f"Current effort: {agent.config.effort or 'auto'}")
        print("Available: auto" + (", " + ", ".join(available) if available else ""))
        return ChatTurn()
    if target == "auto":
        agent.config.effort = None
        print("Effort set to auto for the next provider request.")
        return ChatTurn()
    if target not in available:
        choices = "auto" + (", " + ", ".join(available) if available else "")
        print(
            f"Effort {target!r} is unsupported by {agent.config.model}. "
            f"Choose one of: {choices}."
        )
        return ChatTurn()
    agent.config.effort = target
    print(f"Effort set to {target} for the next provider request.")
    return ChatTurn()


async def _cmd_fast(
    agent: "ReActAgent", ui: AgentUI, args: str, history: list[Message]
) -> ChatTurn:
    raw = args.strip().casefold()
    current = bool(getattr(agent, "fast_mode", False))
    if raw in {"off", "disable", "disabled"} or (not raw and current):
        agent.fast_mode = False
        print("Fast mode off (session only).")
        return ChatTurn()
    if raw not in {"", "on", "enable", "enabled"}:
        print("Usage: /fast [on|off]")
        return ChatTurn()
    if agent.config.provider != CLAUDE_PROVIDER:
        print("Fast mode is available only with the Claude provider.")
        return ChatTurn()
    if agent.config.model != "claude-opus-4-6":
        agent.config.model = "claude-opus-4-6"
        print("Model switched to claude-opus-4-6 for fast mode.")
    agent.fast_mode = True
    print("Fast mode on (session only; takes effect on the next provider request).")
    return ChatTurn()


async def _cmd_rename(
    agent: "ReActAgent", ui: AgentUI, args: str, history: list[Message]
) -> ChatTurn:
    title = " ".join(args.strip().split())
    if not title:
        useful = [message for message in history if message.role in {"user", "assistant"}]
        if not useful:
            print("Cannot generate a title before the session has conversation history.")
            return ChatTurn()
        prompt = Message(
            "user",
            "Create a concise title (at most 8 words) for this conversation. "
            "Return only the title, without quotes.",
        )
        try:
            result = await agent.provider.complete(
                [*useful[-12:], prompt],
                [],
                replace(
                    agent._provider_config(),
                    max_tokens=48,
                    thinking_budget=None,
                    stream=False,
                    speed=None,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - title generation is best effort
            print(f"Session title unchanged: {type(exc).__name__}: {exc}")
            return ChatTurn()
        title = " ".join(result.content.strip().strip("\"'").split())
        if not title:
            print("Session title unchanged: the model returned an empty title.")
            return ChatTurn()
    title = title[:100]
    if agent.transcript is None:
        print("Session title unchanged: transcript persistence is disabled.")
        return ChatTurn()
    await agent.transcript.append_meta("custom-title", {"title": title})
    agent.session_title = title
    print(f"Session renamed to: {title}")
    return ChatTurn()


async def _cmd_sandbox(
    agent: "ReActAgent", ui: AgentUI, args: str, history: list[Message]
) -> ChatTurn:
    raw = args.strip()
    if not raw:
        config = agent.sandbox.config
        print("Sandbox:")
        print(f"  requested   {agent.sandbox.requested}")
        print(f"  effective   {agent.sandbox.is_enabled()}")
        print(f"  prepared    {agent.sandbox.prepared}")
        print(f"  state       {agent.sandbox.preparation_state.value}")
        print(f"  backend     {agent.sandbox.backend_name}")
        print(f"  runtime     {agent.sandbox.runtime or '(host native)'}")
        print(f"  image       {agent.sandbox.image or '(not applicable)'}")
        manifest = agent.sandbox.guest_manifest
        print(
            "  guest       "
            + (
                f"{manifest.guest_os}/{manifest.architecture} protocol={manifest.protocol_version}"
                if manifest is not None else "(none)"
            )
        )
        print(f"  capabilities {', '.join(sorted(agent.sandbox.capabilities)) or '(none)'}")
        print(f"  failure      {agent.sandbox.unavailable_reason() or '(none)'}")
        print(f"  auto-allow  {config.auto_allow_command_if_sandboxed}")
        print("Change with: /sandbox <auto-allow|regular>; use --no-sandbox at startup to disable")
        return ChatTurn()
    if getattr(agent, "_sandbox_cli_locked", False):
        print("Sandbox settings are locked by CLI flags for this session.")
        return ChatTurn()

    action, _, value = raw.partition(" ")
    action = action.casefold()
    old_config = deepcopy(agent.sandbox.config)
    new_config = deepcopy(old_config)
    if action == "auto-allow":
        new_config.enabled = True
        new_config.auto_allow_command_if_sandboxed = True
    elif action == "regular":
        new_config.enabled = True
        new_config.auto_allow_command_if_sandboxed = False
    elif action == "disabled":
        print("Sandbox can only be disabled for a whole session with explicit --no-sandbox.")
        return ChatTurn()
    elif action == "exclude":
        print("Per-command sandbox exclusions are not supported.")
        return ChatTurn()
    else:
        print("Usage: /sandbox <auto-allow|regular>")
        return ChatTurn()

    try:
        from agent_core.sandbox import SandboxManager

        def prepare_replacement() -> SandboxManager:
            candidate = SandboxManager(new_config, workspace=agent.session.workspace)
            candidate.prepare()
            return candidate

        replacement = await asyncio.to_thread(prepare_replacement)
        await asyncio.to_thread(
            update_local_table,
            agent.session.workspace,
            "sandbox",
            {
                "enabled": new_config.enabled,
                "auto_allow_command_if_sandboxed": (
                    new_config.auto_allow_command_if_sandboxed
                ),
                "excluded_commands": new_config.excluded_commands,
            },
        )
    except (LocalConfigError, OSError, RuntimeError) as exc:
        replacement_candidate = locals().get("replacement")
        if replacement_candidate is not None:
            replacement_candidate.teardown()
        print(f"Sandbox settings unchanged: {exc}")
        return ChatTurn()
    old_sandbox = agent.sandbox
    agent.sandbox = replacement
    agent.config.sandbox = new_config
    agent.permissions.sandbox = replacement
    agent.registry.bind_runtime(
        session=agent.session,
        sandbox=replacement,
        web_policy=agent.config.web,
        unattended=agent._is_unattended_mode(),
    )
    worktrees = getattr(agent.session, "worktree_manager", None)
    if worktrees is not None and hasattr(worktrees, "sandbox"):
        worktrees.sandbox = replacement
    agent._retired_sandboxes = [
        *getattr(agent, "_retired_sandboxes", []),
        old_sandbox,
    ]
    print(
        f"Sandbox updated: enabled={new_config.enabled}, "
        f"backend={agent.sandbox.backend_name}, "
        f"auto-allow={new_config.auto_allow_command_if_sandboxed}."
    )
    return ChatTurn()


_PROMPT_ONLY_PLUGIN_COMPONENTS = frozenset(
    {"skills", "agents", "output-styles", "themes", "user-config"}
)


async def _gather_plugin_grants(
    manager: Any, record: Any, ui: AgentUI
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], bool, bool] | None:
    """Interactively grant a plugin's executable capabilities, category by category.

    Returns ``(components, allowed_hooks, allowed_permission_hooks, env_access,
    network_unrestricted)``, or ``None`` when the plugin declares nothing beyond
    prompt-only components (legacy enable path applies).  ``confirm_action`` fails
    closed on non-interactive sinks, so headless runs grant no executable capability.
    """
    from agent_core.plugins import validate_plugin
    from agent_core.plugins.components import (
        declared_hook_ids,
        declared_permission_hooks,
        declared_remote_mcp,
        declared_sensitive_env_refs,
    )

    plugin_id = record.plugin_id
    root = Path(record.path).resolve()
    manifest = (
        dict(record.manifest) if isinstance(record.manifest, dict) else validate_plugin(root)
    )
    declared = set(record.components)
    permission_hooks = declared_permission_hooks(root, manifest, plugin_id)
    env_refs = declared_sensitive_env_refs(root, manifest)
    remote_mcp = declared_remote_mcp(root, manifest)
    executable = sorted(declared - _PROMPT_ONLY_PLUGIN_COMPONENTS)
    if not executable and not permission_hooks and not env_refs and not remote_mcp:
        return None

    granted: list[str] = []
    denied: list[str] = []
    for category in executable:
        if await asyncio.to_thread(
            ui.confirm_action,
            f"Enable {category} components from {plugin_id}? (executable capability)",
        ):
            granted.append(category)
        else:
            denied.append(category)
    allowed_hooks: tuple[str, ...] = ()
    if "hooks" in granted:
        # Grant every declared hook id; PermissionRequest hooks still require the
        # separate grant below and never load through this allowlist.
        allowed_hooks = declared_hook_ids(root, manifest, plugin_id)
    granted_permission_hooks: tuple[str, ...] = ()
    if permission_hooks:
        if await asyncio.to_thread(
            ui.confirm_action,
            f"Plugin {plugin_id} declares PermissionRequest hooks that can auto-approve "
            f"permission prompts ({', '.join(permission_hooks)}). Grant?",
        ):
            granted_permission_hooks = permission_hooks
        else:
            denied.append("permission-request-hooks")
    env_access = False
    if env_refs:
        env_access = await asyncio.to_thread(
            ui.confirm_action,
            f"Plugin {plugin_id} references sensitive host variables "
            f"({', '.join(env_refs)}). Grant env access?",
        )
        if not env_access:
            denied.append("env-access")
    network_unrestricted = False
    if remote_mcp:
        network_unrestricted = await asyncio.to_thread(
            ui.confirm_action,
            f"Plugin {plugin_id} declares remote MCP servers ({', '.join(remote_mcp)}); "
            "they default to public-only (public HTTPS, no credentials). "
            "Grant unrestricted network instead?",
        )
        if not network_unrestricted:
            denied.append("network-unrestricted")

    components = tuple([*sorted(declared & _PROMPT_ONLY_PLUGIN_COMPONENTS), *granted])
    manager.audit.write(
        "capability_granted",
        {
            "plugin_id": plugin_id,
            "components": list(components),
            "permission_hooks": list(granted_permission_hooks),
            "env_access": env_access,
            "network_unrestricted": network_unrestricted,
        },
    )
    if denied:
        manager.audit.write(
            "capability_denied", {"plugin_id": plugin_id, "components": denied}
        )
    return components, allowed_hooks, granted_permission_hooks, env_access, network_unrestricted


async def _cmd_plugin(
    agent: "ReActAgent", ui: AgentUI, args: str, history: list[Message]
) -> ChatTurn:
    from agent_core.plugins import PluginError, PluginManager, validate_plugin

    try:
        parts = [part.strip("\"'") for part in shlex.split(args, posix=False)]
    except ValueError as exc:
        print(f"Invalid /plugin arguments: {exc}")
        return ChatTurn()
    manager = PluginManager(agent.session.workspace)
    action = parts[0].casefold() if parts else "manage"
    rest = parts[1:]
    try:
        if action == "status":
            plugin_state = manager.state_status()
            print(
                f"Plugin schema={plugin_state.schema_version}; reset_required={plugin_state.reset_required}; "
                f"installed={len(manager.records())}; enabled={len(manager.enabled_ids())}."
            )
            for target in plugin_state.targets:
                print(f"  legacy: {target}")
        elif action == "errors":
            path = manager.audit.path
            if not path.is_file():
                print("No capability errors recorded.")
            else:
                failures = []
                for line in reversed(path.read_text(encoding="utf-8").splitlines()):
                    try:
                        item = json.loads(line)
                    except ValueError:
                        continue
                    detail = item.get("detail", {})
                    if item.get("event") == "activation_failed" or (
                        item.get("event") == "catalog_sync" and detail.get("result") == "failed"
                    ):
                        failures.append(item)
                    if len(failures) >= 20:
                        break
                print(
                    "No capability errors recorded."
                    if not failures else json.dumps(failures, ensure_ascii=False, indent=2)
                )
        elif action == "reset":
            preview = manager.reset_preview()
            message = "Clear managed plugin state?\n" + "\n".join(f"  - {item}" for item in preview)
            confirmed = await asyncio.to_thread(ui.confirm_action, message)
            if not confirmed:
                print("Plugin reset was not confirmed.")
                return ChatTurn()
            removed = await asyncio.to_thread(manager.reset_state)
            from agent_core.plugins import reload_plugins

            await asyncio.to_thread(reload_plugins, agent)
            print(f"Plugin state reset ({len(removed)} managed targets).")
        elif action in {"output-style", "theme"}:
            values = (
                agent.plugin_output_styles if action == "output-style" else agent.plugin_themes
            )
            if not rest:
                print(
                    f"Available plugin {action}s: "
                    + (", ".join(sorted(values)) if values else "(none)")
                )
            else:
                name = "" if rest[0].casefold() in {"off", "none", "default"} else rest[0]
                agent.select_plugin_presentation(action, name)
                print(f"Plugin {action} set to {name or 'default'}.")
        elif action in {"manage", "list"}:
            enabled = set(manager.enabled_ids())
            records = manager.records()
            if not records:
                print("No plugins installed.")
            else:
                print("Installed plugins:")
                for plugin_id, record in sorted(records.items()):
                    marker = "enabled" if plugin_id in enabled else "disabled"
                    print(
                        f"  {plugin_id} [{marker}]"
                        + (f" v{record.version}" if record.version else "")
                    )
            print("Changes take effect after /reload-plugins.")
        elif action == "install":
            if not rest:
                raise PluginError("usage: /plugin install <path|name> [marketplace]")
            identifier = rest[0]
            if "@" in identifier and len(rest) == 1 and not Path(identifier).exists():
                plugin_name, marketplace = identifier.rsplit("@", 1)
            else:
                plugin_name = identifier
                marketplace = rest[1] if len(rest) > 1 else "local"
            if marketplace != "local":
                plan = await asyncio.to_thread(
                    manager.dependency_plan, plugin_name, marketplace
                )
            else:
                plan = []
            if len(plan) > 1:
                summary = ", ".join(str(item["plugin_id"]) for item in plan[:-1])
                confirmed = await asyncio.to_thread(
                    ui.confirm_action,
                    f"Install dependency graph for {plugin_name}@{marketplace}: {summary}?",
                )
                if not confirmed:
                    print("Plugin dependency installation was not confirmed.")
                    return ChatTurn()
                record = await asyncio.to_thread(manager.install_dependency_plan, plan)
            else:
                record = await asyncio.to_thread(manager.install, plugin_name, marketplace)
            default_enabled = not (
                isinstance(record.manifest, dict)
                and record.manifest.get("defaultEnabled") is False
            )
            grants = None
            if default_enabled:
                grants = await _gather_plugin_grants(manager, record, ui)
            if not default_enabled:
                state = "installed but disabled"
            elif grants is None:
                await asyncio.to_thread(
                    manager.set_enabled,
                    record.plugin_id,
                    True,
                    scope="project",
                )
                state = "enabled for this project"
            else:
                components, allowed_hooks, permission_hooks, env_access, network = grants
                await asyncio.to_thread(
                    manager.set_enabled, record.plugin_id, True, scope="project"
                )
                await asyncio.to_thread(
                    manager.set_activation,
                    record.plugin_id,
                    components,
                    allowed_hooks=allowed_hooks,
                    allowed_permission_hooks=permission_hooks,
                    env_access=env_access,
                    network_unrestricted=network,
                )
                state = (
                    "enabled for this project with components: "
                    + (", ".join(components) if components else "(none)")
                )
            print(f"Plugin {record.plugin_id} {state}; run /reload-plugins to apply.")
        elif action in {"uninstall", "remove"}:
            if not rest:
                raise PluginError("usage: /plugin uninstall <plugin@marketplace>")
            await asyncio.to_thread(manager.uninstall, rest[0])
            print(f"Uninstalled {rest[0]}; run /reload-plugins to apply.")
        elif action in {"enable", "disable"}:
            if not rest:
                raise PluginError(f"usage: /plugin {action} <plugin@marketplace> [project|user]")
            scope = rest[1].casefold() if len(rest) > 1 else "project"
            grants = None
            if action == "enable":
                target_record = manager.records().get(rest[0])
                if target_record is not None:
                    grants = await _gather_plugin_grants(manager, target_record, ui)
            await asyncio.to_thread(
                manager.set_enabled,
                rest[0],
                action == "enable",
                scope=scope,
            )
            if grants is not None:
                components, allowed_hooks, permission_hooks, env_access, network = grants
                await asyncio.to_thread(
                    manager.set_activation,
                    rest[0],
                    components,
                    allowed_hooks=allowed_hooks,
                    allowed_permission_hooks=permission_hooks,
                    env_access=env_access,
                    network_unrestricted=network,
                )
                print(
                    f"{rest[0]} enabled with components: "
                    + (", ".join(components) if components else "(none)")
                    + "; run /reload-plugins to apply."
                )
            else:
                print(f"{rest[0]} {action}d for {scope}; run /reload-plugins to apply.")
        elif action == "update":
            if not rest:
                raise PluginError("usage: /plugin update <plugin@marketplace>")
            record = await asyncio.to_thread(manager.update, rest[0])
            print(f"Updated {record.plugin_id} to {record.version or record.commit or record.integrity[:12]}.")
        elif action == "details":
            if not rest:
                raise PluginError("usage: /plugin details <plugin@marketplace>")
            print(json.dumps(manager.details(rest[0]), ensure_ascii=False, indent=2))
        elif action == "prune":
            dry_run = "--dry-run" in rest
            orphans = await asyncio.to_thread(manager.prune, dry_run=dry_run)
            print("Nothing to prune." if not orphans else ("Would prune: " if dry_run else "Pruned: ") + ", ".join(orphans))
        elif action == "configure":
            if len(rest) < 2:
                raise PluginError("usage: /plugin configure <plugin@marketplace> key=value ...")
            values = dict(item.split("=", 1) for item in rest[1:] if "=" in item)
            if len(values) != len(rest) - 1:
                raise PluginError("plugin configuration values must use key=value")
            await asyncio.to_thread(manager.configure, rest[0], values)
            print(f"Configured {rest[0]} in user settings.")
        elif action == "validate":
            if not rest:
                raise PluginError("usage: /plugin validate <path|plugin@marketplace>")
            installed = manager.records().get(rest[0])
            target = installed.path if installed is not None else rest[0]
            manifest = await asyncio.to_thread(
                validate_plugin, target, None, strict_warnings="--strict" in rest[1:]
            )
            print(f"Valid plugin: {manifest['name']} ({target})")
            for warning in manifest.get("_warnings", []):
                print(f"  warning: {warning}")
        elif action == "marketplace":
            sub = rest[0].casefold() if rest else "list"
            tail = rest[1:]
            if sub == "list":
                markets = manager.marketplaces()
                if not markets:
                    print("No marketplaces configured.")
                for name, source in sorted(markets.items()):
                    print(f"  {name}  {source}")
            elif sub == "add" and tail:
                if len(tail) >= 2:
                    await asyncio.to_thread(manager.marketplace_add, tail[0], tail[1])
                    added_name = tail[0]
                else:
                    added_name = await asyncio.to_thread(manager.marketplace_add_source, tail[0])
                print(f"Marketplace {added_name} added.")
            elif sub == "remove" and tail:
                await asyncio.to_thread(manager.marketplace_remove, tail[0])
                print(f"Marketplace {tail[0]} removed.")
            elif sub == "update" and tail:
                count = await asyncio.to_thread(manager.marketplace_update, tail[0])
                print(f"Marketplace {tail[0]} validated ({count} plugins).")
            else:
                raise PluginError(
                    "usage: /plugin marketplace <list|add NAME PATH|remove NAME|update NAME>"
                )
        else:
            raise PluginError(
                "usage: /plugin <status|errors|reset|output-style|theme|install|update|manage|details|uninstall|enable|disable|validate|prune|configure|marketplace>"
            )
    except (OSError, PluginError) as exc:
        print(f"Plugin command failed: {exc}")
    return ChatTurn()


async def _cmd_reload_plugins(
    agent: "ReActAgent", ui: AgentUI, args: str, history: list[Message]
) -> ChatTurn:
    from agent_core.plugins import PluginError, reload_plugins

    try:
        skills, hooks, tools = await asyncio.to_thread(reload_plugins, agent)
    except (OSError, PluginError, RuntimeError) as exc:
        print(f"Plugin reload failed; the previous generation remains active: {exc}")
        return ChatTurn()
    print(
        f"Plugins reloaded atomically: {skills} skills/commands/agents, "
        f"{hooks} hooks, {tools} MCP tools."
    )
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
    rule_text = parts[2].strip()
    try:
        from agent_core.permission_rules import parse_rule

        if parse_rule(rule_text) is None:
            raise ValueError("malformed permission rule")
        if destination is not PermissionDestination.SESSION:
            confirmed = await asyncio.to_thread(
                ui.confirm_action,
                f"Persist allow rule {rule_text!r} to {destination.value} configuration?",
            )
            if not confirmed:
                print("Persistent permission rule was not confirmed.")
                return
            persist_allow_rule(rule_text, destination, agent.session.workspace)
        agent.permissions.add_session_rule(rule_text)
    except (OSError, ValueError) as exc:
        print(f"Permission rule was not added: {exc}")
        return
    print(f"Added allow rule to {destination.value}: {rule_text}")


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
    repository = getattr(store, "repository", None)
    if repository is None:
        records = store.all()
        if not records:
            print("No memories stored yet.")
            return ChatTurn()
        print(f"Stored memories ({len(records)}):")
        for record in records[-10:]:
            summary = record.content.strip().splitlines()[0] if record.content.strip() else "(empty)"
            print(f"  [{record.kind}] {summary[:100]}")
        return ChatTurn()
    try:
        parts = shlex.split(args)
    except ValueError as exc:
        print(f"Invalid memory command: {exc}")
        return ChatTurn()
    action = parts.pop(0).lower() if parts else "list"
    if action == "list":
        documents = repository.list()
        if not documents:
            print("No memories stored yet.")
            return ChatTurn()
        print(f"Stored memories ({len(documents)}):")
        for document in documents[-10:]:
            print(
                f"  {document.id} [{document.type}] {document.name} "
                f"(updated {document.updated_at})"
            )
    elif action == "show" and parts:
        document = repository.get(parts[0])
        print(document.content if document is not None else f"No memory {parts[0]}")
    elif action == "search" and parts:
        for document in repository.search(" ".join(parts)):
            print(f"  {document.id} [{document.type}] {document.name}: {document.description}")
    elif action == "add" and parts:
        content = " ".join(parts)
        document = await asyncio.to_thread(
            repository.write,
            name=content[:60],
            description=content[:160],
            type="project",
            content=content,
            explicit=True,
            sources=[f"run:{agent.logger.run_id}"],
        )
        if agent.extractor is not None:
            agent.extractor.mark_direct_write()
        print(f"Added {document.id}")
    elif action == "edit" and len(parts) >= 2:
        document = await asyncio.to_thread(
            repository.update,
            parts[0],
            content=" ".join(parts[1:]),
            explicit=True,
        )
        print(f"Updated {document.id}")
    elif action == "forget" and parts:
        forgotten = await asyncio.to_thread(repository.forget, parts[0])
        print(f"Forgot {parts[0]}" if forgotten else f"No memory {parts[0]}")
    elif action == "validate":
        report = await asyncio.to_thread(repository.validate, repair="--repair" in parts)
        print(f"scanned={report.scanned} valid={report.valid}")
        for error in report.errors:
            print(f"  error: {error}")
    else:
        print("Usage: /memory [list|show ID|search QUERY|add TEXT|edit ID TEXT|forget ID|validate]")
    return ChatTurn()


async def _cmd_resume(agent: "ReActAgent", ui: AgentUI, args: str, history: list[Message]) -> ChatTurn:
    """Switch the live agent to another session of the CURRENT project.

    Frozen product contract: same-project only (cross-project targets are
    rejected with a pointer to ``/branch``); the switch rebuilds the whole
    ``SessionRuntime`` via ``agent.resume_loaded_session`` — session permission
    grants, todos, plans, workflow approvals, read state, scheduler jobs, and
    session counters never carry over.
    """
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

    location = locate_session(session_dir, workspace, target)
    if location is None:
        print(f"No session found with id {target!r}.")
        return ChatTurn()
    if location.workspace.resolve() != workspace.resolve():
        print(
            f"Session {target} belongs to {location.workspace.resolve()}. "
            "Change to that project and resume again; use /branch "
            f"{target} to copy it into the current project."
        )
        return ChatTurn()
    loaded = load_transcript(location.path)
    resumed = await agent.resume_loaded_session(loaded)
    print(f"Resumed session {loaded.session_id} ({len(resumed)} messages).")
    return ChatTurn(history=resumed)


async def _cmd_branch(
    agent: "ReActAgent", ui: AgentUI, args: str, history: list[Message]
) -> ChatTurn:
    """Clone a saved/current chain into a fresh session in the current project."""

    del ui
    if not agent.config.session_dir:
        print("Session persistence is disabled; cannot branch.")
        return ChatTurn()
    target = args.strip()
    if target:
        location = locate_session(agent.config.session_dir, agent.session.workspace, target)
        if location is None:
            print(f"No session found with id {target!r}.")
            return ChatTurn()
        source = load_transcript(location.path)
    else:
        resumable = [
            message
            for message in history
            if message.role != "system" and message.metadata.get("pinned") != "user_context"
        ]
        source = LoadedTranscript(
            path=(agent.transcript.path if agent.transcript is not None else Path(".")),
            session_id=agent.session_id,
            messages={message.uuid: message for message in resumable},
            order=[message.uuid for message in resumable],
            workspace=agent.session.workspace,
        )
    branch_id, cloned = fork_chain(source)
    target_path = project_dir(agent.config.session_dir, agent.session.workspace) / f"{branch_id}.jsonl"
    branch = LoadedTranscript(
        path=target_path,
        session_id=branch_id,
        messages={message.uuid: message for message in cloned},
        order=[message.uuid for message in cloned],
        workspace=agent.session.workspace,
    )
    resumed = await agent.resume_loaded_session(branch)
    if agent.transcript is not None:
        for message in cloned:
            await agent.transcript.append_message(message)
    print(f"Branched into session {branch_id} ({len(resumed)} messages).")
    return ChatTurn(history=resumed)


# Command metadata is the sole authority for whether a command may run while the
# model is streaming. Never infer immediacy from a command being read-only.
_COMMAND_SPECS: dict[str, CommandSpec] = {
    "exit": CommandSpec(None, "Leave the chat.", immediate=True),
    "quit": CommandSpec(None, "Leave the chat.", immediate=True, canonical="exit"),
    "help": CommandSpec(_cmd_help, "Show this help."),
    "skills": CommandSpec(_cmd_skills, "List available skills."),
    "capabilities": CommandSpec(
        _cmd_capabilities,
        "Inspect or search runtime skills, MCP tools, and plugins.",
    ),
    "clear": CommandSpec(_cmd_clear, "Clear conversation history (aliases /reset /new)."),
    "reset": CommandSpec(_cmd_clear, "Clear conversation history.", canonical="clear"),
    "new": CommandSpec(_cmd_clear, "Clear conversation history.", canonical="clear"),
    "status": CommandSpec(_cmd_status, "Show model, session, skill/tool counts.", immediate=True),
    "context": CommandSpec(_cmd_context, "Show context-window usage."),
    "cost": CommandSpec(_cmd_cost, "Show session token usage and duration."),
    "compact": CommandSpec(_cmd_compact, "Compact the conversation now."),
    "model": CommandSpec(_cmd_model, "Show or switch the model.", immediate=True),
    "effort": CommandSpec(_cmd_effort, "Show or set reasoning effort.", immediate=True),
    "fast": CommandSpec(_cmd_fast, "Toggle Claude Opus 4.6 fast mode.", immediate=True),
    "rename": CommandSpec(_cmd_rename, "Rename this saved session.", immediate=True),
    "sandbox": CommandSpec(_cmd_sandbox, "Show or change local sandbox settings.", immediate=True),
    "plugin": CommandSpec(_cmd_plugin, "Install and manage Claude-compatible plugins.", immediate=True),
    "plugins": CommandSpec(
        _cmd_plugin,
        "Manage Claude-compatible plugins.",
        immediate=True,
        canonical="plugin",
    ),
    "reload-plugins": CommandSpec(
        _cmd_reload_plugins,
        "Reload enabled plugins at an idle boundary.",
    ),
    "permissions": CommandSpec(_cmd_permissions, "Show or switch the permission mode."),
    "mcp": CommandSpec(_cmd_mcp, "List configured MCP servers.", immediate=True),
    "memory": CommandSpec(_cmd_memory, "Inspect or curate long-term memory."),
    "resume": CommandSpec(_cmd_resume, "List or resume a saved session (alias /continue)."),
    "continue": CommandSpec(_cmd_resume, "List or resume a saved session.", canonical="resume"),
    "branch": CommandSpec(_cmd_branch, "Clone a session into a new branch."),
}

# Compatibility maps used by completion and integrations.
_COMMANDS: dict[str, Handler] = {
    name: cast(Handler, spec.handler)
    for name, spec in _COMMAND_SPECS.items()
    if spec.handler is not None
}
_COMMAND_HELP: dict[str, tuple[Handler, str]] = {
    name: (cast(Handler, spec.handler), spec.summary)
    for name, spec in _COMMAND_SPECS.items()
    if spec.handler is not None and spec.canonical is None
}


def command_spec(task: str) -> CommandSpec | None:
    """Return built-in command metadata, or ``None`` for prose/skills/unknowns."""

    if task in {"/exit", "/quit"}:
        return _COMMAND_SPECS[task[1:]]
    parsed = parse_slash_command(task)
    if parsed is None or not looks_like_command(parsed.name):
        return None
    return _COMMAND_SPECS.get(parsed.name.lower())


def is_immediate_command(task: str) -> bool:
    spec = command_spec(task)
    return bool(spec and spec.immediate)


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

    spec = _COMMAND_SPECS.get(parsed.name.lower())
    if spec is not None and spec.handler is not None:
        return await spec.handler(agent, ui, parsed.args, history)

    skill = agent.skills.get(parsed.name)
    if skill is None or not skill.user_invocable:
        print(f"Unknown command: /{parsed.name}. Try /skills to list skills or /help.")
        return ChatTurn()

    if agent.session.plugin_skill_invoked is not None:
        await agent.session.plugin_skill_invoked(skill.name)
    ctx = SkillPromptContext.from_session(agent.session, transcript=agent.transcript)
    prompt = await build_skill_prompt(skill, parsed.args, ctx)
    if skill.context is not SkillContext.FORK:
        return ChatTurn(prompt=prompt)  # inline → run through agent.run (enters history)

    factory = agent.session.subagent_factory
    if factory is None:
        print("[error] fork skills need sub-agents, which are unavailable here.")
        return ChatTurn()
    try:
        factory_call = cast(Callable[..., Awaitable[str]], factory)
        if skill.memory != "none":
            answer = await factory_call(
                prompt,
                fork_preset(skill.allowed_tools, skill.disallowed_tools),
                skill.model,
                "shared",
                skill.agent_key or skill.name,
                skill.memory,
            )
        else:
            answer = await factory_call(
                prompt,
                fork_preset(skill.allowed_tools, skill.disallowed_tools),
                skill.model,
            )
    except Exception as exc:  # noqa: BLE001 - a skill failure must not tear down the chat
        print(f"[error] skill /{skill.name} failed: {type(exc).__name__}: {exc}")
        return ChatTurn()
    if not ui.is_live:
        print(answer)
    return ChatTurn()
