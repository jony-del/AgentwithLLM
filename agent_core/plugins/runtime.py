"""Atomic plugin generation swaps on a live agent."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from agent_core.skills import SkillRegistry
from agent_core.tools.base import Tool
from agent_core.tools.registry import DeferredTool
from .models import PluginError, PluginBundle, PluginGeneration
from .store import _atomic_text
from .manager import PluginManager

if TYPE_CHECKING:
    from agent_core.react import ReActAgent


def _prepare_generation(agent: "ReActAgent", bundle: PluginBundle) -> PluginGeneration:
    """Validate a complete candidate without mutating the active generation."""

    try:
        base = agent._load_skills()
        merged = SkillRegistry(base.list())
        # Built-ins remain available unqualified; plugin components are namespaced.
        for skill in bundle.skills:
            merged.add(skill)
        base_hooks = agent._build_hook_pipeline()
        for attr, adapter in bundle.hooks:
            getattr(base_hooks, attr).append(adapter)

        existing_tool_names = {
            *[tool.name for tool in agent.registry.list()],
            *[tool.name for tool in agent.registry.deferred()],
        }
        previous_plugin_names = set(getattr(agent, "_plugin_tool_names", set()))
        collisions = {
            tool.name
            for tool in bundle.mcp_tools
            if tool.name in existing_tool_names - previous_plugin_names
        }
        candidate_names = [tool.name for tool in bundle.mcp_tools]
        collisions.update(
            name for name in candidate_names if candidate_names.count(name) > 1
        )
        if collisions:
            raise PluginError(
                "plugin MCP tools collide with built-ins: " + ", ".join(sorted(collisions))
            )
    except Exception:
        if bundle.mcp_manager is not None:
            bundle.mcp_manager.close()
        raise
    return PluginGeneration(bundle, merged, base_hooks)


def _commit_generation(agent: "ReActAgent", generation: PluginGeneration) -> tuple[int, int, int]:
    """Publish an already-validated generation; the operations below cannot block."""

    bundle = generation.bundle
    merged = generation.skills
    base_hooks = generation.hooks

    old_manager = getattr(agent, "_plugin_mcp_manager", None)
    old_tool_names = set(getattr(agent, "_plugin_tool_names", set()))
    active_tools: list[Tool] = []
    deferred_tools: list[DeferredTool] = []
    for tool in bundle.mcp_tools:

        def factory(bound_tool: Tool = tool) -> Tool:
            return bound_tool

        if getattr(tool, "_always_load", False):
            active_tools.append(tool)
        else:
            deferred_tools.append(
                DeferredTool(
                    tool.name,
                    tool.description,
                    factory,
                    metadata={
                    "kind": "mcp",
                    "server": str(getattr(tool, "_server", "")),
                    "remote": str(getattr(tool, "_remote", "")),
                },
                )
            )
    try:
        agent.registry.replace_group(old_tool_names, active_tools, deferred_tools)
    except Exception:
        if bundle.mcp_manager is not None:
            bundle.mcp_manager.close()
        raise
    if merged.model_invocable():
        try:
            agent.registry.get("skill")
        except KeyError:
            skill_tool = agent.default_registry().get("skill")
            agent.registry.register(skill_tool)
    else:
        agent.registry.unregister("skill")
    agent._plugin_tool_names = {tool.name for tool in bundle.mcp_tools}
    agent._plugin_mcp_manager = bundle.mcp_manager
    agent._plugin_active_ids = frozenset(bundle.plugin_ids)
    agent.plugin_agents = bundle.agents
    plugin_lsp_names = set(getattr(agent, "_plugin_lsp_names", set()))
    base_lsp = [
        item for item in agent.config.tools.lsp.servers
        if item.name not in plugin_lsp_names
    ]
    old_lsp_configs = list(agent.config.tools.lsp.servers)
    agent.config.tools.lsp.servers = [*base_lsp, *bundle.components["lsp"]]
    agent._plugin_lsp_names = {item.name for item in bundle.components["lsp"]}
    if agent.config.tools.lsp.servers:
        try:
            agent.registry.get("lsp")
        except KeyError:
            builtins = agent.default_registry()
            try:
                lsp_tool = builtins.get("lsp")
            except KeyError:
                # LSP is intentionally deferred in the built-in catalog.  A plugin
                # can introduce the first LSP config after agent construction, so
                # instantiate it now and let the live registry bind session/sandbox.
                lsp_tool = builtins.activate("lsp")
            agent.registry.register(lsp_tool)
    else:
        agent.registry.unregister("lsp")
    if old_lsp_configs != agent.config.tools.lsp.servers:
        old_lsp_manager = agent.session.lsp_manager
        agent.session.lsp_manager = None
        if old_lsp_manager is not None:
            _close_async_resource(old_lsp_manager.close())
    agent.plugin_workflows = bundle.components["workflows"]
    agent.session.plugin_workflows = agent.plugin_workflows
    agent.plugin_output_styles = bundle.components["output_styles"]
    agent.plugin_themes = bundle.components["themes"]
    agent._refresh_plugin_presentation()
    agent.plugin_monitors = bundle.components["monitors"]
    agent.plugin_channels = bundle.components["channels"]
    agent.plugin_user_config = bundle.components["user_config"]
    # PATH is intentionally not changed globally. Shell integrations may prepend these
    # per child process after the normal Polaris permission decision.
    agent.plugin_bin_paths = tuple(bundle.components["bin"])
    agent.session.plugin_bin_paths = agent.plugin_bin_paths
    agent.plugin_settings = bundle.components["settings"]
    agent.skills = merged
    agent.session.skills = merged
    agent.hooks = base_hooks
    agent.executor.hooks = base_hooks
    if old_manager is not None:
        old_manager.close()
    agent.session.registered_tool_names = frozenset(tool.name for tool in agent.registry.list())
    PluginManager(agent.session.workspace).audit.write(
        "generation_commit",
        {
            "plugin_ids": list(bundle.plugin_ids),
            "skills": len(bundle.skills),
            "hooks": len(bundle.hooks),
            "mcp_tools": len(bundle.mcp_tools),
        },
    )
    return len(bundle.skills), len(bundle.hooks), len(bundle.mcp_tools)


def _close_async_resource(awaitable: Any) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(awaitable)
    else:
        loop.create_task(awaitable)


def reload_plugins(agent: "ReActAgent") -> tuple[int, int, int]:
    """Build every component first, then atomically swap the live plugin generation."""

    manager = PluginManager(agent.session.workspace)
    return _commit_generation(agent, _prepare_generation(agent, manager.build_bundle(agent)))


def activate_plugin(
    agent: "ReActAgent",
    plugin_id: str,
    *,
    components: tuple[str, ...],
    allowed_hooks: tuple[str, ...] = (),
) -> tuple[int, int, int]:
    """Build a proposed component-scoped state, persist it, then publish it."""

    manager = PluginManager(agent.session.workspace)
    if plugin_id not in manager.records():
        raise PluginError(f"plugin is not installed: {plugin_id}")
    enabled = list(dict.fromkeys([*manager.enabled_ids(), plugin_id]))
    selections = manager.component_selections()
    selections[plugin_id] = tuple(dict.fromkeys(components))
    hook_ids = manager.hook_selections()
    hook_ids[plugin_id] = tuple(dict.fromkeys(allowed_hooks))
    generation = _prepare_generation(
        agent,
        manager.build_bundle(
            agent,
            enabled_ids=enabled,
            component_selections=selections,
            hook_selections=hook_ids,
        ),
    )
    local_path = manager.workspace / "agent.local.toml"
    previous_local = local_path.read_bytes() if local_path.is_file() else None
    try:
        manager.set_activation(
            plugin_id,
            selections[plugin_id],
            allowed_hooks=hook_ids[plugin_id],
        )
        return _commit_generation(agent, generation)
    except Exception:
        try:
            if previous_local is None:
                local_path.unlink(missing_ok=True)
            else:
                _atomic_text(local_path, previous_local.decode("utf-8"))
        except (OSError, UnicodeDecodeError):
            pass
        if generation.bundle.mcp_manager is not None:
            generation.bundle.mcp_manager.close()
        raise
