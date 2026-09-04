"""Load plugin skills, agents, hooks, MCP servers, and passive components."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse
from agent_core.capability_security import TrustTier
from agent_core.hook_adapters import LIFECYCLE_EVENT_ATTRS, build_external_adapter
from agent_core.hooks import ExternalHookSpec
from agent_core.mcp import MCPServerConfig
from agent_core.sandbox import SandboxInvocation
from agent_core.skills import Skill, SkillContext, load_skill_file, parse_frontmatter
from agent_core.tools.base import ExecutionScope
from .models import PluginError, is_safe_plugin_name, _string_tuple, _positive_int, PluginRecord
from .store import _read_json, _inside
from .env import _plugin_option_env, _expand_executable_env, _expand_plugin_vars
from .sandbox import sandboxed_guest_invocation, sandbox_runtime_environment

if TYPE_CHECKING:
    from agent_core.react import ReActAgent


def _clone_skill(skill: Skill, namespace: str) -> Skill:
    return Skill(
        name=f"{namespace}:{skill.name}",
        description=skill.description,
        body=skill.body,
        when_to_use=skill.when_to_use,
        argument_hint=skill.argument_hint,
        allowed_tools=skill.allowed_tools,
        disallowed_tools=skill.disallowed_tools,
        preload_skills=skill.preload_skills,
        capabilities=skill.capabilities,
        hooks=skill.hooks,
        model=skill.model,
        aliases=tuple(f"{namespace}:{alias}" for alias in skill.aliases),
        user_invocable=skill.user_invocable,
        disable_model_invocation=skill.disable_model_invocation,
        context=skill.context,
        memory=skill.memory,
        effort=skill.effort,
        permission_mode=skill.permission_mode,
        isolation=skill.isolation,
        background=skill.background,
        max_turns=skill.max_turns,
        agent_key=f"{namespace}:{skill.agent_key or skill.name}",
        trust_tier=skill.trust_tier,
        source_identity=skill.source_identity,
        plugin_id=skill.plugin_id,
        source_path=skill.source_path,
    )


def _apply_skill_provenance(skills: list[Skill], record: PluginRecord) -> None:
    """Apply host-owned provenance and tiered isolation after parsing frontmatter."""

    remote_untrusted = record.trust_tier in {
        TrustTier.COMMUNITY.value,
        TrustTier.VERIFIED_PUBLISHER.value,
    }
    for skill in skills:
        skill.trust_tier = record.trust_tier
        skill.source_identity = record.source_identity
        skill.plugin_id = record.plugin_id
        if remote_untrusted:
            skill.context = SkillContext.FORK


def _component_paths(
    root: Path,
    manifest: dict[str, Any],
    key: str,
    default: Path,
) -> list[Path]:
    raw = manifest.get(key)
    values = (
        [item for item in raw if isinstance(item, str)]
        if isinstance(raw, list)
        else ([raw] if isinstance(raw, str) else [])
    )
    if not values:
        return [default]
    paths = [(root / str(value)).resolve() for value in values]
    if key == "skills" and not manifest.get("_skills_replace_default") and default not in paths:
        paths.insert(0, default)
    return paths


def _load_plugin_skills(
    root: Path, namespace: str, manifest: dict[str, Any], workspace: Path,
    plugin_data: Path, user_config: dict[str, Any],
) -> list[Skill]:
    result: list[Skill] = []
    candidates: list[Path] = []
    for location in _component_paths(root, manifest, "skills", root / "skills"):
        if location.is_file():
            candidates.append(location)
        else:
            candidates.extend(location.glob("*/SKILL.md"))
            candidates.extend(location.glob("*.md"))
    if not (root / "skills").exists() and "skills" not in manifest and (root / "SKILL.md").is_file():
        candidates.append(root / "SKILL.md")
    for location in _component_paths(root, manifest, "commands", root / "commands"):
        if location.is_file():
            candidates.append(location)
        else:
            candidates.extend(location.glob("*.md"))
    for path in sorted(set(candidates)):
        skill = load_skill_file(path)
        if skill is not None:
            skill = replace(
                skill,
                body=_expand_plugin_vars(
                    skill.body, root, workspace, plugin_data, user_config
                ),
            )
            result.append(_clone_skill(skill, namespace))
    return result


def _load_plugin_agents(
    root: Path, namespace: str, manifest: dict[str, Any], workspace: Path,
    plugin_data: Path, user_config: dict[str, Any],
) -> tuple[dict[str, str], list[Skill]]:
    definitions: dict[str, str] = {}
    skills: list[Skill] = []
    candidates: list[Path] = []
    for location in _component_paths(root, manifest, "agents", root / "agents"):
        candidates.extend([location] if location.is_file() else location.glob("*.md"))
    for path in sorted(candidates):
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if not raw:
            continue
        meta, parsed_body = parse_frontmatter(raw)
        body = parsed_body.strip()
        if not body:
            continue
        local_name = str(meta.get("name") or path.stem).strip()
        if not is_safe_plugin_name(local_name):
            raise PluginError(f"invalid plugin agent name: {local_name}")
        name = f"{namespace}:{local_name}"
        definitions[name] = raw
        memory = str(meta.get("memory", "none")).strip().lower()
        if memory not in {"none", "user", "project", "local"}:
            memory = "none"
        skills.append(
            Skill(
                name=name,
                description=str(meta.get("description") or f"Run the {path.stem} plugin agent."),
                body=_expand_plugin_vars(
                    body, root, workspace, plugin_data, user_config
                ) + "\n\n$ARGUMENTS",
                allowed_tools=_string_tuple(meta.get("tools")),
                disallowed_tools=_string_tuple(meta.get("disallowed_tools")),
                preload_skills=_string_tuple(meta.get("skills")),
                model=(str(meta.get("model")).strip() or None) if meta.get("model") else None,
                context=SkillContext.FORK,
                memory=memory,
                effort=(str(meta.get("effort")).strip() or None) if meta.get("effort") else None,
                # Claude deliberately ignores permissionMode, hooks and mcpServers
                # on plugin-shipped agents.  Plugin agents remain subject to the
                # host's normal Polaris permission policy.
                permission_mode=None,
                isolation=(str(meta.get("isolation")).strip() or None) if meta.get("isolation") else None,
                background=meta.get("background", False) is True,
                max_turns=_positive_int(meta.get("max_turns")),
                agent_key=name,
                source_path=path,
            )
        )
    return definitions, skills


def _load_plugin_hooks(
    root: Path,
    agent: "ReActAgent",
    namespace: str,
    manifest: dict[str, Any],
    *,
    plugin_id: str,
    workspace: Path,
    plugin_data: Path,
    user_config: dict[str, Any],
    runtime_config: dict[str, Any] | None = None,
    allowed_hook_ids: tuple[str, ...] | None = None,
) -> list[tuple[str, Any]]:
    result: list[tuple[str, Any]] = []
    # command_argv is an internal post-sandbox transport and must never be supplied
    # by plugin metadata itself.
    valid_fields = {
        field.name for field in fields(ExternalHookSpec) if field.name != "command_argv"
    }
    raw_hooks = manifest.get("hooks")
    inline_tables: list[dict[str, Any]] = []
    if isinstance(raw_hooks, dict):
        inline_tables.append(raw_hooks.get("hooks", raw_hooks))
    elif isinstance(raw_hooks, list):
        inline_tables.extend(item.get("hooks", item) for item in raw_hooks if isinstance(item, dict))
    tables: list[Any] = list(inline_tables)
    for location in _component_paths(root, manifest, "hooks", root / "hooks"):
        path = location / "hooks.json" if location.is_dir() else location
        data = _read_json(path, {})
        tables.append(data.get("hooks", data) if isinstance(data, dict) else {})
    for table in tables:
        if not isinstance(table, dict):
            continue
        for event, groups in table.items():
            if event not in LIFECYCLE_EVENT_ATTRS or not isinstance(groups, list):
                continue
            for group in groups:
                if not isinstance(group, dict):
                    continue
                entries = group.get("hooks", [group])
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    if allowed_hook_ids is not None:
                        entry_id = str(entry.get("id") or "").strip()
                        qualified = f"{plugin_id}:{entry_id}" if entry_id else ""
                        if not qualified or qualified not in allowed_hook_ids:
                            continue
                    values = dict(entry)
                    values["event"] = event
                    if "if" in values and "condition" not in values:
                        values["condition"] = values.pop("if")
                    if "matcher" not in values and isinstance(group.get("matcher"), str):
                        values["matcher"] = group["matcher"]
                    values = _expand_plugin_vars(
                        values, root, workspace, plugin_data, user_config
                    )
                    values["env"] = {
                        **{
                            str(key): _expand_executable_env(str(value))
                            for key, value in dict(values.get("env") or {}).items()
                        },
                        **_plugin_option_env(runtime_config or user_config),
                    }
                    try:
                        spec = ExternalHookSpec(
                            **{
                                key: value
                                for key, value in values.items()
                                if key in valid_fields
                            }
                        )
                        if allowed_hook_ids is not None:
                            spec = _restrict_autonomous_hook(agent, root, plugin_id, spec)
                        adapter = build_external_adapter(
                            spec,
                            logger=agent.logger,
                            provider=agent.provider,
                            base_config=agent._provider_config(),
                            subagent_factory=agent.session.subagent_factory,
                            limits=agent.config.hooks.limits,
                        )
                    except Exception as exc:
                        raise PluginError(
                            f"{namespace} hook failed validation: {exc}"
                        ) from exc
                    if adapter is not None:
                        result.append((LIFECYCLE_EVENT_ATTRS[event], adapter))
    return result


def _restrict_autonomous_hook(
    agent: "ReActAgent",
    root: Path,
    plugin_id: str,
    spec: ExternalHookSpec,
) -> ExternalHookSpec:
    if spec.type == "command":
        if not spec.command or not agent.sandbox.is_enabled():
            raise PluginError(
                f"autonomous command hook from {plugin_id} requires a real sandbox"
            )
        guest_root = agent.sandbox.translate_path(root)
        command = spec.command.replace(str(root), guest_root)
        scope = ExecutionScope.for_workspace(
            agent.session.workspace,
            read_only_roots=(root,),
            network="deny",
        )
        invocation = SandboxInvocation.create(
            ["/bin/sh", "-c", spec.command],
            guest_argv=["@bash", "-lc", command],
            required_guest_capabilities=("bash",),
            scope=scope,
        )
        wrapped, shell = agent.sandbox.wrap_invocation(invocation, command=command)
        if shell or not isinstance(wrapped, list) or not wrapped:
            raise PluginError(f"sandbox could not wrap autonomous hook from {plugin_id}")
        return replace(spec, command_argv=[str(item) for item in wrapped])
    if spec.type == "http":
        if agent.config.sandbox.enabled:
            raise PluginError(
                f"autonomous HTTP hook from {plugin_id} is unavailable while the "
                "guest sandbox enforces network=deny"
            )
        parsed = urlparse(spec.url or "")
        host = parsed.hostname or ""
        loopback = host in {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme != "https" and not (loopback and parsed.scheme == "http"):
            raise PluginError(
                f"autonomous HTTP hook from {plugin_id} requires HTTPS or loopback"
            )
        if not loopback and not _domain_allowed(host, agent.config.web.allowed_domains):
            raise PluginError(
                f"autonomous HTTP hook domain {host!r} is not in web.allowed_domains"
            )
    return spec


def _load_plugin_mcp(
    root: Path, namespace: str, manifest: dict[str, Any], workspace: Path,
    plugin_data: Path, user_config: dict[str, Any],
    runtime_config: dict[str, Any] | None = None,
) -> list[MCPServerConfig]:
    result: list[MCPServerConfig] = []
    raw_mcp = manifest.get("mcpServers")
    tables: list[Any] = []
    if isinstance(raw_mcp, dict):
        tables.append(raw_mcp.get("mcpServers", raw_mcp))
    elif isinstance(raw_mcp, list):
        tables.extend(
            item.get("mcpServers", item) for item in raw_mcp if isinstance(item, dict)
        )
    for location in _component_paths(root, manifest, "mcpServers", root / ".mcp.json"):
        data = _read_json(location, {})
        tables.append(
            data.get("mcpServers", data.get("servers", {}))
            if isinstance(data, dict)
            else {}
        )
    merged: dict[str, Any] = {}
    for table in tables:
        if not isinstance(table, dict):
            continue
        merged.update(table)
    for name, body in merged.items():
        if not isinstance(body, dict):
            continue
        expanded = _expand_plugin_vars(
            body, root, workspace, plugin_data, user_config
        )
        expanded["env"] = {
            **{
                str(key): _expand_executable_env(str(value))
                for key, value in dict(expanded.get("env") or {}).items()
            },
            **_plugin_option_env(runtime_config or user_config),
        }
        result.append(
            MCPServerConfig.from_dict(f"{namespace}:{name}", expanded)
        )
    return result


def _load_json_component(root: Path, value: Any, default: Path, *, wrapper: str = "") -> Any:
    if isinstance(value, (dict, list)):
        return value
    locations = value if isinstance(value, list) else [value] if isinstance(value, str) else [str(default.relative_to(root)).replace("\\", "/")]
    merged: Any = [] if wrapper == "list" else {}
    for raw in locations:
        if not isinstance(raw, str):
            continue
        path = (root / raw).resolve()
        if not _inside(path, root):
            raise PluginError("component JSON path escapes plugin root")
        loaded = _read_json(path, None)
        if wrapper and isinstance(loaded, dict):
            loaded = loaded.get(wrapper)
        if isinstance(merged, dict) and isinstance(loaded, dict):
            merged.update(loaded)
        elif isinstance(merged, list) and isinstance(loaded, list):
            merged.extend(loaded)
    return merged


def _markdown_components(root: Path, manifest: dict[str, Any], key: str, default: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for location in _component_paths(root, manifest, key, default):
        candidates = [location] if location.is_file() else sorted(location.glob("*")) if location.is_dir() else []
        for path in candidates:
            if not path.is_file() or path.suffix.casefold() not in {".md", ".json", ".js", ".mjs"}:
                continue
            try:
                result[path.stem] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
    return result


def _load_non_mcp_components(
    root: Path,
    namespace: str,
    manifest: dict[str, Any],
    selected: set[str],
    workspace: Path,
    plugin_data: Path,
    user_config: dict[str, Any],
    runtime_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from agent_core.tool_config import LSPServerConfig

    result: dict[str, Any] = {
        "lsp": [], "workflows": {}, "output_styles": {}, "themes": {},
        "monitors": [], "channels": [], "user_config": {}, "bin": [], "settings": {},
    }
    if "lsp" in selected:
        table = _load_json_component(root, manifest.get("lspServers"), root / ".lsp.json")
        if isinstance(table, dict):
            for name, value in table.items():
                if not isinstance(value, dict):
                    continue
                expanded = _expand_plugin_vars(
                    value, root, workspace, plugin_data, user_config
                )
                expanded["env"] = {
                    **{
                        str(key): _expand_executable_env(str(value))
                        for key, value in dict(expanded.get("env") or {}).items()
                    },
                    **_plugin_option_env(runtime_config or user_config),
                }
                expanded["name"] = f"{namespace}:{name}"
                expanded["plugin_root"] = str(root)
                config = LSPServerConfig.from_dict(expanded)
                if config.command and config.extensions:
                    result["lsp"].append(config)
    if "workflows" in selected:
        result["workflows"] = {
            f"{namespace}:{name}": value
            for name, value in _markdown_components(root, manifest, "workflows", root / "workflows").items()
        }
    if "output-styles" in selected:
        result["output_styles"] = {
            f"{namespace}:{name}": value
            for name, value in _markdown_components(root, manifest, "outputStyles", root / "output-styles").items()
        }
    if "themes" in selected:
        theme_value = manifest.get("themes")
        experimental = manifest.get("experimental")
        if theme_value is None and isinstance(experimental, dict):
            theme_value = experimental.get("themes")
        themed = dict(manifest)
        if theme_value is not None:
            themed["themes"] = theme_value
        result["themes"] = {
            f"{namespace}:{name}": value
            for name, value in _markdown_components(root, themed, "themes", root / "themes").items()
        }
    if "monitors" in selected:
        monitor_value = manifest.get("monitors")
        experimental = manifest.get("experimental")
        if monitor_value is None and isinstance(experimental, dict):
            monitor_value = experimental.get("monitors")
        monitors = _load_json_component(
            root, monitor_value, root / "monitors" / "monitors.json", wrapper="list"
        )
        if isinstance(monitors, list):
            for item in monitors:
                if isinstance(item, dict) and item.get("name") and item.get("command"):
                    result["monitors"].append({
                        **_expand_plugin_vars(
                            item, root, workspace, plugin_data, {}, allow_user_config=False
                        ),
                        "name": f"{namespace}:{item['name']}",
                        "plugin": namespace,
                    })
    if "channels" in selected and isinstance(manifest.get("channels"), list):
        mcp_value = manifest.get("mcpServers")
        mcp_names = set(mcp_value) if isinstance(mcp_value, dict) else set()
        if not mcp_names:
            mcp_names = {
                item.name.split(":", 1)[-1]
                for item in _load_plugin_mcp(
                    root, namespace, manifest, workspace, plugin_data, user_config,
                    runtime_config,
                )
            }
        for item in manifest["channels"]:
            if not isinstance(item, dict) or str(item.get("server") or "") not in mcp_names:
                raise PluginError(f"{namespace} channel must bind to its own MCP server")
            result["channels"].append({
                **item,
                "plugin": namespace,
                "qualified_server": f"{namespace}:{item['server']}",
            })
    if "user-config" in selected and isinstance(manifest.get("userConfig"), dict):
        result["user_config"][namespace] = dict(manifest["userConfig"])
    if "bin" in selected and (root / "bin").is_dir():
        result["bin"].append(str((root / "bin").resolve()))
    if "settings" in selected:
        settings = _read_json(root / "settings.json", {})
        inline = manifest.get("settings")
        if isinstance(inline, dict):
            settings = {**inline, **settings}
        if isinstance(settings, dict):
            result["settings"][namespace] = {
                key: value for key, value in settings.items()
                if key in {"agent", "subagentStatusLine"}
            }
    return result


def _domain_allowed(host: str, allowed: list[str]) -> bool:
    normalized = host.casefold().strip(".")
    return any(
        normalized == item.casefold().strip(".")
        or normalized.endswith("." + item.casefold().strip("."))
        for item in allowed
        if item.strip()
    )


def _restrict_autonomous_mcp(
    agent: "ReActAgent", root: Path, servers: list[MCPServerConfig]
) -> list[MCPServerConfig]:
    """Fail closed unless autonomous MCP transport is constrained by host policy."""

    restricted: list[MCPServerConfig] = []
    for server in servers:
        transport = (server.transport or "stdio").casefold()
        if transport in {"streamable-http", "streamable_http", "http", "sse", "ws", "wss", "websocket"}:
            raise PluginError(
                f"autonomous remote MCP {server.name} is unavailable while the "
                "guest sandbox enforces network=deny"
            )

        if transport != "stdio":
            raise PluginError(f"unsupported autonomous MCP transport: {server.transport}")
        if not agent.sandbox.is_enabled():
            raise PluginError(
                f"autonomous stdio MCP {server.name} requires a real sandbox backend"
            )
        if server.env:
            raise PluginError(
                f"autonomous stdio MCP {server.name} cannot inject host environment secrets"
            )
        argv = [server.command, *server.args]
        scope = ExecutionScope.for_workspace(
            agent.session.workspace,
            read_only_roots=(root,),
            network="deny",
        )
        invocation = sandboxed_guest_invocation(
            agent.sandbox, argv, mounted_roots=(root, agent.session.workspace), scope=scope
        )
        wrapped, shell = agent.sandbox.wrap_invocation(invocation)
        if shell or not isinstance(wrapped, list) or not wrapped:
            raise PluginError(f"sandbox could not wrap autonomous MCP {server.name}")
        restricted.append(
            replace(
                server,
                command=str(wrapped[0]),
                args=[str(item) for item in wrapped[1:]],
                env=sandbox_runtime_environment(),
                cwd="",
            )
        )
    return restricted
