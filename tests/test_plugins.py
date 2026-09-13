from __future__ import annotations

import json
from pathlib import Path

import pytest

import agent_core.plugins as plugins_module
from agent_core.memory import MemoryConfig
from agent_core.plugins import PluginError, PluginManager, reload_plugins, validate_plugin
from agent_core.plugins.store import plugin_tree_digest
from agent_core.providers import FakeProvider
from agent_core.react import ReActAgent, ReActConfig


def _write_plugin(root: Path, *, name: str = "demo") -> Path:
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": name, "version": "1.0.0"}),
        encoding="utf-8",
    )
    (root / "commands").mkdir()
    (root / "commands" / "hello.md").write_text(
        "---\ndescription: Say hello\n---\nHello $ARGUMENTS",
        encoding="utf-8",
    )
    return root


def _make_escape_link(link: Path, target: Path) -> str:
    """Create a link at ``link`` pointing at ``target``; returns the kind used.

    Symlinks first (POSIX, or Windows with Developer Mode); junctions as the
    Windows fallback — creating a junction needs no privileges, which is exactly
    why the containment checks must catch them too.
    """
    try:
        link.symlink_to(target)
        return "symlink"
    except OSError:
        pass
    if link.exists() or link.is_symlink():
        raise AssertionError("link creation failed but the path exists")
    import os

    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
        return "junction"
    pytest.skip("symlink creation is unavailable")


def test_plugin_install_enable_and_atomic_skill_reload(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("POLARIS_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    monkeypatch.setenv("POLARIS_SETTINGS_PATH", str(tmp_path / "settings.toml"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = _write_plugin(tmp_path / "source")

    manager = PluginManager(workspace)
    record = manager.install(str(source))
    assert record.plugin_id == "demo@local"
    assert manager.enabled_ids() == []
    manager.set_enabled(record.plugin_id, True)
    assert manager.enabled_ids() == ["demo@local"]

    agent = ReActAgent(
        FakeProvider(),
        ReActConfig(
            run_dir=str(tmp_path / "runs"),
            session_dir="",
            memory=MemoryConfig(enabled=False),
        ),
        workspace=workspace,
    )
    reload_plugins(agent)
    skill = agent.skills.get("demo:hello")
    assert skill is not None
    assert "Hello" in skill.body


def test_plugin_validation_rejects_symlink_escape(
    tmp_path: Path, monkeypatch
) -> None:
    root = _write_plugin(tmp_path / "source")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    kind = _make_escape_link(root / "escape", outside)
    assert kind in {"symlink", "junction"}
    with pytest.raises(PluginError, match="escapes"):
        validate_plugin(root)
    with pytest.raises(PluginError, match="escapes"):
        plugin_tree_digest(root)


def test_marketplace_copy_excludes_escape_links(tmp_path: Path) -> None:
    # An escape link (junction on unprivileged Windows) in a marketplace source
    # must be excluded BEFORE the copy: copytree does not treat a junction as a
    # symlink, so following it would pull host content into the plugin cache.
    from agent_core.plugins.store import copy_marketplace_plugin_tree

    market = tmp_path / "market"
    source = _write_plugin(market / "demo")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    kind = _make_escape_link(source / "escape", outside)
    assert kind in {"symlink", "junction"}

    destination = tmp_path / "cache" / "demo"
    copy_marketplace_plugin_tree(source, destination, market)

    assert (destination / "commands" / "hello.md").is_file()
    assert not (destination / "escape").exists()
    assert not (destination / "escape" / "secret.txt").exists()


def test_no_default_marketplace(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("POLARIS_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    assert PluginManager(tmp_path).marketplaces() == {}


def test_prompt_expansion_cannot_read_arbitrary_host_environment(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PLUGIN_HOST_SECRET", "must-not-leak")
    rendered = plugins_module._expand_plugin_vars(
        "root=${CLAUDE_PLUGIN_ROOT}; public=${user_config.region}; secret=${PLUGIN_HOST_SECRET}",
        tmp_path / "plugin",
        tmp_path / "workspace",
        tmp_path / "data",
        {"region": "eu"},
    )

    assert "public=eu" in rendered
    assert "must-not-leak" not in rendered
    assert "${PLUGIN_HOST_SECRET}" in rendered
    assert plugins_module._expand_executable_env("${PLUGIN_HOST_SECRET}") == "must-not-leak"


def test_plugin_enable_rewrites_multiline_local_array_safely(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("POLARIS_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    monkeypatch.setenv("POLARIS_SETTINGS_PATH", str(tmp_path / "settings.toml"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    record = PluginManager(workspace).install(str(_write_plugin(tmp_path / "source")))
    (workspace / "agent.local.toml").write_text(
        '[plugins]\nenabled = [\n  "older@local",\n]\n\n[sandbox]\nenabled = false\n',
        encoding="utf-8",
    )

    PluginManager(workspace).set_enabled(record.plugin_id, True)

    import tomllib

    parsed = tomllib.loads((workspace / "agent.local.toml").read_text(encoding="utf-8"))
    assert parsed["plugins"]["enabled"] == ["older@local", "demo@local"]
    assert parsed["sandbox"]["enabled"] is False


def test_plugin_bundle_namespaces_agents_hooks_and_mcp(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("POLARIS_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    monkeypatch.setenv("POLARIS_SETTINGS_PATH", str(tmp_path / "settings.toml"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = _write_plugin(tmp_path / "source")
    (source / "agents").mkdir()
    (source / "agents" / "reviewer.md").write_text("Review carefully.", encoding="utf-8")
    (source / "hooks").mkdir()
    (source / "hooks" / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "echo ${CLAUDE_PLUGIN_ROOT}",
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (source / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "helper": {
                        "command": "${CLAUDE_PLUGIN_ROOT}/server",
                        "args": [],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    seen_configs = []

    class FakeMCPManager:
        def __init__(self, config):
            seen_configs.append(config)

        def start(self):
            return self

        def tools(self):
            return []

        def close(self):
            return None

    monkeypatch.setattr(plugins_module, "MCPClientManager", FakeMCPManager)
    manager = PluginManager(workspace)
    record = manager.install(str(source))
    manager.set_enabled(record.plugin_id, True)
    agent = ReActAgent(
        FakeProvider(),
        ReActConfig(
            run_dir=str(tmp_path / "runs"),
            session_dir="",
            memory=MemoryConfig(enabled=False),
        ),
        workspace=workspace,
    )

    skills, hooks, tools = reload_plugins(agent)

    assert skills >= 2 and hooks == 1 and tools == 0
    assert agent.skills.get("demo:reviewer").context.value == "fork"
    assert "demo:reviewer" in agent.plugin_agents
    spec = agent.hooks.session_start_hooks[-1].spec
    assert "${CLAUDE_PLUGIN_ROOT}" not in spec.command
    assert seen_configs[-1].servers[0].name == "demo:helper"
    assert str(Path(record.path)) in seen_configs[-1].servers[0].command
