from __future__ import annotations

import json
from pathlib import Path

import pytest

import agent_core.plugins as plugins_module
from agent_core.memory import MemoryConfig
from agent_core.plugins import PluginError, PluginManager, reload_plugins, validate_plugin
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
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        (root / "escape").symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(PluginError, match="escapes"):
        validate_plugin(root)


def test_no_default_marketplace(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("POLARIS_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    assert PluginManager(tmp_path).marketplaces() == {}


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
