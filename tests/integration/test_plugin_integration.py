"""Offline plugin integration chain (audit phase 5): the full lifecycle, live.

A temp plugin is installed, enabled, granted, loaded into a real agent runtime,
revoked, re-granted and finally disabled — with the runtime capability (its
skill) appearing, disappearing and reappearing exactly as the persisted grant
state says. No marketplace, no network, no host state: everything lives under
temp POLARIS_PLUGIN_HOME / settings paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_core.memory import MemoryConfig
from agent_core.plugins import PluginManager, reload_plugins
from agent_core.providers import FakeProvider
from agent_core.react import ReActAgent, ReActConfig

pytestmark = pytest.mark.integration


def _write_plugin(root: Path, *, name: str = "itg") -> Path:
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": name, "version": "1.0.0"}), encoding="utf-8"
    )
    (root / "commands").mkdir()
    (root / "commands" / "greet.md").write_text(
        "---\ndescription: Greet the operator\n---\nHello $ARGUMENTS",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def plugin_env(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    monkeypatch.setenv("POLARIS_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    monkeypatch.setenv("POLARIS_SETTINGS_PATH", str(tmp_path / "settings.toml"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return tmp_path, workspace


def _agent(workspace: Path) -> ReActAgent:
    return ReActAgent(
        FakeProvider(),
        ReActConfig(
            run_dir=str(workspace / "runs"),
            session_dir="",
            memory=MemoryConfig(enabled=False),
        ),
        workspace=workspace,
    )


def test_plugin_lifecycle_capability_follows_grant_state(plugin_env) -> None:
    tmp_path, workspace = plugin_env
    source = _write_plugin(tmp_path / "source")
    manager = PluginManager(workspace)
    plugin_id = manager.install(str(source)).plugin_id
    assert manager.enabled_ids() == []

    # Disabled plugins install but never reach the runtime.
    agent = _agent(workspace)
    reload_plugins(agent)
    assert agent.skills.get("itg:greet") is None

    # Enable + grant the skills component: the capability goes live.
    manager.set_enabled(plugin_id, True)
    manager.set_activation(plugin_id, ("skills",))
    reload_plugins(agent)
    skill = agent.skills.get("itg:greet")
    assert skill is not None and "Hello" in skill.body

    # A fresh manager on the same workspace sees the same persisted grant state.
    reloaded_manager = PluginManager(workspace)
    assert reloaded_manager.enabled_ids() == [plugin_id]

    # Revoking the component grant removes the runtime capability even while
    # the plugin stays enabled.
    manager.set_activation(plugin_id, ())
    reload_plugins(agent)
    assert agent.skills.get("itg:greet") is None

    # Re-granting brings it back — the revoke was a state change, not deletion.
    manager.set_activation(plugin_id, ("skills",))
    reload_plugins(agent)
    assert agent.skills.get("itg:greet") is not None

    # Disabling wins over any leftover grant: nothing loads.
    manager.set_enabled(plugin_id, False)
    reload_plugins(agent)
    assert agent.skills.get("itg:greet") is None
    assert manager.enabled_ids() == []

    # The installed record is still present and can be cleaned up explicitly.
    manager.uninstall(plugin_id)
    assert manager.records() == {}
