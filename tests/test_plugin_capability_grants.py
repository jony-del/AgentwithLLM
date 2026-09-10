"""Regression tests for per-capability plugin grants (stage-1 security boundary).

Covers: dedicated PermissionRequest hook grants, env-access gating of sensitive
host variables at executable boundaries, and public-only default networking for
plugin remote MCP on every trust tier.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import agent_core.plugins as plugins_module
from agent_core.env_security import (
    expand_host_env,
    is_sensitive_env_name,
    referenced_host_variables,
)
from agent_core.memory import MemoryConfig
from agent_core.plugins import PluginError, PluginManager
from agent_core.plugins.components import _load_plugin_mcp
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


def _write_hooks(root: Path, hooks: dict) -> None:
    (root / "hooks").mkdir(exist_ok=True)
    (root / "hooks" / "hooks.json").write_text(
        json.dumps({"hooks": hooks}), encoding="utf-8"
    )


def _make_agent(tmp_path: Path, workspace: Path) -> ReActAgent:
    return ReActAgent(
        FakeProvider(),
        ReActConfig(
            run_dir=str(tmp_path / "runs"),
            session_dir="",
            memory=MemoryConfig(enabled=False),
        ),
        workspace=workspace,
    )


@pytest.fixture
def plugin_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("POLARIS_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    monkeypatch.setenv("POLARIS_SETTINGS_PATH", str(tmp_path / "settings.toml"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return tmp_path, workspace


def _audit_events(manager: PluginManager) -> list[dict]:
    if not manager.audit.path.is_file():
        return []
    return [
        json.loads(line)
        for line in manager.audit.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --- PermissionRequest hook gating --------------------------------------------


def _install_permission_hook_plugin(tmp_path: Path, workspace: Path):
    source = _write_plugin(tmp_path / "source")
    _write_hooks(
        source,
        {
            "PermissionRequest": [
                {
                    "hooks": [
                        {
                            "id": "auto-approve",
                            "type": "prompt",
                            "prompt": "Approve safe requests.",
                        }
                    ]
                }
            ]
        },
    )
    manager = PluginManager(workspace)
    record = manager.install(str(source))
    return manager, record


def test_permission_request_hook_rejected_without_dedicated_grant(plugin_env) -> None:
    tmp_path, workspace = plugin_env
    manager, record = _install_permission_hook_plugin(tmp_path, workspace)
    manager.set_enabled(record.plugin_id, True)
    agent = _make_agent(tmp_path, workspace)

    bundle = manager.build_bundle(agent)

    assert [attr for attr, _ in bundle.hooks if attr == "permission_request_hooks"] == []
    events = _audit_events(manager)
    assert any(
        item["event"] == "permission_hook_rejected"
        and item["detail"]["hook_id"] == "demo@local:auto-approve"
        for item in events
    )


def test_permission_request_hook_loads_with_dedicated_grant(plugin_env) -> None:
    tmp_path, workspace = plugin_env
    manager, record = _install_permission_hook_plugin(tmp_path, workspace)
    manager.set_activation(
        record.plugin_id,
        ("skills", "hooks"),
        allowed_permission_hooks=("demo@local:auto-approve",),
    )
    agent = _make_agent(tmp_path, workspace)

    bundle = manager.build_bundle(agent)

    assert any(attr == "permission_request_hooks" for attr, _ in bundle.hooks)


def test_generic_hook_allowlist_does_not_grant_permission_hooks(plugin_env) -> None:
    tmp_path, workspace = plugin_env
    manager, record = _install_permission_hook_plugin(tmp_path, workspace)
    # Even when the generic allowlist names the hook, the PermissionRequest event
    # loads only through allowed_permission_hooks.
    manager.set_activation(
        record.plugin_id,
        ("skills", "hooks"),
        allowed_hooks=("demo@local:auto-approve",),
    )
    agent = _make_agent(tmp_path, workspace)

    bundle = manager.build_bundle(agent)

    assert [attr for attr, _ in bundle.hooks if attr == "permission_request_hooks"] == []


def test_set_activation_persists_and_revokes_grants(plugin_env) -> None:
    import tomllib

    tmp_path, workspace = plugin_env
    manager, record = _install_permission_hook_plugin(tmp_path, workspace)

    manager.set_activation(
        record.plugin_id,
        ("skills", "hooks"),
        allowed_hooks=("demo@local:other",),
        allowed_permission_hooks=("demo@local:auto-approve",),
        env_access=True,
        network_unrestricted=True,
    )

    parsed = tomllib.loads((workspace / "agent.local.toml").read_text(encoding="utf-8"))
    plugins = parsed["plugins"]
    assert plugins["components"]["demo@local"] == ["skills", "hooks"]
    assert plugins["allowed_permission_hooks"]["demo@local"] == ["demo@local:auto-approve"]
    assert plugins["env_access"] == ["demo@local"]
    assert plugins["network_unrestricted"] == ["demo@local"]

    fresh = PluginManager(workspace)
    assert fresh.permission_hook_selections()["demo@local"] == ("demo@local:auto-approve",)
    assert "demo@local" in fresh.env_access_grants()
    assert "demo@local" in fresh.network_unrestricted_grants()

    # set_activation writes the complete grant state: grants not repeated are revoked.
    manager.set_activation(record.plugin_id, ("skills",), env_access=False)
    assert "demo@local" not in manager.env_access_grants()
    assert "demo@local" not in manager.network_unrestricted_grants()
    assert manager.permission_hook_selections().get("demo@local") == ()


# --- env-access gating ---------------------------------------------------------


def test_plugin_hook_env_blocks_sensitive_variable_without_grant(plugin_env, monkeypatch) -> None:
    tmp_path, workspace = plugin_env
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-leak")
    source = _write_plugin(tmp_path / "source")
    _write_hooks(
        source,
        {
            "SessionStart": [
                {
                    "hooks": [
                        {
                            "type": "prompt",
                            "prompt": "hi",
                            "env": {"PLUGIN_OUT": "${OPENAI_API_KEY}"},
                        }
                    ]
                }
            ]
        },
    )
    manager = PluginManager(workspace)
    record = manager.install(str(source))
    manager.set_enabled(record.plugin_id, True)
    agent = _make_agent(tmp_path, workspace)

    with pytest.raises(PluginError) as excinfo:
        manager.build_bundle(agent)
    assert "OPENAI_API_KEY" in str(excinfo.value)
    assert "sk-must-not-leak" not in str(excinfo.value)


def test_plugin_hook_env_expands_with_grant_and_audit(plugin_env, monkeypatch) -> None:
    tmp_path, workspace = plugin_env
    monkeypatch.setenv("OPENAI_API_KEY", "sk-granted")
    source = _write_plugin(tmp_path / "source")
    _write_hooks(
        source,
        {
            "SessionStart": [
                {
                    "hooks": [
                        {
                            "id": "greeter",
                            "type": "prompt",
                            "prompt": "hi",
                            "env": {"PLUGIN_OUT": "${OPENAI_API_KEY}"},
                        }
                    ]
                }
            ]
        },
    )
    manager = PluginManager(workspace)
    record = manager.install(str(source))
    manager.set_activation(
        record.plugin_id,
        ("skills", "hooks"),
        allowed_hooks=("demo@local:greeter",),
        env_access=True,
    )
    agent = _make_agent(tmp_path, workspace)

    bundle = manager.build_bundle(agent)

    session_hooks = [adapter for attr, adapter in bundle.hooks if attr == "session_start_hooks"]
    assert session_hooks and session_hooks[0].spec.env["PLUGIN_OUT"] == "sk-granted"
    events = _audit_events(manager)
    assert any(
        item["event"] == "env_expansion"
        and item["detail"]["variables"] == ["OPENAI_API_KEY"]
        for item in events
    )


def test_plugin_hook_env_allows_non_sensitive_variables(plugin_env, monkeypatch) -> None:
    tmp_path, workspace = plugin_env
    monkeypatch.setenv("PLUGIN_REGION", "eu")
    source = _write_plugin(tmp_path / "source")
    _write_hooks(
        source,
        {
            "SessionStart": [
                {
                    "hooks": [
                        {
                            "type": "prompt",
                            "prompt": "hi",
                            "env": {"REGION": "${PLUGIN_REGION:-us}"},
                        }
                    ]
                }
            ]
        },
    )
    manager = PluginManager(workspace)
    record = manager.install(str(source))
    manager.set_enabled(record.plugin_id, True)
    agent = _make_agent(tmp_path, workspace)

    bundle = manager.build_bundle(agent)

    session_hooks = [adapter for attr, adapter in bundle.hooks if attr == "session_start_hooks"]
    assert session_hooks and session_hooks[0].spec.env["REGION"] == "eu"


# --- plugin remote MCP public-only --------------------------------------------


class _FakeAudit:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def write(self, event: str, detail=None) -> None:
        self.events.append((event, dict(detail or {})))


def _remote_manifest(**server_extra) -> dict:
    server = {
        "type": "streamable-http",
        "url": "https://mcp.example.com/sse",
        **server_extra,
    }
    return {"name": "demo", "mcpServers": {"remote": server}}


def test_plugin_remote_mcp_defaults_public_only_on_local_tier(tmp_path: Path) -> None:
    root = _write_plugin(tmp_path / "source")
    servers = _load_plugin_mcp(
        root,
        "demo",
        _remote_manifest(),
        tmp_path,
        tmp_path / "data",
        {},
        plugin_id="demo@local",
    )
    assert servers[0].network_policy == "public-only"


def test_plugin_remote_mcp_explicit_default_downgraded_without_grant(tmp_path: Path) -> None:
    root = _write_plugin(tmp_path / "source")
    audit = _FakeAudit()
    servers = _load_plugin_mcp(
        root,
        "demo",
        _remote_manifest(network_policy="default"),
        tmp_path,
        tmp_path / "data",
        {},
        audit=audit,
        plugin_id="demo@local",
    )
    assert servers[0].network_policy == "public-only"
    assert any(event == "mcp_network_policy_downgraded" for event, _ in audit.events)


def test_plugin_remote_mcp_keeps_default_with_network_grant(tmp_path: Path) -> None:
    root = _write_plugin(tmp_path / "source")
    servers = _load_plugin_mcp(
        root,
        "demo",
        _remote_manifest(network_policy="default"),
        tmp_path,
        tmp_path / "data",
        {},
        plugin_id="demo@local",
        network_unrestricted_granted=True,
    )
    assert servers[0].network_policy == "default"


def test_plugin_stdio_mcp_network_policy_untouched(tmp_path: Path) -> None:
    root = _write_plugin(tmp_path / "source")
    manifest = {"name": "demo", "mcpServers": {"local": {"command": "./server"}}}
    servers = _load_plugin_mcp(
        root, "demo", manifest, tmp_path, tmp_path / "data", {}, plugin_id="demo@local"
    )
    assert servers[0].network_policy == "default"


def test_plugin_mcp_headers_block_sensitive_without_grant(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MCP_API_TOKEN", "tok-must-not-leak")
    root = _write_plugin(tmp_path / "source")
    manifest = _remote_manifest(headers={"Authorization": "Bearer ${MCP_API_TOKEN}"})
    with pytest.raises(PluginError) as excinfo:
        _load_plugin_mcp(
            root, "demo", manifest, tmp_path, tmp_path / "data", {}, plugin_id="demo@local"
        )
    assert "MCP_API_TOKEN" in str(excinfo.value)
    assert "tok-must-not-leak" not in str(excinfo.value)


def test_plugin_mcp_env_expands_with_grant(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MCP_API_TOKEN", "tok-granted")
    root = _write_plugin(tmp_path / "source")
    manifest = _remote_manifest(network_policy="default")
    manifest["mcpServers"]["remote"]["env"] = {"AUTH": "${MCP_API_TOKEN}"}
    audit = _FakeAudit()
    servers = _load_plugin_mcp(
        root,
        "demo",
        manifest,
        tmp_path,
        tmp_path / "data",
        {},
        env_access_granted=True,
        audit=audit,
        plugin_id="demo@local",
        network_unrestricted_granted=True,
    )
    assert servers[0].env["AUTH"] == "tok-granted"
    assert any(event == "env_expansion" for event, _ in audit.events)


# --- env_security primitives ----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "sensitive"),
    [
        ("OPENAI_API_KEY", True),
        ("AWS_SECRET_ACCESS_KEY", True),
        ("GITHUB_TOKEN", True),
        ("MY_COOKIE", True),
        ("PATH", False),
        ("PATHEXT", False),
        ("MONKEY", False),
        ("PLUGIN_REGION", False),
    ],
)
def test_is_sensitive_env_name(name: str, sensitive: bool) -> None:
    assert is_sensitive_env_name(name) is sensitive


def test_expand_host_env_blocks_sensitive_only(monkeypatch) -> None:
    monkeypatch.setenv("MY_API_KEY", "secret")
    monkeypatch.setenv("MY_REGION", "eu")
    assert expand_host_env("${MY_API_KEY}", allow_sensitive=True) == "secret"
    with pytest.raises(ValueError) as excinfo:
        expand_host_env("${MY_API_KEY}", allow_sensitive=False)
    assert "MY_API_KEY" in str(excinfo.value)
    assert "secret" not in str(excinfo.value)
    assert expand_host_env("${MY_REGION}", allow_sensitive=False) == "eu"
    assert expand_host_env("${MISSING:-fallback}", allow_sensitive=False) == "fallback"


def test_referenced_host_variables_order() -> None:
    assert referenced_host_variables("a=${B} b=${A:-x} c=${B}") == ("B", "A")


def test_user_configured_mcp_env_expansion_unchanged() -> None:
    # Non-plugin callers keep the legacy behavior: no plugin context, no blocking.
    assert plugins_module._expand_executable_env("${MISSING_VAR:-dflt}") == "dflt"
