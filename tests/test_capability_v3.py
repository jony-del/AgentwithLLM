from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_core.mcp_packages as package_module
from agent_core.capabilities import CapabilitiesConfig
from agent_core.capability_audit import CapabilityAuditLog
from agent_core.mcp.adapter import MCPTool
from agent_core.mcp.client import _minimal_stdio_env
from agent_core.mcp.config import MCPServerConfig
from agent_core.mcp_packages import MCPPackageManager
from agent_core.mcp_registry import MCPRegistryClient, MCPRegistryConfig, RegistryServerRecord
from agent_core.memory import MemoryConfig
from agent_core.models import ToolResult, ToolRisk
from agent_core.plugins import (
    PluginBundle,
    PluginError,
    PluginManager,
    PluginRecord,
    _commit_generation,
    _apply_skill_provenance,
    _prepare_generation,
    plugin_tree_digest,
)
from agent_core.providers import FakeProvider
from agent_core.react import ReActAgent, ReActConfig
from agent_core.skills import Skill, SkillContext
from agent_core.tool_config import LSPServerConfig
from agent_core.tools.base import Tool
from agent_core.tools.registry import DeferredTool, ToolRegistry


def _market(root: Path, name: str, plugins: list[dict] | None = None) -> Path:
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin" / "marketplace.json").write_text(
        json.dumps({"name": name, "plugins": plugins or []}), encoding="utf-8"
    )
    return root


def _agent_config(tmp_path: Path, capabilities: CapabilitiesConfig) -> ReActConfig:
    return ReActConfig(
        run_dir=str(tmp_path / "runs"),
        session_dir="",
        memory=MemoryConfig(enabled=False),
        capabilities=capabilities,
    )


def test_reserved_marketplace_name_cannot_be_rebound(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("POLARIS_PLUGIN_HOME", str(tmp_path / "home"))
    source = _market(tmp_path / "market", "claude-plugins-official")

    with pytest.raises(PluginError, match="reserved marketplace"):
        PluginManager(tmp_path).marketplace_add("claude-plugins-official", str(source))


def test_marketplace_name_cannot_inherit_trust_from_another_source(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("POLARIS_PLUGIN_HOME", str(tmp_path / "home"))
    first = _market(tmp_path / "one", "company")
    second = _market(tmp_path / "two", "company")
    manager = PluginManager(tmp_path)
    manager.marketplace_add("company", str(first))

    with pytest.raises(PluginError, match="identity mismatch"):
        manager.marketplace_add("company", str(second))


def test_legacy_state_is_never_loaded_and_explicit_reset_is_idempotent(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("POLARIS_PLUGIN_HOME", str(home))
    monkeypatch.setenv("POLARIS_SETTINGS_PATH", str(tmp_path / "settings.toml"))
    (home / "installed.json").write_text(
        json.dumps({"malicious@official": {"plugin_id": "malicious@official"}}),
        encoding="utf-8",
    )
    (home / "marketplaces.json").write_text(
        json.dumps({"schema_version": 2, "marketplaces": {}}), encoding="utf-8"
    )
    manager = PluginManager(tmp_path)

    assert manager.state_status().reset_required is True
    assert manager.records() == {}
    manager.reset_state()
    manager.reset_state()
    assert manager.state_status().reset_required is False
    assert json.loads((home / "installed.json").read_text(encoding="utf-8"))["schema_version"] == 3


def test_activation_plan_freezes_bytes_without_marking_plugin_installed(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("POLARIS_PLUGIN_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("POLARIS_SETTINGS_PATH", str(tmp_path / "settings.toml"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    plugin = tmp_path / "plugin"
    (plugin / ".claude-plugin").mkdir(parents=True)
    (plugin / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": "xlsx-helper", "description": "Edit Excel workbooks"}),
        encoding="utf-8",
    )
    (plugin / "skills" / "xlsx").mkdir(parents=True)
    skill_file = plugin / "skills" / "xlsx" / "SKILL.md"
    skill_file.write_text("---\ndescription: Edit xlsx files\n---\nORIGINAL", encoding="utf-8")
    market = _market(
        tmp_path / "market",
        "company",
        [
            {
                "name": "xlsx-helper",
                "description": "Spreadsheet tools",
                "keywords": ["excel", "xlsx"],
                "skills": ["./skills/xlsx/SKILL.md"],
                "source": str(plugin),
                "sha256": plugin_tree_digest(plugin),
            }
        ],
    )
    PluginManager(workspace).marketplace_add("company", str(market))
    config = CapabilitiesConfig(
        mode="autonomous-trusted",
        trusted_marketplaces=("company",),
        auto_components=("skills",),
    )
    agent = ReActAgent(FakeProvider(), _agent_config(tmp_path, config), workspace=workspace)
    match = next(
        item
        for item in agent.capability_manager.search("xlsx", kinds=["skill"])["matches"]
        if item["id"].startswith("component:")
    )
    plan = agent.capability_manager.create_plan(
        match["id"], match["catalog_digest"], sandbox_enabled=False
    )

    assert PluginManager(workspace).records() == {}
    assert Path(plan["artifacts"][0]["path"]).is_file() is False
    assert Path(plan["artifacts"][0]["path"]).is_dir()
    skill_file.write_text("TAMPERED", encoding="utf-8")
    queued = agent.capability_manager.request_plan_activation(
        plan["plan_id"], plan["plan_digest"]
    )
    assert queued["status"] == "queued"
    outcome = agent.capability_manager.commit_pending()[match["id"]]
    assert outcome["status"] == "activated"
    assert "ORIGINAL" in agent.skills.get("xlsx-helper:xlsx").body


def test_community_skill_provenance_forces_fork() -> None:
    skill = Skill("review", "Review", "instructions", context=SkillContext.INLINE)
    record = PluginRecord(
        "review@community", "review", "community", "x", "x",
        trust_tier="community", source_identity="source-1",
    )

    _apply_skill_provenance([skill], record)

    assert skill.context is SkillContext.FORK
    assert skill.source_identity == "source-1"
    assert skill.plugin_id == "review@community"


def test_stdio_mcp_does_not_inherit_unrelated_host_secret(monkeypatch) -> None:
    monkeypatch.setenv("POLARIS_CANARY_SECRET", "do-not-leak")
    monkeypatch.setenv("PATH", "test-path")

    env = _minimal_stdio_env(MCPServerConfig(name="server"))

    assert env["PATH"] == "test-path"
    assert "POLARIS_CANARY_SECRET" not in env


def test_discovered_community_mcp_cannot_lower_its_permission_risk() -> None:
    descriptor = SimpleNamespace(
        name="lookup",
        description="Lookup",
        inputSchema={"type": "object", "properties": {}},
    )
    server = MCPServerConfig(
        name="community",
        risk="read",
        discovered=True,
        trust_tier="community",
    )

    tool = MCPTool(SimpleNamespace(), server, descriptor)

    assert tool.risk is ToolRisk.DANGEROUS


def test_registry_normalizes_every_official_package_type(tmp_path: Path) -> None:
    client = MCPRegistryClient(tmp_path, MCPRegistryConfig(enabled=True))
    response = {
        "servers": [
            {
                "server": {
                    "name": "io.example/everything",
                    "version": "1.2.3",
                    "description": "Everything server",
                    "packages": [
                        {"registryType": kind, "identifier": f"example-{kind}", "version": "1.2.3"}
                        for kind in ("npm", "pypi", "nuget", "oci", "mcpb")
                    ],
                    "remotes": [{"type": "streamable-http", "url": "https://example.com/mcp"}],
                }
            }
        ]
    }

    record = client._normalize(response, 10)[0]

    assert set(record.installable_types) == {"remote", "npm", "pypi", "nuget", "oci", "mcpb"}
    assert record.public()["requires_approval"] is True


def test_registry_resolvers_pin_npm_pypi_nuget_mcpb_and_oci(
    tmp_path: Path, monkeypatch
) -> None:
    npm_body = b"npm-tarball"
    wheel_body = b"wheel"
    nuget_body = b"nuget"
    mcpb_body = b"mcpb"

    def fake_download(url: str, *, maximum: int = 512 * 1024 * 1024) -> bytes:
        del maximum
        if "registry.npmjs.org" in url:
            return json.dumps({"dist": {"tarball": "https://example.com/npm.tgz"}}).encode()
        if url.endswith("npm.tgz"):
            return npm_body
        if "pypi.org" in url:
            return json.dumps(
                {
                    "urls": [
                        {
                            "packagetype": "bdist_wheel",
                            "filename": "example-1.2.3-py3-none-any.whl",
                            "url": "https://example.com/package.whl",
                            "digests": {"sha256": hashlib.sha256(wheel_body).hexdigest()},
                        }
                    ]
                }
            ).encode()
        if url.endswith("package.whl"):
            return wheel_body
        if "nuget.org" in url:
            return nuget_body
        if url.endswith("bundle.mcpb"):
            return mcpb_body
        raise AssertionError(url)

    monkeypatch.setattr(package_module, "_download", fake_download)
    monkeypatch.setattr(package_module.shutil, "which", lambda name: f"/usr/bin/{name}" if name == "podman" else None)
    monkeypatch.setattr(
        package_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout='["registry.example/server@sha256:' + "a" * 64 + '"]',
            stderr="",
        ),
    )
    packages = [
        {"registryType": "npm", "identifier": "example-npm", "version": "1.2.3"},
        {"registryType": "pypi", "identifier": "example-pypi", "version": "1.2.3"},
        {"registryType": "nuget", "identifier": "Example.NuGet", "version": "1.2.3"},
        {
            "registryType": "mcpb",
            "identifier": "https://example.com/bundle.mcpb",
            "url": "https://example.com/bundle.mcpb",
            "version": "1.2.3",
            "fileSha256": hashlib.sha256(mcpb_body).hexdigest(),
        },
        {"registryType": "oci", "identifier": "registry.example/server", "version": "1.2.3"},
    ]
    record = RegistryServerRecord(
        "io.example/server",
        "1.2.3",
        "server",
        packages=tuple(packages),
        installable_types=("npm", "pypi", "nuget", "mcpb", "oci"),
    )
    manager = MCPPackageManager(tmp_path)

    plans = {kind: manager.plan(record, kind) for kind in record.installable_types}

    assert all(plan.installable for plan in plans.values())
    assert all(len(plan.digest) == 64 for plan in plans.values())
    assert all(Path(plan.artifact_path).is_file() for kind, plan in plans.items() if kind != "oci")
    assert "@sha256:" in plans["oci"].artifact_path


class _Tool(Tool):
    description = "test"

    def __init__(self, name: str) -> None:
        self.name = name

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        return ToolResult(self.name, "ok")


def test_registry_group_swap_rejects_duplicate_candidate_without_partial_mutation() -> None:
    registry = ToolRegistry()
    original = _Tool("original")
    registry.register(original)

    with pytest.raises(ValueError, match="Duplicate"):
        registry.replace_group(
            {"original"},
            [_Tool("duplicate")],
            [DeferredTool("duplicate", "x", lambda: _Tool("duplicate"))],
        )

    assert registry.get("original") is original


def _empty_components(*, lsp: list[LSPServerConfig] | None = None) -> dict[str, object]:
    return {
        "lsp": lsp or [],
        "workflows": {},
        "output_styles": {},
        "themes": {},
        "monitors": [],
        "channels": [],
        "user_config": {},
        "bin": [],
        "settings": {},
    }


def test_plugin_generation_hot_adds_and_removes_lsp_tool(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("POLARIS_PLUGIN_HOME", str(tmp_path / "home"))
    agent = ReActAgent(
        FakeProvider(),
        _agent_config(tmp_path, CapabilitiesConfig()),
        workspace=tmp_path,
    )
    with pytest.raises(KeyError):
        agent.registry.get("lsp")

    added = PluginBundle(
        [], [], None, [], {},
        _empty_components(lsp=[LSPServerConfig("test-lsp", "missing-lsp")]),
        ("lsp@test",),
    )
    _commit_generation(agent, _prepare_generation(agent, added))

    assert agent.registry.get("lsp").name == "lsp"
    assert [item.name for item in agent.config.tools.lsp.servers] == ["test-lsp"]

    removed = PluginBundle([], [], None, [], {}, _empty_components(), ())
    _commit_generation(agent, _prepare_generation(agent, removed))

    with pytest.raises(KeyError):
        agent.registry.get("lsp")
    assert agent.config.tools.lsp.servers == []


def test_capability_audit_redacts_secret_values(tmp_path: Path) -> None:
    audit = CapabilityAuditLog(tmp_path)
    audit.write("configuration", {"api_token": "secret", "nested": {"password": "secret"}})

    event = json.loads(audit.path.read_text(encoding="utf-8"))
    assert event["detail"]["api_token"] == "[REDACTED]"
    assert event["detail"]["nested"]["password"] == "[REDACTED]"
