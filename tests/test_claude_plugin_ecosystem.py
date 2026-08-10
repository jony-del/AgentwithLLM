from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tomllib
import urllib.request
import zipfile

import pytest

from agent_core.capabilities import CapabilitiesConfig
from agent_core.hook_adapters import (
    CommandHookAdapter,
    LIFECYCLE_EVENT_ATTRS,
    outcome_from_output,
)
from agent_core.hooks import ExternalHookSpec, HookContext, HookEvent, HookPipeline
from agent_core.local_config import update_toml_table
from agent_core.mcp.config import MCPServerConfig
import agent_core.plugins as plugin_module
from agent_core.plugin_spec import (
    MarketplaceSourceConfig,
    PluginSourceConfig,
    SpecError,
    resolve_plugin_manifest,
)
from agent_core.plugins import PluginError, PluginManager, _extract_zip_bytes, validate_plugin
from agent_core.semver import satisfies, select_highest
from agent_core.workflow_runtime import WorkflowError, WorkflowRuntime


def test_marketplace_and_plugin_sources_normalize_claude_syntax(tmp_path: Path) -> None:
    github = MarketplaceSourceConfig.from_value("anthropics/skills@main")
    assert (github.kind, github.repo, github.ref) == ("github", "anthropics/skills", "main")
    assert MarketplaceSourceConfig.from_value("https://example.com/marketplace.json").kind == "url"
    assert MarketplaceSourceConfig.from_value("https://example.com/plugins.git").kind == "git"
    assert MarketplaceSourceConfig.from_value(
        {"source": "directory", "path": "catalog"}, base_dir=tmp_path
    ).path == str((tmp_path / "catalog").resolve())
    git_market = MarketplaceSourceConfig.from_value(
        {"source": "git", "url": "https://example.com/plugins.git", "path": "catalog", "ref": "stable"}
    )
    assert git_market.path == "catalog"
    inline = MarketplaceSourceConfig.from_value({
        "source": "settings", "name": "inline", "plugins": [
            {"name": "demo", "source": {"source": "github", "repo": "acme/demo"}}
        ],
    })
    assert inline.name == "inline" and inline.plugins[0]["name"] == "demo"

    source = PluginSourceConfig.from_value(
        {"source": "git-subdir", "url": "https://example.com/repo.git", "path": "plugins/a", "ref": "v1"}
    )
    assert source.kind == "git-subdir" and source.path == "plugins/a"
    official_url_subdir = PluginSourceConfig.from_value({
        "source": "url", "url": "https://example.com/repo.git", "path": "plugin/a",
        "sha": "a" * 40,
    })
    assert official_url_subdir.kind == "git-subdir"
    assert PluginSourceConfig.from_value(
        {"source": "npm", "package": "@acme/plugin", "version": "2.1.0"}
    ).immutable
    with pytest.raises(SpecError, match="escapes"):
        PluginSourceConfig.from_value(
            {"source": "git-subdir", "url": "https://example.com/r", "path": "../escape"}
        )


def test_manifestless_root_skill_and_strict_merge(tmp_path: Path) -> None:
    root = tmp_path / "root-skill"
    root.mkdir()
    (root / "SKILL.md").write_text("---\nname: root-review\ndescription: Review\n---\nReview.", encoding="utf-8")

    manifest = validate_plugin(root)
    assert manifest["name"] == "root-skill"

    merged, _warnings = resolve_plugin_manifest(
        root,
        {"name": "catalog-name", "source": "./", "strict": True, "skills": ["./"]},
    )
    assert merged["name"] == "catalog-name"


def test_strict_false_rejects_plugin_component_conflict(tmp_path: Path) -> None:
    root = tmp_path / "plugin"
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": "demo", "skills": "./skills"}), encoding="utf-8"
    )
    with pytest.raises(SpecError, match="strict:false"):
        resolve_plugin_manifest(
            root, {"name": "demo", "source": "./", "strict": False, "agents": "./agents"}
        )


def test_marketplace_name_mismatch_is_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("POLARIS_PLUGIN_HOME", str(tmp_path / "home"))
    market = tmp_path / "market"
    (market / ".claude-plugin").mkdir(parents=True)
    (market / ".claude-plugin" / "marketplace.json").write_text(
        json.dumps({"name": "real-name", "owner": {"name": "A"}, "plugins": []}),
        encoding="utf-8",
    )
    with pytest.raises(PluginError, match="name mismatch"):
        PluginManager(tmp_path).marketplace_add("wrong-name", str(market))


def test_manifestless_marketplace_plugin_installs_with_merged_inventory(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("POLARIS_PLUGIN_HOME", str(tmp_path / "home"))
    plugin = tmp_path / "market" / "plugins" / "demo"
    (plugin / "skills" / "review").mkdir(parents=True)
    (plugin / "skills" / "review" / "SKILL.md").write_text("Review.", encoding="utf-8")
    (plugin / ".lsp.json").write_text(
        json.dumps({"python": {"command": "pyright-langserver", "args": ["--stdio"], "extensionToLanguage": {".py": "python"}}}),
        encoding="utf-8",
    )
    market = tmp_path / "market"
    (market / ".claude-plugin").mkdir(parents=True)
    (market / ".claude-plugin" / "marketplace.json").write_text(
        json.dumps({
            "name": "test-market", "owner": {"name": "A"},
            "plugins": [{"name": "demo", "source": "./plugins/demo", "strict": False}],
        }),
        encoding="utf-8",
    )
    manager = PluginManager(tmp_path)
    manager.marketplace_add("test-market", str(market))
    record = manager.install("demo", "test-market")
    assert record.plugin_id == "demo@test-market"
    assert set(record.components) >= {"skills", "lsp"}


def test_archive_zip_slip_is_rejected(tmp_path: Path) -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    with pytest.raises(PluginError, match="unsafe|escapes"):
        _extract_zip_bytes(payload.getvalue(), tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_mcp_claude_fields_and_transports_parse() -> None:
    config = MCPServerConfig.from_dict(
        "remote",
        {
            "type": "sse", "url": "https://example.com/sse", "headersHelper": "auth-helper",
            "alwaysLoad": True, "timeout": 5000, "oauth": {"scopes": ["read"]},
            "roots": ["${CLAUDE_PROJECT_DIR}"],
        },
    )
    assert config.transport == "sse"
    assert config.headers_helper == "auth-helper"
    assert config.always_load is True and config.timeout == 5


def test_node_semver_ranges_intersect_and_prereleases_opt_in() -> None:
    assert satisfies("2.4.0", "^2.0 >=2.1")
    assert not satisfies("3.0.0", "^2.0")
    assert not satisfies("2.0.0-beta.1", "^2.0.0")
    assert satisfies("2.0.0-beta.1", "^2.0.0-0")
    assert select_highest(["2.0.0", "2.2.0", "3.0.0"], ["^2.0", ">=2.1"]) == "2.2.0"


def test_capabilities_config_parses_typed_marketplaces() -> None:
    config = CapabilitiesConfig.from_dict({
        "mode": "autonomous-trusted",
        "trusted_marketplaces": ["official"],
        "marketplaces": {"official": {"kind": "github", "repo": "anthropics/skills", "ref": "main"}},
    })
    assert config.marketplaces is not None
    assert config.marketplaces["official"].repo == "anthropics/skills"


@pytest.mark.asyncio
async def test_tool_hooks_parse_deny_and_input_output_rewrites() -> None:
    class Pre:
        async def on_pre_tool(self, ctx: HookContext):
            assert ctx.trigger == "shell"
            return outcome_from_output(json.dumps({
                "hookSpecificOutput": {"updatedInput": {"command": "safe"}}
            }), 0)

    class Post:
        async def on_post_tool(self, ctx: HookContext):
            return outcome_from_output(json.dumps({
                "hookSpecificOutput": {"updatedMCPToolOutput": "filtered"}
            }), 0)

    pipeline = HookPipeline(
        external_pre_tool_hooks=[Pre()], external_post_tool_hooks=[Post()]
    )
    context = HookContext(
        event=HookEvent.PRE_TOOL_USE, messages=[], trigger="shell", detail={}
    )
    assert (await pipeline.run_external_pre_tool(context)).metadata["updated_input"] == {
        "command": "safe"
    }
    assert (await pipeline.run_external_post_tool(context)).metadata["updated_output"] == "filtered"
    denied = outcome_from_output(json.dumps({
        "hookSpecificOutput": {
            "permissionDecision": "deny", "permissionDecisionReason": "unsafe"
        }
    }), 0)
    assert denied.block and denied.decision == "deny" and denied.reason == "unsafe"
    assert LIFECYCLE_EVENT_ATTRS["PreToolUse"] == "external_pre_tool_hooks"
    assert LIFECYCLE_EVENT_ATTRS["FileChanged"] == "unhandled_hooks"


def test_hook_if_uses_permission_rule_and_claude_tool_alias() -> None:
    adapter = CommandHookAdapter(
        ExternalHookSpec(
            event="PreToolUse", type="command", condition="Bash(git *)"
        ),
        logger=object(),  # logger is only touched when the transport executes
    )
    assert adapter._matches(HookContext(
        event=HookEvent.PRE_TOOL_USE,
        messages=[],
        trigger="shell_command",
        detail={"tool_input": {"command": "git status"}},
    ))
    assert not adapter._matches(HookContext(
        event=HookEvent.PRE_TOOL_USE,
        messages=[],
        trigger="shell_command",
        detail={"tool_input": {"command": "npm test"}},
    ))


def test_dependency_tag_uses_highest_intersection(tmp_path: Path, monkeypatch) -> None:
    manager = PluginManager(tmp_path)
    monkeypatch.setattr(
        manager,
        "marketplace_sources",
        lambda: {
            "official": MarketplaceSourceConfig(
                "git", url="https://example.com/catalog.git"
            )
        },
    )
    monkeypatch.setattr(
        manager,
        "_marketplace_entry",
        lambda name, marketplace: {"name": name, "source": "./plugins/demo"},
    )
    commits = {"2.0.0": "1" * 40, "2.4.0": "2" * 40, "3.0.0": "3" * 40}
    monkeypatch.setattr(
        plugin_module,
        "_git_output",
        lambda args: "".join(
            f"{commit}\trefs/tags/demo--v{version}\n"
            for version, commit in commits.items()
        ),
    )
    assert manager._dependency_tag("demo", "official", ["^2.0", ">=2.1"]) == (
        "2.4.0",
        commits["2.4.0"],
    )


def test_dependency_cycle_is_reported_without_installing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("POLARIS_PLUGIN_HOME", str(tmp_path / "home"))
    market = tmp_path / "market"
    (market / ".claude-plugin").mkdir(parents=True)
    (market / ".claude-plugin" / "marketplace.json").write_text(
        json.dumps({
            "name": "cycle-market",
            "owner": {"name": "A"},
            "plugins": [
                {"name": "a", "source": "./a", "version": "1.0.0", "dependencies": ["b"]},
                {"name": "b", "source": "./b", "version": "1.0.0", "dependencies": ["a"]},
            ],
        }),
        encoding="utf-8",
    )
    manager = PluginManager(tmp_path)
    manager.marketplace_add("cycle-market", str(market))
    with pytest.raises(PluginError, match="dependency cycle"):
        manager.dependency_plan("a", "cycle-market")
    assert manager.records() == {}


def test_plugin_config_ids_are_quoted_as_toml_keys(tmp_path: Path) -> None:
    target = tmp_path / "settings.toml"
    update_toml_table(target, "plugin_configs", {"demo@official": {"endpoint": "x"}})
    parsed = tomllib.loads(target.read_text(encoding="utf-8"))
    assert parsed["plugin_configs"]["demo@official"]["endpoint"] == "x"


def test_user_config_is_typed_and_sensitive_values_stay_environment_backed(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("POLARIS_PLUGIN_HOME", str(tmp_path / "home"))
    settings = tmp_path / "settings.toml"
    monkeypatch.setenv("POLARIS_SETTINGS_PATH", str(settings))
    monkeypatch.setattr(plugin_module.secret_store, "available", lambda: False)
    source = tmp_path / "source"
    (source / ".claude-plugin").mkdir(parents=True)
    (source / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({
            "name": "configured",
            "userConfig": {
                "token": {
                    "type": "string", "title": "Token", "description": "API token",
                    "sensitive": True, "required": True,
                },
                "retries": {
                    "type": "number", "title": "Retries", "description": "Retry count",
                    "min": 1, "max": 5, "default": 2,
                },
            },
        }),
        encoding="utf-8",
    )
    manager = PluginManager(tmp_path)
    record = manager.install(str(source))
    with pytest.raises(PluginError, match="environment variable"):
        manager.configure(record.plugin_id, {"token": "plaintext"})
    manager.configure(record.plugin_id, {"token": "${PLUGIN_TEST_TOKEN}", "retries": "4"})
    monkeypatch.setenv("PLUGIN_TEST_TOKEN", "runtime-secret")
    resolved, public = manager.configured_options(record.plugin_id, record.manifest or {})
    assert resolved == {"token": "runtime-secret", "retries": 4}
    assert public == {"retries": 4}
    on_disk = settings.read_text(encoding="utf-8")
    assert "runtime-secret" not in on_disk and "${PLUGIN_TEST_TOKEN}" in on_disk


def test_sensitive_user_config_uses_system_store_when_available(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("POLARIS_PLUGIN_HOME", str(tmp_path / "home"))
    settings = tmp_path / "settings.toml"
    monkeypatch.setenv("POLARIS_SETTINGS_PATH", str(settings))
    secrets: dict[str, str] = {}
    monkeypatch.setattr(plugin_module.secret_store, "available", lambda: True)
    monkeypatch.setattr(plugin_module.secret_store, "put", secrets.__setitem__)
    monkeypatch.setattr(plugin_module.secret_store, "get", secrets.get)
    source = tmp_path / "source"
    (source / ".claude-plugin").mkdir(parents=True)
    (source / ".claude-plugin" / "plugin.json").write_text(json.dumps({
        "name": "secure-config",
        "userConfig": {
            "token": {
                "type": "string", "title": "Token", "description": "API token",
                "sensitive": True, "required": True,
            }
        },
    }), encoding="utf-8")
    manager = PluginManager(tmp_path)
    record = manager.install(str(source))
    manager.configure(record.plugin_id, {"token": "super-secret"})
    resolved, public = manager.configured_options(record.plugin_id, record.manifest or {})
    assert resolved == {"token": "super-secret"} and public == {}
    assert "super-secret" not in settings.read_text(encoding="utf-8")
    assert "keychain://Polaris/plugin-config/" in settings.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_workflow_runtime_isolated_pipeline_and_schema() -> None:
    calls: list[str] = []

    async def factory(prompt: str, preset: str, model: str | None) -> str:
        calls.append(prompt)
        if "JSON Schema" in prompt:
            return '{"files":["a.py","b.py"]}'
        return prompt.rsplit(" ", 1)[-1]

    source = """
export const meta = {name: 'audit'}
const found = await agent('list files', {
  schema: {type:'object', required:['files'], properties:{files:{type:'array', items:{type:'string'}}}}
})
return await pipeline(found.files, file => agent(`review ${file}`))
"""
    result = await WorkflowRuntime().run(source, {}, factory, timeout=10)
    assert result == ["a.py", "b.py"]
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_workflow_runtime_rejects_module_loading() -> None:
    async def factory(prompt: str, preset: str, model: str | None) -> str:
        return "unused"

    with pytest.raises(WorkflowError, match="module loading"):
        await WorkflowRuntime().run("await import('node:fs')", {}, factory, timeout=10)


@pytest.mark.skipif(
    os.getenv("POLARIS_ONLINE_MARKETPLACE_TEST") != "1",
    reason="set POLARIS_ONLINE_MARKETPLACE_TEST=1 for live catalog compatibility",
)
def test_live_anthropic_catalogs_normalize() -> None:
    catalogs = {
        "claude-plugins-official": "anthropics/claude-plugins-official",
        "anthropic-agent-skills": "anthropics/skills",
        "knowledge-work-plugins": "anthropics/knowledge-work-plugins",
    }
    for expected_name, repo in catalogs.items():
        url = f"https://raw.githubusercontent.com/{repo}/main/.claude-plugin/marketplace.json"
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - fixed HTTPS hosts
            manifest = json.load(response)
        assert manifest["name"] == expected_name
        for entry in manifest["plugins"]:
            PluginSourceConfig.from_value(entry["source"])
