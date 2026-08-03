from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_core.capabilities import CapabilitiesConfig
from agent_core.memory import MemoryConfig
from agent_core.mcp.adapter import MCPTool
from agent_core.mcp.config import MCPServerConfig
from agent_core.models import LLMResult, ToolCall
from agent_core.plugins import PluginError, PluginManager, plugin_tree_digest, reload_plugins
from agent_core.providers import FakeProvider
from agent_core.react import ReActAgent, ReActConfig
from agent_core.storage import JSONLRunLogger
from agent_core.tools.registry import ToolRegistry


def _config(tmp_path: Path, capabilities: CapabilitiesConfig) -> ReActConfig:
    return ReActConfig(
        run_dir=str(tmp_path / "runs"),
        session_dir="",
        memory=MemoryConfig(enabled=False),
        capabilities=capabilities,
    )


def _write_plugin(
    root: Path,
    *,
    name: str = "lint-helper",
    hooks: bool = False,
    hook_type: str = "command",
) -> Path:
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin" / "plugin.json").write_text(
        json.dumps(
            {
                "name": name,
                "version": "1.0.0",
                "description": "Review Python lint failures",
                "keywords": ["lint", "python", "检查"],
            }
        ),
        encoding="utf-8",
    )
    (root / "skills").mkdir()
    (root / "skills" / "review.md").write_text(
        "---\ndescription: Review lint output\n---\nReview $ARGUMENTS",
        encoding="utf-8",
    )
    if hooks:
        (root / "hooks").mkdir()
        hook = (
            {"id": "observe", "type": "http", "url": "http://127.0.0.1:8765/hook"}
            if hook_type == "http"
            else {"id": "observe", "type": "command", "command": "echo hook"}
        )
        (root / "hooks" / "hooks.json").write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionStart": [{"hooks": [hook]}]
                    }
                }
            ),
            encoding="utf-8",
        )
    return root


def _marketplace(
    root: Path,
    source: Path,
    *,
    sha256: str | None,
    hooks: bool = False,
    commit: str | None = None,
) -> Path:
    (root / ".claude-plugin").mkdir(parents=True)
    entry = {
        "name": "lint-helper",
        "description": "Find and fix Python lint problems",
        "keywords": ["lint", "python", "检查"],
        "components": ["skills", "hooks"] if hooks else ["skills"],
        "source": str(source),
    }
    if sha256 is not None:
        entry["sha256"] = sha256
    if commit is not None:
        entry["commit"] = commit
    (root / ".claude-plugin" / "marketplace.json").write_text(
        json.dumps({"plugins": [entry]}),
        encoding="utf-8",
    )
    return root


def _trusted_agent(
    tmp_path: Path,
    monkeypatch,
    *,
    pinned: bool = True,
    hooks: bool = False,
    hook_type: str = "command",
    auto_components: tuple[str, ...] = ("skills",),
    allowed_hooks: tuple[str, ...] = (),
):
    monkeypatch.setenv("POLARIS_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    monkeypatch.setenv("POLARIS_SETTINGS_PATH", str(tmp_path / "settings.toml"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = _write_plugin(tmp_path / "source", hooks=hooks, hook_type=hook_type)
    sha256 = plugin_tree_digest(source) if pinned else None
    market = _marketplace(tmp_path / "market", source, sha256=sha256, hooks=hooks)
    PluginManager(workspace).marketplace_add("company", str(market))
    config = CapabilitiesConfig(
        mode="autonomous-trusted",
        trusted_marketplaces=("company",),
        auto_components=auto_components,
        allowed_hooks=allowed_hooks,
    )
    agent = ReActAgent(FakeProvider(), _config(tmp_path, config), workspace=workspace)
    return agent, sha256


def test_search_indexes_local_skills_and_supports_chinese(tmp_path: Path) -> None:
    skill_dir = tmp_path / ".polaris" / "skills" / "security-cn"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\ndescription: 执行安全审查和漏洞检查\n---\n检查 $ARGUMENTS",
        encoding="utf-8",
    )
    agent = ReActAgent(
        FakeProvider(),
        _config(tmp_path, CapabilitiesConfig()),
        workspace=tmp_path,
    )

    result = agent.capability_manager.search("安全审查", kinds=["skill"])

    assert result["catalog_digest"]
    assert any(item["kind"] == "skill" for item in result["matches"])


def test_trusted_pinned_plugin_activates_component_scoped(tmp_path: Path, monkeypatch) -> None:
    agent, sha256 = _trusted_agent(tmp_path, monkeypatch)
    search = agent.capability_manager.search("lint", kinds=["plugin"])
    match = search["matches"][0]
    assert match["integrity"] == f"sha256:{sha256}"
    allowed, _ = agent.capability_manager.authorization(
        match["id"], match["catalog_digest"], sandbox_enabled=False
    )
    assert allowed is True  # skill-only plugins need no process sandbox

    agent.capability_manager.request_activation(match["id"], match["catalog_digest"])
    outcome = agent.capability_manager.commit_pending()[match["id"]]

    assert outcome["status"] == "activated"
    assert agent.skills.get("lint-helper:review") is not None
    manager = PluginManager(agent.session.workspace)
    assert manager.enabled_ids() == ["lint-helper@company"]
    assert manager.component_selections()["lint-helper@company"] == ("skills",)


async def test_unpinned_marketplace_entry_is_denied(tmp_path: Path, monkeypatch) -> None:
    agent, _ = _trusted_agent(tmp_path, monkeypatch, pinned=False)
    match = agent.capability_manager.search("lint", kinds=["plugin"])["matches"][0]

    allowed, reason = agent.capability_manager.authorization(
        match["id"], match["catalog_digest"], sandbox_enabled=True
    )

    assert allowed is False
    assert "pinned" in reason


def test_mutable_git_ref_is_not_accepted_as_an_integrity_pin(tmp_path: Path, monkeypatch) -> None:
    agent, _ = _trusted_agent(tmp_path, monkeypatch, pinned=False)
    manifest_path = tmp_path / "market" / ".claude-plugin" / "marketplace.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["plugins"][0]["commit"] = "main"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    match = agent.capability_manager.search("lint", kinds=["plugin"])["matches"][0]

    allowed, reason = agent.capability_manager.authorization(
        match["id"], match["catalog_digest"], sandbox_enabled=True
    )

    assert allowed is False
    assert "full immutable object id" in reason


def test_stale_catalog_digest_is_rejected(tmp_path: Path, monkeypatch) -> None:
    agent, _ = _trusted_agent(tmp_path, monkeypatch)
    match = agent.capability_manager.search("lint", kinds=["plugin"])["matches"][0]

    allowed, reason = agent.capability_manager.authorization(
        match["id"], "0" * 64, sandbox_enabled=True
    )

    assert allowed is False
    assert "stale" in reason


def test_integrity_failure_keeps_old_generation_and_activation_state(
    tmp_path: Path, monkeypatch
) -> None:
    agent, _ = _trusted_agent(tmp_path, monkeypatch)
    original_skills = {skill.name for skill in agent.skills.list()}
    search = agent.capability_manager.search("lint", kinds=["plugin"])
    match = search["matches"][0]
    # Alter the source after the trusted catalog digest was created.
    source_skill = tmp_path / "source" / "skills" / "review.md"
    source_skill.write_text("tampered", encoding="utf-8")

    agent.capability_manager.request_activation(match["id"], match["catalog_digest"])
    outcome = agent.capability_manager.commit_pending()[match["id"]]

    assert outcome["status"] == "failed"
    assert {skill.name for skill in agent.skills.list()} == original_skills
    assert PluginManager(agent.session.workspace).enabled_ids() == []


def test_tampered_installed_cache_is_rejected_without_replacing_live_generation(
    tmp_path: Path, monkeypatch
) -> None:
    agent, _ = _trusted_agent(tmp_path, monkeypatch)
    match = agent.capability_manager.search("lint", kinds=["plugin"])["matches"][0]
    agent.capability_manager.request_activation(match["id"], match["catalog_digest"])
    assert agent.capability_manager.commit_pending()[match["id"]]["status"] == "activated"
    live_skill = agent.skills.get("lint-helper:review")
    assert live_skill is not None
    manager = PluginManager(agent.session.workspace)
    installed = manager.records()["lint-helper@company"]
    (Path(installed.path) / "skills" / "review.md").write_text(
        "tampered cache", encoding="utf-8"
    )

    with pytest.raises(PluginError, match="integrity verification"):
        reload_plugins(agent)

    assert agent.skills.get("lint-helper:review") is live_skill


def test_hook_component_is_not_enabled_by_safe_default(tmp_path: Path, monkeypatch) -> None:
    agent, _ = _trusted_agent(tmp_path, monkeypatch, hooks=True)
    match = agent.capability_manager.search("lint", kinds=["plugin"])["matches"][0]
    before = len(agent.hooks.session_start_hooks)

    agent.capability_manager.request_activation(match["id"], match["catalog_digest"])
    outcome = agent.capability_manager.commit_pending()[match["id"]]

    assert outcome["status"] == "activated"
    assert len(agent.hooks.session_start_hooks) == before


def test_exactly_allowlisted_loopback_hook_can_be_activated(tmp_path: Path, monkeypatch) -> None:
    plugin_id = "lint-helper@company"
    agent, _ = _trusted_agent(
        tmp_path,
        monkeypatch,
        hooks=True,
        hook_type="http",
        auto_components=("skills", "hooks"),
        allowed_hooks=(f"{plugin_id}:observe",),
    )
    match = agent.capability_manager.search("lint", kinds=["plugin"])["matches"][0]
    before = len(agent.hooks.session_start_hooks)

    agent.capability_manager.request_activation(match["id"], match["catalog_digest"])
    outcome = agent.capability_manager.commit_pending()[match["id"]]

    assert outcome["status"] == "activated"
    assert len(agent.hooks.session_start_hooks) == before + 1
    assert PluginManager(agent.session.workspace).hook_selections()[plugin_id] == (
        f"{plugin_id}:observe",
    )


def test_installed_plugin_without_current_marketplace_pin_is_not_trusted(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("POLARIS_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    monkeypatch.setenv("POLARIS_SETTINGS_PATH", str(tmp_path / "settings.toml"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = _write_plugin(tmp_path / "source")
    market = _marketplace(
        tmp_path / "market",
        source,
        sha256=plugin_tree_digest(source),
    )
    manager = PluginManager(workspace)
    manager.marketplace_add("company", str(market))
    manager.install(str(source), "company")
    (market / ".claude-plugin" / "marketplace.json").write_text(
        json.dumps({"plugins": []}), encoding="utf-8"
    )
    agent = ReActAgent(
        FakeProvider(),
        _config(
            tmp_path,
            CapabilitiesConfig(
                mode="autonomous-trusted",
                trusted_marketplaces=("company",),
            ),
        ),
        workspace=workspace,
    )
    match = agent.capability_manager.search("lint", kinds=["plugin"])["matches"][0]

    allowed, reason = agent.capability_manager.authorization(
        match["id"], match["catalog_digest"], sandbox_enabled=True
    )

    assert allowed is False
    assert "pinned" in reason


class _Descriptor:
    name = "lookup"
    description = "Look up package metadata"
    inputSchema = {"type": "object", "properties": {}}


class _Manager:
    def call_tool(self, server, tool, arguments):  # pragma: no cover - activation only
        raise AssertionError((server, tool, arguments))


def test_connected_mcp_tool_is_searchable_and_activated_lazily(tmp_path: Path) -> None:
    registry = ToolRegistry()
    tool = MCPTool(_Manager(), MCPServerConfig(name="packages", risk="read"), _Descriptor())
    registry.register_deferred(
        tool.name,
        tool.description,
        lambda: tool,
        metadata={"kind": "mcp", "server": "packages", "remote": "lookup"},
    )
    agent = ReActAgent(
        FakeProvider(),
        _config(tmp_path, CapabilitiesConfig()),
        tools=registry,
        workspace=tmp_path,
    )
    match = agent.capability_manager.search("package metadata", kinds=["mcp"])["matches"][0]
    assert match["state"] == "deferred"

    agent.capability_manager.request_activation(match["id"], match["catalog_digest"])
    outcome = agent.capability_manager.commit_pending()[match["id"]]

    assert outcome["status"] == "activated"
    assert registry.get("packages__lookup") is tool


class _BoundaryProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.capability_id = ""
        self.catalog_digest = ""
        self.second_turn_tools: list[str] = []
        self.second_turn_tool_messages: list[str] = []

    async def complete(self, messages, tools, config, stream=None, should_cancel=None):
        self.calls += 1
        tool_names = [str(tool.get("name")) for tool in tools]
        if self.calls == 1:
            assert "packages__lookup" not in tool_names
            return LLMResult(
                "activate the catalog result",
                tool_calls=[
                    ToolCall(
                        "capability_activate",
                        {"id": self.capability_id, "catalog_digest": self.catalog_digest},
                        id="capability-1",
                    )
                ],
                stop_reason="tool_use",
            )
        self.second_turn_tools = tool_names
        self.second_turn_tool_messages = [
            message.content for message in messages if message.role == "tool"
        ]
        return LLMResult("done", stop_reason="end")


async def test_react_commits_activation_before_the_next_model_call(tmp_path: Path) -> None:
    registry = ReActAgent.default_registry()
    tool = MCPTool(_Manager(), MCPServerConfig(name="packages", risk="read"), _Descriptor())
    registry.register_deferred(
        tool.name,
        tool.description,
        lambda: tool,
        metadata={"kind": "mcp", "server": "packages", "remote": "lookup"},
    )
    provider = _BoundaryProvider()
    logger = JSONLRunLogger(tmp_path / "runs", run_id="boundary")
    agent = ReActAgent(
        provider,
        _config(tmp_path, CapabilitiesConfig()),
        tools=registry,
        logger=logger,
        workspace=tmp_path,
    )
    match = agent.capability_manager.search("package metadata", kinds=["mcp"])["matches"][0]
    provider.capability_id = match["id"]
    provider.catalog_digest = match["catalog_digest"]

    result = await agent.run("use package metadata")

    assert result.answer == "done"
    assert "packages__lookup" in provider.second_turn_tools
    assert any('"status": "activated"' in item for item in provider.second_turn_tool_messages)
    events = [json.loads(line)["event"] for line in logger.path.read_text(encoding="utf-8").splitlines()]
    assert "capability_activation" in events
