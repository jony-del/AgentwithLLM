"""Runtime capability discovery and trusted, boundary-safe activation.

The catalog intentionally exposes identifiers, never executable source parameters.  A
model can search metadata and request activation of a returned id; installation and
generation swaps remain policy-owned host operations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from agent_core.capability_security import (
    ActivationPlan,
    ResolvedArtifact,
    TrustTier,
    marketplace_source_id,
    plugin_trust_tier,
)
from agent_core.mcp_registry import MCPRegistryClient, MCPRegistryConfig, RegistryServerRecord
from agent_core.mcp_packages import MCPPackageManager, MCPPackagePlan
from agent_core.mcp import MCPAdapter, MCPClientManager, MCPConfig
from agent_core.plugin_spec import PluginSourceConfig, SpecError
from agent_core.plugins import (
    PluginError,
    PluginManager,
    MarketplaceSourceConfig,
    activate_plugin,
    is_git_commit_pin,
    is_safe_plugin_name,
    is_sha256_pin,
    plugin_tree_digest,
)
from agent_core.tools.base import ExecutionScope, Tool
from agent_core.tools.registry import DeferredTool

if TYPE_CHECKING:
    from agent_core.react import ReActAgent


CapabilityKind = Literal["skill", "mcp", "plugin"]
CapabilityState = Literal["active", "deferred", "installed", "available"]
_KINDS = frozenset({"skill", "mcp", "plugin"})
_COMPONENTS = frozenset({
    "skills", "agents", "mcp", "lsp", "hooks", "workflows", "monitors",
    "channels", "output-styles", "themes", "user-config", "bin", "settings",
})
_CONFIRM_COMPONENTS = frozenset({
    "hooks", "workflows", "monitors", "channels", "themes", "bin", "settings",
})
_CONTROL_TAG = re.compile(r"(?i)</?(?:system-reminder|tool_output_ref|untrusted-data)[^>]*>")
_WORD = re.compile(r"[\w.+:@/-]+", re.UNICODE)
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_AUTO_DISCOVERY_HINT = re.compile(
    r"(?i)(?:\.(?:xlsx?|csv|docx?|pdf|pptx?|parquet|sqlite)\b|"
    r"\b(?:skill|plugin|mcp|spreadsheet|excel|powerpoint|browser|database|slack|github|jira)\b|"
    r"(?:技能|插件|工具|电子表格|表格|文档|幻灯片|浏览器|数据库|安装|搜索社区))"
)
logger = logging.getLogger(__name__)


def _configuration_env_name(value: str) -> str:
    text = value.strip()
    if text.startswith("${") and text.endswith("}"):
        text = text[2:-1].split(":-", 1)[0]
    return text


@dataclass(slots=True)
class CapabilitiesConfig:
    """Policy for model-facing discovery and autonomous trusted activation."""

    mode: str = "local"  # disabled | local | autonomous-trusted
    trusted_marketplaces: tuple[str, ...] = ()
    require_integrity: bool = True
    auto_components: tuple[str, ...] = ("skills", "agents", "mcp")
    allowed_hooks: tuple[str, ...] = ()
    max_results: int = 8
    marketplace_refresh_ttl_seconds: int = 86_400
    marketplaces: dict[str, MarketplaceSourceConfig] = field(default_factory=dict)
    auto_discovery: bool = True
    auto_install_tiers: tuple[str, ...] = (TrustTier.ANTHROPIC_FIRST_PARTY.value,)
    auto_install_max_risk: str = "low"
    remote_skill_policy: str = "tiered"
    legacy_state_policy: str = "prompt_interactive"
    plan_ttl_seconds: int = 900
    mcp_registry: MCPRegistryConfig = field(default_factory=MCPRegistryConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CapabilitiesConfig":
        raw = data or {}
        mode = str(raw.get("mode", "local")).strip().lower()
        if mode not in {"disabled", "local", "autonomous-trusted"}:
            mode = "local"

        def strings(key: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
            value = raw.get(key, default)
            if not isinstance(value, (list, tuple)):
                return default
            return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))

        components = tuple(item for item in strings("auto_components", cls().auto_components) if item in _COMPONENTS)
        try:
            max_results = max(1, min(20, int(raw.get("max_results", 8))))
        except (TypeError, ValueError):
            max_results = 8
        try:
            refresh = max(0, int(raw.get("marketplace_refresh_ttl_seconds", 86_400)))
        except (TypeError, ValueError):
            refresh = 86_400
        try:
            plan_ttl = max(60, min(3600, int(raw.get("plan_ttl_seconds", 900))))
        except (TypeError, ValueError):
            plan_ttl = 900
        marketplace_sources: dict[str, MarketplaceSourceConfig] = {}
        configured = raw.get("marketplaces", {})
        if isinstance(configured, dict):
            for name, value in configured.items():
                if not is_safe_plugin_name(str(name)):
                    continue
                try:
                    marketplace_sources[str(name)] = MarketplaceSourceConfig.from_value(value)
                except (PluginError, ValueError):
                    continue
        return cls(
            mode=mode,
            trusted_marketplaces=strings("trusted_marketplaces"),
            require_integrity=bool(raw.get("require_integrity", True)),
            auto_components=components,
            allowed_hooks=strings("allowed_hooks"),
            max_results=max_results,
            marketplace_refresh_ttl_seconds=refresh,
            marketplaces=marketplace_sources,
            auto_discovery=bool(raw.get("auto_discovery", True)),
            auto_install_tiers=strings(
                "auto_install_tiers", (TrustTier.ANTHROPIC_FIRST_PARTY.value,)
            ),
            auto_install_max_risk=str(raw.get("auto_install_max_risk", "low")),
            remote_skill_policy=str(raw.get("remote_skill_policy", "tiered")),
            legacy_state_policy=str(raw.get("legacy_state_policy", "prompt_interactive")),
            plan_ttl_seconds=plan_ttl,
            mcp_registry=MCPRegistryConfig.from_dict(
                raw.get("mcp_registry") if isinstance(raw.get("mcp_registry"), dict) else None,
                default_enabled=True,
            ),
        )

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"

    @property
    def autonomous(self) -> bool:
        return self.mode == "autonomous-trusted"


@dataclass(frozen=True, slots=True)
class CapabilityRecord:
    id: str
    kind: CapabilityKind
    name: str
    description: str
    state: CapabilityState
    source: str
    keywords: tuple[str, ...] = ()
    marketplace: str = ""
    plugin_id: str = ""
    tool_name: str = ""
    components: tuple[str, ...] = ()
    sha256: str = ""
    commit: str = ""
    snapshot: str = ""
    version: str = ""
    risk: str = "low"
    dependencies: tuple[str, ...] = ()
    trust_tier: str = TrustTier.LOCAL_USER_DECLARED.value
    source_identity: str = ""
    publisher: str = ""
    homepage: str = ""
    installable: bool = True
    requires_approval: bool = False
    package_types: tuple[str, ...] = ()

    @property
    def integrity(self) -> str:
        if self.sha256:
            return f"sha256:{self.sha256}"
        if self.commit:
            return f"git:{self.commit}"
        if self.snapshot:
            return f"marketplace:{self.snapshot}"
        return ""

    def public(self, catalog_digest: str) -> dict[str, Any]:
        invoke: dict[str, str] | None = None
        if self.kind == "skill" and self.state == "active":
            invoke = {"tool": "skill", "command": self.name}
        elif self.kind == "mcp" and self.state == "active":
            invoke = {"tool": self.tool_name}
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "description": self.description,
            "state": self.state,
            "source": self.source,
            "marketplace": self.marketplace or None,
            "components": list(self.components),
            "risk": self.risk,
            "version": self.version or None,
            "dependencies": list(self.dependencies),
            "integrity": self.integrity or None,
            "catalog_digest": catalog_digest,
            "invoke": invoke,
            "trust_tier": self.trust_tier,
            "source_identity": self.source_identity or None,
            "publisher": self.publisher or None,
            "homepage": self.homepage or None,
            "installable": self.installable,
            "requires_approval": self.requires_approval,
            "package_types": list(self.package_types),
        }


def _bounded_text(value: object, limit: int = 1000) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = "".join(char if char in "\t\n" or ord(char) >= 32 else " " for char in text)
    text = _CONTROL_TAG.sub("[control tag removed]", text)
    return text[:limit].strip()


def _terms(text: str) -> set[str]:
    normalized = _bounded_text(text, 4000).casefold()
    result = {match.group(0) for match in _WORD.finditer(normalized)}
    for match in _CJK.finditer(normalized):
        value = match.group(0)
        result.add(value)
        result.update(value[index : index + 2] for index in range(max(0, len(value) - 1)))
    return result


def _score(record: CapabilityRecord, query: str) -> int:
    query_text = _bounded_text(query, 500).casefold()
    if not query_text:
        return 0
    query_terms = _terms(query_text)
    name = f"{record.id} {record.name}".casefold()
    keywords = " ".join(record.keywords).casefold()
    description = record.description.casefold()
    score = 100 if query_text in name else 0
    for term in query_terms:
        if term in name:
            score += 20
        if term in keywords:
            score += 10
        if term in description:
            score += 3
    return score


def _component_candidates(entry: dict[str, Any], plugin_name: str) -> list[tuple[str, str, str]]:
    """Return (component, display-name, searchable-description) without fetching code."""

    fields = {
        "skills": "skills", "commands": "skills", "agents": "agents",
        "mcpServers": "mcp", "lspServers": "lsp", "workflows": "workflows",
        "outputStyles": "output-styles", "themes": "themes", "monitors": "monitors",
    }
    result: list[tuple[str, str, str]] = []
    for field_name, component in fields.items():
        raw = entry.get(field_name)
        if isinstance(raw, dict):
            for name, value in raw.items():
                result.append((component, str(name), _bounded_text(value, 500)))
        elif isinstance(raw, list):
            for value in raw:
                if isinstance(value, str):
                    name = Path(value.rstrip("/.")).stem or plugin_name
                    result.append((component, name, value))
                elif isinstance(value, dict):
                    name = str(value.get("name") or value.get("id") or plugin_name)
                    result.append((component, name, _bounded_text(value, 500)))
        elif isinstance(raw, str):
            name = Path(raw.rstrip("/.")).stem or plugin_name
            result.append((component, name, raw))
    return result


class CapabilityCatalog:
    """A bounded, deterministic snapshot assembled from the agent's live registries."""

    def __init__(self, agent: "ReActAgent", config: CapabilitiesConfig) -> None:
        self.agent = agent
        self.config = config

    def records(self) -> list[CapabilityRecord]:
        records: dict[str, CapabilityRecord] = {}
        for skill in self.agent.skills.model_invocable():
            record = CapabilityRecord(
                id=f"skill:{skill.name}",
                kind="skill",
                name=skill.name,
                description=_bounded_text(skill.description or skill.when_to_use),
                state="active",
                source=str(skill.source_path or "builtin"),
                keywords=tuple(_terms(f"{skill.name} {skill.when_to_use}")),
                trust_tier=str(getattr(skill, "trust_tier", TrustTier.LOCAL_USER_DECLARED.value)),
                source_identity=str(getattr(skill, "source_identity", "")),
            )
            records[record.id] = record

        active_names = {tool.name for tool in self.agent.registry.list()}
        for tool in self.agent.registry.list():
            server = str(getattr(tool, "_server", ""))
            remote = str(getattr(tool, "_remote", ""))
            if not server or not remote:
                continue
            record = CapabilityRecord(
                id=f"mcp:{server}/{remote}",
                kind="mcp",
                name=f"{server}/{remote}",
                description=_bounded_text(tool.description),
                state="active",
                source=f"mcp:{server}",
                tool_name=tool.name,
                keywords=tuple(_terms(f"{server} {remote}")),
            )
            records[record.id] = record
        for item in self.agent.registry.deferred():
            metadata = getattr(item, "metadata", {}) or {}
            if metadata.get("kind") != "mcp":
                continue
            server = str(metadata.get("server", ""))
            remote = str(metadata.get("remote", ""))
            record = CapabilityRecord(
                id=f"mcp:{server}/{remote}",
                kind="mcp",
                name=f"{server}/{remote}",
                description=_bounded_text(item.description),
                state="active" if item.name in active_names else "deferred",
                source=f"mcp:{server}",
                tool_name=item.name,
                keywords=tuple(_terms(f"{server} {remote}")),
            )
            records[record.id] = record

        manager = PluginManager(self.agent.session.workspace)
        # Persisted enablement is desired state; only a successfully committed runtime
        # generation is active in this session.
        enabled = set(getattr(self.agent, "_plugin_active_ids", frozenset()))
        configured_components = manager.component_selections()
        for plugin_id, installed in manager.records().items():
            components = configured_components.get(
                plugin_id,
                installed.components or ("skills", "agents", "hooks", "mcp"),
            )
            record = CapabilityRecord(
                id=f"plugin:{plugin_id}",
                kind="plugin",
                name=_bounded_text(installed.name, 100),
                description=_bounded_text(installed.description),
                state="active" if plugin_id in enabled else "installed",
                source=_bounded_text(installed.source, 500),
                marketplace=installed.marketplace,
                plugin_id=plugin_id,
                components=tuple(components),
                # An observed cache digest is not a trust anchor. A disabled installed
                # plugin becomes autonomously activatable only when a current trusted
                # marketplace entry below supplies the expected pin.
                sha256="",
                commit="",
                version=installed.version,
                risk="confirmation" if set(components) & _CONFIRM_COMPONENTS else "sandboxed" if set(components) & {"mcp", "lsp"} else "low",
                dependencies=installed.dependencies,
                keywords=tuple(_bounded_text(item, 100) for item in installed.keywords[:50]),
                trust_tier=installed.trust_tier,
                source_identity=installed.source_identity,
                requires_approval=installed.trust_tier not in {
                    TrustTier.ANTHROPIC_FIRST_PARTY.value,
                    TrustTier.LOCAL_USER_DECLARED.value,
                },
            )
            records[record.id] = record

        known_markets = manager.marketplaces()
        snapshots = manager.marketplace_snapshots()
        identities = manager.marketplace_identities()
        for marketplace in self.config.trusted_marketplaces:
            if marketplace not in known_markets:
                continue
            configured_source = self.config.marketplaces.get(marketplace)
            identity = identities.get(marketplace)
            if configured_source is not None and (
                identity is None or identity.source_id != marketplace_source_id(configured_source)
            ):
                logger.error(
                    "marketplace %s source identity differs from capability configuration; ignored",
                    marketplace,
                )
                continue
            try:
                entries = manager.marketplace_plugins(marketplace)
            except PluginError:
                continue
            for entry in entries:
                name = _bounded_text(entry.get("name"), 100)
                if not name or not is_safe_plugin_name(name):
                    continue
                plugin_id = f"{name}@{marketplace}"
                record_id = f"plugin:{plugin_id}"
                if record_id in records and records[record_id].state == "active":
                    continue
                components_raw = entry.get("components", [])
                components_list = list(
                    str(item) for item in components_raw
                    if isinstance(item, str) and item in _COMPONENTS
                ) if isinstance(components_raw, list) else []
                component_fields = {
                    "skills": "skills", "commands": "skills", "agents": "agents",
                    "hooks": "hooks", "mcpServers": "mcp", "lspServers": "lsp",
                    "workflows": "workflows", "outputStyles": "output-styles",
                    "themes": "themes", "monitors": "monitors", "userConfig": "user-config",
                    "channels": "channels", "settings": "settings",
                }
                for field_name, component in component_fields.items():
                    if entry.get(field_name) not in (None, [], {}):
                        components_list.append(component)
                experimental = entry.get("experimental")
                if isinstance(experimental, dict):
                    if experimental.get("themes") not in (None, [], {}):
                        components_list.append("themes")
                    if experimental.get("monitors") not in (None, [], {}):
                        components_list.append("monitors")
                components = tuple(dict.fromkeys(components_list))
                keywords_raw = entry.get("keywords", [])
                keywords = tuple(_bounded_text(item, 100) for item in keywords_raw) if isinstance(keywords_raw, list) else ()
                source_value = entry.get("source")
                source_sha = source_value.get("sha") if isinstance(source_value, dict) else ""
                archive_sha = source_value.get("sha256") if isinstance(source_value, dict) else ""
                identity = identities.get(marketplace)
                market_tier = identity.trust_tier if identity is not None else TrustTier.COMMUNITY
                try:
                    if not isinstance(source_value, (str, dict)):
                        raise SpecError("plugin source is missing")
                    plugin_source = PluginSourceConfig.from_value(source_value)
                    tier = plugin_trust_tier(market_tier, plugin_source)
                except (SpecError, TypeError):
                    tier = TrustTier.COMMUNITY
                source_identity = hashlib.sha256(
                    (
                        (identity.source_id if identity is not None else marketplace)
                        + ":"
                        + json.dumps(source_value, sort_keys=True, ensure_ascii=False)
                    ).encode("utf-8")
                ).hexdigest()
                author = entry.get("author")
                publisher = (
                    str(author.get("name") or "") if isinstance(author, dict) else str(author or "")
                )
                tags = entry.get("tags", [])
                relevance = entry.get("relevance", [])
                extra_terms = [entry.get("category", ""), publisher]
                if isinstance(tags, list):
                    extra_terms.extend(tags)
                if isinstance(relevance, list):
                    extra_terms.extend(relevance)
                records[record_id] = CapabilityRecord(
                    id=record_id,
                    kind="plugin",
                    name=name,
                    description=_bounded_text(entry.get("description")),
                    state="installed" if plugin_id in manager.records() else "available",
                    source=f"marketplace:{marketplace}",
                    marketplace=marketplace,
                    plugin_id=plugin_id,
                    components=components,
                    sha256=_bounded_text(entry.get("sha256") or archive_sha, 128).lower(),
                    commit=_bounded_text(entry.get("commit") or source_sha, 128).lower(),
                    snapshot=snapshots.get(marketplace, ""),
                    version=_bounded_text(entry.get("version"), 100),
                    risk=(
                        "confirmation" if set(components) & _CONFIRM_COMPONENTS
                        else "sandboxed" if set(components) & {"mcp", "lsp"}
                        else "low" if components else "unknown"
                    ),
                    dependencies=tuple(
                        str(item.get("name") or "") if isinstance(item, dict) else str(item)
                        for item in entry.get("dependencies", [])
                        if (isinstance(item, str) and item.strip())
                        or (isinstance(item, dict) and str(item.get("name") or "").strip())
                    ),
                    keywords=tuple(dict.fromkeys([*keywords, *(_bounded_text(item, 100) for item in extra_terms if item)])),
                    trust_tier=tier.value,
                    source_identity=source_identity,
                    publisher=_bounded_text(publisher, 300),
                    homepage=_bounded_text(entry.get("homepage") or entry.get("repository"), 1000),
                    requires_approval=(
                        tier not in {
                            TrustTier.ANTHROPIC_FIRST_PARTY,
                            TrustTier.LOCAL_USER_DECLARED,
                        }
                        or bool(set(components) & (_CONFIRM_COMPONENTS | {"mcp", "lsp"}))
                    ),
                )
                base_record = records[record_id]
                for component, component_name, component_description in _component_candidates(
                    entry, name
                ):
                    suffix = hashlib.sha256(
                        f"{component}:{component_name}".encode("utf-8")
                    ).hexdigest()[:12]
                    component_id = f"component:{plugin_id}:{component}:{suffix}"
                    component_kind: CapabilityKind = (
                        "skill" if component in {"skills", "agents"}
                        else "mcp" if component == "mcp"
                        else "plugin"
                    )
                    records[component_id] = CapabilityRecord(
                        id=component_id,
                        kind=component_kind,
                        name=f"{name}:{_bounded_text(component_name, 100)}",
                        description=_bounded_text(
                            f"{entry.get('description', '')} {component_description}"
                        ),
                        state=base_record.state,
                        source=base_record.source,
                        marketplace=marketplace,
                        plugin_id=plugin_id,
                        components=(component,),
                        sha256=base_record.sha256,
                        commit=base_record.commit,
                        snapshot=base_record.snapshot,
                        version=base_record.version,
                        risk=(
                            "confirmation" if component in _CONFIRM_COMPONENTS
                            else "sandboxed" if component in {"mcp", "lsp"}
                            else "low"
                        ),
                        dependencies=base_record.dependencies,
                        keywords=tuple(
                            dict.fromkeys(
                                [*base_record.keywords, component, component_name, component_description]
                            )
                        ),
                        trust_tier=base_record.trust_tier,
                        source_identity=base_record.source_identity,
                        publisher=base_record.publisher,
                        homepage=base_record.homepage,
                        installable=base_record.installable,
                        requires_approval=base_record.requires_approval,
                    )
        return sorted(records.values(), key=lambda item: item.id)

    @staticmethod
    def digest(records: list[CapabilityRecord]) -> str:
        encoded = json.dumps(
            [asdict(record) for record in records],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(slots=True)
class _PendingActivation:
    record: CapabilityRecord
    catalog_digest: str
    components: tuple[str, ...] = ()
    plan: ActivationPlan | None = None


class CapabilityManager:
    """Own catalog snapshots and queued activation requests for one agent session."""

    def __init__(self, agent: "ReActAgent", config: CapabilitiesConfig) -> None:
        self.agent = agent
        self.config = config
        self._lock = threading.RLock()
        self._last_records: dict[str, CapabilityRecord] = {}
        self._last_digest = ""
        self._pending: list[_PendingActivation] = []
        self._last_marketplace_refresh = 0.0
        self._plans: dict[str, ActivationPlan] = {}
        self._registry_records: dict[str, RegistryServerRecord] = {}
        plugin_manager = PluginManager(self.agent.session.workspace)
        self._registry = MCPRegistryClient(plugin_manager.root / "cache", config.mcp_registry)
        self._packages = MCPPackageManager(plugin_manager.root)

    def _refresh_marketplaces(self) -> None:
        ttl = self.config.marketplace_refresh_ttl_seconds
        now = time.monotonic()
        if not self.config.autonomous:
            return
        if ttl > 0 and self._last_marketplace_refresh and now - self._last_marketplace_refresh < ttl:
            return
        manager = PluginManager(self.agent.session.workspace)
        if manager.state_status().reset_required:
            return
        known = manager.marketplaces()
        added: set[str] = set()
        succeeded = False
        for marketplace, source in (self.config.marketplaces or {}).items():
            if marketplace not in self.config.trusted_marketplaces:
                continue
            if marketplace in known:
                identity = manager.marketplace_identities().get(marketplace)
                if identity is None or identity.source_id != marketplace_source_id(source):
                    logger.error(
                        "configured marketplace %s has a different persisted source identity; ignored",
                        marketplace,
                    )
                continue
            try:
                manager.marketplace_add(marketplace, source)
                added.add(marketplace)
                succeeded = True
            except (OSError, PluginError) as exc:
                logger.warning(
                    "trusted marketplace %s initial sync failed: %s: %s",
                    marketplace, type(exc).__name__, exc,
                )
        known = manager.marketplaces()
        for marketplace in self.config.trusted_marketplaces:
            if marketplace not in known or marketplace in added:
                continue
            try:
                manager.marketplace_update(marketplace)
                succeeded = True
            except (OSError, PluginError) as exc:
                logger.warning(
                    "trusted marketplace %s refresh failed; using cached index: %s: %s",
                    marketplace,
                    type(exc).__name__,
                    exc,
                )
        if succeeded:
            self._last_marketplace_refresh = now

    def snapshot(self, query: str = "") -> tuple[list[CapabilityRecord], str]:
        self._refresh_marketplaces()
        records = CapabilityCatalog(self.agent, self.config).records()
        registry_records = self._registry.search(query, limit=self.config.max_results) if query else []
        self._registry_records = {item.id: item for item in registry_records}
        for item in registry_records:
            public = item.public()
            records.append(
                CapabilityRecord(
                    id=item.id,
                    kind="mcp",
                    name=str(public["name"]),
                    description=item.description,
                    state=(
                        "active"
                        if item.id in set(getattr(self.agent, "_registry_active_ids", frozenset()))
                        else "available"
                    ),
                    source="mcp-registry",
                    version=item.version,
                    risk="dangerous",
                    keywords=item.keywords,
                    trust_tier=TrustTier.COMMUNITY.value,
                    publisher=item.publisher,
                    homepage=item.website or item.repository,
                    installable=bool(item.installable_types),
                    requires_approval=True,
                    package_types=item.installable_types,
                    source_identity=hashlib.sha256(item.id.encode("utf-8")).hexdigest(),
                )
            )
        records.sort(key=lambda item: item.id)
        digest = CapabilityCatalog.digest(records)
        with self._lock:
            self._last_records = {record.id: record for record in records}
            self._last_digest = digest
        return records, digest

    def search(
        self,
        query: str,
        *,
        kinds: list[str] | None = None,
        max_results: int | None = None,
    ) -> dict[str, Any]:
        if not self.config.enabled:
            return {"matches": [], "catalog_digest": "", "reason": "capability discovery is disabled"}
        records, digest = self.snapshot(query)
        allowed = {item for item in (kinds or []) if item in _KINDS} or set(_KINDS)
        scored = [(_score(record, query), record) for record in records if record.kind in allowed]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: (-item[0], item[1].id))
        limit = max_results if max_results is not None else self.config.max_results
        limit = max(1, min(20, int(limit)))
        return {
            "matches": [
                {**record.public(digest), "relevance_score": score_value}
                for score_value, record in scored[:limit]
            ],
            "catalog_digest": digest,
        }

    def auto_discover(self, task: str) -> dict[str, Any]:
        """One bounded pre-model discovery pass with safe first-party auto-activation."""

        if (
            not self.config.enabled
            or not self.config.autonomous
            or not self.config.auto_discovery
            or not _AUTO_DISCOVERY_HINT.search(task[:4000])
        ):
            return {"searched": False, "matches": [], "activations": {}}
        result = self.search(task, max_results=min(5, self.config.max_results))
        activations: dict[str, Any] = {}
        for match in result.get("matches", []):
            if not isinstance(match, dict):
                continue
            if int(match.get("relevance_score") or 0) < 10:
                continue
            if match.get("state") not in {"available", "installed"}:
                continue
            if match.get("requires_approval") is True:
                continue
            capability_id = str(match.get("id") or "")
            try:
                plan = self.create_plan(
                    capability_id,
                    str(result.get("catalog_digest") or ""),
                    sandbox_enabled=self.agent.sandbox.is_enabled(),
                )
                if plan.get("requires_approval") or not plan.get("installable"):
                    continue
                queued = self.request_plan_activation(
                    str(plan["plan_id"]), str(plan["plan_digest"])
                )
                if queued.get("status") == "queued":
                    activations.update(self.commit_pending())
                break
            except (OSError, PluginError, ValueError) as exc:
                activations[capability_id] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return {
            "searched": True,
            "matches": result.get("matches", []),
            "catalog_digest": result.get("catalog_digest", ""),
            "activations": activations,
        }

    def create_plan(
        self,
        capability_id: str,
        catalog_digest: str,
        *,
        components: tuple[str, ...] = (),
        sandbox_enabled: bool,
        package_type: str = "",
    ) -> dict[str, Any]:
        """Freeze one catalog result into a short-lived, host-owned activation plan."""

        with self._lock:
            record = self._last_records.get(capability_id)
            current_digest = self._last_digest
        if record is None or not catalog_digest or catalog_digest != current_digest:
            raise PluginError("unknown or stale capability id; call capability_search again")
        selected = tuple(dict.fromkeys(components or self.config.auto_components))
        invalid = [item for item in selected if item not in _COMPONENTS]
        if invalid:
            raise PluginError("unknown plugin components: " + ", ".join(invalid))
        if record.components:
            selected = tuple(item for item in selected if item in record.components)
        if not selected and record.plugin_id:
            raise PluginError("none of the requested components are provided by this plugin")

        artifacts: tuple[ResolvedArtifact, ...] = ()
        metadata: dict[str, Any] = {}
        permissions: list[str] = []
        configuration: tuple[str, ...] = ()
        installable = record.installable
        reason = ""
        selected_set = set(selected)
        explicitly_allowed_hooks = (
            "hooks" not in selected_set
            or any(item.startswith(record.plugin_id + ":") for item in self.config.allowed_hooks)
        )
        selected_risk = (
            "confirmation" if selected_set & _CONFIRM_COMPONENTS and not explicitly_allowed_hooks
            else "sandboxed" if selected_set & {"mcp", "lsp"}
            else "low"
        )
        if record.plugin_id and record.state != "active":
            manager = PluginManager(self.agent.session.workspace)
            plugin_name = record.plugin_id.rsplit("@", 1)[0]
            artifact = manager.prepare_artifact(
                plugin_name,
                record.marketplace,
                expected_sha256=record.sha256,
                expected_commit=record.commit,
                expected_marketplace_snapshot=record.snapshot,
            )
            if record.source_identity and artifact.source_identity != record.source_identity:
                raise PluginError("source identity changed while resolving the activation plan")
            artifacts = (
                ResolvedArtifact(
                    kind="plugin-tree",
                    source=artifact.source_identity,
                    digest=artifact.digest,
                    path=artifact.path,
                    commit=artifact.commit,
                    version=record.version,
                ),
            )
            metadata = {
                "plugin_id": record.plugin_id,
                "plugin_name": plugin_name,
                "marketplace": record.marketplace,
                "marketplace_snapshot": artifact.marketplace_snapshot,
            }
            if set(selected) & {"mcp", "lsp", "hooks", "workflows", "monitors", "channels", "bin"}:
                permissions.append("execute_subprocess_or_plugin_code")
            if set(selected) & {"mcp", "lsp"} and not sandbox_enabled:
                installable = False
                reason = "sandbox_unavailable"
            if "user-config" in record.components:
                configuration = ("user-config",)
        elif record.source == "mcp-registry":
            registry = self._registry_records.get(record.id)
            if registry is None:
                raise PluginError("MCP Registry result is stale; search again")
            package_plan = self._packages.plan(registry, package_type)
            metadata = {
                "registry_name": registry.name,
                "registry_version": registry.version,
                "mcp_package_plan": package_plan.public(),
            }
            permissions.extend(("install_or_connect_mcp", "network"))
            configuration = tuple(package_plan.environment)
            if package_plan.digest:
                artifacts = (
                    ResolvedArtifact(
                        kind=f"mcp-{package_plan.package_type}",
                        source=package_plan.identifier,
                        digest=package_plan.digest,
                        path=package_plan.artifact_path,
                        version=package_plan.version,
                    ),
                )
            if not package_plan.installable:
                installable = False
                reason = package_plan.reason or "dependency_unavailable"
            elif package_plan.package_type != "remote" and not sandbox_enabled:
                installable = False
                reason = "sandbox_unavailable"

        safe_components = selected_set <= {"skills", "agents"} or (
            selected_set <= {"skills", "agents", "hooks"} and explicitly_allowed_hooks
        )
        auto_tier = record.trust_tier in set(self.config.auto_install_tiers) or (
            record.trust_tier == TrustTier.LOCAL_USER_DECLARED.value
            and record.marketplace in self.config.trusted_marketplaces
        )
        requires_approval = not (
            bool(record.plugin_id)
            and auto_tier
            and selected_risk == self.config.auto_install_max_risk
            and safe_components
            and not record.dependencies
            and not configuration
        )
        if record.source == "mcp-registry":
            requires_approval = True
        elif record.state in {"active", "deferred"}:
            requires_approval = False
        plan = ActivationPlan.issue(
            capability_id=record.id,
            catalog_digest=catalog_digest,
            trust_tier=TrustTier(record.trust_tier),
            risk=selected_risk,
            components=selected,
            artifacts=artifacts,
            dependencies=record.dependencies,
            permissions=tuple(permissions),
            configuration=configuration,
            requires_approval=requires_approval,
            installable=installable,
            reason=reason,
            metadata=metadata,
            lifetime_seconds=self.config.plan_ttl_seconds,
        )
        with self._lock:
            now = time.time()
            self._plans = {
                key: value for key, value in self._plans.items() if value.expires_at > now
            }
            self._plans[plan.plan_id] = plan
        PluginManager(self.agent.session.workspace).audit.write(
            "capability_plan",
            {
                "plan_id": plan.plan_id,
                "capability_id": plan.capability_id,
                "plan_digest": plan.plan_digest,
                "trust_tier": plan.trust_tier.value,
                "requires_approval": plan.requires_approval,
                "installable": plan.installable,
            },
        )
        return plan.public()

    def authorization_plan(
        self, plan_id: str, plan_digest: str, *, sandbox_enabled: bool
    ) -> tuple[bool, str, bool]:
        with self._lock:
            plan = self._plans.get(plan_id)
        if plan is None or plan.plan_digest != plan_digest:
            return False, "unknown or altered activation plan", False
        if plan.expires_at <= time.time():
            return False, "activation plan expired; create a new plan", False
        if not plan.installable:
            return False, plan.reason or "capability is not installable", False
        if set(plan.components) & {"mcp", "lsp"} and not sandbox_enabled:
            return False, "sandbox_unavailable", False
        return True, "activation plan verified", plan.requires_approval

    def request_plan_activation(
        self,
        plan_id: str,
        plan_digest: str,
        *,
        host_approved: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            plan = self._plans.get(plan_id)
            record = self._last_records.get(plan.capability_id) if plan is not None else None
        if plan is None or record is None or plan.plan_digest != plan_digest:
            raise PluginError("unknown or altered activation plan")
        if plan.expires_at <= time.time():
            raise PluginError("activation plan expired; create a new plan")
        if not plan.installable:
            raise PluginError(plan.reason or "capability is not installable")
        missing_configuration = [
            name for name in plan.configuration if not os.getenv(_configuration_env_name(name))
        ]
        if missing_configuration:
            return {
                "status": "configuration_required",
                "id": plan.capability_id,
                "plan_id": plan.plan_id,
                "configuration": missing_configuration,
                "reason": "set the named environment variables or configure them through the host",
            }
        if plan.requires_approval and not host_approved:
            return {
                "status": "approval_required",
                "id": plan.capability_id,
                "plan_id": plan.plan_id,
                "reason": "host approval is required",
            }
        with self._lock:
            self._pending.append(
                _PendingActivation(record, plan.catalog_digest, plan.components, plan)
            )
        PluginManager(self.agent.session.workspace).audit.write(
            "activation_queued",
            {"plan_id": plan.plan_id, "capability_id": plan.capability_id},
        )
        return {
            "status": "queued",
            "id": plan.capability_id,
            "plan_id": plan.plan_id,
            "components": list(plan.components),
            "activated_next_turn": True,
        }

    def authorization(
        self,
        capability_id: str,
        catalog_digest: str,
        *,
        sandbox_enabled: bool,
        components: tuple[str, ...] = (),
    ) -> tuple[bool, str]:
        with self._lock:
            record = self._last_records.get(capability_id)
            current_digest = self._last_digest
        if record is None:
            return False, "capability id was not returned by the current catalog"
        if not catalog_digest or catalog_digest != current_digest:
            return False, "catalog snapshot is stale; search again"
        if record.state == "active":
            return True, "capability is already active"
        if record.kind == "mcp" and record.state == "deferred":
            return True, "activating an already-connected deferred MCP tool"
        if record.kind != "plugin":
            return False, "this capability cannot be activated"
        if not self.config.autonomous:
            return False, "autonomous activation is disabled; set capabilities.mode='autonomous-trusted'"
        if record.marketplace not in self.config.trusted_marketplaces:
            return False, "plugin marketplace is not trusted by capability policy"
        if record.trust_tier not in {
            TrustTier.ANTHROPIC_FIRST_PARTY.value,
            TrustTier.LOCAL_USER_DECLARED.value,
        }:
            return False, "community capability requires an activation plan and host approval"
        if record.sha256 and not is_sha256_pin(record.sha256):
            return False, "trusted catalog contains a malformed sha256 pin"
        if record.commit and not is_git_commit_pin(record.commit):
            return False, "trusted catalog commit must be a full immutable object id"
        if self.config.require_integrity and not (record.sha256 or record.commit):
            return False, "trusted activation requires a pinned artifact; create a plan to freeze it"
        selected = set(components or self.config.auto_components)
        wants_process = bool(selected & {"mcp", "lsp"}) and (
            not record.components or bool(selected & set(record.components))
        )
        if wants_process and not sandbox_enabled:
            return False, "autonomous MCP/LSP activation requires a real sandbox"
        return True, "trusted, integrity-pinned capability is allowed by policy"

    def request_activation(
        self,
        capability_id: str,
        catalog_digest: str,
        *,
        components: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        with self._lock:
            record = self._last_records.get(capability_id)
            if record is None or catalog_digest != self._last_digest:
                raise PluginError("unknown or stale capability id; call capability_search again")
            if record.state == "active":
                return {"status": "already_active", "id": capability_id}
        try:
            plan_value = self.create_plan(
                capability_id,
                catalog_digest,
                components=components,
                sandbox_enabled=self.agent.sandbox.is_enabled(),
            )
        except PluginError:
            # Compatibility for the original internal API: a caller that bypasses the
            # model tool still observes verification failure at the commit boundary.
            # The public capability_plan path remains fail-fast and never issues a plan.
            if record.plugin_id and record.trust_tier in {
                TrustTier.ANTHROPIC_FIRST_PARTY.value,
                TrustTier.LOCAL_USER_DECLARED.value,
            } and (record.sha256 or record.commit):
                selected = tuple(dict.fromkeys(components or self.config.auto_components))
                if record.components:
                    selected = tuple(item for item in selected if item in record.components)
                with self._lock:
                    self._pending.append(_PendingActivation(record, catalog_digest, selected))
                return {
                    "status": "queued",
                    "id": capability_id,
                    "components": list(selected),
                    "activated_next_turn": True,
                }
            raise
        if plan_value["configuration"]:
            return {
                "status": "configuration_required",
                "id": capability_id,
                "plan_id": plan_value["plan_id"],
                "configuration": plan_value["configuration"],
                "reason": "plugin declares enable-time userConfig",
            }
        if plan_value["requires_approval"]:
            return {
                "status": "confirmation_required",
                "id": capability_id,
                "plan_id": plan_value["plan_id"],
                "plan_digest": plan_value["plan_digest"],
                "components": plan_value["components"],
                "dependencies": plan_value["dependencies"],
                "permissions": plan_value["permissions"],
                "reason": "host approval is required for this activation plan",
            }
        return self.request_plan_activation(
            str(plan_value["plan_id"]), str(plan_value["plan_digest"])
        )

    def commit_pending(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            pending = self._pending
            self._pending = []
        outcomes: dict[str, dict[str, Any]] = {}
        for request in pending:
            record = request.record
            try:
                if record.kind == "mcp" and record.state == "deferred":
                    self.agent.registry.activate(record.tool_name)
                    outcomes[record.id] = {
                        "status": "activated",
                        "id": record.id,
                        "tools": [record.tool_name],
                    }
                    continue
                if not record.plugin_id:
                    if record.source == "mcp-registry" and request.plan is not None:
                        tool_names = self._activate_registry_plan(request.plan)
                        outcomes[record.id] = {
                            "status": "activated",
                            "id": record.id,
                            "plan_id": request.plan.plan_id,
                            "tools": tool_names,
                        }
                        continue
                    raise PluginError("capability does not support activation")
                plugin_manager = PluginManager(self.agent.session.workspace)
                installed = plugin_manager.records().get(record.plugin_id)
                if (
                    installed is not None
                    and record.snapshot
                    and installed.marketplace_snapshot.casefold() != record.snapshot.casefold()
                ):
                    installed = None
                if installed is None:
                    prepared = request.plan.artifacts[0] if request.plan and request.plan.artifacts else None
                    if prepared is not None:
                        assert request.plan is not None
                        activation_plan = request.plan
                        installed = plugin_manager.install(
                            str(activation_plan.metadata.get("plugin_name") or record.plugin_id.rsplit("@", 1)[0]),
                            record.marketplace,
                            expected_commit=prepared.commit,
                            expected_marketplace_snapshot=str(
                                activation_plan.metadata.get("marketplace_snapshot", "")
                            ),
                            prepared_path=prepared.path,
                            expected_prepared_digest=prepared.digest,
                        )
                    else:
                        installed = plugin_manager.install(
                            record.plugin_id.rsplit("@", 1)[0],
                            record.marketplace,
                            expected_sha256=record.sha256,
                            expected_commit=record.commit,
                            expected_marketplace_snapshot=record.snapshot,
                        )
                if request.plan and request.plan.artifacts:
                    prepared = request.plan.artifacts[0]
                    actual = plugin_tree_digest(installed.path)
                    if actual.casefold() != prepared.digest.casefold():
                        # Reinstall strictly from the content-addressed plan artifact.
                        installed = plugin_manager.install(
                            str(request.plan.metadata.get("plugin_name") or record.plugin_id.rsplit("@", 1)[0]),
                            record.marketplace,
                            expected_commit=prepared.commit,
                            expected_marketplace_snapshot=str(
                                request.plan.metadata.get("marketplace_snapshot", "")
                            ),
                            prepared_path=prepared.path,
                            expected_prepared_digest=prepared.digest,
                        )
                if record.sha256 and (
                    installed.artifact_digest or installed.integrity
                ).casefold() != record.sha256.casefold():
                    raise PluginError(
                        "installed plugin integrity does not match the trusted catalog entry"
                    )
                if record.commit and installed.commit.casefold() != record.commit.casefold():
                    raise PluginError(
                        "installed plugin commit does not match the trusted catalog entry"
                    )
                components = tuple(
                    item for item in (request.components or self.config.auto_components)
                    if not record.components or item in record.components
                )
                hooks_allowed = tuple(
                    item for item in self.config.allowed_hooks
                    if item.startswith(record.plugin_id + ":")
                )
                counts = activate_plugin(
                    self.agent,
                    installed.plugin_id,
                    components=components,
                    allowed_hooks=hooks_allowed,
                )
                outcomes[record.id] = {
                    "status": "activated",
                    "id": record.id,
                    "plugin_id": installed.plugin_id,
                    "components": list(components),
                    "source_commit": installed.commit or None,
                    "artifact_digest": installed.artifact_digest or installed.integrity,
                    "marketplace_snapshot": installed.marketplace_snapshot or None,
                    "loaded": {"skills": counts[0], "hooks": counts[1], "mcp_tools": counts[2]},
                    "plan_id": request.plan.plan_id if request.plan else None,
                }
            except Exception as exc:  # noqa: BLE001 - failed candidates must not sink the run
                outcomes[record.id] = {
                    "status": "failed",
                    "id": record.id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                PluginManager(self.agent.session.workspace).audit.write(
                    "activation_failed",
                    {
                        "capability_id": record.id,
                        "plan_id": request.plan.plan_id if request.plan else "",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
        self.snapshot()
        return outcomes

    def _activate_registry_plan(self, plan: ActivationPlan) -> list[str]:
        raw = plan.metadata.get("mcp_package_plan")
        if not isinstance(raw, dict):
            raise PluginError("activation plan has no normalized MCP package")
        try:
            package = MCPPackagePlan(
                package_type=str(raw.get("package_type") or ""),
                identifier=str(raw.get("identifier") or ""),
                version=str(raw.get("version") or ""),
                digest=str(raw.get("digest") or ""),
                artifact_path=str(raw.get("artifact_path") or ""),
                command_hint=str(raw.get("command_hint") or ""),
                arguments=tuple(str(item) for item in raw.get("arguments", [])),
                environment=tuple(str(item) for item in raw.get("environment", [])),
                transport=str(raw.get("transport") or "stdio"),
                url=str(raw.get("url") or ""),
                headers=tuple(
                    (str(item[0]), str(item[1]))
                    for item in raw.get("headers", [])
                    if isinstance(item, (list, tuple)) and len(item) == 2
                ),
                installable=bool(raw.get("installable", True)),
                reason=str(raw.get("reason") or ""),
                extra=dict(raw.get("extra") or {}),
            )
        except (TypeError, ValueError) as exc:
            raise PluginError("activation plan contains invalid MCP package metadata") from exc
        environment = {
            name: os.environ[_configuration_env_name(name)]
            for name in package.environment
            if os.getenv(_configuration_env_name(name)) is not None
        }
        headers = {
            header: os.environ[_configuration_env_name(variable)]
            for header, variable in package.headers
            if os.getenv(_configuration_env_name(variable)) is not None
        }
        server_name = "registry_" + re.sub(r"[^A-Za-z0-9_.-]", "_", plan.capability_id)[-60:]
        server = self._packages.activate(
            package,
            server_name=server_name,
            configured_env=environment,
            configured_headers=headers,
        )
        if package.package_type != "remote":
            if not self.agent.sandbox.is_enabled():
                raise PluginError("sandbox_unavailable")
            root = Path(package.artifact_path).parent if package.artifact_path else self._packages.installs
            scope = ExecutionScope.for_workspace(
                self.agent.session.workspace,
                read_only_roots=(root, self._packages.installs),
                network="deny",
            )
            wrapped, shell = self.agent.sandbox.wrap(
                [server.command, *server.args], False, scope=scope
            )
            if shell or not isinstance(wrapped, list) or not wrapped:
                raise PluginError("sandbox could not wrap Registry MCP server")
            server = replace(
                server,
                command=str(wrapped[0]),
                args=[str(item) for item in wrapped[1:]],
                cwd="",
            )
        manager = MCPClientManager(MCPConfig([server]))
        try:
            manager.start()
            tools = MCPAdapter(manager).list_tools()
            tool_groups = dict(getattr(self.agent, "_registry_mcp_tool_groups", {}))
            old_names = set(tool_groups.get(plan.capability_id, set()))
            active: list[Tool] = []
            deferred: list[DeferredTool] = []
            for tool in tools:
                def factory(bound_tool: Tool = tool) -> Tool:
                    return bound_tool

                deferred.append(
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
            self.agent.registry.replace_group(old_names, active, deferred)
        except Exception:
            manager.close()
            raise
        managers = dict(getattr(self.agent, "_registry_mcp_managers", {}))
        previous = managers.get(plan.capability_id)
        managers[plan.capability_id] = manager
        tool_groups[plan.capability_id] = {tool.name for tool in tools}
        self.agent._registry_mcp_managers = managers
        self.agent._registry_mcp_tool_groups = tool_groups
        self.agent._registry_active_ids = frozenset(managers)
        if previous is not None:
            previous.close()
        PluginManager(self.agent.session.workspace).audit.write(
            "generation_commit",
            {
                "registry_capability": plan.capability_id,
                "package_type": package.package_type,
                "tools": sorted(tool_groups[plan.capability_id]),
            },
        )
        return sorted(tool_groups[plan.capability_id])
