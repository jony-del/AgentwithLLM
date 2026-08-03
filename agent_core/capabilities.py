"""Runtime capability discovery and trusted, boundary-safe activation.

The catalog intentionally exposes identifiers, never executable source parameters.  A
model can search metadata and request activation of a returned id; installation and
generation swaps remain policy-owned host operations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

from agent_core.plugins import (
    PluginError,
    PluginManager,
    activate_plugin,
    is_git_commit_pin,
    is_safe_plugin_name,
    is_sha256_pin,
)

if TYPE_CHECKING:
    from agent_core.react import ReActAgent


CapabilityKind = Literal["skill", "mcp", "plugin"]
CapabilityState = Literal["active", "deferred", "installed", "available"]
_KINDS = frozenset({"skill", "mcp", "plugin"})
_COMPONENTS = frozenset({"skills", "agents", "mcp", "hooks"})
_CONTROL_TAG = re.compile(r"(?i)</?(?:system-reminder|tool_output_ref|untrusted-data)[^>]*>")
_WORD = re.compile(r"[\w.+:@/-]+", re.UNICODE)
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
logger = logging.getLogger(__name__)


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
        return cls(
            mode=mode,
            trusted_marketplaces=strings("trusted_marketplaces"),
            require_integrity=bool(raw.get("require_integrity", True)),
            auto_components=components,
            allowed_hooks=strings("allowed_hooks"),
            max_results=max_results,
            marketplace_refresh_ttl_seconds=refresh,
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

    @property
    def integrity(self) -> str:
        if self.sha256:
            return f"sha256:{self.sha256}"
        if self.commit:
            return f"git:{self.commit}"
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
            "components": list(self.components),
            "integrity": self.integrity or None,
            "catalog_digest": catalog_digest,
            "invoke": invoke,
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
        enabled = set(manager.enabled_ids())
        configured_components = manager.component_selections()
        for plugin_id, installed in manager.records().items():
            components = configured_components.get(plugin_id, ("skills", "agents", "hooks", "mcp"))
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
                keywords=tuple(_bounded_text(item, 100) for item in installed.keywords[:50]),
            )
            records[record.id] = record

        known_markets = manager.marketplaces()
        for marketplace in self.config.trusted_marketplaces:
            if marketplace not in known_markets:
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
                components = tuple(
                    str(item) for item in components_raw
                    if isinstance(item, str) and item in _COMPONENTS
                ) if isinstance(components_raw, list) else ()
                keywords_raw = entry.get("keywords", [])
                keywords = tuple(_bounded_text(item, 100) for item in keywords_raw) if isinstance(keywords_raw, list) else ()
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
                    sha256=_bounded_text(entry.get("sha256"), 128).lower(),
                    commit=_bounded_text(entry.get("commit"), 128).lower(),
                    keywords=keywords,
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

    def _refresh_marketplaces(self) -> None:
        ttl = self.config.marketplace_refresh_ttl_seconds
        now = time.monotonic()
        if not self.config.autonomous or ttl <= 0:
            return
        if self._last_marketplace_refresh and now - self._last_marketplace_refresh < ttl:
            return
        manager = PluginManager(self.agent.session.workspace)
        for marketplace in self.config.trusted_marketplaces:
            if marketplace not in manager.marketplaces():
                continue
            try:
                manager.marketplace_update(marketplace)
            except (OSError, PluginError) as exc:
                logger.warning(
                    "trusted marketplace %s refresh failed; using cached index: %s: %s",
                    marketplace,
                    type(exc).__name__,
                    exc,
                )
        self._last_marketplace_refresh = now

    def snapshot(self) -> tuple[list[CapabilityRecord], str]:
        self._refresh_marketplaces()
        records = CapabilityCatalog(self.agent, self.config).records()
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
        records, digest = self.snapshot()
        allowed = {item for item in (kinds or []) if item in _KINDS} or set(_KINDS)
        scored = [(_score(record, query), record) for record in records if record.kind in allowed]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: (-item[0], item[1].id))
        limit = max_results if max_results is not None else self.config.max_results
        limit = max(1, min(20, int(limit)))
        return {
            "matches": [record.public(digest) for _score_value, record in scored[:limit]],
            "catalog_digest": digest,
        }

    def authorization(self, capability_id: str, catalog_digest: str, *, sandbox_enabled: bool) -> tuple[bool, str]:
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
        if record.sha256 and not is_sha256_pin(record.sha256):
            return False, "trusted catalog contains a malformed sha256 pin"
        if record.commit and not is_git_commit_pin(record.commit):
            return False, "trusted catalog commit must be a full immutable object id"
        if self.config.require_integrity and not (record.sha256 or record.commit):
            return False, "trusted activation requires a pinned sha256 or git commit"
        wants_mcp = "mcp" in self.config.auto_components and (
            not record.components or "mcp" in record.components
        )
        if wants_mcp and not sandbox_enabled:
            return False, "autonomous MCP activation requires a real sandbox"
        return True, "trusted, integrity-pinned capability is allowed by policy"

    def request_activation(self, capability_id: str, catalog_digest: str) -> dict[str, Any]:
        with self._lock:
            record = self._last_records.get(capability_id)
            if record is None or catalog_digest != self._last_digest:
                raise PluginError("unknown or stale capability id; call capability_search again")
            if record.state == "active":
                return {"status": "already_active", "id": capability_id}
            self._pending.append(_PendingActivation(record, catalog_digest))
        return {"status": "queued", "id": capability_id}

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
                if record.kind != "plugin":
                    raise PluginError("capability does not support activation")
                plugin_manager = PluginManager(self.agent.session.workspace)
                installed = plugin_manager.records().get(record.plugin_id)
                if installed is None:
                    installed = plugin_manager.install(
                        record.name,
                        record.marketplace,
                        expected_sha256=record.sha256,
                        expected_commit=record.commit,
                    )
                if record.sha256 and installed.integrity.casefold() != record.sha256.casefold():
                    raise PluginError(
                        "installed plugin integrity does not match the trusted catalog entry"
                    )
                if record.commit and installed.commit.casefold() != record.commit.casefold():
                    raise PluginError(
                        "installed plugin commit does not match the trusted catalog entry"
                    )
                components = tuple(
                    item for item in self.config.auto_components
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
                    "loaded": {"skills": counts[0], "hooks": counts[1], "mcp_tools": counts[2]},
                }
            except Exception as exc:  # noqa: BLE001 - failed candidates must not sink the run
                outcomes[record.id] = {
                    "status": "failed",
                    "id": record.id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        self.snapshot()
        return outcomes
