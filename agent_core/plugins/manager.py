"""PluginManager: marketplace registry, installation, and bundle building."""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import tempfile
import time
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse
from agent_core.capability_audit import CapabilityAuditLog
from agent_core.capability_security import (
    CAPABILITY_STATE_SCHEMA_VERSION,
    MarketplaceIdentity,
    TrustTier,
    plugin_trust_tier,
    validate_reserved_marketplace,
)
from agent_core.config import user_settings_path
from agent_core.file_lock import FileLock
from agent_core.local_config import update_local_table, update_toml_table
from agent_core.mcp import MCPAdapter, MCPClientManager, MCPConfig, MCPServerConfig
from agent_core.plugin_spec import (
    MarketplaceSourceConfig,
    PluginSourceConfig,
    SpecError,
    declared_components,
    resolve_plugin_manifest,
)
from agent_core import secret_store
from agent_core.skills import Skill
from .models import (
    _SAFE_NAME,
    _REMOTE_SOURCE,
    PluginError,
    is_safe_plugin_name,
    is_sha256_pin,
    is_git_commit_pin,
    PluginRecord,
    MarketplaceRecord,
    PluginStateStatus,
    PreparedPluginArtifact,
    PluginBundle,
)
from .store import (
    _PLUGIN_COMPONENTS,
    _KEYCHAIN_PREFIX,
    plugin_home,
    _read_json,
    _atomic_json,
    _serializable_marketplace_source,
    _resolve_marketplace_headers,
    _remove_toml_tables,
    _inside,
    plugin_tree_digest,
    validate_plugin,
    copy_marketplace_plugin_tree,
    _read_plugin_list,
    _changed_enabled,
)
from .sources import (
    _git_head,
    _git_head_if_repository,
    _clone_remote,
    _download_https,
    _extract_archive_bytes,
    _single_package_root,
    _materialize_npm_package,
    _dependency_labels,
)
from .env import _plugin_secret_targets
from .components import (
    _apply_skill_provenance,
    _load_plugin_skills,
    _load_plugin_agents,
    _load_plugin_hooks,
    _load_plugin_mcp,
    _load_non_mcp_components,
    _restrict_autonomous_mcp,
)

if TYPE_CHECKING:
    from agent_core.react import ReActAgent


class PluginManager:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()
        self.root = plugin_home()
        self.records_path = self.root / "installed.json"
        self.marketplaces_path = self.root / "marketplaces.json"
        self.state_lock_path = self.root / ".state.lock"
        self.audit = CapabilityAuditLog(self.root)

    def state_status(self) -> PluginStateStatus:
        """Describe legacy state without ever loading it as trusted configuration."""

        targets: list[str] = []
        versions: list[int] = []
        for path in (self.records_path, self.marketplaces_path):
            if not path.is_file():
                continue
            raw = _read_json(path, None)
            version = int(raw.get("schema_version", 0)) if isinstance(raw, dict) else 0
            versions.append(version)
            if version != CAPABILITY_STATE_SCHEMA_VERSION:
                targets.append(str(path))
        version = min(versions) if versions else CAPABILITY_STATE_SCHEMA_VERSION
        return PluginStateStatus(
            schema_version=version,
            legacy=bool(targets),
            reset_required=bool(targets),
            targets=tuple(targets),
        )

    def reset_preview(self) -> tuple[str, ...]:
        """Return exact managed targets used by interactive and CLI reset flows."""

        targets = [
            self.records_path,
            self.marketplaces_path,
            self.root / "cache",
            self.root / "sources",
            self.root / "marketplaces",
            self.root / "marketplace-commits",
            self.root / "staging",
            self.root / "artifacts",
            self.root / "plans",
            self.workspace / "agent.local.toml",
        ]
        try:
            targets.append(user_settings_path())
        except RuntimeError:
            pass
        return tuple(str(item.resolve()) for item in targets if item.exists())

    def reset_state(self) -> tuple[str, ...]:
        """Clear v1/v2 plugin state after an explicit host/user confirmation.

        This method never follows a path outside the managed root and only removes
        plugin-owned TOML tables from settings files.  Callers own the confirmation.
        """

        removed: list[str] = []
        with FileLock(self.state_lock_path):
            secret_targets = _plugin_secret_targets(self.workspace)
            managed_dirs = (
                "cache", "sources", "marketplaces", "marketplace-commits",
                "staging", "artifacts", "plans", "data",
            )
            for name in managed_dirs:
                target = (self.root / name).resolve()
                if target.exists() and _inside(target, self.root):
                    shutil.rmtree(target)
                    removed.append(str(target))
            for target in (self.records_path, self.marketplaces_path):
                resolved = target.resolve()
                if resolved.exists() and _inside(resolved, self.root):
                    resolved.unlink()
                    removed.append(str(resolved))
            config_paths = [self.workspace / "agent.local.toml"]
            try:
                config_paths.append(user_settings_path())
            except RuntimeError:
                pass
            for path in config_paths:
                if _remove_toml_tables(path, ("plugins", "plugin_configs")):
                    removed.append(str(path.resolve()))
            for secret_target in secret_targets:
                try:
                    secret_store.delete(secret_target)
                except secret_store.SecretStoreError:
                    continue
            self._save_records({})
            self._save_marketplaces({})
        self.audit.write("legacy_state_reset", {"removed": removed})
        return tuple(removed)

    def records(self) -> dict[str, PluginRecord]:
        raw = _read_json(self.records_path, {})
        if not isinstance(raw, dict):
            return {}
        if raw and raw.get("schema_version") != CAPABILITY_STATE_SCHEMA_VERSION:
            return {}
        table = raw.get("plugins", {})
        if not isinstance(table, dict):
            return {}
        records: dict[str, PluginRecord] = {}
        valid = {field.name for field in fields(PluginRecord)}
        for plugin_id, item in table.items():
            if not isinstance(item, dict):
                continue
            try:
                values = {key: value for key, value in item.items() if key in valid}
                for tuple_field in ("keywords", "components", "dependencies", "required_by"):
                    sequence = values.get(tuple_field)
                    if isinstance(sequence, list):
                        values[tuple_field] = tuple(str(value) for value in sequence)
                record = PluginRecord(**values)
            except TypeError:
                continue
            records[str(plugin_id)] = record
        return records

    def _save_records(self, records: dict[str, PluginRecord]) -> None:
        _atomic_json(
            self.records_path,
            {
                "schema_version": CAPABILITY_STATE_SCHEMA_VERSION,
                "plugins": {key: asdict(value) for key, value in records.items()},
            },
        )

    def _marketplace_records(self) -> dict[str, MarketplaceRecord]:
        raw = _read_json(self.marketplaces_path, {})
        if not isinstance(raw, dict):
            return {}
        if raw and raw.get("schema_version") != CAPABILITY_STATE_SCHEMA_VERSION:
            return {}
        table = raw.get("marketplaces", {})
        if not isinstance(table, dict):
            return {}
        result: dict[str, MarketplaceRecord] = {}
        for name, value in table.items():
            if not is_safe_plugin_name(str(name)):
                continue
            if not isinstance(value, dict):
                continue
            try:
                result[str(name)] = MarketplaceRecord(
                    name=str(name),
                    source=dict(value.get("source") or {}),
                    snapshot_path=str(value.get("snapshot_path") or ""),
                    snapshot_commit=str(value.get("snapshot_commit") or ""),
                    catalog_digest=str(value.get("catalog_digest") or ""),
                    refreshed_at=float(value.get("refreshed_at") or 0.0),
                    source_id=str(value.get("source_id") or ""),
                    canonical_source=str(value.get("canonical_source") or ""),
                    trust_tier=str(value.get("trust_tier") or TrustTier.COMMUNITY.value),
                    last_attempt=float(value.get("last_attempt") or 0.0),
                    last_success=float(value.get("last_success") or 0.0),
                    failure_count=int(value.get("failure_count") or 0),
                    next_retry=float(value.get("next_retry") or 0.0),
                    last_error=str(value.get("last_error") or ""),
                )
            except (TypeError, ValueError):
                continue
        return result

    def _save_marketplaces(self, records: dict[str, MarketplaceRecord]) -> None:
        _atomic_json(
            self.marketplaces_path,
            {
                "schema_version": CAPABILITY_STATE_SCHEMA_VERSION,
                "marketplaces": {key: asdict(value) for key, value in records.items()},
            },
        )

    def marketplaces(self) -> dict[str, str]:
        """Return compatible name -> active snapshot paths."""

        return {key: value.snapshot_path for key, value in self._marketplace_records().items()}

    def marketplace_sources(self) -> dict[str, MarketplaceSourceConfig]:
        result: dict[str, MarketplaceSourceConfig] = {}
        for name, record in self._marketplace_records().items():
            try:
                result[name] = MarketplaceSourceConfig.from_value(record.source)
            except SpecError:
                continue
        return result

    def marketplace_snapshots(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for name, record in self._marketplace_records().items():
            try:
                source = MarketplaceSourceConfig.from_value(record.source)
            except SpecError:
                continue
            if source.remote:
                result[name] = record.snapshot_commit or record.catalog_digest
        return result

    def marketplace_identities(self) -> dict[str, MarketplaceIdentity]:
        result: dict[str, MarketplaceIdentity] = {}
        for name, record in self._marketplace_records().items():
            try:
                source = MarketplaceSourceConfig.from_value(record.source)
                identity = MarketplaceIdentity.create(
                    name, source, record.snapshot_commit or record.catalog_digest
                )
            except (SpecError, ValueError):
                continue
            if identity.source_id != record.source_id:
                continue
            result[name] = identity
        return result

    def marketplace_add(
        self, name: str, source: str | dict[str, Any] | MarketplaceSourceConfig
    ) -> None:
        if not _SAFE_NAME.fullmatch(name):
            raise PluginError("marketplace name must be filesystem-safe")
        try:
            config = source if isinstance(source, MarketplaceSourceConfig) else MarketplaceSourceConfig.from_value(source)
        except SpecError as exc:
            raise PluginError(str(exc)) from exc
        try:
            validate_reserved_marketplace(name, config)
        except ValueError as exc:
            raise PluginError(str(exc)) from exc
        identity = MarketplaceIdentity.create(name, config)
        records = self._marketplace_records()
        previous = records.get(name)
        if previous is not None and previous.source_id != identity.source_id:
            raise PluginError(
                f"marketplace source identity mismatch for {name!r}; remove it explicitly before rebinding"
            )
        source_path, commit = self._materialize_marketplace(name, config)
        manifest = self._load_marketplace_manifest(source_path)
        actual_name = str(manifest.get("name") or "").strip()
        if actual_name and actual_name != name:
            raise PluginError(
                f"marketplace manifest name mismatch: configured {name!r}, received {actual_name!r}"
            )
        new_record = MarketplaceRecord(
            name=name,
            source=_serializable_marketplace_source(config),
            snapshot_path=str(source_path),
            snapshot_commit=commit,
            catalog_digest=self._catalog_digest(source_path),
            refreshed_at=time.time(),
            source_id=identity.source_id,
            canonical_source=identity.canonical_source,
            trust_tier=identity.trust_tier.value,
            last_attempt=time.time(),
            last_success=time.time(),
        )
        with FileLock(self.state_lock_path):
            records = self._marketplace_records()
            previous = records.get(name)
            if previous is not None and previous.source_id != identity.source_id:
                raise PluginError(f"marketplace source identity mismatch for {name!r}")
            records[name] = new_record
            self._save_marketplaces(records)
        self.audit.write(
            "catalog_sync",
            {
                "marketplace": name,
                "source_id": identity.source_id,
                "snapshot": commit or new_record.catalog_digest,
                "result": "success",
            },
        )

    def marketplace_add_source(
        self, source: str | dict[str, Any] | MarketplaceSourceConfig, name: str = ""
    ) -> str:
        """Claude-style add where the catalog's real name is authoritative."""

        try:
            config = source if isinstance(source, MarketplaceSourceConfig) else MarketplaceSourceConfig.from_value(source)
        except SpecError as exc:
            raise PluginError(str(exc)) from exc
        probe_name = name or f"probe-{hashlib.sha256(config.display().encode()).hexdigest()[:12]}"
        path, _commit = self._materialize_marketplace(probe_name, config)
        manifest = self._load_marketplace_manifest(path)
        actual = str(manifest.get("name") or "").strip()
        chosen = name or actual
        if not chosen:
            raise PluginError("marketplace manifest must declare name")
        self.marketplace_add(chosen, config)
        return chosen

    def marketplace_remove(self, name: str) -> None:
        with FileLock(self.state_lock_path):
            values = self._marketplace_records()
            if name not in values:
                raise PluginError(f"unknown marketplace: {name}")
            del values[name]
            self._save_marketplaces(values)

    def marketplace_update(self, name: str) -> int:
        values = self._marketplace_records()
        if name not in values:
            raise PluginError(f"unknown marketplace: {name}")
        try:
            source = MarketplaceSourceConfig.from_value(values[name].source)
        except SpecError as exc:
            raise PluginError(str(exc)) from exc
        try:
            validate_reserved_marketplace(name, source)
        except ValueError as exc:
            raise PluginError(str(exc)) from exc
        identity = MarketplaceIdentity.create(name, source)
        record = values[name]
        if record.source_id != identity.source_id:
            raise PluginError(f"marketplace source identity mismatch for {name!r}")
        now = time.time()
        if record.next_retry and now < record.next_retry:
            raise PluginError(
                f"marketplace {name!r} refresh is in backoff until {record.next_retry:.0f}"
            )
        # Local sources are validated in place. Remote sources are built in staging;
        # the registry pointer changes only after name/schema validation succeeds.
        try:
            path, commit = self._materialize_marketplace(name, source, refresh=True)
            manifest = self._load_marketplace_manifest(path)
            actual = str(manifest.get("name") or "").strip()
            if actual and actual != name:
                raise PluginError(
                    f"marketplace manifest name mismatch: configured {name!r}, received {actual!r}"
                )
        except Exception as exc:
            failures = record.failure_count + 1
            delay = min(3600.0, 5.0 * (2 ** min(failures - 1, 9)))
            values[name] = replace(
                record,
                last_attempt=now,
                failure_count=failures,
                next_retry=now + delay,
                last_error=f"{type(exc).__name__}: {exc}"[:1000],
            )
            with FileLock(self.state_lock_path):
                latest = self._marketplace_records()
                latest[name] = values[name]
                self._save_marketplaces(latest)
            self.audit.write(
                "catalog_sync",
                {"marketplace": name, "source_id": record.source_id, "result": "failed", "error": str(exc)},
            )
            raise
        values[name] = MarketplaceRecord(
            name=name,
            source=_serializable_marketplace_source(source),
            snapshot_path=str(path),
            snapshot_commit=commit,
            catalog_digest=self._catalog_digest(path),
            refreshed_at=now,
            source_id=identity.source_id,
            canonical_source=identity.canonical_source,
            trust_tier=identity.trust_tier.value,
            last_attempt=now,
            last_success=now,
        )
        with FileLock(self.state_lock_path):
            latest = self._marketplace_records()
            current = latest.get(name)
            if current is None or current.source_id != identity.source_id:
                raise PluginError(f"marketplace source identity changed during refresh: {name!r}")
            latest[name] = values[name]
            self._save_marketplaces(latest)
        self.audit.write(
            "catalog_sync",
            {
                "marketplace": name,
                "source_id": identity.source_id,
                "snapshot": commit or values[name].catalog_digest,
                "result": "success",
            },
        )
        return len([item for item in manifest.get("plugins", []) if isinstance(item, dict)])

    def marketplace_plugins(self, name: str) -> list[dict[str, Any]]:
        """Read one configured marketplace's bounded manifest entries."""

        values = self._marketplace_records()
        if name not in values:
            raise PluginError(f"unknown marketplace: {name}")
        return self._load_marketplace(Path(values[name].snapshot_path))

    def _materialize_marketplace(
        self,
        name: str,
        config: MarketplaceSourceConfig,
        *,
        refresh: bool = False,
    ) -> tuple[Path, str]:
        if config.kind in {"file", "directory"}:
            path = Path(config.path).expanduser().resolve()
            if not path.exists():
                raise PluginError(f"marketplace source does not exist: {path}")
            return path, _git_head(path) if (path / ".git").is_dir() else ""
        if config.kind == "settings":
            manifest = {
                "name": config.name,
                "owner": {"name": "settings"},
                "plugins": [dict(item) for item in config.plugins],
            }
            body = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
            digest = hashlib.sha256(body).hexdigest()
            destination = self.root / "marketplaces" / name / digest
            if not destination.exists():
                destination.mkdir(parents=True, exist_ok=False)
                (destination / "marketplace.json").write_bytes(body)
            return destination, digest
        if config.kind == "url":
            body = _download_https(
                config.url,
                headers=_resolve_marketplace_headers(config.headers),
                max_bytes=16 * 1024 * 1024,
            )
            digest = hashlib.sha256(body).hexdigest()
            destination = self.root / "marketplaces" / name / digest
            if not destination.exists():
                destination.mkdir(parents=True, exist_ok=False)
                temporary = destination / ".marketplace.json.tmp"
                temporary.write_bytes(body)
                os.replace(temporary, destination / "marketplace.json")
            return destination, digest
        url = (
            f"https://github.com/{config.repo}.git"
            if config.kind == "github" else config.url
        )
        staging = self.root / "staging"
        staging.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f"marketplace-{name}.", dir=str(staging)))
        checkout = temporary / "source"
        try:
            _clone_remote(url, checkout, ref=config.ref or None)
            commit = _git_head(checkout)
            destination = self.root / "marketplaces" / name / commit
            if destination.exists():
                root = (destination / config.path).resolve() if config.path else destination.resolve()
                if not _inside(root, destination):
                    raise PluginError("marketplace source path escapes repository")
                return root, commit
            destination.parent.mkdir(parents=True, exist_ok=True)
            # A git marketplace may place the catalog below the repository root.
            # Validate that exact configured root before publishing the immutable
            # snapshot, otherwise a valid ``path`` source would be rejected merely
            # because the checkout root has no marketplace.json of its own.
            checkout_root = (
                (checkout / config.path).resolve() if config.path else checkout.resolve()
            )
            if not _inside(checkout_root, checkout):
                raise PluginError("marketplace source path escapes repository")
            self._load_marketplace_manifest(checkout_root)
            os.replace(checkout, destination)
            root = (destination / config.path).resolve() if config.path else destination.resolve()
            if not _inside(root, destination):
                raise PluginError("marketplace source path escapes repository")
            return root, commit
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    @staticmethod
    def _catalog_digest(source: Path) -> str:
        try:
            manifest = PluginManager._load_marketplace_manifest(source)
        except PluginError:
            return ""
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _load_marketplace_manifest(source: Path) -> dict[str, Any]:
        path = source / ".claude-plugin" / "marketplace.json" if source.is_dir() else source
        if not path.is_file() and source.is_dir():
            path = source / "marketplace.json"
        data = _read_json(path, None)
        if not isinstance(data, dict) or not isinstance(data.get("plugins"), list):
            raise PluginError(f"invalid marketplace manifest: {path}")
        if data.get("name") is not None and not is_safe_plugin_name(str(data["name"])):
            raise PluginError("marketplace manifest name must be filesystem-safe")
        return data

    @staticmethod
    def _load_marketplace(source: Path) -> list[dict[str, Any]]:
        data = PluginManager._load_marketplace_manifest(source)
        return [item for item in data["plugins"] if isinstance(item, dict)]

    def _marketplace_entry(self, name: str, marketplace: str) -> dict[str, Any]:
        if not is_safe_plugin_name(name):
            raise PluginError("plugin name must be filesystem-safe")
        markets = self.marketplaces()
        if marketplace not in markets:
            raise PluginError(f"unknown marketplace: {marketplace}")
        market_source = Path(markets[marketplace])
        manifest = self._load_marketplace_manifest(market_source)
        renames = manifest.get("renames", {})
        if isinstance(renames, dict) and name in renames:
            renamed = renames[name]
            if renamed is None:
                raise PluginError(f"plugin {name!r} was removed from marketplace {marketplace!r}")
            name = str(renamed)
        for entry in manifest["plugins"]:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("name") or "") != name:
                continue
            return entry
        raise PluginError(f"plugin {name!r} not found in marketplace {marketplace!r}")

    def _resolve_marketplace_plugin(
        self,
        name: str,
        marketplace: str,
        *,
        expected_sha256: str = "",
        expected_commit: str = "",
    ) -> Path:
        markets = self._marketplace_records()
        if marketplace not in markets:
            raise PluginError(f"unknown marketplace: {marketplace}")
        market_source = Path(markets[marketplace].snapshot_path)
        entry = self._marketplace_entry(name, marketplace)
        source_value = entry.get("source") or entry.get("path")
        if not isinstance(source_value, (str, dict)):
            raise PluginError(f"plugin {name!r} has no valid source")
        try:
            source = PluginSourceConfig.from_value(source_value)
        except SpecError as exc:
            raise PluginError(f"invalid source for plugin {name!r}: {exc}") from exc
        if source.kind == "relative":
            marketplace_config = self.marketplace_sources().get(marketplace)
            if marketplace_config is not None and marketplace_config.kind in {"url", "settings"}:
                raise PluginError(
                    f"relative plugin sources are unavailable for {marketplace_config.kind} marketplaces"
                )
            market_record = markets[marketplace]
            if (
                expected_commit
                and expected_commit.casefold() != market_record.snapshot_commit.casefold()
            ):
                if marketplace_config is None or marketplace_config.kind not in {"github", "git"}:
                    raise PluginError(
                        "version-tagged relative dependencies require a git-backed marketplace"
                    )
                marketplace_url = (
                    f"https://github.com/{marketplace_config.repo}.git"
                    if marketplace_config.kind == "github" else marketplace_config.url
                )
                checkout = self.root / "marketplace-commits" / marketplace / expected_commit
                _clone_remote(marketplace_url, checkout, commit=expected_commit)
                market_source = (
                    (checkout / marketplace_config.path).resolve()
                    if marketplace_config.path else checkout.resolve()
                )
                if not _inside(market_source, checkout):
                    raise PluginError("marketplace source path escapes tagged checkout")
                tagged_manifest = self._load_marketplace_manifest(market_source)
                tagged_name = str(tagged_manifest.get("name") or "").strip()
                if tagged_name and tagged_name != marketplace:
                    raise PluginError("tagged marketplace manifest name mismatch")
                tagged_entry = next(
                    (
                        item for item in tagged_manifest.get("plugins", [])
                        if isinstance(item, dict) and str(item.get("name") or "") == name
                    ),
                    None,
                )
                if tagged_entry is None:
                    raise PluginError(f"tagged marketplace does not contain plugin {name!r}")
                tagged_source_value = tagged_entry.get("source") or tagged_entry.get("path")
                if not isinstance(tagged_source_value, (str, dict)):
                    raise PluginError(f"tagged plugin {name!r} has no valid source")
                try:
                    source = PluginSourceConfig.from_value(tagged_source_value)
                except SpecError as exc:
                    raise PluginError(f"invalid tagged source for plugin {name!r}: {exc}") from exc
                if source.kind != "relative":
                    raise PluginError("marketplace version tags require a relative plugin source")
            candidate = Path(source.path).expanduser()
            if candidate.is_absolute():
                if marketplace_config is None or marketplace_config.kind not in {"file", "directory"}:
                    raise PluginError("remote marketplace relative plugin source must not be absolute")
                return candidate.resolve()
            market_root = market_source
            if market_source.is_file():
                market_root = market_source.parent
                if market_root.name == ".claude-plugin":
                    market_root = market_root.parent
            plugin_root = ""
            manifest = self._load_marketplace_manifest(market_source)
            metadata = manifest.get("metadata")
            if isinstance(metadata, dict):
                plugin_root = str(metadata.get("pluginRoot") or "")
            candidate = (market_root / plugin_root / source.path).resolve()
            if not _inside(candidate, market_root):
                raise PluginError("relative plugin source escapes marketplace snapshot")
            return candidate
        if source.kind in {"github", "url", "git-subdir"}:
            url = (
                f"https://github.com/{source.repo}.git"
                if source.kind == "github" else source.url
            )
            commit = source.sha or expected_commit
            source_root = self.root / "sources" / marketplace / name
            if commit:
                destination = source_root / commit
                if not destination.exists():
                    _clone_remote(url, destination, commit=commit)
            else:
                staging_root = self.root / "staging"
                staging_root.mkdir(parents=True, exist_ok=True)
                temporary = Path(tempfile.mkdtemp(prefix=f"plugin-{name}.", dir=str(staging_root)))
                checkout = temporary / "source"
                try:
                    _clone_remote(url, checkout, ref=source.ref or None)
                    resolved_commit = _git_head(checkout)
                    destination = source_root / resolved_commit
                    if not destination.exists():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(checkout, destination)
                finally:
                    shutil.rmtree(temporary, ignore_errors=True)
            if source.kind == "git-subdir":
                candidate = (destination / source.path).resolve()
                if not _inside(candidate, destination):
                    raise PluginError("git-subdir plugin path escapes repository")
                return candidate
            return destination.resolve()
        if source.kind == "archive":
            archive_headers = dict(source.headers)
            market_config = self.marketplace_sources().get(marketplace)
            if not archive_headers and market_config is not None and market_config.kind == "url":
                market_origin = urlparse(market_config.url)
                archive_origin = urlparse(source.url)
                if (
                    market_origin.scheme.casefold(), market_origin.hostname,
                    market_origin.port or 443,
                ) == (
                    archive_origin.scheme.casefold(), archive_origin.hostname,
                    archive_origin.port or 443,
                ):
                    archive_headers = dict(market_config.headers)
            body = _download_https(
                source.url,
                headers=archive_headers,
                max_bytes=256 * 1024 * 1024,
            )
            digest = hashlib.sha256(body).hexdigest()
            pin = source.sha256 or expected_sha256
            if pin and digest.casefold() != pin.casefold():
                raise PluginError(f"archive sha256 mismatch: expected {pin}, got {digest}")
            destination = self.root / "sources" / marketplace / name / digest
            if not destination.exists():
                _extract_archive_bytes(body, destination)
            return _single_package_root(destination)
        if source.kind == "npm":
            return _materialize_npm_package(
                source, self.root / "sources" / marketplace / name
            )
        raise PluginError(f"unsupported plugin source: {source.kind}")

    def prepare_artifact(
        self,
        name: str,
        marketplace: str,
        *,
        expected_sha256: str = "",
        expected_commit: str = "",
        expected_marketplace_snapshot: str = "",
    ) -> PreparedPluginArtifact:
        """Resolve, validate and freeze a plugin without marking it installed."""

        market = self._marketplace_records().get(marketplace)
        if market is None:
            raise PluginError(f"unknown marketplace: {marketplace}")
        current_snapshot = market.snapshot_commit or market.catalog_digest
        if expected_marketplace_snapshot and (
            current_snapshot.casefold() != expected_marketplace_snapshot.casefold()
        ):
            raise PluginError("marketplace snapshot changed after capability search; search again")
        entry = self._marketplace_entry(name, marketplace)
        source = entry.get("source") or entry.get("path")
        if not isinstance(source, (str, dict)):
            raise PluginError(f"plugin {name!r} has no valid source")
        try:
            source_config = PluginSourceConfig.from_value(source)
        except SpecError as exc:
            raise PluginError(f"invalid source for plugin {name!r}: {exc}") from exc
        source_path = self._resolve_marketplace_plugin(
            name,
            marketplace,
            expected_sha256=expected_sha256,
            expected_commit=expected_commit,
        ).resolve()
        resolved_commit = _git_head_if_repository(source_path)
        if expected_commit and resolved_commit.casefold() != expected_commit.casefold():
            # Relative sources inherit the immutable marketplace checkout commit.
            if source_config.kind != "relative" or current_snapshot.casefold() != expected_commit.casefold():
                raise PluginError(
                    f"plugin commit mismatch: expected {expected_commit}, got {resolved_commit or current_snapshot}"
                )
            resolved_commit = expected_commit
        staging = self.root / "staging"
        staging.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix="artifact.", dir=str(staging)))
        try:
            market_path = Path(market.snapshot_path)
            market_root = market_path if market_path.is_dir() else market_path.parent
            if market_root.name == ".claude-plugin":
                market_root = market_root.parent
            normalized = temporary / "plugin"
            copy_marketplace_plugin_tree(source_path, normalized, market_root)
            validate_plugin(normalized, entry)
            digest = plugin_tree_digest(normalized)
            archive_pin = source_config.kind == "archive"
            if expected_sha256 and not archive_pin and digest.casefold() != expected_sha256.casefold():
                raise PluginError(
                    f"plugin sha256 mismatch: expected {expected_sha256}, got {digest}"
                )
            artifact_root = (self.root / "artifacts" / digest).resolve()
            destination = artifact_root / "plugin"
            if not _inside(destination, self.root / "artifacts"):
                raise PluginError("computed artifact path escaped managed artifact cache")
            if not destination.exists():
                artifact_root.parent.mkdir(parents=True, exist_ok=True)
                publish = Path(
                    tempfile.mkdtemp(prefix=f".{digest[:12]}.", dir=str(artifact_root.parent))
                )
                try:
                    shutil.copytree(normalized, publish / "plugin", symlinks=True)
                    validate_plugin(publish / "plugin", entry)
                    try:
                        os.replace(publish, artifact_root)
                    except OSError:
                        if not artifact_root.exists():
                            raise
                finally:
                    shutil.rmtree(publish, ignore_errors=True)
            if plugin_tree_digest(destination) != digest:
                raise PluginError("prepared artifact cache failed digest verification")
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        try:
            tier = plugin_trust_tier(TrustTier(market.trust_tier), source_config)
        except ValueError:
            tier = TrustTier.COMMUNITY
        encoded_source = json.dumps(
            source, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        identity = hashlib.sha256(
            f"{market.source_id}:{encoded_source}".encode("utf-8")
        ).hexdigest()
        artifact = PreparedPluginArtifact(
            name=name,
            marketplace=marketplace,
            path=str(destination),
            digest=digest,
            commit=resolved_commit or (current_snapshot if source_config.kind == "relative" else ""),
            marketplace_snapshot=current_snapshot,
            source_identity=identity,
            trust_tier=tier.value,
        )
        self.audit.write(
            "artifact_resolve",
            {
                "plugin": f"{name}@{marketplace}",
                "source_identity": identity,
                "digest": digest,
                "commit": artifact.commit,
            },
        )
        return artifact

    def install(
        self,
        source: str | dict[str, Any],
        marketplace: str = "local",
        *,
        expected_sha256: str = "",
        expected_commit: str = "",
        resolved_version: str = "",
        expected_marketplace_snapshot: str = "",
        prepared_path: str = "",
        expected_prepared_digest: str = "",
    ) -> PluginRecord:
        expected_sha256 = expected_sha256.strip().lower()
        expected_commit = expected_commit.strip().lower()
        expected_marketplace_snapshot = expected_marketplace_snapshot.strip().lower()
        if expected_sha256 and not is_sha256_pin(expected_sha256):
            raise PluginError("plugin sha256 pin must contain exactly 64 hexadecimal characters")
        if expected_commit and not is_git_commit_pin(expected_commit):
            raise PluginError("plugin commit pin must be a full 40- or 64-character object id")
        if marketplace != "local" and expected_marketplace_snapshot:
            market_record = self._marketplace_records().get(marketplace)
            current_snapshot = (
                market_record.snapshot_commit or market_record.catalog_digest
                if market_record is not None else ""
            )
            if current_snapshot.casefold() != expected_marketplace_snapshot:
                raise PluginError("marketplace snapshot changed after capability search; search again")
        temporary_source: Path | None = None
        source_text = source if isinstance(source, str) else json.dumps(source, sort_keys=True)
        entry: dict[str, Any] | None = None
        prepared = bool(prepared_path)
        if prepared:
            if marketplace == "local" or not isinstance(source, str):
                raise PluginError("prepared plugin installation requires a marketplace plugin name")
            source_path = Path(prepared_path).expanduser().resolve()
            artifact_root = (self.root / "artifacts").resolve()
            if not _inside(source_path, artifact_root):
                raise PluginError("prepared plugin path is outside the managed artifact cache")
            entry = self._marketplace_entry(source, marketplace)
            source_text = f"prepared:{expected_prepared_digest or source_path.parent.name}"
        elif isinstance(source, str) and _REMOTE_SOURCE.match(source):
            staging = self.root / "staging"
            staging.mkdir(parents=True, exist_ok=True)
            temporary_source = Path(tempfile.mkdtemp(prefix="plugin.", dir=str(staging)))
            source_path = temporary_source / "source"
            _clone_remote(source, source_path, commit=expected_commit or None)
        elif isinstance(source, dict):
            if marketplace == "local":
                raise PluginError("structured plugin sources require a marketplace")
            name = str(source.get("name") or "").strip()
            if not name:
                raise PluginError("structured plugin install requires a plugin name")
            entry = self._marketplace_entry(name, marketplace)
            source_path = self._resolve_marketplace_plugin(
                name, marketplace,
                expected_sha256=expected_sha256,
                expected_commit=expected_commit,
            )
        else:
            source_path = Path(source).expanduser()
        if not source_path.exists() and marketplace != "local":
            source_name = str(source)
            entry = self._marketplace_entry(source_name, marketplace)
            source_spec = entry.get("source")
            source_sha = source_spec.get("sha") if isinstance(source_spec, dict) else ""
            source_archive_sha = source_spec.get("sha256") if isinstance(source_spec, dict) else ""
            expected_sha256 = (
                expected_sha256 or str(entry.get("sha256") or source_archive_sha or "")
            ).strip().lower()
            expected_commit = (
                expected_commit or str(entry.get("commit") or source_sha or "")
            ).strip().lower()
            if expected_sha256 and not is_sha256_pin(expected_sha256):
                raise PluginError("plugin sha256 pin must contain exactly 64 hexadecimal characters")
            if expected_commit and not is_git_commit_pin(expected_commit):
                raise PluginError("plugin commit pin must be a full 40- or 64-character object id")
            source_path = self._resolve_marketplace_plugin(
                source_name,
                marketplace,
                expected_sha256=expected_sha256,
                expected_commit=expected_commit,
            )
            if expected_commit:
                market_config = self.marketplace_sources().get(marketplace)
                tagged_checkout = self.root / "marketplace-commits" / marketplace / expected_commit
                if tagged_checkout.is_dir() and market_config is not None:
                    tagged_root = (
                        (tagged_checkout / market_config.path).resolve()
                        if market_config.path else tagged_checkout.resolve()
                    )
                    tagged_manifest = self._load_marketplace_manifest(tagged_root)
                    entry = next(
                        (
                            item for item in tagged_manifest.get("plugins", [])
                            if isinstance(item, dict)
                            and str(item.get("name") or "") == source_name
                        ),
                        entry,
                    )
        source_path = source_path.resolve()
        # Keep the owning repository revision before copying a marketplace subdirectory
        # into a normalized, git-less staging tree. Dependency tag resolution and the
        # install audit record must retain that immutable marketplace commit.
        resolved_source_commit = _git_head_if_repository(source_path)
        normalized_source: Path | None = None
        if entry is not None and not prepared:
            staging = self.root / "staging"
            staging.mkdir(parents=True, exist_ok=True)
            normalized_source = Path(tempfile.mkdtemp(prefix="normalized-plugin.", dir=str(staging)))
            market_record = self._marketplace_records().get(marketplace)
            market_path = Path(market_record.snapshot_path) if market_record is not None else source_path
            market_root = market_path if market_path.is_dir() else market_path.parent
            tagged_checkout = self.root / "marketplace-commits" / marketplace / expected_commit
            if expected_commit and tagged_checkout.is_dir():
                market_config = self.marketplace_sources().get(marketplace)
                market_root = (
                    (tagged_checkout / market_config.path).resolve()
                    if market_config is not None and market_config.path
                    else tagged_checkout.resolve()
                )
            if market_root.name == ".claude-plugin":
                market_root = market_root.parent
            copy_marketplace_plugin_tree(source_path, normalized_source / "plugin", market_root)
            source_path = (normalized_source / "plugin").resolve()
        try:
            manifest = validate_plugin(source_path, entry)
            manifest.pop("_warnings", None)
            name = str(manifest["name"])
            version = str(manifest.get("version") or "")
            actual_commit = (
                expected_commit
                if prepared and expected_commit
                else resolved_source_commit or _git_head_if_repository(source_path)
            )
            if expected_commit and actual_commit.casefold() != expected_commit.casefold():
                raise PluginError(
                    f"plugin commit mismatch: expected {expected_commit}, got {actual_commit or '(none)'}"
                )
            source_digest = plugin_tree_digest(source_path)
            if expected_prepared_digest and source_digest.casefold() != expected_prepared_digest.casefold():
                raise PluginError(
                    "prepared plugin digest changed between planning and installation"
                )
            archive_pin = bool(
                entry is not None
                and isinstance(entry.get("source"), dict)
                and entry["source"].get("source") == "archive"
            )
            if expected_sha256 and not archive_pin and source_digest.casefold() != expected_sha256.casefold():
                raise PluginError(
                    f"plugin sha256 mismatch: expected {expected_sha256}, got {source_digest}"
                )
            plugin_id = f"{name}@{marketplace}"
            cache_root = (self.root / "cache").resolve()
            cache_key = (
                expected_commit[:12] or expected_sha256[:12]
                or actual_commit[:12] or source_digest[:12] or version
            )
            destination = cache_root / marketplace / name / cache_key
            if not _inside(destination, cache_root):
                raise PluginError("computed cache path escaped managed plugin cache")
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = Path(
                    tempfile.mkdtemp(prefix=f".{name}.", dir=str(destination.parent))
                )
                try:
                    shutil.copytree(source_path, temporary / "plugin", symlinks=True)
                    validate_plugin(temporary / "plugin", entry)
                    os.replace(temporary / "plugin", destination)
                finally:
                    shutil.rmtree(temporary, ignore_errors=True)
            cached_digest = plugin_tree_digest(destination)
            if cached_digest != source_digest:
                raise PluginError(
                    "plugin cache content differs from the validated source; remove the stale cache entry"
                )
            keywords_raw = manifest.get("keywords", [])
            keywords = tuple(
                str(item).strip() for item in keywords_raw
                if isinstance(item, str) and str(item).strip()
            ) if isinstance(keywords_raw, list) else ()
            current_market = self._marketplace_records().get(marketplace)
            trust_tier = TrustTier.LOCAL_USER_DECLARED
            source_identity = "local:" + os.path.normcase(str(source_path))
            if current_market is not None:
                try:
                    market_tier = TrustTier(current_market.trust_tier)
                except ValueError:
                    market_tier = TrustTier.COMMUNITY
                artifact_source: PluginSourceConfig | None = None
                if entry is not None and isinstance(entry.get("source"), (str, dict)):
                    try:
                        artifact_source = PluginSourceConfig.from_value(entry["source"])
                    except SpecError:
                        pass
                trust_tier = (
                    plugin_trust_tier(market_tier, artifact_source)
                    if artifact_source is not None else market_tier
                )
                encoded_source = json.dumps(
                    entry.get("source") if entry is not None else source_text,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                source_identity = hashlib.sha256(
                    f"{current_market.source_id}:{encoded_source}".encode("utf-8")
                ).hexdigest()
            record = PluginRecord(
                plugin_id=plugin_id,
                name=name,
                marketplace=marketplace,
                path=str(destination),
                source=source_text,
                version=version,
                installed_at=time.time(),
                description=str(manifest.get("description") or "").strip(),
                keywords=keywords,
                integrity=cached_digest,
                artifact_digest=expected_sha256 or cached_digest,
                commit=actual_commit,
                source_kind=(
                    PluginSourceConfig.from_value(entry["source"]).kind
                    if entry is not None else ("git" if _REMOTE_SOURCE.match(source_text) else "local")
                ),
                source_ref=(
                    str(entry.get("source", {}).get("ref") or "")
                    if entry is not None and isinstance(entry.get("source"), dict) else ""
                ),
                resolved_version=resolved_version,
                marketplace_snapshot=(
                    (market_record.snapshot_commit or market_record.catalog_digest) if (
                        market_record := self._marketplace_records().get(marketplace)
                    ) is not None else ""
                ),
                components=declared_components(source_path, manifest),
                dependencies=tuple(_dependency_labels(manifest.get("dependencies"))),
                manifest=manifest,
                source_identity=source_identity,
                trust_tier=trust_tier.value,
            )
            with FileLock(self.state_lock_path):
                records = self.records()
                records[plugin_id] = record
                self._save_records(records)
            self.audit.write(
                "artifact_verify",
                {
                    "plugin_id": plugin_id,
                    "source_identity": source_identity,
                    "artifact_digest": cached_digest,
                    "commit": actual_commit,
                    "result": "success",
                },
            )
            self.audit.write(
                "install_complete",
                {
                    "plugin_id": plugin_id,
                    "trust_tier": trust_tier.value,
                    "components": list(record.components),
                },
            )
            return record
        finally:
            if temporary_source is not None:
                shutil.rmtree(temporary_source, ignore_errors=True)
            if normalized_source is not None:
                shutil.rmtree(normalized_source, ignore_errors=True)

    def uninstall(self, plugin_id: str) -> None:
        records = self.records()
        record = records.get(plugin_id)
        if record is None:
            raise PluginError(f"plugin is not installed: {plugin_id}")
        target = Path(record.path)
        cache_root = (self.root / "cache").resolve()
        if not _inside(target, cache_root):
            raise PluginError("refusing to remove a path outside the managed plugin cache")
        # Enforce dependency constraints while the record still exists, then remove
        # enablement at both scopes before deleting the installation record.
        self.set_enabled(plugin_id, False, scope="project")
        self.set_enabled(plugin_id, False, scope="user")
        if target.exists():
            shutil.rmtree(target)
        with FileLock(self.state_lock_path):
            latest = self.records()
            latest.pop(plugin_id, None)
            self._save_records(latest)

    def update(self, plugin_id: str) -> PluginRecord:
        record = self.records().get(plugin_id)
        if record is None:
            raise PluginError(f"plugin is not installed: {plugin_id}")
        if record.marketplace == "local":
            return self.install(record.source, "local")
        return self.install(record.name, record.marketplace)

    def _dependency_tag(
        self, plugin_name: str, marketplace: str, ranges: list[str]
    ) -> tuple[str, str] | None:
        """Resolve Claude's ``{plugin}--v{version}`` tags for relative git plugins."""

        # Deferred: tests monkeypatch the package-level _git_output.
        from agent_core.plugins import _git_output

        if not ranges:
            return None
        config = self.marketplace_sources().get(marketplace)
        if config is None or config.kind not in {"github", "git"}:
            return None
        entry = self._marketplace_entry(plugin_name, marketplace)
        raw_source = entry.get("source") or entry.get("path")
        if not isinstance(raw_source, (str, dict)):
            raise PluginError(f"dependency {plugin_name!r} has no valid source")
        try:
            plugin_source = PluginSourceConfig.from_value(raw_source)
        except SpecError as exc:
            raise PluginError(f"invalid dependency source for {plugin_name}: {exc}") from exc
        # npm versions are fixed by the package source and checked after materialization.
        # External repositories likewise own their own immutable source pin; marketplace
        # tags select versions only for plugins stored in the marketplace repository.
        if plugin_source.kind != "relative":
            return None
        url = f"https://github.com/{config.repo}.git" if config.kind == "github" else config.url
        prefix = f"refs/tags/{plugin_name}--v"
        output = _git_output(["ls-remote", "--refs", "--tags", url, f"{prefix}*"])
        versions: dict[str, str] = {}
        for line in output.splitlines():
            pieces = line.split(maxsplit=1)
            if len(pieces) != 2 or not pieces[1].startswith(prefix):
                continue
            version = pieces[1][len(prefix):]
            if is_git_commit_pin(pieces[0]):
                versions[version] = pieces[0].casefold()
        from agent_core.semver import select_highest

        selected = select_highest(versions, ranges)
        if selected is None:
            available = ", ".join(sorted(versions)) or "(none)"
            raise PluginError(
                f"no-matching-tag: {plugin_name}@{marketplace} has no tag satisfying "
                f"{' & '.join(ranges)}; available: {available}"
            )
        return selected, versions[selected]

    def dependency_plan(self, name: str, marketplace: str) -> list[dict[str, Any]]:
        """Return a dependency-first install plan without executing plugin content."""

        from agent_core.semver import satisfies

        root_record = self._marketplace_records().get(marketplace)
        if root_record is None:
            raise PluginError(f"unknown marketplace: {marketplace}")
        root_manifest = self._load_marketplace_manifest(Path(root_record.snapshot_path))
        cross_allowed = {
            str(item) for item in root_manifest.get("allowCrossMarketplaceDependenciesOn", [])
            if isinstance(item, str)
        }
        visiting: list[str] = []
        complete: set[str] = set()
        plan: list[dict[str, Any]] = []
        constraints: dict[str, list[str]] = {}

        def visit(plugin_name: str, market_name: str, required_by: str = "") -> None:
            plugin_id = f"{plugin_name}@{market_name}"
            if plugin_id in visiting:
                cycle = " -> ".join([*visiting[visiting.index(plugin_id):], plugin_id])
                raise PluginError(f"dependency cycle: {cycle}")
            if plugin_id in complete:
                return
            if market_name not in self._marketplace_records():
                raise PluginError(f"dependency marketplace is not configured: {market_name}")
            visiting.append(plugin_id)
            entry = self._marketplace_entry(plugin_name, market_name)
            resolved_manifest = dict(entry)
            resolved_commit = ""
            resolved_tag_version = ""
            tag = self._dependency_tag(
                plugin_name, market_name, constraints.get(plugin_id, [])
            )
            if tag is not None:
                resolved_tag_version, resolved_commit = tag
            if "dependencies" not in entry or "version" not in entry or tag is not None:
                try:
                    candidate = self._resolve_marketplace_plugin(
                        plugin_name, market_name, expected_commit=resolved_commit
                    )
                    resolved_commit = resolved_commit or _git_head_if_repository(candidate)
                    resolved_manifest, _warnings = resolve_plugin_manifest(candidate, entry)
                except (OSError, SpecError, PluginError) as exc:
                    raise PluginError(
                        f"could not inspect dependency metadata for {plugin_id}: {exc}"
                    ) from exc
            dependencies = resolved_manifest.get("dependencies", [])
            if not isinstance(dependencies, list):
                raise PluginError(f"dependencies for {plugin_id} must be an array")
            for dependency in dependencies:
                if isinstance(dependency, str):
                    dep_name, dep_market, version_range = dependency, market_name, ""
                elif isinstance(dependency, dict):
                    dep_name = str(dependency.get("name") or "").strip()
                    dep_market = str(dependency.get("marketplace") or market_name).strip()
                    version_range = str(dependency.get("version") or "").strip()
                else:
                    raise PluginError(f"invalid dependency in {plugin_id}")
                if not is_safe_plugin_name(dep_name):
                    raise PluginError(f"invalid dependency name in {plugin_id}")
                if dep_market != market_name and dep_market not in cross_allowed:
                    raise PluginError(
                        f"cross-marketplace dependency {dep_name}@{dep_market} is not allowed; "
                        "add it to allowCrossMarketplaceDependenciesOn"
                    )
                dep_id = f"{dep_name}@{dep_market}"
                if version_range:
                    constraints.setdefault(dep_id, []).append(version_range)
                visit(dep_name, dep_market, plugin_id)
            visiting.pop()
            version = resolved_tag_version or str(resolved_manifest.get("version") or "")
            for constraint in constraints.get(plugin_id, []):
                if version:
                    try:
                        matched = satisfies(version, constraint)
                    except ValueError as exc:
                        raise PluginError(f"invalid dependency semver range {constraint!r}") from exc
                    if not matched:
                        raise PluginError(
                            f"dependency-version-unsatisfied: {plugin_id} {version} does not satisfy {constraint}"
                        )
            complete.add(plugin_id)
            plan.append({
                "name": plugin_name,
                "marketplace": market_name,
                "plugin_id": plugin_id,
                "version": version or None,
                "required_by": required_by or None,
                "constraints": list(constraints.get(plugin_id, [])),
                "source_commit": resolved_commit or None,
                "resolved_version": resolved_tag_version or None,
            })

        visit(name, marketplace)
        for item in plan:
            version = item.get("version")
            if not isinstance(version, str):
                continue
            for constraint in constraints.get(str(item["plugin_id"]), []):
                try:
                    matched = satisfies(version, constraint)
                except ValueError as exc:
                    raise PluginError(f"invalid dependency semver range {constraint!r}") from exc
                if not matched:
                    raise PluginError(
                        f"range-conflict: {item['plugin_id']} {version} does not satisfy {constraint}"
                    )
        return plan

    def install_dependency_plan(self, plan: list[dict[str, Any]]) -> PluginRecord:
        """Install a previously reviewed graph transactionally at the record layer."""

        if not plan:
            raise PluginError("dependency plan is empty")
        before = self.records()
        installed: list[PluginRecord] = []
        try:
            for index, item in enumerate(plan):
                record = self.install(
                    str(item["name"]),
                    str(item["marketplace"]),
                    expected_commit=str(item.get("source_commit") or ""),
                    resolved_version=str(item.get("resolved_version") or ""),
                )
                record.auto_installed = index < len(plan) - 1 and record.plugin_id not in before
                required_by = str(item.get("required_by") or "")
                if required_by:
                    record.required_by = tuple(dict.fromkeys([*record.required_by, required_by]))
                with FileLock(self.state_lock_path):
                    records = self.records()
                    records[record.plugin_id] = record
                    self._save_records(records)
                installed.append(record)
        except Exception:
            with FileLock(self.state_lock_path):
                self._save_records(before)
            raise
        return installed[-1]

    def details(self, plugin_id: str) -> dict[str, Any]:
        record = self.records().get(plugin_id)
        if record is None:
            raise PluginError(f"plugin is not installed: {plugin_id}")
        token_chars = 0
        root = Path(record.path)
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.casefold() in {".md", ".json", ".js", ".mjs"}:
                try:
                    token_chars += len(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError):
                    continue
        return {
            "plugin_id": plugin_id,
            "version": record.version or None,
            "resolved_version": record.resolved_version or None,
            "marketplace": record.marketplace,
            "source_kind": record.source_kind,
            "integrity": record.integrity,
            "commit": record.commit or None,
            "marketplace_snapshot": record.marketplace_snapshot or None,
            "components": list(record.components),
            "dependencies": list(record.dependencies),
            "enabled": plugin_id in self.enabled_ids(),
            "projected_tokens": (token_chars + 3) // 4,
        }

    def prune(self, *, dry_run: bool = False) -> list[str]:
        records = self.records()
        required = {
            dependency.split(" ", 1)[0]
            for item in records.values()
            for dependency in item.dependencies
        }
        orphans = sorted(
            plugin_id for plugin_id, item in records.items()
            if item.auto_installed
            and item.name not in required
            and plugin_id not in self.enabled_ids()
        )
        if not dry_run:
            for plugin_id in orphans:
                self.uninstall(plugin_id)
        return orphans

    def configure(self, plugin_id: str, values: dict[str, str]) -> None:
        """Store userConfig without ever persisting a secret in plaintext."""

        record = self.records().get(plugin_id)
        if record is None or not isinstance(record.manifest, dict):
            raise PluginError(f"plugin is not installed: {plugin_id}")
        schema = record.manifest.get("userConfig", {})
        if not isinstance(schema, dict):
            raise PluginError("plugin does not declare userConfig")
        safe: dict[str, Any] = {}
        for key, value in values.items():
            option = schema.get(key)
            if not isinstance(option, dict):
                raise PluginError(f"unknown plugin config key: {key}")
            if option.get("sensitive") is True:
                if value.startswith("${") and value.endswith("}"):
                    environment_name = value[2:-1]
                    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", environment_name):
                        raise PluginError(f"invalid environment reference for {key}")
                    safe[key] = value
                    continue
                if not secret_store.available():
                    raise PluginError(
                        f"sensitive plugin config {key} must reference an environment variable "
                        "because no system secret store is available"
                    )
                target = f"Polaris/plugin-config/{plugin_id}/{key}"
                try:
                    secret_store.put(target, value)
                except secret_store.SecretStoreError as exc:
                    raise PluginError(f"could not store sensitive plugin config {key}: {exc}") from exc
                safe[key] = _KEYCHAIN_PREFIX + target.removeprefix("Polaris/plugin-config/")
                continue
            option_type = str(option.get("type") or "string")
            try:
                if option.get("multiple") is True:
                    decoded = json.loads(value)
                    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
                        raise ValueError
                    safe[key] = decoded
                elif option_type == "boolean":
                    if value.casefold() not in {"true", "false"}:
                        raise ValueError
                    safe[key] = value.casefold() == "true"
                elif option_type == "number":
                    number = float(value)
                    minimum, maximum = option.get("min"), option.get("max")
                    if minimum is not None and number < float(minimum):
                        raise ValueError
                    if maximum is not None and number > float(maximum):
                        raise ValueError
                    safe[key] = int(number) if number.is_integer() else number
                elif option_type in {"string", "directory", "file"}:
                    safe[key] = value
                else:
                    raise ValueError
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise PluginError(f"invalid {option_type} value for plugin config {key}") from exc
        try:
            settings_path = user_settings_path()
        except RuntimeError as exc:
            raise PluginError("user home directory is unavailable") from exc
        previous: dict[str, Any] = {}
        if settings_path.is_file():
            try:
                import tomllib

                loaded = tomllib.loads(settings_path.read_text(encoding="utf-8"))
                table = loaded.get("plugin_configs", {})
                existing = table.get(plugin_id, {}) if isinstance(table, dict) else {}
                if isinstance(existing, dict):
                    previous = dict(existing)
            except (OSError, UnicodeDecodeError, ValueError):
                pass
        with FileLock(settings_path.with_suffix(settings_path.suffix + ".plugins.lock")):
            # Re-read inside the lock so two plugin configuration updates do not lose
            # each other's keys.
            latest: dict[str, Any] = {}
            if settings_path.is_file():
                try:
                    import tomllib

                    loaded = tomllib.loads(settings_path.read_text(encoding="utf-8"))
                    table = loaded.get("plugin_configs", {})
                    current = table.get(plugin_id, {}) if isinstance(table, dict) else {}
                    if isinstance(current, dict):
                        latest = dict(current)
                except (OSError, UnicodeDecodeError, ValueError):
                    latest = previous
            update_toml_table(settings_path, "plugin_configs", {plugin_id: {**latest, **safe}})

    def configured_options(
        self, plugin_id: str, manifest: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Resolve defaults and system-store/environment-backed secrets."""

        schema = manifest.get("userConfig", {})
        if not isinstance(schema, dict):
            return {}, {}
        stored: dict[str, Any] = {}
        try:
            import tomllib

            path = user_settings_path()
            loaded = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            configs = loaded.get("plugin_configs", {})
            value = configs.get(plugin_id, {}) if isinstance(configs, dict) else {}
            if isinstance(value, dict):
                stored = value
        except (OSError, RuntimeError, UnicodeDecodeError, ValueError):
            stored = {}
        resolved: dict[str, Any] = {}
        public: dict[str, Any] = {}
        missing: list[str] = []
        for key, option in schema.items():
            if not isinstance(option, dict):
                continue
            value = stored.get(key, option.get("default"))
            sensitive = option.get("sensitive") is True
            if sensitive and isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                value = os.getenv(value[2:-1])
            elif sensitive and isinstance(value, str) and value.startswith(_KEYCHAIN_PREFIX):
                target = "Polaris/plugin-config/" + value.removeprefix(_KEYCHAIN_PREFIX)
                try:
                    value = secret_store.get(target)
                except secret_store.SecretStoreError:
                    value = None
            if value is None or value == "":
                if option.get("required") is True:
                    missing.append(str(key))
                continue
            resolved[str(key)] = value
            if not sensitive:
                public[str(key)] = value
        if missing:
            raise PluginError(
                f"plugin {plugin_id} requires configuration: {', '.join(missing)}"
            )
        return resolved, public

    def enabled_ids(self) -> list[str]:
        try:
            settings = user_settings_path()
        except RuntimeError:
            settings = Path("__no_user_plugin_settings__")
        user = _read_plugin_list(settings, "enabled")
        local = _read_plugin_list(self.workspace / "agent.local.toml", "enabled")
        disabled = set(
            _read_plugin_list(self.workspace / "agent.local.toml", "disabled")
        )
        return [
            item
            for item in dict.fromkeys([*user, *local])
            if item not in disabled
        ]

    def component_selections(self) -> dict[str, tuple[str, ...]]:
        """Return per-project component filters; absence preserves legacy all-components."""

        path = self.workspace / "agent.local.toml"
        if not path.is_file():
            return {}
        try:
            import tomllib

            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            return {}
        plugins = data.get("plugins")
        raw = plugins.get("components", {}) if isinstance(plugins, dict) else {}
        if not isinstance(raw, dict):
            return {}
        allowed = _PLUGIN_COMPONENTS
        result: dict[str, tuple[str, ...]] = {}
        for plugin_id, values in raw.items():
            if not isinstance(values, list):
                continue
            result[str(plugin_id)] = tuple(
                dict.fromkeys(str(value) for value in values if str(value) in allowed)
            )
        return result

    def hook_selections(self) -> dict[str, tuple[str, ...]]:
        path = self.workspace / "agent.local.toml"
        if not path.is_file():
            return {}
        try:
            import tomllib

            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            return {}
        plugins = data.get("plugins")
        raw = plugins.get("allowed_hooks", {}) if isinstance(plugins, dict) else {}
        if not isinstance(raw, dict):
            return {}
        return {
            str(plugin_id): tuple(str(value) for value in values if isinstance(value, str))
            for plugin_id, values in raw.items()
            if isinstance(values, list)
        }

    def set_enabled(
        self,
        plugin_id: str,
        enabled: bool,
        *,
        scope: str = "project",
        _seen: set[str] | None = None,
    ) -> None:
        records = self.records()
        if plugin_id not in records:
            raise PluginError(f"plugin is not installed: {plugin_id}")
        seen = _seen or set()
        if plugin_id in seen:
            return
        seen.add(plugin_id)
        record = records[plugin_id]
        dependency_ids = []
        for label in record.dependencies:
            target = label.split(" ", 1)[0]
            dependency_ids.append(
                target if "@" in target else f"{target}@{record.marketplace}"
            )
        if enabled:
            missing = [item for item in dependency_ids if item not in records]
            if missing:
                raise PluginError(
                    "dependency-unsatisfied: install " + ", ".join(missing)
                )
            for dependency_id in dependency_ids:
                self.set_enabled(
                    dependency_id, True, scope=scope, _seen=seen
                )
        else:
            enabled_now = set(self.enabled_ids())
            dependents = []
            for other_id, other in records.items():
                if other_id == plugin_id or other_id not in enabled_now:
                    continue
                for label in other.dependencies:
                    target = label.split(" ", 1)[0]
                    target_id = target if "@" in target else f"{target}@{other.marketplace}"
                    if target_id == plugin_id:
                        dependents.append(other_id)
            if dependents:
                raise PluginError(
                    f"{plugin_id} is still required by " + ", ".join(sorted(dependents))
                )
        if scope == "user":
            try:
                path = user_settings_path()
            except RuntimeError as exc:
                raise PluginError("user home directory is unavailable") from exc
            with FileLock(path.with_suffix(path.suffix + ".plugins.lock")):
                raw = _read_plugin_list(path, "enabled")
                values = _changed_enabled(raw, plugin_id, enabled)
                update_toml_table(path, "plugins", {"enabled": values})
        elif scope == "project":
            local_path = self.workspace / "agent.local.toml"
            with FileLock(local_path.with_suffix(local_path.suffix + ".plugins.lock")):
                raw = _read_plugin_list(local_path, "enabled")
                disabled = _read_plugin_list(local_path, "disabled")
                values = _changed_enabled(raw, plugin_id, enabled)
                disabled_values = _changed_enabled(disabled, plugin_id, not enabled)
                update_local_table(
                    self.workspace,
                    "plugins",
                    {"enabled": values, "disabled": disabled_values},
                )
        else:
            raise PluginError("scope must be project or user")

    def set_activation(
        self,
        plugin_id: str,
        components: tuple[str, ...],
        *,
        allowed_hooks: tuple[str, ...] = (),
    ) -> None:
        """Atomically enable a plugin with a project-local component selection."""

        if plugin_id not in self.records():
            raise PluginError(f"plugin is not installed: {plugin_id}")
        local_path = self.workspace / "agent.local.toml"
        with FileLock(local_path.with_suffix(local_path.suffix + ".plugins.lock")):
            enabled = _changed_enabled(
                _read_plugin_list(local_path, "enabled"), plugin_id, True
            )
            disabled = _changed_enabled(
                _read_plugin_list(local_path, "disabled"), plugin_id, False
            )
            selected = self.component_selections()
            selected[plugin_id] = tuple(dict.fromkeys(components))
            hook_ids = self.hook_selections()
            hook_ids[plugin_id] = tuple(dict.fromkeys(allowed_hooks))
            update_local_table(
                self.workspace,
                "plugins",
                {
                    "enabled": enabled,
                    "disabled": disabled,
                    "components": {key: list(value) for key, value in selected.items()},
                    "allowed_hooks": {key: list(value) for key, value in hook_ids.items()},
                },
            )

    def executable_components(self, plugin_id: str) -> bool:
        record = self.records().get(plugin_id)
        if record is None:
            raise PluginError(f"plugin is not installed: {plugin_id}")
        risky = {
            "hooks", "mcp", "lsp", "workflows", "monitors", "channels", "bin", "settings"
        }
        return bool(set(record.components) & risky)

    def build_bundle(
        self,
        agent: "ReActAgent",
        *,
        enabled_ids: list[str] | None = None,
        component_selections: dict[str, tuple[str, ...]] | None = None,
        hook_selections: dict[str, tuple[str, ...]] | None = None,
    ) -> PluginBundle:
        # Deferred: tests monkeypatch the package-level MCPClientManager.
        from agent_core.plugins import MCPClientManager
        records = self.records()
        skills: list[Skill] = []
        hook_pairs: list[tuple[str, Any]] = []
        agents: dict[str, str] = {}
        component_registry: dict[str, Any] = {
            "lsp": [], "workflows": {}, "output_styles": {}, "themes": {},
            "monitors": [], "channels": [], "user_config": {}, "bin": [],
            "settings": {},
        }
        mcp_servers: list[MCPServerConfig] = []
        selected_components = (
            self.component_selections() if component_selections is None else component_selections
        )
        selected_hooks = self.hook_selections() if hook_selections is None else hook_selections
        for plugin_id in self.enabled_ids() if enabled_ids is None else enabled_ids:
            record = records.get(plugin_id)
            if record is None:
                raise PluginError(f"enabled plugin is not installed: {plugin_id}")
            root = Path(record.path).resolve()
            manifest = dict(record.manifest) if isinstance(record.manifest, dict) else validate_plugin(root)
            if record.integrity:
                actual_digest = plugin_tree_digest(root)
                if actual_digest != record.integrity:
                    raise PluginError(
                        f"installed plugin {plugin_id} failed integrity verification: "
                        f"expected {record.integrity}, got {actual_digest}"
                    )
            namespace = str(manifest["name"])
            configured, public_config = self.configured_options(plugin_id, manifest)
            plugin_data = self.root / "data" / record.marketplace / record.name
            plugin_data.mkdir(parents=True, exist_ok=True)
            components = set(
                selected_components.get(
                    plugin_id,
                    (
                        "skills", "agents", "hooks", "mcp", "lsp", "workflows",
                        "monitors", "channels", "output-styles", "themes",
                        "user-config", "bin", "settings",
                    ),
                )
            )
            if "skills" in components:
                loaded_skills = _load_plugin_skills(
                    root, namespace, manifest, agent.session.workspace,
                    plugin_data, public_config,
                )
                _apply_skill_provenance(loaded_skills, record)
                skills.extend(loaded_skills)
            if "agents" in components:
                plugin_agents, agent_skills = _load_plugin_agents(
                    root, namespace, manifest, agent.session.workspace,
                    plugin_data, public_config,
                )
                agents.update(plugin_agents)
                _apply_skill_provenance(agent_skills, record)
                skills.extend(agent_skills)
            if "hooks" in components:
                allowed = selected_hooks.get(plugin_id)
                hook_pairs.extend(
                    _load_plugin_hooks(
                        root,
                        agent,
                        namespace,
                        manifest,
                        plugin_id=plugin_id,
                        allowed_hook_ids=allowed,
                        workspace=agent.session.workspace,
                        plugin_data=plugin_data,
                        user_config=public_config,
                        runtime_config=configured,
                    )
                )
            if "mcp" in components:
                plugin_mcp = _load_plugin_mcp(
                    root, namespace, manifest, agent.session.workspace,
                    plugin_data, public_config, configured,
                )
                if plugin_id in selected_components:
                    plugin_mcp = _restrict_autonomous_mcp(agent, root, plugin_mcp)
                for server in plugin_mcp:
                    server.discovered = True
                    server.trust_tier = record.trust_tier
                    if record.trust_tier not in {
                        TrustTier.ANTHROPIC_FIRST_PARTY.value,
                        TrustTier.LOCAL_USER_DECLARED.value,
                    }:
                        server.risk = "dangerous"
                        if (server.transport or "stdio").casefold() != "stdio":
                            server.network_policy = "public-only"
                mcp_servers.extend(plugin_mcp)

            metadata = _load_non_mcp_components(
                root, namespace, manifest, components, agent.session.workspace,
                plugin_data, public_config, configured,
            )
            if plugin_id in selected_components and metadata["lsp"]:
                if not agent.sandbox.is_enabled():
                    raise PluginError(f"autonomous LSP from {plugin_id} requires a real sandbox")
                if any(item.env for item in metadata["lsp"]):
                    raise PluginError(f"autonomous LSP from {plugin_id} cannot inject host environment secrets")
            for key in ("lsp", "monitors", "channels", "bin"):
                component_registry[key].extend(metadata[key])
            for key in ("workflows", "output_styles", "themes", "user_config", "settings"):
                component_registry[key].update(metadata[key])

        manager: MCPClientManager | None = None
        tools: list[Any] = []
        if mcp_servers:
            channel_servers = {
                str(item.get("qualified_server") or "")
                for item in component_registry["channels"]
                if isinstance(item, dict)
            }
            active_channels = {item for item in channel_servers if item}
            manager = (
                MCPClientManager(
                    MCPConfig(mcp_servers),
                    channel_servers=active_channels,
                    notification_sink=agent._queue_plugin_channel,
                )
                if active_channels
                else MCPClientManager(MCPConfig(mcp_servers))
            )
            manager.start()
            tools = MCPAdapter(manager).list_tools()
        active_ids = tuple(
            plugin_id
            for plugin_id in (self.enabled_ids() if enabled_ids is None else enabled_ids)
            if plugin_id in records
        )
        return PluginBundle(skills, hook_pairs, manager, tools, agents, component_registry, active_ids)
