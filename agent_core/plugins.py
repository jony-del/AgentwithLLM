"""Claude-compatible plugin installation, validation, and generation swaps.

No marketplace is preloaded. Installation only copies/records files; executable
components (hooks and MCP servers) are activated solely by an explicit enable followed
by ``/reload-plugins``, or by the policy-gated capability manager at a turn boundary.
"""

from __future__ import annotations

import asyncio
import json
import hashlib
import ipaddress
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass, fields, replace
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
from agent_core.hook_adapters import LIFECYCLE_EVENT_ATTRS, build_external_adapter
from agent_core.hooks import ExternalHookSpec
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
from agent_core.sandbox import GuestCapabilityUnavailable, SandboxInvocation
from agent_core.skills import Skill, SkillContext, SkillRegistry, load_skill_file, parse_frontmatter
from agent_core.tools.base import ExecutionScope, Tool
from agent_core.tools.registry import DeferredTool

if TYPE_CHECKING:
    from agent_core.react import ReActAgent

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_REMOTE_SOURCE = re.compile(r"^(?:https?|ssh|git)://|^git@")
_SHA256_PIN = re.compile(r"^[0-9a-fA-F]{64}$")
_IMMUTABLE_GIT_COMMIT = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_PLUGIN_COMPONENTS = frozenset({
    "skills", "agents", "hooks", "mcp", "lsp", "workflows", "monitors",
    "channels", "output-styles", "themes", "user-config", "bin", "settings",
})
_KEYCHAIN_PREFIX = "keychain://Polaris/plugin-config/"


class PluginError(RuntimeError):
    pass


def is_safe_plugin_name(value: str) -> bool:
    return bool(_SAFE_NAME.fullmatch(value))


def is_sha256_pin(value: str) -> bool:
    return bool(_SHA256_PIN.fullmatch(value))


def is_git_commit_pin(value: str) -> bool:
    """Return true only for a full immutable Git object id, never a branch/tag/ref."""

    return bool(_IMMUTABLE_GIT_COMMIT.fullmatch(value))


def _string_tuple(value: object) -> tuple[str, ...]:
    values = value if isinstance(value, list) else str(value or "").split(",")
    return tuple(str(item).strip() for item in values if str(item).strip())


def _positive_int(value: object) -> int | None:
    text = str(value or "").strip()
    return int(text) if text.isdigit() and int(text) > 0 else None


@dataclass(slots=True)
class PluginRecord:
    plugin_id: str
    name: str
    marketplace: str
    path: str
    source: str
    version: str = ""
    installed_at: float = 0.0
    description: str = ""
    keywords: tuple[str, ...] = ()
    integrity: str = ""
    artifact_digest: str = ""
    commit: str = ""
    source_kind: str = ""
    source_ref: str = ""
    resolved_version: str = ""
    marketplace_snapshot: str = ""
    components: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    manifest: dict[str, Any] | None = None
    auto_installed: bool = False
    required_by: tuple[str, ...] = ()
    source_identity: str = ""
    trust_tier: str = TrustTier.COMMUNITY.value


@dataclass(slots=True)
class MarketplaceRecord:
    name: str
    source: dict[str, Any]
    snapshot_path: str
    snapshot_commit: str = ""
    catalog_digest: str = ""
    refreshed_at: float = 0.0
    source_id: str = ""
    canonical_source: str = ""
    trust_tier: str = TrustTier.COMMUNITY.value
    last_attempt: float = 0.0
    last_success: float = 0.0
    failure_count: int = 0
    next_retry: float = 0.0
    last_error: str = ""


@dataclass(frozen=True, slots=True)
class PluginStateStatus:
    schema_version: int
    legacy: bool
    reset_required: bool
    targets: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreparedPluginArtifact:
    name: str
    marketplace: str
    path: str
    digest: str
    commit: str
    marketplace_snapshot: str
    source_identity: str
    trust_tier: str


@dataclass(slots=True)
class PluginBundle:
    skills: list[Skill]
    hooks: list[tuple[str, Any]]
    mcp_manager: MCPClientManager | None
    mcp_tools: list[Any]
    agents: dict[str, str]
    components: dict[str, Any]
    plugin_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class PluginGeneration:
    bundle: PluginBundle
    skills: SkillRegistry
    hooks: Any


def plugin_home() -> Path:
    override = os.getenv("POLARIS_PLUGIN_HOME")
    if override:
        return Path(override).expanduser().resolve()
    try:
        base = Path.home()
    except RuntimeError:
        base = Path.cwd()
    return (base / ".polaris" / "plugins").resolve()


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return default


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _serializable_marketplace_source(config: MarketplaceSourceConfig) -> dict[str, Any]:
    """Persist header references, never literal marketplace credentials."""

    value = config.to_dict()
    headers = value.get("headers")
    if not isinstance(headers, dict):
        return value
    safe: dict[str, str] = {}
    for key, item in headers.items():
        text = str(item)
        if not (
            re.fullmatch(r"\$\{[A-Za-z_][A-Za-z0-9_]*(?::-[^}]*)?\}", text)
            or text.startswith("keychain://")
        ):
            raise PluginError(
                f"marketplace header {key!r} must use an environment or keychain reference"
            )
        safe[str(key)] = text
    value["headers"] = safe
    return value


def _resolve_marketplace_headers(headers: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    if not isinstance(headers, dict):
        return result
    for key, value in headers.items():
        text = str(value)
        match = re.fullmatch(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}", text
        )
        if match:
            result[str(key)] = os.getenv(match.group(1), match.group(2) or "")
            continue
        if text.startswith("keychain://"):
            target = text.removeprefix("keychain://")
            try:
                secret = secret_store.get(target)
            except secret_store.SecretStoreError as exc:
                raise PluginError(f"could not load marketplace header {key!r}: {exc}") from exc
            if secret is None:
                raise PluginError(f"marketplace header {key!r} is not configured")
            result[str(key)] = secret
            continue
        raise PluginError(f"marketplace header {key!r} is not a safe reference")
    return result


def _plugin_secret_targets(workspace: Path) -> tuple[str, ...]:
    targets: set[str] = set()
    try:
        import tomllib

        path = user_settings_path()
        loaded = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        configs = loaded.get("plugin_configs", {})
        if isinstance(configs, dict):
            for values in configs.values():
                if not isinstance(values, dict):
                    continue
                for value in values.values():
                    if isinstance(value, str) and value.startswith(_KEYCHAIN_PREFIX):
                        targets.add(
                            "Polaris/plugin-config/" + value.removeprefix(_KEYCHAIN_PREFIX)
                        )
    except (OSError, RuntimeError, UnicodeDecodeError, ValueError):
        pass
    return tuple(sorted(targets))


def _remove_toml_tables(path: Path, names: tuple[str, ...]) -> bool:
    if not path.is_file():
        return False
    try:
        import tomlkit

        document = tomlkit.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    changed = False
    for name in names:
        if name in document:
            del document[name]
            changed = True
    if changed:
        _atomic_text(path.resolve(), tomlkit.dumps(document))
    return changed


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def plugin_tree_digest(root: str | Path) -> str:
    """Return a deterministic SHA-256 for plugin contents, excluding VCS metadata."""

    base = Path(root).resolve()
    digest = hashlib.sha256()
    for path in sorted(base.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(base)
        if ".git" in relative.parts:
            continue
        if path.is_symlink():
            try:
                target = path.resolve(strict=True)
            except OSError as exc:
                raise PluginError(f"broken symlink: {relative}") from exc
            if not _inside(target, base):
                raise PluginError(f"symlink escapes plugin root: {relative}")
            digest.update(relative.as_posix().encode("utf-8"))
            digest.update(b"\0symlink\0")
            digest.update(os.readlink(path).encode("utf-8"))
            digest.update(b"\0")
            continue
        if path.is_dir():
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        try:
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
        except OSError as exc:
            raise PluginError(f"could not hash plugin file {relative}: {exc}") from exc
        digest.update(b"\0")
    return digest.hexdigest()


def _manifest_path(root: Path) -> Path:
    return root / ".claude-plugin" / "plugin.json"


def validate_plugin(
    root: str | Path,
    marketplace_entry: dict[str, Any] | None = None,
    *,
    strict_warnings: bool = False,
) -> dict[str, Any]:
    """Validate a plugin without executing it.

    Claude plugin manifests are optional.  A marketplace entry is accepted here so
    validation and installation use the exact same ``strict`` merge semantics.
    Warnings are returned in the private ``_warnings`` key; callers may promote them
    to errors with ``strict_warnings`` (matching ``claude plugin validate --strict``).
    """

    plugin_root = Path(root).expanduser().resolve()
    if not plugin_root.is_dir():
        raise PluginError(f"plugin root is not a directory: {plugin_root}")
    try:
        manifest, warnings = resolve_plugin_manifest(plugin_root, marketplace_entry)
    except SpecError as exc:
        raise PluginError(str(exc)) from exc
    name = str(manifest.get("name") or "").strip()
    if not _SAFE_NAME.fullmatch(name):
        raise PluginError("plugin name must be filesystem-safe")
    for path in plugin_root.rglob("*"):
        if path.is_symlink():
            try:
                target = path.resolve(strict=True)
            except OSError as exc:
                raise PluginError(f"broken symlink: {path}") from exc
            if not _inside(target, plugin_root):
                raise PluginError(f"symlink escapes plugin root: {path}")
    if strict_warnings and warnings:
        raise PluginError("; ".join(warnings))
    if warnings:
        manifest["_warnings"] = list(warnings)
    return manifest


def copy_marketplace_plugin_tree(source: Path, destination: Path, marketplace_root: Path) -> None:
    """Copy with Claude symlink semantics and a marketplace containment boundary."""

    source = source.resolve()
    marketplace_root = marketplace_root.resolve()
    if not _inside(source, marketplace_root):
        # External Git/npm/archive sources are their own immutable artifact boundary.
        marketplace_root = source
    shutil.copytree(source, destination, symlinks=True)
    for copied in sorted(destination.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if not copied.is_symlink():
            continue
        relative = copied.relative_to(destination)
        original = source / relative
        try:
            target = original.resolve(strict=True)
        except OSError:
            copied.unlink(missing_ok=True)
            continue
        if _inside(target, source):
            # Relative, internal links remain links in the immutable cache.
            if Path(os.readlink(original)).is_absolute():
                copied.unlink(missing_ok=True)
                if target.is_dir():
                    shutil.copytree(target, copied, symlinks=True)
                else:
                    shutil.copy2(target, copied)
            continue
        copied.unlink(missing_ok=True)
        if not _inside(target, marketplace_root):
            # A marketplace cannot smuggle host files into an installed plugin.
            continue
        if target.is_dir():
            shutil.copytree(target, copied, symlinks=False)
        else:
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, copied)


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


def _read_plugin_list(path: Path, key: str) -> list[str]:
    if not path.is_file():
        return []
    try:
        import tomllib

        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return []
    table = data.get("plugins")
    raw = table.get(key, []) if isinstance(table, dict) else []
    return [str(item) for item in raw if isinstance(item, str)]


def _changed_enabled(values: list[str], plugin_id: str, enabled: bool) -> list[str]:
    result = [item for item in values if item != plugin_id]
    if enabled:
        result.append(plugin_id)
    return list(dict.fromkeys(result))


def _git(args: list[str]) -> None:
    try:
        completed = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PluginError(f"git failed: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "git failed").strip()
        raise PluginError(detail[:1000])


def _git_output(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args], capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=120, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PluginError(f"git failed: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "git failed").strip()
        raise PluginError(detail[:1000])
    return completed.stdout


def _git_head(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PluginError(f"could not read plugin git revision: {exc}") from exc
    if completed.returncode != 0:
        raise PluginError((completed.stderr or "could not read plugin git revision")[:1000])
    return completed.stdout.strip()


def _git_head_if_repository(root: Path) -> str:
    try:
        return _git_head(root)
    except PluginError:
        return ""


def _clone_remote(
    source: str,
    destination: Path,
    *,
    commit: str | None = None,
    ref: str | None = None,
) -> None:
    _validate_git_source(source)
    if ref and (ref.startswith("-") or not re.fullmatch(r"[A-Za-z0-9._/@+-]{1,200}", ref)):
        raise PluginError("git ref contains unsafe characters")
    if commit and not is_git_commit_pin(commit):
        raise PluginError("git commit must be a full immutable object id")
    if destination.exists():
        if commit:
            actual = _git_head(destination)
            if actual.casefold() != commit.casefold():
                raise PluginError(
                    f"cached source commit mismatch: expected {commit}, got {actual}"
                )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        clone_args = ["clone"]
        if not commit:
            clone_args += ["--depth", "1"]
        if ref and not commit:
            clone_args += ["--branch", ref]
        clone_args += ["--", source, str(temporary)]
        _git(clone_args)
        if commit:
            # Fetching by full object id first closes the branch-ref TOCTOU window.
            _git(["-C", str(temporary), "fetch", "--depth", "1", "origin", commit])
            _git(["-C", str(temporary), "checkout", "--detach", commit])
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _validate_git_source(source: str) -> None:
    if not source or source.startswith("-") or any(ord(char) < 32 for char in source):
        raise PluginError("git source is invalid")
    if re.fullmatch(r"git@[A-Za-z0-9.-]+:[A-Za-z0-9_./-]+(?:\.git)?", source):
        return
    parsed = urlparse(source)
    if parsed.scheme not in {"https", "ssh"} or not parsed.hostname:
        raise PluginError("git source must use credential-free HTTPS or SSH")
    if parsed.username or parsed.password:
        raise PluginError("git source URL must not contain credentials")


def _download_https(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    max_bytes: int,
) -> bytes:
    """Bounded HTTPS download with redirect/private-network rejection."""

    import socket
    import httpx

    parsed = urlparse(url)
    if parsed.scheme.casefold() != "https" or not parsed.hostname:
        raise PluginError("remote archives and catalogs require HTTPS")

    def public_host(host: str) -> bool:
        try:
            addresses = {str(item[4][0]) for item in socket.getaddrinfo(host, None)}
        except OSError as exc:
            raise PluginError(f"could not resolve remote host: {exc}") from exc
        for address in addresses:
            ip = ipaddress.ip_address(address.split("%", 1)[0])
            if not ip.is_global:
                return False
        return bool(addresses)

    if not public_host(parsed.hostname):
        raise PluginError("remote URL resolves to a private or non-global address")
    origin = (parsed.scheme.casefold(), parsed.hostname.casefold(), parsed.port or 443)
    current = url
    current_headers = dict(headers or {})
    with httpx.Client(follow_redirects=False, timeout=30.0) as client:
        for _ in range(6):
            with client.stream("GET", current, headers=current_headers) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise PluginError("remote redirect omitted Location")
                    redirected = httpx.URL(current).join(location)
                    next_url = str(redirected)
                    next_parsed = urlparse(next_url)
                    if next_parsed.scheme.casefold() != "https" or not next_parsed.hostname:
                        raise PluginError("remote redirect must remain on HTTPS")
                    if not public_host(next_parsed.hostname):
                        raise PluginError("remote redirect resolves to a private address")
                    next_origin = (
                        next_parsed.scheme.casefold(), next_parsed.hostname.casefold(),
                        next_parsed.port or 443,
                    )
                    if next_origin != origin:
                        current_headers = {}
                    current = next_url
                    continue
                response.raise_for_status()
                expected = response.headers.get("content-length")
                if expected and int(expected) > max_bytes:
                    raise PluginError("remote artifact exceeds compressed size limit")
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        raise PluginError("remote artifact exceeds compressed size limit")
                    chunks.append(chunk)
                return b"".join(chunks)
    raise PluginError("too many remote redirects")


def _safe_member_path(destination: Path, raw: str) -> Path:
    normalized = raw.replace("\\", "/").lstrip("/")
    if (
        not normalized
        or "\x00" in normalized
        or re.match(r"^[A-Za-z]:", normalized)
        or ".." in Path(normalized).parts
    ):
        raise PluginError(f"archive contains unsafe path: {raw}")
    target = (destination / normalized).resolve()
    if not _inside(target, destination):
        raise PluginError(f"archive path escapes destination: {raw}")
    return target


def _extract_zip_bytes(body: bytes, destination: Path) -> None:
    import io

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent)))
    total = 0
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            entries = archive.infolist()
            if len(entries) > 100_000:
                raise PluginError("archive contains too many entries")
            for member in entries:
                total += member.file_size
                if total > 1024 * 1024 * 1024:
                    raise PluginError("archive exceeds uncompressed size limit")
                target = _safe_member_path(temporary, member.filename)
                # Unix symlinks in zip files can redirect later extraction outside.
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise PluginError("archive contains a symbolic link")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source_stream, target.open("wb") as target_stream:
                    shutil.copyfileobj(source_stream, target_stream, 1024 * 1024)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _extract_archive_bytes(body: bytes, destination: Path) -> None:
    """Extract a bounded ZIP or tar archive through the hardened extractors."""

    import io

    if zipfile.is_zipfile(io.BytesIO(body)):
        _extract_zip_bytes(body, destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".archive", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
        if not tarfile.is_tarfile(temporary):
            raise PluginError("archive is neither ZIP nor a supported tar format")
        _extract_tar(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _extract_tar(path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent)))
    total = 0
    try:
        with tarfile.open(path, "r:*") as archive:
            members = archive.getmembers()
            if len(members) > 100_000:
                raise PluginError("npm package contains too many entries")
            for member in members:
                total += max(0, member.size)
                if total > 1024 * 1024 * 1024:
                    raise PluginError("npm package exceeds uncompressed size limit")
                target = _safe_member_path(temporary, member.name)
                if member.issym() or member.islnk():
                    raise PluginError("npm package contains a symbolic link")
                if member.isdev() or member.isfifo():
                    raise PluginError("npm package contains a special file")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source_stream = archive.extractfile(member)
                    if source_stream is None:
                        raise PluginError(f"could not extract npm member: {member.name}")
                    with source_stream, target.open("wb") as target_stream:
                        shutil.copyfileobj(source_stream, target_stream, 1024 * 1024)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _single_package_root(root: Path) -> Path:
    children = [item for item in root.iterdir() if item.name not in {"__MACOSX"}]
    if len(children) == 1 and children[0].is_dir():
        candidate = children[0]
        if candidate.name == "package" or not _manifest_path(root).is_file():
            return candidate
    return root


def _materialize_npm_package(source: PluginSourceConfig, destination_root: Path) -> Path:
    staging = destination_root / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    spec = source.package + (f"@{source.version}" if source.version else "")
    command = ["npm", "pack", spec, "--ignore-scripts", "--json", "--pack-destination", str(staging)]
    if source.registry:
        command += ["--registry", source.registry]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=120, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PluginError(f"npm pack failed: {exc}") from exc
    if completed.returncode != 0:
        raise PluginError((completed.stderr or completed.stdout or "npm pack failed")[:1000])
    try:
        result = json.loads(completed.stdout)
        item = result[0]
        filename = str(item["filename"])
        version = str(item.get("version") or source.version or "unknown")
        integrity = str(item.get("integrity") or "")
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise PluginError("npm pack returned invalid metadata") from exc
    tarball = staging / filename
    if not tarball.is_file() or tarball.stat().st_size > 256 * 1024 * 1024:
        raise PluginError("npm tarball is missing or exceeds size limit")
    key = hashlib.sha256((version + "\0" + integrity).encode()).hexdigest()
    destination = destination_root / key
    if not destination.exists():
        _extract_tar(tarball, destination)
    try:
        tarball.unlink()
    except OSError:
        pass
    return _single_package_root(destination)


def _dependency_labels(value: Any) -> list[str]:
    result: list[str] = []
    if not isinstance(value, list):
        return result
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
        elif isinstance(item, dict) and str(item.get("name") or "").strip():
            name = str(item["name"]).strip()
            marketplace = str(item.get("marketplace") or "").strip()
            version = str(item.get("version") or "").strip()
            result.append(name + (f"@{marketplace}" if marketplace else "") + (f" {version}" if version else ""))
    return result


def _clone_skill(skill: Skill, namespace: str) -> Skill:
    return Skill(
        name=f"{namespace}:{skill.name}",
        description=skill.description,
        body=skill.body,
        when_to_use=skill.when_to_use,
        argument_hint=skill.argument_hint,
        allowed_tools=skill.allowed_tools,
        disallowed_tools=skill.disallowed_tools,
        preload_skills=skill.preload_skills,
        capabilities=skill.capabilities,
        hooks=skill.hooks,
        model=skill.model,
        aliases=tuple(f"{namespace}:{alias}" for alias in skill.aliases),
        user_invocable=skill.user_invocable,
        disable_model_invocation=skill.disable_model_invocation,
        context=skill.context,
        memory=skill.memory,
        effort=skill.effort,
        permission_mode=skill.permission_mode,
        isolation=skill.isolation,
        background=skill.background,
        max_turns=skill.max_turns,
        agent_key=f"{namespace}:{skill.agent_key or skill.name}",
        trust_tier=skill.trust_tier,
        source_identity=skill.source_identity,
        plugin_id=skill.plugin_id,
        source_path=skill.source_path,
    )


def _apply_skill_provenance(skills: list[Skill], record: PluginRecord) -> None:
    """Apply host-owned provenance and tiered isolation after parsing frontmatter."""

    remote_untrusted = record.trust_tier in {
        TrustTier.COMMUNITY.value,
        TrustTier.VERIFIED_PUBLISHER.value,
    }
    for skill in skills:
        skill.trust_tier = record.trust_tier
        skill.source_identity = record.source_identity
        skill.plugin_id = record.plugin_id
        if remote_untrusted:
            skill.context = SkillContext.FORK


def _component_paths(
    root: Path,
    manifest: dict[str, Any],
    key: str,
    default: Path,
) -> list[Path]:
    raw = manifest.get(key)
    values = (
        [item for item in raw if isinstance(item, str)]
        if isinstance(raw, list)
        else ([raw] if isinstance(raw, str) else [])
    )
    if not values:
        return [default]
    paths = [(root / str(value)).resolve() for value in values]
    if key == "skills" and not manifest.get("_skills_replace_default") and default not in paths:
        paths.insert(0, default)
    return paths


def _load_plugin_skills(
    root: Path, namespace: str, manifest: dict[str, Any], workspace: Path,
    plugin_data: Path, user_config: dict[str, Any],
) -> list[Skill]:
    result: list[Skill] = []
    candidates: list[Path] = []
    for location in _component_paths(root, manifest, "skills", root / "skills"):
        if location.is_file():
            candidates.append(location)
        else:
            candidates.extend(location.glob("*/SKILL.md"))
            candidates.extend(location.glob("*.md"))
    if not (root / "skills").exists() and "skills" not in manifest and (root / "SKILL.md").is_file():
        candidates.append(root / "SKILL.md")
    for location in _component_paths(root, manifest, "commands", root / "commands"):
        if location.is_file():
            candidates.append(location)
        else:
            candidates.extend(location.glob("*.md"))
    for path in sorted(set(candidates)):
        skill = load_skill_file(path)
        if skill is not None:
            skill = replace(
                skill,
                body=_expand_plugin_vars(
                    skill.body, root, workspace, plugin_data, user_config
                ),
            )
            result.append(_clone_skill(skill, namespace))
    return result


def _load_plugin_agents(
    root: Path, namespace: str, manifest: dict[str, Any], workspace: Path,
    plugin_data: Path, user_config: dict[str, Any],
) -> tuple[dict[str, str], list[Skill]]:
    definitions: dict[str, str] = {}
    skills: list[Skill] = []
    candidates: list[Path] = []
    for location in _component_paths(root, manifest, "agents", root / "agents"):
        candidates.extend([location] if location.is_file() else location.glob("*.md"))
    for path in sorted(candidates):
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if not raw:
            continue
        meta, parsed_body = parse_frontmatter(raw)
        body = parsed_body.strip()
        if not body:
            continue
        local_name = str(meta.get("name") or path.stem).strip()
        if not is_safe_plugin_name(local_name):
            raise PluginError(f"invalid plugin agent name: {local_name}")
        name = f"{namespace}:{local_name}"
        definitions[name] = raw
        memory = str(meta.get("memory", "none")).strip().lower()
        if memory not in {"none", "user", "project", "local"}:
            memory = "none"
        skills.append(
            Skill(
                name=name,
                description=str(meta.get("description") or f"Run the {path.stem} plugin agent."),
                body=_expand_plugin_vars(
                    body, root, workspace, plugin_data, user_config
                ) + "\n\n$ARGUMENTS",
                allowed_tools=_string_tuple(meta.get("tools")),
                disallowed_tools=_string_tuple(meta.get("disallowed_tools")),
                preload_skills=_string_tuple(meta.get("skills")),
                model=(str(meta.get("model")).strip() or None) if meta.get("model") else None,
                context=SkillContext.FORK,
                memory=memory,
                effort=(str(meta.get("effort")).strip() or None) if meta.get("effort") else None,
                # Claude deliberately ignores permissionMode, hooks and mcpServers
                # on plugin-shipped agents.  Plugin agents remain subject to the
                # host's normal Polaris permission policy.
                permission_mode=None,
                isolation=(str(meta.get("isolation")).strip() or None) if meta.get("isolation") else None,
                background=meta.get("background", False) is True,
                max_turns=_positive_int(meta.get("max_turns")),
                agent_key=name,
                source_path=path,
            )
        )
    return definitions, skills


def _load_plugin_hooks(
    root: Path,
    agent: "ReActAgent",
    namespace: str,
    manifest: dict[str, Any],
    *,
    plugin_id: str,
    workspace: Path,
    plugin_data: Path,
    user_config: dict[str, Any],
    runtime_config: dict[str, Any] | None = None,
    allowed_hook_ids: tuple[str, ...] | None = None,
) -> list[tuple[str, Any]]:
    result: list[tuple[str, Any]] = []
    # command_argv is an internal post-sandbox transport and must never be supplied
    # by plugin metadata itself.
    valid_fields = {
        field.name for field in fields(ExternalHookSpec) if field.name != "command_argv"
    }
    raw_hooks = manifest.get("hooks")
    inline_tables: list[dict[str, Any]] = []
    if isinstance(raw_hooks, dict):
        inline_tables.append(raw_hooks.get("hooks", raw_hooks))
    elif isinstance(raw_hooks, list):
        inline_tables.extend(item.get("hooks", item) for item in raw_hooks if isinstance(item, dict))
    tables: list[Any] = list(inline_tables)
    for location in _component_paths(root, manifest, "hooks", root / "hooks"):
        path = location / "hooks.json" if location.is_dir() else location
        data = _read_json(path, {})
        tables.append(data.get("hooks", data) if isinstance(data, dict) else {})
    for table in tables:
        if not isinstance(table, dict):
            continue
        for event, groups in table.items():
            if event not in LIFECYCLE_EVENT_ATTRS or not isinstance(groups, list):
                continue
            for group in groups:
                if not isinstance(group, dict):
                    continue
                entries = group.get("hooks", [group])
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    if allowed_hook_ids is not None:
                        entry_id = str(entry.get("id") or "").strip()
                        qualified = f"{plugin_id}:{entry_id}" if entry_id else ""
                        if not qualified or qualified not in allowed_hook_ids:
                            continue
                    values = dict(entry)
                    values["event"] = event
                    if "if" in values and "condition" not in values:
                        values["condition"] = values.pop("if")
                    if "matcher" not in values and isinstance(group.get("matcher"), str):
                        values["matcher"] = group["matcher"]
                    values = _expand_plugin_vars(
                        values, root, workspace, plugin_data, user_config
                    )
                    values["env"] = {
                        **{
                            str(key): _expand_executable_env(str(value))
                            for key, value in dict(values.get("env") or {}).items()
                        },
                        **_plugin_option_env(runtime_config or user_config),
                    }
                    try:
                        spec = ExternalHookSpec(
                            **{
                                key: value
                                for key, value in values.items()
                                if key in valid_fields
                            }
                        )
                        if allowed_hook_ids is not None:
                            spec = _restrict_autonomous_hook(agent, root, plugin_id, spec)
                        adapter = build_external_adapter(
                            spec,
                            logger=agent.logger,
                            provider=agent.provider,
                            base_config=agent._provider_config(),
                            subagent_factory=agent.session.subagent_factory,
                        )
                    except Exception as exc:
                        raise PluginError(
                            f"{namespace} hook failed validation: {exc}"
                        ) from exc
                    if adapter is not None:
                        result.append((LIFECYCLE_EVENT_ATTRS[event], adapter))
    return result


def _restrict_autonomous_hook(
    agent: "ReActAgent",
    root: Path,
    plugin_id: str,
    spec: ExternalHookSpec,
) -> ExternalHookSpec:
    if spec.type == "command":
        if not spec.command or not agent.sandbox.is_enabled():
            raise PluginError(
                f"autonomous command hook from {plugin_id} requires a real sandbox"
            )
        guest_root = agent.sandbox.translate_path(root)
        command = spec.command.replace(str(root), guest_root)
        scope = ExecutionScope.for_workspace(
            agent.session.workspace,
            read_only_roots=(root,),
            network="deny",
        )
        invocation = SandboxInvocation.create(
            ["/bin/sh", "-c", spec.command],
            guest_argv=["@bash", "-lc", command],
            required_guest_capabilities=("bash",),
            scope=scope,
        )
        wrapped, shell = agent.sandbox.wrap_invocation(invocation, command=command)
        if shell or not isinstance(wrapped, list) or not wrapped:
            raise PluginError(f"sandbox could not wrap autonomous hook from {plugin_id}")
        return replace(spec, command_argv=[str(item) for item in wrapped])
    if spec.type == "http":
        if agent.config.sandbox.enabled:
            raise PluginError(
                f"autonomous HTTP hook from {plugin_id} is unavailable while the "
                "guest sandbox enforces network=deny"
            )
        parsed = urlparse(spec.url or "")
        host = parsed.hostname or ""
        loopback = host in {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme != "https" and not (loopback and parsed.scheme == "http"):
            raise PluginError(
                f"autonomous HTTP hook from {plugin_id} requires HTTPS or loopback"
            )
        if not loopback and not _domain_allowed(host, agent.config.web.allowed_domains):
            raise PluginError(
                f"autonomous HTTP hook domain {host!r} is not in web.allowed_domains"
            )
    return spec


def _load_plugin_mcp(
    root: Path, namespace: str, manifest: dict[str, Any], workspace: Path,
    plugin_data: Path, user_config: dict[str, Any],
    runtime_config: dict[str, Any] | None = None,
) -> list[MCPServerConfig]:
    result: list[MCPServerConfig] = []
    raw_mcp = manifest.get("mcpServers")
    tables: list[Any] = []
    if isinstance(raw_mcp, dict):
        tables.append(raw_mcp.get("mcpServers", raw_mcp))
    elif isinstance(raw_mcp, list):
        tables.extend(
            item.get("mcpServers", item) for item in raw_mcp if isinstance(item, dict)
        )
    for location in _component_paths(root, manifest, "mcpServers", root / ".mcp.json"):
        data = _read_json(location, {})
        tables.append(
            data.get("mcpServers", data.get("servers", {}))
            if isinstance(data, dict)
            else {}
        )
    merged: dict[str, Any] = {}
    for table in tables:
        if not isinstance(table, dict):
            continue
        merged.update(table)
    for name, body in merged.items():
        if not isinstance(body, dict):
            continue
        expanded = _expand_plugin_vars(
            body, root, workspace, plugin_data, user_config
        )
        expanded["env"] = {
            **{
                str(key): _expand_executable_env(str(value))
                for key, value in dict(expanded.get("env") or {}).items()
            },
            **_plugin_option_env(runtime_config or user_config),
        }
        result.append(
            MCPServerConfig.from_dict(f"{namespace}:{name}", expanded)
        )
    return result


def _load_json_component(root: Path, value: Any, default: Path, *, wrapper: str = "") -> Any:
    if isinstance(value, (dict, list)):
        return value
    locations = value if isinstance(value, list) else [value] if isinstance(value, str) else [str(default.relative_to(root)).replace("\\", "/")]
    merged: Any = [] if wrapper == "list" else {}
    for raw in locations:
        if not isinstance(raw, str):
            continue
        path = (root / raw).resolve()
        if not _inside(path, root):
            raise PluginError("component JSON path escapes plugin root")
        loaded = _read_json(path, None)
        if wrapper and isinstance(loaded, dict):
            loaded = loaded.get(wrapper)
        if isinstance(merged, dict) and isinstance(loaded, dict):
            merged.update(loaded)
        elif isinstance(merged, list) and isinstance(loaded, list):
            merged.extend(loaded)
    return merged


def _markdown_components(root: Path, manifest: dict[str, Any], key: str, default: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for location in _component_paths(root, manifest, key, default):
        candidates = [location] if location.is_file() else sorted(location.glob("*")) if location.is_dir() else []
        for path in candidates:
            if not path.is_file() or path.suffix.casefold() not in {".md", ".json", ".js", ".mjs"}:
                continue
            try:
                result[path.stem] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
    return result


def _load_non_mcp_components(
    root: Path,
    namespace: str,
    manifest: dict[str, Any],
    selected: set[str],
    workspace: Path,
    plugin_data: Path,
    user_config: dict[str, Any],
    runtime_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from agent_core.tool_config import LSPServerConfig

    result: dict[str, Any] = {
        "lsp": [], "workflows": {}, "output_styles": {}, "themes": {},
        "monitors": [], "channels": [], "user_config": {}, "bin": [], "settings": {},
    }
    if "lsp" in selected:
        table = _load_json_component(root, manifest.get("lspServers"), root / ".lsp.json")
        if isinstance(table, dict):
            for name, value in table.items():
                if not isinstance(value, dict):
                    continue
                expanded = _expand_plugin_vars(
                    value, root, workspace, plugin_data, user_config
                )
                expanded["env"] = {
                    **{
                        str(key): _expand_executable_env(str(value))
                        for key, value in dict(expanded.get("env") or {}).items()
                    },
                    **_plugin_option_env(runtime_config or user_config),
                }
                expanded["name"] = f"{namespace}:{name}"
                expanded["plugin_root"] = str(root)
                config = LSPServerConfig.from_dict(expanded)
                if config.command and config.extensions:
                    result["lsp"].append(config)
    if "workflows" in selected:
        result["workflows"] = {
            f"{namespace}:{name}": value
            for name, value in _markdown_components(root, manifest, "workflows", root / "workflows").items()
        }
    if "output-styles" in selected:
        result["output_styles"] = {
            f"{namespace}:{name}": value
            for name, value in _markdown_components(root, manifest, "outputStyles", root / "output-styles").items()
        }
    if "themes" in selected:
        theme_value = manifest.get("themes")
        experimental = manifest.get("experimental")
        if theme_value is None and isinstance(experimental, dict):
            theme_value = experimental.get("themes")
        themed = dict(manifest)
        if theme_value is not None:
            themed["themes"] = theme_value
        result["themes"] = {
            f"{namespace}:{name}": value
            for name, value in _markdown_components(root, themed, "themes", root / "themes").items()
        }
    if "monitors" in selected:
        monitor_value = manifest.get("monitors")
        experimental = manifest.get("experimental")
        if monitor_value is None and isinstance(experimental, dict):
            monitor_value = experimental.get("monitors")
        monitors = _load_json_component(
            root, monitor_value, root / "monitors" / "monitors.json", wrapper="list"
        )
        if isinstance(monitors, list):
            for item in monitors:
                if isinstance(item, dict) and item.get("name") and item.get("command"):
                    result["monitors"].append({
                        **_expand_plugin_vars(
                            item, root, workspace, plugin_data, {}, allow_user_config=False
                        ),
                        "name": f"{namespace}:{item['name']}",
                        "plugin": namespace,
                    })
    if "channels" in selected and isinstance(manifest.get("channels"), list):
        mcp_value = manifest.get("mcpServers")
        mcp_names = set(mcp_value) if isinstance(mcp_value, dict) else set()
        if not mcp_names:
            mcp_names = {
                item.name.split(":", 1)[-1]
                for item in _load_plugin_mcp(
                    root, namespace, manifest, workspace, plugin_data, user_config,
                    runtime_config,
                )
            }
        for item in manifest["channels"]:
            if not isinstance(item, dict) or str(item.get("server") or "") not in mcp_names:
                raise PluginError(f"{namespace} channel must bind to its own MCP server")
            result["channels"].append({
                **item,
                "plugin": namespace,
                "qualified_server": f"{namespace}:{item['server']}",
            })
    if "user-config" in selected and isinstance(manifest.get("userConfig"), dict):
        result["user_config"][namespace] = dict(manifest["userConfig"])
    if "bin" in selected and (root / "bin").is_dir():
        result["bin"].append(str((root / "bin").resolve()))
    if "settings" in selected:
        settings = _read_json(root / "settings.json", {})
        inline = manifest.get("settings")
        if isinstance(inline, dict):
            settings = {**inline, **settings}
        if isinstance(settings, dict):
            result["settings"][namespace] = {
                key: value for key, value in settings.items()
                if key in {"agent", "subagentStatusLine"}
            }
    return result


def _domain_allowed(host: str, allowed: list[str]) -> bool:
    normalized = host.casefold().strip(".")
    return any(
        normalized == item.casefold().strip(".")
        or normalized.endswith("." + item.casefold().strip("."))
        for item in allowed
        if item.strip()
    )


def _translate_plugin_path(value: str, root: Path, guest_root: str) -> str:
    try:
        candidate = Path(value)
        if not candidate.is_absolute():
            return value
        relative = candidate.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return value
    return str(Path(guest_root) / relative).replace("\\", "/")


_GUEST_COMMAND_ALIASES = {
    "bash": "bash",
    "pwsh": "pwsh",
    "powershell": "pwsh",
    "python": "python",
    "python3": "python",
    "node": "node",
    "npm": "npm",
    "npx": "npx",
    "pyright-langserver": "pyright-langserver",
}
_WINDOWS_COMMAND_SUFFIXES = {".exe", ".cmd", ".bat"}


def sandboxed_guest_invocation(
    sandbox: Any,
    host_argv: list[str],
    *,
    mounted_roots: tuple[Path, ...],
    scope: ExecutionScope,
) -> SandboxInvocation:
    """Build a fail-closed invocation for a plugin/Hook/MCP guest process."""

    if not host_argv:
        raise GuestCapabilityUnavailable(
            "guest_capability_unavailable: empty guest process command"
        )
    command = host_argv[0]
    basename = max((Path(command).name, Path(command.replace("\\", "/")).name), key=len)
    suffix = Path(basename).suffix.casefold()
    if suffix in _WINDOWS_COMMAND_SUFFIXES:
        raise GuestCapabilityUnavailable(
            f"guest_capability_unavailable: Windows command is unsupported: {command!r}"
        )
    alias = _GUEST_COMMAND_ALIASES.get(basename.casefold())
    required: tuple[str, ...] = ()
    if alias is not None:
        if alias not in sandbox.capabilities:
            raise GuestCapabilityUnavailable(
                f"guest_capability_unavailable: image does not declare {alias!r}"
            )
        guest_command = f"@{alias}"
        required = (alias,)
    else:
        guest_command = _mounted_guest_path(command, mounted_roots, sandbox)
    guest_args = [
        _mounted_guest_argument(value, mounted_roots, sandbox)
        for value in host_argv[1:]
    ]
    return SandboxInvocation.create(
        host_argv,
        guest_argv=[guest_command, *guest_args],
        required_guest_capabilities=required,
        scope=scope,
    )


def sandbox_runtime_environment() -> dict[str, str]:
    """Host environment needed by an OCI client, never forwarded into its guest."""

    names = (
        "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "XDG_CONFIG_HOME",
        "XDG_RUNTIME_DIR", "CONTAINER_HOST", "CONTAINERS_CONF",
        "CONTAINERS_STORAGE_CONF", "DOCKER_HOST", "DOCKER_CONTEXT",
    )
    return {name: value for name in names if (value := os.environ.get(name)) is not None}


def _mounted_guest_path(value: str, roots: tuple[Path, ...], sandbox: Any) -> str:
    candidate = Path(value)
    candidates = [candidate] if candidate.is_absolute() else [root / candidate for root in roots]
    for item in candidates:
        try:
            resolved = item.resolve()
        except OSError:
            continue
        for root in roots:
            try:
                relative = resolved.relative_to(root.resolve())
            except (OSError, ValueError):
                continue
            return str(Path(sandbox.translate_path(root.resolve())) / relative).replace("\\", "/")
    raise GuestCapabilityUnavailable(
        "guest_capability_unavailable: command is neither a declared image alias nor "
        f"a script under a mounted root: {value!r}"
    )


def _mounted_guest_argument(value: str, roots: tuple[Path, ...], sandbox: Any) -> str:
    candidate = Path(value)
    if not candidate.is_absolute():
        return value
    return _mounted_guest_path(value, roots, sandbox)


def _restrict_autonomous_mcp(
    agent: "ReActAgent", root: Path, servers: list[MCPServerConfig]
) -> list[MCPServerConfig]:
    """Fail closed unless autonomous MCP transport is constrained by host policy."""

    restricted: list[MCPServerConfig] = []
    for server in servers:
        transport = (server.transport or "stdio").casefold()
        if transport in {"streamable-http", "streamable_http", "http", "sse", "ws", "wss", "websocket"}:
            raise PluginError(
                f"autonomous remote MCP {server.name} is unavailable while the "
                "guest sandbox enforces network=deny"
            )

        if transport != "stdio":
            raise PluginError(f"unsupported autonomous MCP transport: {server.transport}")
        if not agent.sandbox.is_enabled():
            raise PluginError(
                f"autonomous stdio MCP {server.name} requires a real sandbox backend"
            )
        if server.env:
            raise PluginError(
                f"autonomous stdio MCP {server.name} cannot inject host environment secrets"
            )
        argv = [server.command, *server.args]
        scope = ExecutionScope.for_workspace(
            agent.session.workspace,
            read_only_roots=(root,),
            network="deny",
        )
        invocation = sandboxed_guest_invocation(
            agent.sandbox, argv, mounted_roots=(root, agent.session.workspace), scope=scope
        )
        wrapped, shell = agent.sandbox.wrap_invocation(invocation)
        if shell or not isinstance(wrapped, list) or not wrapped:
            raise PluginError(f"sandbox could not wrap autonomous MCP {server.name}")
        restricted.append(
            replace(
                server,
                command=str(wrapped[0]),
                args=[str(item) for item in wrapped[1:]],
                env=sandbox_runtime_environment(),
                cwd="",
            )
        )
    return restricted


def _plugin_option_env(values: dict[str, Any]) -> dict[str, str]:
    return {
        f"CLAUDE_PLUGIN_OPTION_{key}": (
            json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        )
        for key, value in values.items()
    }


def _expand_executable_env(value: str) -> str:
    """Resolve host variables only for an explicit executable environment value."""

    return re.sub(
        r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}",
        lambda match: os.getenv(match.group(1), match.group(2) or ""),
        value,
    )


def _expand_plugin_vars(
    value: Any,
    root: Path,
    workspace: Path,
    plugin_data: Path,
    user_config: dict[str, Any],
    *,
    allow_user_config: bool = True,
) -> Any:
    if isinstance(value, str):
        result = (
            value.replace("${CLAUDE_PLUGIN_ROOT}", str(root))
            .replace("${CLAUDE_PLUGIN_DATA}", str(plugin_data))
            .replace("${CLAUDE_PROJECT_DIR}", str(workspace))
        )
        if not allow_user_config and "${user_config." in result:
            raise PluginError("monitor commands cannot reference user_config values")
        if allow_user_config:
            result = re.sub(
                r"\$\{user_config\.([A-Za-z_][A-Za-z0-9_]*)\}",
                lambda match: str(user_config.get(match.group(1), "")),
                result,
            )
        # Arbitrary host environment variables are deliberately left literal here.
        # Prompt-bearing plugin content only receives framework paths and public
        # ``user_config``. Executable components resolve environment references solely
        # in their explicit ``env``/``headers`` maps at the process/transport boundary.
        return result
    if isinstance(value, list):
        return [
            _expand_plugin_vars(
                item, root, workspace, plugin_data, user_config,
                allow_user_config=allow_user_config,
            )
            for item in value
        ]
    if isinstance(value, dict):
        return {
            str(key): _expand_plugin_vars(
                item, root, workspace, plugin_data, user_config,
                allow_user_config=allow_user_config,
            )
            for key, item in value.items()
        }
    return value


def _prepare_generation(agent: "ReActAgent", bundle: PluginBundle) -> PluginGeneration:
    """Validate a complete candidate without mutating the active generation."""

    try:
        base = agent._load_skills()
        merged = SkillRegistry(base.list())
        # Built-ins remain available unqualified; plugin components are namespaced.
        for skill in bundle.skills:
            merged.add(skill)
        base_hooks = agent._build_hook_pipeline()
        for attr, adapter in bundle.hooks:
            getattr(base_hooks, attr).append(adapter)

        existing_tool_names = {
            *[tool.name for tool in agent.registry.list()],
            *[tool.name for tool in agent.registry.deferred()],
        }
        previous_plugin_names = set(getattr(agent, "_plugin_tool_names", set()))
        collisions = {
            tool.name
            for tool in bundle.mcp_tools
            if tool.name in existing_tool_names - previous_plugin_names
        }
        candidate_names = [tool.name for tool in bundle.mcp_tools]
        collisions.update(
            name for name in candidate_names if candidate_names.count(name) > 1
        )
        if collisions:
            raise PluginError(
                "plugin MCP tools collide with built-ins: " + ", ".join(sorted(collisions))
            )
    except Exception:
        if bundle.mcp_manager is not None:
            bundle.mcp_manager.close()
        raise
    return PluginGeneration(bundle, merged, base_hooks)


def _commit_generation(agent: "ReActAgent", generation: PluginGeneration) -> tuple[int, int, int]:
    """Publish an already-validated generation; the operations below cannot block."""

    bundle = generation.bundle
    merged = generation.skills
    base_hooks = generation.hooks

    old_manager = getattr(agent, "_plugin_mcp_manager", None)
    old_tool_names = set(getattr(agent, "_plugin_tool_names", set()))
    active_tools: list[Tool] = []
    deferred_tools: list[DeferredTool] = []
    for tool in bundle.mcp_tools:

        def factory(bound_tool: Tool = tool) -> Tool:
            return bound_tool

        if getattr(tool, "_always_load", False):
            active_tools.append(tool)
        else:
            deferred_tools.append(
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
    try:
        agent.registry.replace_group(old_tool_names, active_tools, deferred_tools)
    except Exception:
        if bundle.mcp_manager is not None:
            bundle.mcp_manager.close()
        raise
    if merged.model_invocable():
        try:
            agent.registry.get("skill")
        except KeyError:
            skill_tool = agent.default_registry().get("skill")
            agent.registry.register(skill_tool)
    else:
        agent.registry.unregister("skill")
    agent._plugin_tool_names = {tool.name for tool in bundle.mcp_tools}
    agent._plugin_mcp_manager = bundle.mcp_manager
    agent._plugin_active_ids = frozenset(bundle.plugin_ids)
    agent.plugin_agents = bundle.agents
    plugin_lsp_names = set(getattr(agent, "_plugin_lsp_names", set()))
    base_lsp = [
        item for item in agent.config.tools.lsp.servers
        if item.name not in plugin_lsp_names
    ]
    old_lsp_configs = list(agent.config.tools.lsp.servers)
    agent.config.tools.lsp.servers = [*base_lsp, *bundle.components["lsp"]]
    agent._plugin_lsp_names = {item.name for item in bundle.components["lsp"]}
    if agent.config.tools.lsp.servers:
        try:
            agent.registry.get("lsp")
        except KeyError:
            builtins = agent.default_registry()
            try:
                lsp_tool = builtins.get("lsp")
            except KeyError:
                # LSP is intentionally deferred in the built-in catalog.  A plugin
                # can introduce the first LSP config after agent construction, so
                # instantiate it now and let the live registry bind session/sandbox.
                lsp_tool = builtins.activate("lsp")
            agent.registry.register(lsp_tool)
    else:
        agent.registry.unregister("lsp")
    if old_lsp_configs != agent.config.tools.lsp.servers:
        old_lsp_manager = agent.session.lsp_manager
        agent.session.lsp_manager = None
        if old_lsp_manager is not None:
            _close_async_resource(old_lsp_manager.close())
    agent.plugin_workflows = bundle.components["workflows"]
    agent.session.plugin_workflows = agent.plugin_workflows
    agent.plugin_output_styles = bundle.components["output_styles"]
    agent.plugin_themes = bundle.components["themes"]
    agent._refresh_plugin_presentation()
    agent.plugin_monitors = bundle.components["monitors"]
    agent.plugin_channels = bundle.components["channels"]
    agent.plugin_user_config = bundle.components["user_config"]
    # PATH is intentionally not changed globally. Shell integrations may prepend these
    # per child process after the normal Polaris permission decision.
    agent.plugin_bin_paths = tuple(bundle.components["bin"])
    agent.session.plugin_bin_paths = agent.plugin_bin_paths
    agent.plugin_settings = bundle.components["settings"]
    agent.skills = merged
    agent.session.skills = merged
    agent.hooks = base_hooks
    agent.executor.hooks = base_hooks
    if old_manager is not None:
        old_manager.close()
    agent.session.registered_tool_names = frozenset(tool.name for tool in agent.registry.list())
    PluginManager(agent.session.workspace).audit.write(
        "generation_commit",
        {
            "plugin_ids": list(bundle.plugin_ids),
            "skills": len(bundle.skills),
            "hooks": len(bundle.hooks),
            "mcp_tools": len(bundle.mcp_tools),
        },
    )
    return len(bundle.skills), len(bundle.hooks), len(bundle.mcp_tools)


def _close_async_resource(awaitable: Any) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(awaitable)
    else:
        loop.create_task(awaitable)


def reload_plugins(agent: "ReActAgent") -> tuple[int, int, int]:
    """Build every component first, then atomically swap the live plugin generation."""

    manager = PluginManager(agent.session.workspace)
    return _commit_generation(agent, _prepare_generation(agent, manager.build_bundle(agent)))


def activate_plugin(
    agent: "ReActAgent",
    plugin_id: str,
    *,
    components: tuple[str, ...],
    allowed_hooks: tuple[str, ...] = (),
) -> tuple[int, int, int]:
    """Build a proposed component-scoped state, persist it, then publish it."""

    manager = PluginManager(agent.session.workspace)
    if plugin_id not in manager.records():
        raise PluginError(f"plugin is not installed: {plugin_id}")
    enabled = list(dict.fromkeys([*manager.enabled_ids(), plugin_id]))
    selections = manager.component_selections()
    selections[plugin_id] = tuple(dict.fromkeys(components))
    hook_ids = manager.hook_selections()
    hook_ids[plugin_id] = tuple(dict.fromkeys(allowed_hooks))
    generation = _prepare_generation(
        agent,
        manager.build_bundle(
            agent,
            enabled_ids=enabled,
            component_selections=selections,
            hook_selections=hook_ids,
        ),
    )
    local_path = manager.workspace / "agent.local.toml"
    previous_local = local_path.read_bytes() if local_path.is_file() else None
    try:
        manager.set_activation(
            plugin_id,
            selections[plugin_id],
            allowed_hooks=hook_ids[plugin_id],
        )
        return _commit_generation(agent, generation)
    except Exception:
        try:
            if previous_local is None:
                local_path.unlink(missing_ok=True)
            else:
                _atomic_text(local_path, previous_local.decode("utf-8"))
        except (OSError, UnicodeDecodeError):
            pass
        if generation.bundle.mcp_manager is not None:
            generation.bundle.mcp_manager.close()
        raise
