"""Claude-compatible plugin installation, validation, and generation swaps.

No marketplace is preloaded. Installation only copies/records files; executable
components (hooks and MCP servers) are activated solely by an explicit enable followed
by ``/reload-plugins``, or by the policy-gated capability manager at a turn boundary.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from agent_core.config import user_settings_path
from agent_core.hook_adapters import LIFECYCLE_EVENT_ATTRS, build_external_adapter
from agent_core.hooks import ExternalHookSpec
from agent_core.local_config import update_local_table, update_toml_table
from agent_core.mcp import MCPAdapter, MCPClientManager, MCPConfig, MCPServerConfig
from agent_core.skills import Skill, SkillContext, SkillRegistry, load_skill_file, parse_frontmatter
from agent_core.tools.base import ExecutionScope, Tool

if TYPE_CHECKING:
    from agent_core.react import ReActAgent

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_REMOTE_SOURCE = re.compile(r"^(?:https?|ssh|git)://|^git@")
_SHA256_PIN = re.compile(r"^[0-9a-fA-F]{64}$")
_IMMUTABLE_GIT_COMMIT = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")


class PluginError(RuntimeError):
    pass


def is_safe_plugin_name(value: str) -> bool:
    return bool(_SAFE_NAME.fullmatch(value))


def is_sha256_pin(value: str) -> bool:
    return bool(_SHA256_PIN.fullmatch(value))


def is_git_commit_pin(value: str) -> bool:
    """Return true only for a full immutable Git object id, never a branch/tag/ref."""

    return bool(_IMMUTABLE_GIT_COMMIT.fullmatch(value))


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
    commit: str = ""


@dataclass(slots=True)
class PluginBundle:
    skills: list[Skill]
    hooks: list[tuple[str, Any]]
    mcp_manager: MCPClientManager | None
    mcp_tools: list[Any]
    agents: dict[str, str]


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
        if ".git" in relative.parts or path.is_dir():
            continue
        if path.is_symlink():
            raise PluginError(f"cannot hash symlinked plugin content: {relative}")
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


def validate_plugin(root: str | Path) -> dict[str, Any]:
    """Validate a plugin without executing it."""

    plugin_root = Path(root).expanduser().resolve()
    manifest_path = _manifest_path(plugin_root)
    if not manifest_path.is_file():
        raise PluginError(f"missing {manifest_path}")
    manifest = _read_json(manifest_path, None)
    if not isinstance(manifest, dict):
        raise PluginError("plugin.json must contain a JSON object")
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
    for key in ("commands", "skills", "agents", "hooks", "mcpServers"):
        value = manifest.get(key)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, str):
                candidate = (plugin_root / item).resolve()
                if not _inside(candidate, plugin_root):
                    raise PluginError(f"manifest path escapes plugin root: {key}")
    return manifest


class PluginManager:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()
        self.root = plugin_home()
        self.records_path = self.root / "installed.json"
        self.marketplaces_path = self.root / "marketplaces.json"

    def records(self) -> dict[str, PluginRecord]:
        raw = _read_json(self.records_path, {})
        if not isinstance(raw, dict):
            return {}
        records: dict[str, PluginRecord] = {}
        valid = {field.name for field in fields(PluginRecord)}
        for plugin_id, item in raw.items():
            if not isinstance(item, dict):
                continue
            try:
                values = {key: value for key, value in item.items() if key in valid}
                keywords = values.get("keywords")
                if isinstance(keywords, list):
                    values["keywords"] = tuple(str(value) for value in keywords)
                record = PluginRecord(**values)
            except TypeError:
                continue
            records[str(plugin_id)] = record
        return records

    def _save_records(self, records: dict[str, PluginRecord]) -> None:
        _atomic_json(self.records_path, {key: asdict(value) for key, value in records.items()})

    def marketplaces(self) -> dict[str, str]:
        raw = _read_json(self.marketplaces_path, {})
        return (
            {str(key): str(value) for key, value in raw.items()}
            if isinstance(raw, dict)
            else {}
        )

    def marketplace_add(self, name: str, source: str) -> None:
        if not _SAFE_NAME.fullmatch(name):
            raise PluginError("marketplace name must be filesystem-safe")
        if _REMOTE_SOURCE.match(source):
            source_path = self.root / "marketplaces" / name
            _clone_remote(source, source_path)
        else:
            source_path = Path(source).expanduser().resolve()
        self._load_marketplace(source_path)
        values = self.marketplaces()
        values[name] = str(source_path)
        _atomic_json(self.marketplaces_path, values)

    def marketplace_remove(self, name: str) -> None:
        values = self.marketplaces()
        if name not in values:
            raise PluginError(f"unknown marketplace: {name}")
        del values[name]
        _atomic_json(self.marketplaces_path, values)

    def marketplace_update(self, name: str) -> int:
        values = self.marketplaces()
        if name not in values:
            raise PluginError(f"unknown marketplace: {name}")
        path = Path(values[name])
        if (path / ".git").is_dir():
            _git(["-C", str(path), "pull", "--ff-only"])
        return len(self._load_marketplace(path))

    def marketplace_plugins(self, name: str) -> list[dict[str, Any]]:
        """Read one configured marketplace's bounded manifest entries."""

        values = self.marketplaces()
        if name not in values:
            raise PluginError(f"unknown marketplace: {name}")
        return self._load_marketplace(Path(values[name]))

    @staticmethod
    def _load_marketplace(source: Path) -> list[dict[str, Any]]:
        path = (
            source / ".claude-plugin" / "marketplace.json"
            if source.is_dir()
            else source
        )
        if not path.is_file() and source.is_dir():
            path = source / "marketplace.json"
        data = _read_json(path, None)
        if not isinstance(data, dict) or not isinstance(data.get("plugins"), list):
            raise PluginError(f"invalid marketplace manifest: {path}")
        return [item for item in data["plugins"] if isinstance(item, dict)]

    def _marketplace_entry(self, name: str, marketplace: str) -> dict[str, Any]:
        if not is_safe_plugin_name(name):
            raise PluginError("plugin name must be filesystem-safe")
        markets = self.marketplaces()
        if marketplace not in markets:
            raise PluginError(f"unknown marketplace: {marketplace}")
        market_source = Path(markets[marketplace])
        for entry in self._load_marketplace(market_source):
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
        markets = self.marketplaces()
        if marketplace not in markets:
            raise PluginError(f"unknown marketplace: {marketplace}")
        market_source = Path(markets[marketplace])
        entry = self._marketplace_entry(name, marketplace)
        source = entry.get("source") or entry.get("path")
        if isinstance(source, dict):
            if source.get("source") == "github" and source.get("repo"):
                source = f"https://github.com/{source['repo']}.git"
            else:
                source = source.get("path") or source.get("url") or source.get("repo")
        if not isinstance(source, str):
            raise PluginError(f"plugin {name!r} has no local source")
        if _REMOTE_SOURCE.match(source):
            source_key = expected_commit[:12] or expected_sha256[:12] or "current"
            destination = self.root / "sources" / marketplace / name / source_key
            _clone_remote(source, destination, commit=expected_commit or None)
            return destination.resolve()
        candidate = Path(source).expanduser()
        if not candidate.is_absolute():
            candidate = market_source.resolve().parent / candidate
            if market_source.is_dir():
                candidate = market_source / source
        return candidate.resolve()

    def install(
        self,
        source: str,
        marketplace: str = "local",
        *,
        expected_sha256: str = "",
        expected_commit: str = "",
    ) -> PluginRecord:
        expected_sha256 = expected_sha256.strip().lower()
        expected_commit = expected_commit.strip().lower()
        if expected_sha256 and not is_sha256_pin(expected_sha256):
            raise PluginError("plugin sha256 pin must contain exactly 64 hexadecimal characters")
        if expected_commit and not is_git_commit_pin(expected_commit):
            raise PluginError("plugin commit pin must be a full 40- or 64-character object id")
        temporary_source: Path | None = None
        if _REMOTE_SOURCE.match(source):
            staging = self.root / "staging"
            staging.mkdir(parents=True, exist_ok=True)
            temporary_source = Path(tempfile.mkdtemp(prefix="plugin.", dir=str(staging)))
            source_path = temporary_source / "source"
            _clone_remote(source, source_path, commit=expected_commit or None)
        else:
            source_path = Path(source).expanduser()
        if not source_path.exists() and marketplace != "local":
            entry = self._marketplace_entry(source, marketplace)
            expected_sha256 = (expected_sha256 or str(entry.get("sha256") or "")).strip().lower()
            expected_commit = (expected_commit or str(entry.get("commit") or "")).strip().lower()
            if expected_sha256 and not is_sha256_pin(expected_sha256):
                raise PluginError("plugin sha256 pin must contain exactly 64 hexadecimal characters")
            if expected_commit and not is_git_commit_pin(expected_commit):
                raise PluginError("plugin commit pin must be a full 40- or 64-character object id")
            source_path = self._resolve_marketplace_plugin(
                source,
                marketplace,
                expected_sha256=expected_sha256,
                expected_commit=expected_commit,
            )
        source_path = source_path.resolve()
        try:
            manifest = validate_plugin(source_path)
            name = str(manifest["name"])
            version = str(manifest.get("version") or "")
            actual_commit = _git_head(source_path) if (source_path / ".git").is_dir() else ""
            if expected_commit and actual_commit.casefold() != expected_commit.casefold():
                raise PluginError(
                    f"plugin commit mismatch: expected {expected_commit}, got {actual_commit or '(none)'}"
                )
            source_digest = plugin_tree_digest(source_path)
            if expected_sha256 and source_digest.casefold() != expected_sha256.casefold():
                raise PluginError(
                    f"plugin sha256 mismatch: expected {expected_sha256}, got {source_digest}"
                )
            plugin_id = f"{name}@{marketplace}"
            cache_root = (self.root / "cache").resolve()
            cache_key = expected_commit[:12] or expected_sha256[:12] or version or "current"
            destination = cache_root / marketplace / name / cache_key
            if not _inside(destination, cache_root):
                raise PluginError("computed cache path escaped managed plugin cache")
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = Path(
                    tempfile.mkdtemp(prefix=f".{name}.", dir=str(destination.parent))
                )
                try:
                    shutil.copytree(source_path, temporary / "plugin", symlinks=False)
                    validate_plugin(temporary / "plugin")
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
            record = PluginRecord(
                plugin_id=plugin_id,
                name=name,
                marketplace=marketplace,
                path=str(destination),
                source=source,
                version=version,
                installed_at=time.time(),
                description=str(manifest.get("description") or "").strip(),
                keywords=keywords,
                integrity=cached_digest,
                commit=actual_commit,
            )
            records = self.records()
            records[plugin_id] = record
            self._save_records(records)
            return record
        finally:
            if temporary_source is not None:
                shutil.rmtree(temporary_source, ignore_errors=True)

    def uninstall(self, plugin_id: str) -> None:
        records = self.records()
        record = records.get(plugin_id)
        if record is None:
            raise PluginError(f"plugin is not installed: {plugin_id}")
        target = Path(record.path)
        cache_root = (self.root / "cache").resolve()
        if not _inside(target, cache_root):
            raise PluginError("refusing to remove a path outside the managed plugin cache")
        if target.exists():
            shutil.rmtree(target)
        del records[plugin_id]
        self._save_records(records)
        self.set_enabled(plugin_id, False, scope="project")
        self.set_enabled(plugin_id, False, scope="user")

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
        allowed = {"skills", "agents", "hooks", "mcp"}
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

    def set_enabled(self, plugin_id: str, enabled: bool, *, scope: str = "project") -> None:
        if plugin_id not in self.records():
            raise PluginError(f"plugin is not installed: {plugin_id}")
        if scope == "user":
            try:
                path = user_settings_path()
            except RuntimeError as exc:
                raise PluginError("user home directory is unavailable") from exc
            raw = _read_plugin_list(path, "enabled")
            values = _changed_enabled(raw, plugin_id, enabled)
            update_toml_table(path, "plugins", {"enabled": values})
        elif scope == "project":
            local_path = self.workspace / "agent.local.toml"
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
        enabled = _changed_enabled(
            _read_plugin_list(self.workspace / "agent.local.toml", "enabled"),
            plugin_id,
            True,
        )
        disabled = _changed_enabled(
            _read_plugin_list(self.workspace / "agent.local.toml", "disabled"),
            plugin_id,
            False,
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
        root = Path(record.path)
        return (root / "hooks").exists() or (root / ".mcp.json").is_file()

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
            manifest = validate_plugin(root)
            if record.integrity:
                actual_digest = plugin_tree_digest(root)
                if actual_digest != record.integrity:
                    raise PluginError(
                        f"installed plugin {plugin_id} failed integrity verification: "
                        f"expected {record.integrity}, got {actual_digest}"
                    )
            namespace = str(manifest["name"])
            components = set(
                selected_components.get(
                    plugin_id,
                    ("skills", "agents", "hooks", "mcp"),
                )
            )
            if "skills" in components:
                skills.extend(_load_plugin_skills(root, namespace, manifest))
            if "agents" in components:
                plugin_agents, agent_skills = _load_plugin_agents(
                    root, namespace, manifest
                )
                agents.update(plugin_agents)
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
                    )
                )
            if "mcp" in components:
                plugin_mcp = _load_plugin_mcp(root, namespace, manifest)
                if plugin_id in selected_components:
                    plugin_mcp = _restrict_autonomous_mcp(agent, root, plugin_mcp)
                mcp_servers.extend(plugin_mcp)

        manager: MCPClientManager | None = None
        tools: list[Any] = []
        if mcp_servers:
            manager = MCPClientManager(MCPConfig(mcp_servers))
            manager.start()
            tools = MCPAdapter(manager).list_tools()
        return PluginBundle(skills, hook_pairs, manager, tools, agents)


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


def _clone_remote(source: str, destination: Path, *, commit: str | None = None) -> None:
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
        clone_args += [source, str(temporary)]
        _git(clone_args)
        if commit:
            _git(["-C", str(temporary), "checkout", "--detach", commit])
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _clone_skill(skill: Skill, namespace: str) -> Skill:
    return Skill(
        name=f"{namespace}:{skill.name}",
        description=skill.description,
        body=skill.body,
        when_to_use=skill.when_to_use,
        argument_hint=skill.argument_hint,
        allowed_tools=skill.allowed_tools,
        capabilities=skill.capabilities,
        hooks=skill.hooks,
        model=skill.model,
        aliases=tuple(f"{namespace}:{alias}" for alias in skill.aliases),
        user_invocable=skill.user_invocable,
        disable_model_invocation=skill.disable_model_invocation,
        context=skill.context,
        memory=skill.memory,
        agent_key=f"{namespace}:{skill.agent_key or skill.name}",
        source_path=skill.source_path,
    )


def _component_paths(
    root: Path,
    manifest: dict[str, Any],
    key: str,
    default: Path,
) -> list[Path]:
    raw = manifest.get(key)
    values = raw if isinstance(raw, list) else ([raw] if isinstance(raw, str) else [])
    if not values:
        return [default]
    return [(root / str(value)).resolve() for value in values]


def _load_plugin_skills(
    root: Path, namespace: str, manifest: dict[str, Any]
) -> list[Skill]:
    result: list[Skill] = []
    candidates: list[Path] = []
    for location in _component_paths(root, manifest, "skills", root / "skills"):
        if location.is_file():
            candidates.append(location)
        else:
            candidates.extend(location.glob("*/SKILL.md"))
            candidates.extend(location.glob("*.md"))
    for location in _component_paths(root, manifest, "commands", root / "commands"):
        if location.is_file():
            candidates.append(location)
        else:
            candidates.extend(location.glob("*.md"))
    for path in sorted(set(candidates)):
        skill = load_skill_file(path)
        if skill is not None:
            result.append(_clone_skill(skill, namespace))
    return result


def _load_plugin_agents(
    root: Path, namespace: str, manifest: dict[str, Any]
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
        name = f"{namespace}:{path.stem}"
        definitions[name] = raw
        memory = str(meta.get("memory", "none")).strip().lower()
        if memory not in {"none", "user", "project", "local"}:
            memory = "none"
        skills.append(
            Skill(
                name=name,
                description=f"Run the {path.stem} plugin agent.",
                body=body + "\n\n$ARGUMENTS",
                context=SkillContext.FORK,
                memory=memory,
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
    allowed_hook_ids: tuple[str, ...] | None = None,
) -> list[tuple[str, Any]]:
    result: list[tuple[str, Any]] = []
    # command_argv is an internal post-sandbox transport and must never be supplied
    # by plugin metadata itself.
    valid_fields = {
        field.name for field in fields(ExternalHookSpec) if field.name != "command_argv"
    }
    locations = _component_paths(root, manifest, "hooks", root / "hooks")
    for location in locations:
        path = location / "hooks.json" if location.is_dir() else location
        data = _read_json(path, {})
        table = data.get("hooks", data) if isinstance(data, dict) else {}
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
                    if "matcher" not in values and isinstance(group.get("matcher"), str):
                        values["matcher"] = group["matcher"]
                    if isinstance(values.get("command"), str):
                        values["command"] = values["command"].replace(
                            "${CLAUDE_PLUGIN_ROOT}", str(root)
                        )
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
        wrapped, shell = agent.sandbox.wrap(command, True, command=command, scope=scope)
        if shell or not isinstance(wrapped, list) or not wrapped:
            raise PluginError(f"sandbox could not wrap autonomous hook from {plugin_id}")
        return replace(spec, command_argv=[str(item) for item in wrapped])
    if spec.type == "http":
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
    root: Path, namespace: str, manifest: dict[str, Any]
) -> list[MCPServerConfig]:
    result: list[MCPServerConfig] = []
    locations = _component_paths(root, manifest, "mcpServers", root / ".mcp.json")
    for location in locations:
        data = _read_json(location, {})
        table = (
            data.get("mcpServers", data.get("servers", {}))
            if isinstance(data, dict)
            else {}
        )
        if not isinstance(table, dict):
            continue
        for name, body in table.items():
            if not isinstance(body, dict):
                continue
            expanded = _expand_plugin_root(body, root)
            result.append(
                MCPServerConfig.from_dict(f"{namespace}:{name}", expanded)
            )
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


def _restrict_autonomous_mcp(
    agent: "ReActAgent", root: Path, servers: list[MCPServerConfig]
) -> list[MCPServerConfig]:
    """Fail closed unless autonomous MCP transport is constrained by host policy."""

    restricted: list[MCPServerConfig] = []
    for server in servers:
        transport = (server.transport or "stdio").casefold()
        if transport in {"streamable-http", "streamable_http", "http"}:
            parsed = urlparse(server.url)
            host = parsed.hostname or ""
            loopback = host in {"localhost", "127.0.0.1", "::1"}
            if parsed.scheme != "https" and not (loopback and parsed.scheme == "http"):
                raise PluginError(
                    f"autonomous HTTP MCP {server.name} requires HTTPS or a loopback URL"
                )
            if not loopback and not _domain_allowed(host, agent.config.web.allowed_domains):
                raise PluginError(
                    f"autonomous HTTP MCP domain {host!r} is not in web.allowed_domains"
                )
            restricted.append(server)
            continue

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
        guest_root = agent.sandbox.translate_path(root)
        argv = [
            _translate_plugin_path(server.command, root, guest_root),
            *[_translate_plugin_path(item, root, guest_root) for item in server.args],
        ]
        scope = ExecutionScope.for_workspace(
            agent.session.workspace,
            read_only_roots=(root,),
            network="deny",
        )
        wrapped, shell = agent.sandbox.wrap(argv, False, scope=scope)
        if shell or not isinstance(wrapped, list) or not wrapped:
            raise PluginError(f"sandbox could not wrap autonomous MCP {server.name}")
        restricted.append(
            replace(
                server,
                command=str(wrapped[0]),
                args=[str(item) for item in wrapped[1:]],
                env={},
                cwd="",
            )
        )
    return restricted


def _expand_plugin_root(value: Any, root: Path) -> Any:
    if isinstance(value, str):
        return value.replace("${CLAUDE_PLUGIN_ROOT}", str(root))
    if isinstance(value, list):
        return [_expand_plugin_root(item, root) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _expand_plugin_root(item, root)
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
    for name in getattr(agent, "_plugin_tool_names", set()):
        agent.registry.unregister(name)
    for tool in bundle.mcp_tools:

        def factory(bound_tool: Tool = tool) -> Tool:
            return bound_tool

        agent.registry.register_deferred(
            tool.name,
            tool.description,
            factory,
            metadata={
                "kind": "mcp",
                "server": str(getattr(tool, "_server", "")),
                "remote": str(getattr(tool, "_remote", "")),
            },
        )
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
    agent.plugin_agents = bundle.agents
    agent.skills = merged
    agent.session.skills = merged
    agent.hooks = base_hooks
    agent.executor.hooks = base_hooks
    if old_manager is not None:
        old_manager.close()
    return len(bundle.skills), len(bundle.hooks), len(bundle.mcp_tools)


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
    try:
        manager.set_activation(
            plugin_id,
            selections[plugin_id],
            allowed_hooks=hook_ids[plugin_id],
        )
    except Exception:
        if generation.bundle.mcp_manager is not None:
            generation.bundle.mcp_manager.close()
        raise
    return _commit_generation(agent, generation)
