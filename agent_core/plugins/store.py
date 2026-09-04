"""Plugin state storage: atomic JSON writes, validation, and tree helpers."""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any
from agent_core.plugin_spec import MarketplaceSourceConfig, SpecError, resolve_plugin_manifest
from agent_core import secret_store
from .models import _SAFE_NAME, PluginError

_PLUGIN_COMPONENTS = frozenset({
    "skills", "agents", "hooks", "mcp", "lsp", "workflows", "monitors",
    "channels", "output-styles", "themes", "user-config", "bin", "settings",
})


_KEYCHAIN_PREFIX = "keychain://Polaris/plugin-config/"


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
