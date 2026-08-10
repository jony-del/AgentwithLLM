"""Claude marketplace/plugin schema normalization.

This module is deliberately side-effect free.  Network and filesystem mutations live in
``plugins.py``; everything here can therefore be used by validation, catalog search and
installation without accidentally fetching executable content.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


class SpecError(ValueError):
    pass


_GITHUB = re.compile(r"^(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(?:@(?P<ref>[^\s]+))?$")
_GIT_URL = re.compile(r"^(?:git@|ssh://|git://)|(?:\.git)(?:[#?].*)?$", re.I)
_HTTP = re.compile(r"^https?://", re.I)
_COMPONENT_FIELDS = (
    "skills", "commands", "agents", "hooks", "mcpServers", "lspServers",
    "workflows", "outputStyles", "themes", "monitors",
)
_KNOWN_MANIFEST_FIELDS = frozenset({
    "$schema", "name", "displayName", "version", "description", "author",
    "homepage", "repository", "license", "keywords", "metadata", "skills",
    "commands", "agents", "hooks", "mcpServers", "lspServers", "workflows",
    "outputStyles", "themes", "monitors", "experimental", "dependencies",
    "userConfig", "channels", "settings", "defaultEnabled",
})


def _clean_map(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items()}


@dataclass(frozen=True, slots=True)
class MarketplaceSourceConfig:
    """Normalized Claude marketplace source.

    ``url`` means a remote marketplace JSON document while ``git`` means a Git
    repository URL.  The distinction matters because relative plugin sources are not
    valid for a JSON-only snapshot.
    """

    kind: str
    repo: str = ""
    url: str = ""
    path: str = ""
    ref: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)
    name: str = ""
    plugins: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_value(
        cls, value: str | Mapping[str, Any], *, base_dir: str | Path | None = None
    ) -> "MarketplaceSourceConfig":
        base = Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                raise SpecError("marketplace source is empty")
            github = _GITHUB.fullmatch(raw)
            if github:
                return cls(
                    "github",
                    repo=f"{github.group('owner')}/{github.group('repo')}",
                    ref=github.group("ref") or "",
                )
            if _HTTP.match(raw):
                kind = "git" if _GIT_URL.search(raw) else "url"
                return cls(kind, url=raw)
            if raw.startswith(("git@", "ssh://", "git://")):
                return cls("git", url=raw)
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = base / path
            # A non-existent .json path is still unambiguously a file source.
            kind = "file" if path.suffix.casefold() == ".json" or path.is_file() else "directory"
            return cls(kind, path=str(path.resolve()))
        if not isinstance(value, Mapping):
            raise SpecError("marketplace source must be a string or table")
        data = _clean_map(value)
        kind = str(data.get("kind") or data.get("source") or "").strip().casefold()
        aliases = {"http": "url", "remote": "url", "path": "directory"}
        kind = aliases.get(kind, kind)
        if kind not in {"github", "git", "url", "file", "directory", "settings"}:
            raise SpecError(f"unsupported marketplace source kind: {kind or '(missing)'}")
        repo = str(data.get("repo") or "").strip()
        url = str(data.get("url") or "").strip()
        raw_path = str(data.get("path") or data.get("file") or data.get("directory") or "").strip()
        if kind == "github" and not _GITHUB.fullmatch(repo):
            raise SpecError("github marketplace repo must use owner/repo syntax")
        if kind in {"git", "url"} and not url:
            raise SpecError(f"{kind} marketplace source requires url")
        if kind in {"file", "directory"} and not raw_path:
            raise SpecError(f"{kind} marketplace source requires path")
        inline_name = str(data.get("name") or "").strip()
        plugins_raw = data.get("plugins", [])
        if kind == "settings" and (
            not inline_name or not isinstance(plugins_raw, list)
            or not all(isinstance(item, Mapping) for item in plugins_raw)
        ):
            raise SpecError("settings marketplace source requires name and plugins")
        resolved_path = ""
        if raw_path and kind in {"file", "directory"}:
            candidate = Path(raw_path).expanduser()
            if not candidate.is_absolute():
                candidate = base / candidate
            resolved_path = str(candidate.resolve())
        elif raw_path:
            _validate_relative_path(raw_path, allow_dot=True)
            resolved_path = raw_path
        headers_raw = data.get("headers", {})
        headers = (
            {str(key): str(item) for key, item in headers_raw.items()}
            if isinstance(headers_raw, Mapping) else {}
        )
        return cls(
            kind, repo=repo, url=url, path=resolved_path,
            ref=str(data.get("ref") or ""), headers=headers,
            name=inline_name,
            plugins=tuple(_clean_map(item) for item in plugins_raw if isinstance(item, Mapping)),
        )

    @property
    def remote(self) -> bool:
        return self.kind in {"github", "git", "url"}

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"source": self.kind}
        for key, value in asdict(self).items():
            if key != "kind" and value:
                result[key] = value
        return result

    def display(self) -> str:
        if self.kind == "github":
            return self.repo + (f"@{self.ref}" if self.ref else "")
        return self.url or self.path or f"settings:{self.name}"


@dataclass(frozen=True, slots=True)
class PluginSourceConfig:
    kind: str
    path: str = ""
    repo: str = ""
    url: str = ""
    ref: str = ""
    sha: str = ""
    package: str = ""
    version: str = ""
    registry: str = ""
    sha256: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: str | Mapping[str, Any]) -> "PluginSourceConfig":
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                raise SpecError("plugin source is empty")
            return cls("relative", path=raw)
        if not isinstance(value, Mapping):
            raise SpecError("plugin source must be a relative path or object")
        data = _clean_map(value)
        kind = str(data.get("source") or data.get("kind") or "").strip().casefold()
        if kind == "git":
            kind = "url"
        if kind == "url" and str(data.get("path") or "").strip():
            kind = "git-subdir"
        if kind not in {"github", "url", "git-subdir", "npm", "archive"}:
            raise SpecError(f"unsupported plugin source kind: {kind or '(missing)'}")
        headers_raw = data.get("headers", {})
        headers = (
            {str(key): str(item) for key, item in headers_raw.items()}
            if isinstance(headers_raw, Mapping) else {}
        )
        result = cls(
            kind,
            path=str(data.get("path") or "").strip(),
            repo=str(data.get("repo") or "").strip(),
            url=str(data.get("url") or "").strip(),
            ref=str(data.get("ref") or "").strip(),
            sha=str(data.get("sha") or data.get("commit") or "").strip().lower(),
            package=str(data.get("package") or "").strip(),
            version=str(data.get("version") or "").strip(),
            registry=str(data.get("registry") or "").strip(),
            sha256=str(data.get("sha256") or "").strip().lower(),
            headers=headers,
        )
        required = {
            "github": ("repo",), "url": ("url",), "git-subdir": ("url", "path"),
            "npm": ("package",), "archive": ("url",),
        }[kind]
        if any(not getattr(result, field_name) for field_name in required):
            raise SpecError(f"{kind} plugin source requires {', '.join(required)}")
        if result.path:
            _validate_relative_path(result.path, allow_dot=False)
        return result

    @property
    def immutable(self) -> bool:
        return bool(self.sha or self.sha256 or (self.kind == "npm" and self.version and not any(c in self.version for c in "*xX^~<>=| ")))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"source": self.kind}
        for key, value in asdict(self).items():
            if key != "kind" and value:
                result[key] = value
        return result


def _validate_relative_path(raw: str, *, allow_dot: bool = True) -> None:
    normalized = raw.replace("\\", "/")
    if allow_dot and normalized in {".", "./"}:
        return
    # Absolute paths remain accepted for legacy local-only installations.  Catalog
    # component paths, checked by ``validate_component_paths``, are stricter.
    pure = PurePosixPath(normalized)
    if ".." in pure.parts:
        raise SpecError(f"path escapes plugin root: {raw}")


def _values(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else ([] if value is None else [value])


def _merge_component(left: Any, right: Any) -> Any:
    if left is None:
        return right
    if right is None:
        return left
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        merged = _clean_map(left)
        merged.update(_clean_map(right))
        return merged
    return [*_values(left), *_values(right)]


def resolve_plugin_manifest(
    root: str | Path, marketplace_entry: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Load the optional manifest and apply Claude's marketplace strict semantics."""

    import json

    plugin_root = Path(root).resolve()
    path = plugin_root / ".claude-plugin" / "plugin.json"
    manifest: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise SpecError(f"invalid plugin manifest: {exc}") from exc
        if not isinstance(loaded, dict):
            raise SpecError("plugin.json must contain a JSON object")
        manifest = _clean_map(loaded)
    entry = _clean_map(marketplace_entry or {})
    strict = entry.get("strict", True) is not False
    entry_components = {key: entry[key] for key in _COMPONENT_FIELDS if key in entry}
    experimental = entry.get("experimental")
    if isinstance(experimental, Mapping):
        for key in ("themes", "monitors"):
            if key in experimental:
                entry_components[key] = experimental[key]
    manifest_components = {key: manifest[key] for key in _COMPONENT_FIELDS if key in manifest}
    manifest_experimental = manifest.get("experimental")
    if isinstance(manifest_experimental, Mapping):
        for key in ("themes", "monitors"):
            if key in manifest_experimental:
                manifest_components[key] = manifest_experimental[key]
    if not strict and manifest_components:
        raise SpecError("strict:false conflicts with component declarations in plugin.json")
    if strict:
        resolved = dict(manifest)
        for key, value in entry.items():
            if key not in _COMPONENT_FIELDS and key not in {"source", "strict", "components", "sha256", "commit", "category", "tags", "relevance"}:
                resolved.setdefault(key, value)
        for key, value in entry_components.items():
            resolved[key] = _merge_component(manifest_components.get(key), value)
    else:
        resolved = {
            key: value for key, value in entry.items()
            if key not in {"source", "strict", "components", "sha256", "commit", "category", "tags", "relevance"}
        }
        resolved.update(entry_components)
    resolved["name"] = str(entry.get("name") or resolved.get("name") or plugin_root.name).strip()
    entry_source = entry.get("source")
    if isinstance(entry_source, str) and entry_source in {".", "./"} and "skills" in entry:
        skill_paths = [item for item in _values(entry.get("skills")) if isinstance(item, str)]
        full_scan = any(item in {".", "./", "./skills", "./skills/"} for item in skill_paths)
        if not full_scan and any((plugin_root / item).exists() for item in skill_paths):
            resolved["_skills_replace_default"] = True
    if not resolved["name"]:
        raise SpecError("plugin name is empty")
    warnings = tuple(
        f"unrecognized plugin manifest field: {key}"
        for key in sorted(set(manifest) - _KNOWN_MANIFEST_FIELDS)
    )
    validate_component_paths(plugin_root, resolved)
    return resolved, warnings


def validate_component_paths(root: Path, manifest: Mapping[str, Any]) -> None:
    for key in _COMPONENT_FIELDS:
        value = manifest.get(key)
        if isinstance(value, Mapping):
            continue
        for item in _values(value):
            if key in {"hooks", "mcpServers", "lspServers"} and isinstance(item, Mapping):
                continue
            if not isinstance(item, str):
                raise SpecError(f"{key} must contain relative path strings")
            if not (item == "." or item.startswith("./")):
                raise SpecError(f"{key} path must start with './': {item}")
            _validate_relative_path(item)
            candidate = (root / item).resolve()
            try:
                candidate.relative_to(root.resolve())
            except ValueError as exc:
                raise SpecError(f"{key} path escapes plugin root: {item}") from exc


def declared_components(root: str | Path, manifest: Mapping[str, Any]) -> tuple[str, ...]:
    base = Path(root)
    found: list[str] = []
    defaults = {
        "skills": (base / "skills", base / "SKILL.md"),
        "commands": (base / "commands",), "agents": (base / "agents",),
        "hooks": (base / "hooks" / "hooks.json",), "mcp": (base / ".mcp.json",),
        "lsp": (base / ".lsp.json",), "workflows": (base / "workflows",),
        "output-styles": (base / "output-styles",), "themes": (base / "themes",),
        "monitors": (base / "monitors" / "monitors.json",), "bin": (base / "bin",),
        "settings": (base / "settings.json",),
    }
    fields = {
        "skills": "skills", "commands": "skills", "agents": "agents", "hooks": "hooks",
        "mcpServers": "mcp", "lspServers": "lsp", "workflows": "workflows",
        "outputStyles": "output-styles", "themes": "themes", "monitors": "monitors",
        "userConfig": "user-config", "channels": "channels", "settings": "settings",
    }
    for key, component in fields.items():
        if key in manifest and manifest[key] not in (None, [], {}):
            found.append(component)
    # Themes and monitors currently live under ``experimental`` in Claude plugin
    # manifests.  Treat them exactly like their future top-level counterparts so
    # installation records do not silently omit a component the runtime can load.
    experimental = manifest.get("experimental")
    if isinstance(experimental, Mapping):
        if experimental.get("themes") not in (None, [], {}):
            found.append("themes")
        if experimental.get("monitors") not in (None, [], {}):
            found.append("monitors")
    for component, candidates in defaults.items():
        if any(path.exists() for path in candidates):
            found.append(component)
    return tuple(dict.fromkeys(found))
