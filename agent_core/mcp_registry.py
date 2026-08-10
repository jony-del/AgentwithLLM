"""Search-only MCP Registry provider and normalized package/remote metadata.

Installation is intentionally separate from discovery.  Registry responses are
untrusted metadata and cannot supply Polaris trust/risk decisions.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import httpx

from agent_core.file_lock import FileLock


REGISTRY_PACKAGE_TYPES = frozenset({"remote", "npm", "pypi", "nuget", "oci", "mcpb"})


@dataclass(slots=True)
class MCPRegistryConfig:
    enabled: bool = False
    endpoint: str = "https://registry.modelcontextprotocol.io"
    mode: str = "search_only"
    allowed_package_types: tuple[str, ...] = tuple(sorted(REGISTRY_PACKAGE_TYPES))
    refresh_ttl_seconds: int = 3600
    request_timeout_seconds: float = 10.0
    max_results: int = 20

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None, *, default_enabled: bool = True) -> "MCPRegistryConfig":
        raw = dict(data or {})
        endpoint = str(raw.get("endpoint") or cls().endpoint).rstrip("/")
        parsed = urlsplit(endpoint)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            endpoint = cls().endpoint
        package_types = raw.get("allowed_package_types", tuple(sorted(REGISTRY_PACKAGE_TYPES)))
        allowed = tuple(
            dict.fromkeys(
                str(item).casefold()
                for item in package_types
                if str(item).casefold() in REGISTRY_PACKAGE_TYPES
            )
        ) if isinstance(package_types, (list, tuple)) else tuple(sorted(REGISTRY_PACKAGE_TYPES))
        try:
            ttl = max(0, int(raw.get("refresh_ttl_seconds", 3600)))
        except (TypeError, ValueError):
            ttl = 3600
        try:
            timeout = max(1.0, min(60.0, float(raw.get("request_timeout_seconds", 10.0))))
        except (TypeError, ValueError):
            timeout = 10.0
        try:
            maximum = max(1, min(100, int(raw.get("max_results", 20))))
        except (TypeError, ValueError):
            maximum = 20
        return cls(
            enabled=bool(raw.get("enabled", default_enabled)),
            endpoint=endpoint,
            mode="search_only",
            allowed_package_types=allowed,
            refresh_ttl_seconds=ttl,
            request_timeout_seconds=timeout,
            max_results=maximum,
        )


@dataclass(frozen=True, slots=True)
class RegistryServerRecord:
    name: str
    version: str
    description: str
    title: str = ""
    repository: str = ""
    website: str = ""
    packages: tuple[Mapping[str, Any], ...] = ()
    remotes: tuple[Mapping[str, Any], ...] = ()
    keywords: tuple[str, ...] = ()
    publisher: str = ""
    installable_types: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        return f"mcp-registry:{self.name}@{self.version or 'latest'}"

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": "mcp",
            "name": self.title or self.name,
            "description": self.description,
            "state": "available",
            "source": "mcp-registry",
            "publisher": self.publisher or None,
            "homepage": self.website or self.repository or None,
            "version": self.version or None,
            "package_types": list(self.installable_types),
            "installable": bool(self.installable_types),
            "trust_tier": "community",
            "risk": "dangerous",
            "requires_approval": True,
        }


class MCPRegistryClient:
    def __init__(self, root: str | Path, config: MCPRegistryConfig) -> None:
        self.root = Path(root)
        self.config = config
        self.cache_root = self.root / "registry"

    def search(self, query: str, *, limit: int | None = None) -> list[RegistryServerRecord]:
        if not self.config.enabled or not query.strip():
            return []
        maximum = max(1, min(limit or self.config.max_results, self.config.max_results))
        key = hashlib.sha256(query.strip().casefold().encode("utf-8")).hexdigest()
        cache = self.cache_root / "search" / f"{key}.json"
        cached = self._read_cache(cache)
        now = time.time()
        if cached and now - float(cached.get("fetched_at", 0.0)) < self.config.refresh_ttl_seconds:
            return self._normalize(cached.get("response"), maximum)
        response: Any = None
        try:
            with httpx.Client(
                timeout=self.config.request_timeout_seconds,
                follow_redirects=False,
                headers={"Accept": "application/json", "User-Agent": "Polaris-Capability-Discovery/3"},
            ) as client:
                result = client.get(
                    self.config.endpoint + "/v0.1/servers",
                    params={"search": query[:500], "version": "latest", "limit": maximum},
                )
                result.raise_for_status()
                if len(result.content) > 8 * 1024 * 1024:
                    raise ValueError("MCP Registry response exceeds 8 MiB")
                response = result.json()
            self._write_cache(cache, {"fetched_at": now, "response": response})
        except (OSError, ValueError, httpx.HTTPError):
            if cached:
                response = cached.get("response")
            else:
                return []
        return self._normalize(response, maximum)

    def _read_cache(self, path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def _write_cache(self, path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_suffix(".lock")
        temporary = path.with_suffix(f".{time.time_ns()}.tmp")
        with FileLock(lock):
            try:
                temporary.write_text(
                    json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8"
                )
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)

    def _normalize(self, response: Any, limit: int) -> list[RegistryServerRecord]:
        values = response.get("servers", []) if isinstance(response, dict) else []
        records: list[RegistryServerRecord] = []
        for wrapper in values if isinstance(values, list) else []:
            if not isinstance(wrapper, dict):
                continue
            server = wrapper.get("server", wrapper)
            if not isinstance(server, dict):
                continue
            name = str(server.get("name") or "").strip()
            if not name or len(name) > 300:
                continue
            packages = tuple(item for item in server.get("packages", []) if isinstance(item, dict))
            remotes = tuple(item for item in server.get("remotes", []) if isinstance(item, dict))
            types = {
                str(item.get("registryType") or item.get("registry_type") or "").casefold()
                for item in packages
            }
            if remotes:
                types.add("remote")
            types &= set(self.config.allowed_package_types)
            repository = server.get("repository")
            repository_url = (
                str(repository.get("url") or "") if isinstance(repository, dict) else str(repository or "")
            )
            meta = wrapper.get("_meta", {})
            publisher = ""
            if isinstance(meta, dict):
                publisher = str(meta.get("publisher") or meta.get("publisherId") or "")
            records.append(
                RegistryServerRecord(
                    name=name,
                    version=str(server.get("version") or ""),
                    description=str(server.get("description") or "")[:4000],
                    title=str(server.get("title") or "")[:300],
                    repository=repository_url[:2000],
                    website=str(server.get("websiteUrl") or server.get("website") or "")[:2000],
                    packages=packages,
                    remotes=remotes,
                    keywords=tuple(str(item)[:100] for item in server.get("keywords", []) if isinstance(item, str)),
                    publisher=publisher[:300],
                    installable_types=tuple(sorted(types)),
                )
            )
            if len(records) >= limit:
                break
        return records
