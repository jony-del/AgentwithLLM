"""Conservative resolvers/installers for MCP Registry package types.

Resolvers fetch or pull immutable bytes into a content-addressed plan cache.  The
activation step consumes only that cache and returns an MCPServerConfig; it never
re-resolves a mutable tag/version between approval and launch.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import venv
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote, urlsplit

import httpx

from agent_core.mcp.config import MCPServerConfig
from agent_core.mcp_registry import RegistryServerRecord


class MCPPackageError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MCPPackagePlan:
    package_type: str
    identifier: str
    version: str
    digest: str = ""
    artifact_path: str = ""
    command_hint: str = ""
    arguments: tuple[str, ...] = ()
    environment: tuple[str, ...] = ()
    transport: str = "stdio"
    url: str = ""
    headers: tuple[tuple[str, str], ...] = ()
    installable: bool = True
    reason: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return asdict(self)


def _public_https(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise MCPPackageError("artifact or remote URL must be credential-free HTTPS")
    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
        }
    except OSError as exc:
        raise MCPPackageError(f"could not resolve artifact host: {parsed.hostname}") from exc
    for raw in addresses:
        address = ipaddress.ip_address(raw)
        if not address.is_global:
            raise MCPPackageError(f"artifact host resolves to a non-public address: {address}")
    return url


def _download(url: str, *, maximum: int = 512 * 1024 * 1024) -> bytes:
    safe = _public_https(url)
    with httpx.Client(timeout=60.0, follow_redirects=False) as client:
        response = client.get(safe, headers={"User-Agent": "Polaris-MCP-Resolver/3"})
        response.raise_for_status()
        if len(response.content) > maximum:
            raise MCPPackageError("MCP package exceeds the configured size limit")
        return response.content


def _argument_values(item: Mapping[str, Any]) -> tuple[str, ...]:
    raw = item.get("packageArguments", item.get("package_arguments", []))
    result: list[str] = []
    for value in raw if isinstance(raw, list) else []:
        if isinstance(value, str):
            result.append(value)
        elif isinstance(value, Mapping):
            name = str(value.get("name") or "")
            raw_value = value.get("value")
            if name:
                result.append(name)
            if raw_value is not None and raw_value is not True:
                result.append(str(raw_value))
    return tuple(result)


def _environment_names(item: Mapping[str, Any]) -> tuple[str, ...]:
    raw = item.get("environmentVariables", item.get("environment_variables", []))
    result: list[str] = []
    for value in raw if isinstance(raw, list) else []:
        name = str(value.get("name") or "") if isinstance(value, Mapping) else str(value)
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            result.append(name)
    return tuple(dict.fromkeys(result))


class MCPPackageManager:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.artifacts = self.root / "artifacts" / "mcp"
        self.installs = self.root / "mcp-installs"

    def plan(
        self, record: RegistryServerRecord, package_type: str = ""
    ) -> MCPPackagePlan:
        chosen = package_type.casefold().strip()
        available = list(record.installable_types)
        if chosen and chosen not in available:
            raise MCPPackageError(
                f"MCP server does not provide requested package type {chosen!r}"
            )
        if not chosen:
            chosen = next(
                (item for item in ("remote", "npm", "pypi", "nuget", "oci", "mcpb") if item in available),
                "",
            )
        if not chosen:
            return MCPPackagePlan("", record.name, record.version, installable=False, reason="dependency_unavailable")
        if chosen == "remote":
            remote = next((item for item in record.remotes if isinstance(item, Mapping)), None)
            if remote is None:
                raise MCPPackageError("Registry record has no remote endpoint")
            url = _public_https(str(remote.get("url") or ""))
            transport = str(remote.get("type") or remote.get("transport") or "streamable-http")
            if transport != "streamable-http":
                raise MCPPackageError(f"unsupported Registry remote transport: {transport}")
            header_names: list[tuple[str, str]] = []
            for header in remote.get("headers", []) if isinstance(remote.get("headers"), list) else []:
                if not isinstance(header, Mapping):
                    continue
                name = str(header.get("name") or "")
                variable = str(header.get("value") or header.get("variable") or "")
                if name and variable:
                    header_names.append((name, variable))
            digest = hashlib.sha256(f"{transport}:{url}".encode("utf-8")).hexdigest()
            return MCPPackagePlan(
                "remote", record.name, record.version, digest=digest,
                transport=transport, url=url, headers=tuple(header_names),
                environment=tuple(value for _name, value in header_names),
            )
        package = next(
            (
                item for item in record.packages
                if str(item.get("registryType") or item.get("registry_type") or "").casefold() == chosen
            ),
            None,
        )
        if package is None:
            raise MCPPackageError(f"Registry record has no {chosen} package")
        identifier = str(package.get("identifier") or package.get("name") or "").strip()
        version = str(package.get("version") or record.version or "").strip()
        if not identifier or not version or any(char in version for char in "*^~<>=| "):
            raise MCPPackageError(f"{chosen} package requires an exact identifier and version")
        if chosen == "npm":
            body, expected = self._resolve_npm(identifier, version)
            return self._store(chosen, identifier, version, body, package, expected)
        if chosen == "pypi":
            body, expected, filename = self._resolve_pypi(identifier, version)
            return self._store(chosen, identifier, version, body, package, expected, {"filename": filename})
        if chosen == "nuget":
            url = (
                "https://api.nuget.org/v3-flatcontainer/"
                f"{quote(identifier.casefold(), safe='')}/{quote(version.casefold(), safe='')}/"
                f"{quote(identifier.casefold(), safe='')}.{quote(version.casefold(), safe='')}.nupkg"
            )
            return self._store(chosen, identifier, version, _download(url), package)
        if chosen == "mcpb":
            url = str(package.get("url") or identifier)
            expected = str(package.get("fileSha256") or package.get("sha256") or "").casefold()
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise MCPPackageError("MCPB package requires fileSha256")
            return self._store(chosen, identifier, version, _download(url), package, expected)
        if chosen == "oci":
            return self._resolve_oci(identifier, version, package)
        raise MCPPackageError(f"unsupported MCP Registry package type: {chosen}")

    def _resolve_npm(self, identifier: str, version: str) -> tuple[bytes, str]:
        encoded = quote(identifier, safe="@")
        metadata = json.loads(_download(f"https://registry.npmjs.org/{encoded}/{quote(version, safe='')}", maximum=16 * 1024 * 1024))
        dist = metadata.get("dist", {}) if isinstance(metadata, dict) else {}
        tarball = str(dist.get("tarball") or "") if isinstance(dist, dict) else ""
        body = _download(tarball)
        # npm's sha512 integrity is verified by npm again during install; Polaris pins
        # the exact downloaded bytes with SHA-256 for the activation plan.
        return body, ""

    def _resolve_pypi(self, identifier: str, version: str) -> tuple[bytes, str, str]:
        metadata = json.loads(
            _download(
                f"https://pypi.org/pypi/{quote(identifier, safe='')}/{quote(version, safe='')}/json",
                maximum=16 * 1024 * 1024,
            )
        )
        urls = metadata.get("urls", []) if isinstance(metadata, dict) else []
        candidates = [
            item for item in urls
            if isinstance(item, dict) and not item.get("yanked") and item.get("packagetype") == "bdist_wheel"
        ]
        if not candidates:
            raise MCPPackageError("PyPI package has no wheel; source builds are not executed automatically")
        universal = next(
            (item for item in candidates if str(item.get("filename", "")).endswith(("-any.whl", "-win_amd64.whl"))),
            candidates[0],
        )
        expected = str((universal.get("digests") or {}).get("sha256") or "").casefold()
        return _download(str(universal.get("url") or "")), expected, str(universal.get("filename") or "package.whl")

    def _store(
        self,
        package_type: str,
        identifier: str,
        version: str,
        body: bytes,
        metadata: Mapping[str, Any],
        expected: str = "",
        extra: Mapping[str, Any] | None = None,
    ) -> MCPPackagePlan:
        digest = hashlib.sha256(body).hexdigest()
        if expected and digest.casefold() != expected.casefold():
            raise MCPPackageError(f"{package_type} artifact digest mismatch")
        destination = self.artifacts / digest / "package"
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(f".{os.getpid()}.tmp")
            try:
                temporary.write_bytes(body)
                try:
                    os.replace(temporary, destination)
                except OSError:
                    if not destination.exists():
                        raise
            finally:
                temporary.unlink(missing_ok=True)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise MCPPackageError("content-addressed MCP artifact is corrupt")
        return MCPPackagePlan(
            package_type,
            identifier,
            version,
            digest=digest,
            artifact_path=str(destination),
            command_hint=str(metadata.get("runtimeHint") or metadata.get("runtime_hint") or ""),
            arguments=_argument_values(metadata),
            environment=_environment_names(metadata),
            extra=dict(extra or {}),
        )

    def _resolve_oci(
        self, identifier: str, version: str, metadata: Mapping[str, Any]
    ) -> MCPPackagePlan:
        runtime = shutil.which("podman") or shutil.which("docker")
        if runtime is None:
            return MCPPackagePlan(
                "oci", identifier, version, installable=False, reason="container_runtime_unavailable"
            )
        image = identifier if "@sha256:" in identifier else f"{identifier}:{version}"
        if "@sha256:" not in image:
            completed = subprocess.run(
                [runtime, "pull", image], capture_output=True, text=True, timeout=300, check=False
            )
            if completed.returncode != 0:
                raise MCPPackageError("container runtime could not pull OCI package")
        inspected = subprocess.run(
            [runtime, "image", "inspect", image, "--format", "{{json .RepoDigests}}"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        try:
            digests = json.loads(inspected.stdout)
        except ValueError:
            digests = []
        pinned = next((str(item) for item in digests if "@sha256:" in str(item)), "")
        if not pinned and "@sha256:" in image:
            pinned = image
        if not pinned:
            raise MCPPackageError("OCI runtime did not provide an immutable repository digest")
        digest = pinned.rsplit("@sha256:", 1)[-1]
        return MCPPackagePlan(
            "oci", identifier, version, digest=digest,
            artifact_path=pinned, command_hint=Path(runtime).name,
            arguments=_argument_values(metadata), environment=_environment_names(metadata),
            extra={"runtime": runtime},
        )

    def activate(
        self,
        plan: MCPPackagePlan,
        *,
        server_name: str,
        configured_env: Mapping[str, str] | None = None,
        configured_headers: Mapping[str, str] | None = None,
    ) -> MCPServerConfig:
        if not plan.installable:
            raise MCPPackageError(plan.reason or "MCP package is not installable")
        env = {key: str(value) for key, value in (configured_env or {}).items() if key in plan.environment}
        if plan.package_type == "remote":
            headers = {
                name: str((configured_headers or {}).get(name, ""))
                for name, _variable in plan.headers
                if (configured_headers or {}).get(name)
            }
            return MCPServerConfig(
                name=server_name,
                transport=plan.transport,
                url=plan.url,
                headers=headers,
                risk="dangerous",
                discovered=True,
                trust_tier="community",
                network_policy="public-only",
            )
        artifact = Path(plan.artifact_path)
        if plan.package_type != "oci":
            if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != plan.digest:
                raise MCPPackageError("planned MCP package artifact is missing or changed")
        destination = self.installs / plan.package_type / plan.digest
        if plan.package_type == "npm":
            npm = shutil.which("npm")
            if npm is None:
                raise MCPPackageError("npm runtime is unavailable")
            if not destination.exists():
                destination.mkdir(parents=True)
                completed = subprocess.run(
                    [npm, "install", "--ignore-scripts", "--no-audit", "--no-fund", "--prefix", str(destination), str(artifact)],
                    capture_output=True, text=True, timeout=300, check=False,
                )
                if completed.returncode != 0:
                    raise MCPPackageError("npm could not install the pinned MCP artifact")
            return MCPServerConfig(
                name=server_name, command=npm,
                args=["exec", "--offline", "--prefix", str(destination), "--", plan.identifier, *plan.arguments],
                env=env, cwd=str(destination), risk="dangerous", discovered=True, trust_tier="community",
            )
        if plan.package_type == "pypi":
            python = destination / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            if not python.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                venv.EnvBuilder(with_pip=True, clear=True).create(destination)
                completed = subprocess.run(
                    [str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(artifact)],
                    capture_output=True, text=True, timeout=300, check=False,
                )
                if completed.returncode != 0:
                    raise MCPPackageError("pip could not install the pinned MCP wheel")
            scripts = destination / ("Scripts" if os.name == "nt" else "bin")
            excluded = {"python", "python3", "pip", "pip3", "activate"}
            executable = next(
                (
                    item for item in scripts.iterdir()
                    if item.is_file() and item.stem.casefold() not in excluded
                    and not item.name.casefold().startswith("activate")
                ),
                None,
            )
            if executable is None:
                raise MCPPackageError("PyPI MCP package exposes no console entry point")
            return MCPServerConfig(
                name=server_name, command=str(executable), args=list(plan.arguments), env=env,
                cwd=str(destination), risk="dangerous", discovered=True, trust_tier="community",
            )
        if plan.package_type == "nuget":
            dotnet = shutil.which("dotnet")
            if dotnet is None:
                raise MCPPackageError("dotnet runtime is unavailable")
            if not destination.exists():
                feed = destination.parent / f"feed-{plan.digest}"
                feed.mkdir(parents=True, exist_ok=True)
                package_file = feed / f"{plan.identifier}.{plan.version}.nupkg"
                if not package_file.exists():
                    shutil.copy2(artifact, package_file)
                destination.mkdir(parents=True)
                completed = subprocess.run(
                    [dotnet, "tool", "install", "--tool-path", str(destination), "--add-source", str(feed), plan.identifier, "--version", plan.version],
                    capture_output=True, text=True, timeout=300, check=False,
                )
                if completed.returncode != 0:
                    raise MCPPackageError("dotnet could not install the pinned NuGet MCP tool")
            executable = next((item for item in destination.iterdir() if item.is_file() and item.suffix.casefold() in {"", ".exe"}), None)
            if executable is None:
                raise MCPPackageError("NuGet MCP package exposes no dotnet tool entry point")
            return MCPServerConfig(
                name=server_name, command=str(executable), args=list(plan.arguments), env=env,
                cwd=str(destination), risk="dangerous", discovered=True, trust_tier="community",
            )
        if plan.package_type == "oci":
            runtime = str(plan.extra.get("runtime") or shutil.which("podman") or shutil.which("docker") or "")
            if not runtime:
                raise MCPPackageError("container runtime is unavailable")
            return MCPServerConfig(
                name=server_name, command=runtime,
                args=["run", "--rm", "-i", "--read-only", "--network", "none", "--cap-drop", "ALL", plan.artifact_path, *plan.arguments],
                env={}, risk="dangerous", discovered=True, trust_tier="community",
            )
        if plan.package_type == "mcpb":
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = Path(tempfile.mkdtemp(prefix="mcpb.", dir=str(destination.parent)))
                try:
                    with zipfile.ZipFile(artifact) as bundle:
                        if len(bundle.infolist()) > 100_000:
                            raise MCPPackageError("MCPB archive contains too many entries")
                        total = 0
                        for info in bundle.infolist():
                            total += info.file_size
                            if total > 1024 * 1024 * 1024:
                                raise MCPPackageError("MCPB archive exceeds the extraction limit")
                            mode = (info.external_attr >> 16) & 0o170000
                            if mode == 0o120000:
                                raise MCPPackageError("MCPB archive contains a symbolic link")
                            target = (temporary / info.filename).resolve()
                            try:
                                target.relative_to(temporary.resolve())
                            except ValueError as exc:
                                raise MCPPackageError("MCPB archive contains a path traversal") from exc
                            if info.is_dir():
                                target.mkdir(parents=True, exist_ok=True)
                            else:
                                target.parent.mkdir(parents=True, exist_ok=True)
                                with bundle.open(info) as source, target.open("wb") as output:
                                    shutil.copyfileobj(source, output)
                    os.replace(temporary, destination)
                finally:
                    shutil.rmtree(temporary, ignore_errors=True)
            manifests = [destination / "manifest.json", destination / ".mcpb" / "manifest.json"]
            manifest_path = next((item for item in manifests if item.is_file()), None)
            if manifest_path is None:
                raise MCPPackageError("MCPB archive has no manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw_server = manifest.get("server", manifest.get("mcp_config", {})) if isinstance(manifest, dict) else {}
            server = raw_server if isinstance(raw_server, dict) else {}
            command = str(server.get("command") or "")
            command_path = (destination / command).resolve()
            try:
                command_path.relative_to(destination.resolve())
            except ValueError as exc:
                raise MCPPackageError("MCPB entry point escapes the package root") from exc
            if not command_path.is_file():
                raise MCPPackageError("MCPB entry point is missing")
            args = [str(item) for item in server.get("args", [])] if isinstance(server.get("args"), list) else []
            return MCPServerConfig(
                name=server_name, command=str(command_path), args=args, env=env,
                cwd=str(destination), risk="dangerous", discovered=True, trust_tier="community",
            )
        raise MCPPackageError(f"unsupported MCP package type: {plan.package_type}")
