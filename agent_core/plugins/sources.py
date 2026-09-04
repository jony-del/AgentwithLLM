"""Plugin source materialization: git clones, HTTPS downloads, archives, npm."""

from __future__ import annotations

import json
import hashlib
import ipaddress
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from agent_core.plugin_spec import PluginSourceConfig
from .models import PluginError, is_git_commit_pin
from .store import _inside, _manifest_path

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
