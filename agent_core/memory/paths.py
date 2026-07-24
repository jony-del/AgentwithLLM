from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePath

from agent_core.memory.models import MemoryScope


class UnsafeMemoryPathError(ValueError):
    pass


def canonical_agent_key(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise UnsafeMemoryPathError("named-agent memory requires an agent_key")
    if re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", raw):
        return raw
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-._")[:48] or "agent"
    suffix = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"{slug}-{suffix}"


def _git_path(workspace: Path, argument: str) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", argument],
            capture_output=True,
            check=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    if not value:
        return None
    path = Path(value)
    return (workspace / path).resolve() if not path.is_absolute() else path.resolve()


def validate_memory_root(path: str | Path, *, allowed_parent: str | Path | None = None) -> Path:
    raw = os.fspath(path)
    if "\0" in raw:
        raise UnsafeMemoryPathError("memory path contains a null byte")
    if raw.startswith(("\\\\", "//")):
        raise UnsafeMemoryPathError("UNC/network memory paths are not allowed")
    candidate = Path(raw).expanduser()
    if any(part == ".." for part in PurePath(raw).parts):
        raise UnsafeMemoryPathError("memory path may not contain '..'")
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate == Path(candidate.anchor):
        raise UnsafeMemoryPathError("a filesystem or drive root cannot be a memory directory")
    resolved = candidate.resolve(strict=False)
    if allowed_parent is not None:
        parent = Path(allowed_parent).expanduser().resolve(strict=False)
        try:
            resolved.relative_to(parent)
        except ValueError as exc:
            raise UnsafeMemoryPathError(f"memory path escapes allowed parent: {parent}") from exc
    # ``resolve(strict=False)`` resolves every existing symlink component even when
    # the final directory does not exist, so the containment check above also catches
    # symlink escapes without rejecting a not-yet-created allowed parent.
    return resolved


@dataclass(slots=True)
class MemoryPathResolver:
    workspace: Path
    user_root: Path
    private_override: Path | None = None

    def __init__(
        self,
        workspace: str | Path = ".",
        *,
        user_root: str | Path | None = None,
        private_override: str | Path | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        if user_root is not None:
            self.user_root = Path(user_root).expanduser()
        else:
            try:
                home = Path.home()
            except RuntimeError:
                home = Path.cwd()
            self.user_root = home / ".polaris"
        self.private_override = Path(private_override).expanduser() if private_override is not None else None

    @property
    def checkout_root(self) -> Path:
        return _git_path(self.workspace, "--show-toplevel") or self.workspace

    @property
    def project_id(self) -> str:
        identity = _git_path(self.workspace, "--git-common-dir") or self.workspace
        canonical = os.path.normcase(str(identity.resolve()))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]

    def resolve(
        self,
        scope: MemoryScope = "private",
        *,
        agent_key: str | None = None,
    ) -> Path:
        if scope == "team":
            root = self.checkout_root / ".polaris" / "memory" / "team"
            return validate_memory_root(root, allowed_parent=self.checkout_root)
        if scope == "private":
            if self.private_override is not None:
                return validate_memory_root(self.private_override)
            root = self.user_root / "projects" / self.project_id / "memory" / "private"
            return validate_memory_root(root, allowed_parent=self.user_root)
        if not agent_key:
            raise UnsafeMemoryPathError("named-agent memory requires an agent_key")
        agent_key = canonical_agent_key(agent_key)
        if scope == "user":
            root = self.user_root / "agents" / agent_key / "memory"
            return validate_memory_root(root, allowed_parent=self.user_root)
        if scope == "project":
            root = self.user_root / "projects" / self.project_id / "agents" / agent_key / "memory"
            return validate_memory_root(root, allowed_parent=self.user_root)
        if scope == "local":
            root = self.checkout_root / ".polaris" / "agents" / agent_key / "memory"
            return validate_memory_root(root, allowed_parent=self.checkout_root)
        raise UnsafeMemoryPathError(f"unsupported memory scope: {scope}")
