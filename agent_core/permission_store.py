"""Atomic persistence for user-approved permission rule updates."""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from agent_core.permission_types import PermissionDestination


class PermissionPersistenceError(OSError):
    pass


def destination_path(destination: PermissionDestination, workspace: Path) -> Path:
    if destination is PermissionDestination.LOCAL:
        return workspace / "agent.local.toml"
    if destination is PermissionDestination.PROJECT:
        return workspace / "agent.toml"
    if destination is PermissionDestination.USER:
        try:
            return Path.home() / ".polaris" / "agent.toml"
        except RuntimeError as exc:
            raise PermissionPersistenceError("user home directory is unavailable") from exc
    raise PermissionPersistenceError("session grants are not written to disk")


def persist_allow_rule(rule: str, destination: PermissionDestination, workspace: Path) -> Path:
    """Add one allow rule without duplicating it, using lock + fsync + atomic replace."""
    return persist_allow_rules((rule,), destination, workspace)


def persist_allow_rules(
    rules: tuple[str, ...], destination: PermissionDestination, workspace: Path
) -> Path:
    """Atomically add a batch of allow rules to one destination."""
    path = destination_path(destination, workspace.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".permissions.lock")
    with _exclusive_lock(lock_path):
        original = path.read_text(encoding="utf-8") if path.exists() else ""
        updated = original
        for rule in rules:
            updated = _update_permissions_allow(updated, rule)
        if updated == original:
            return path
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                stream.write(updated)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    if destination is PermissionDestination.PROJECT:
        _record_project_trust(path)
    return path


@contextmanager
def _exclusive_lock(path: Path, timeout: float = 5.0) -> Iterator[None]:
    deadline = time.monotonic() + timeout
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            try:
                if time.time() - path.stat().st_mtime > 60:
                    path.unlink()
                    continue
            except (FileNotFoundError, OSError):
                pass
            if time.monotonic() >= deadline:
                raise PermissionPersistenceError(f"timed out locking {path}") from exc
            time.sleep(0.02)
    try:
        os.write(fd, f"{os.getpid()}\n".encode())
        yield
    finally:
        os.close(fd)
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _update_permissions_allow(text: str, rule: str) -> str:
    try:
        import tomlkit
    except ModuleNotFoundError:
        tomlkit = None  # type: ignore[assignment]
    if tomlkit is not None:
        try:
            document = tomlkit.parse(text) if text.strip() else tomlkit.document()
            permissions = document.get("permissions")
            if permissions is None:
                permissions = tomlkit.table()
                document["permissions"] = permissions
            allow = permissions.get("allow")
            if allow is None:
                allow = tomlkit.array().multiline(True)
                permissions["allow"] = allow
            existing = [str(item) for item in allow]
            if rule not in existing:
                allow.append(rule)
            return tomlkit.dumps(document)
        except Exception as exc:
            raise PermissionPersistenceError(f"cannot update TOML: {exc}") from exc
    try:
        import tomllib

        parsed = tomllib.loads(text) if text.strip() else {}
    except (UnicodeDecodeError, ValueError) as exc:
        raise PermissionPersistenceError(f"cannot update malformed TOML: {exc}") from exc
    permissions = parsed.get("permissions", {})
    if permissions is not None and not isinstance(permissions, dict):
        raise PermissionPersistenceError("[permissions] is not a TOML table")
    existing = permissions.get("allow", []) if isinstance(permissions, dict) else []
    if not isinstance(existing, list) or any(not isinstance(item, str) for item in existing):
        raise PermissionPersistenceError("permissions.allow is not an array of strings")
    if rule in existing:
        return text
    values = [*existing, rule]
    rendered = "allow = [\n" + "".join(f"    {json.dumps(item, ensure_ascii=False)},\n" for item in values) + "]"

    section = re.search(r"(?m)^\[permissions\][ \t]*(?:#.*)?$", text)
    if section is None:
        prefix = text.rstrip()
        return (prefix + "\n\n" if prefix else "") + "[permissions]\n" + rendered + "\n"
    next_section = re.search(r"(?m)^\[[^\]\n]+\][ \t]*(?:#.*)?$", text[section.end() :])
    end = section.end() + (next_section.start() if next_section else len(text[section.end() :]))
    body = text[section.end() : end]
    assignment = re.search(r"(?m)^[ \t]*allow[ \t]*=", body)
    if assignment is None:
        insertion = section.end()
        return text[:insertion] + "\n" + rendered + text[insertion:]
    value_start = assignment.end()
    value_end = _toml_value_end(body, value_start)
    return text[: section.end() + assignment.start()] + rendered + body[value_end:] + text[end:]


def _toml_value_end(text: str, start: int) -> int:
    index = start
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or text[index] != "[":
        return text.find("\n", index) if "\n" in text[index:] else len(text)
    depth = 0
    quote: str | None = None
    escaped = False
    for pos in range(index, len(text)):
        char = text[pos]
        if escaped:
            escaped = False
            continue
        if quote is not None:
            if char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return pos + 1
    raise PermissionPersistenceError("unterminated permissions.allow array")


def _record_project_trust(path: Path) -> None:
    try:
        import tomllib

        from agent_core.trust import TrustStore, fingerprint, widening_subset

        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        subset = widening_subset(raw)
        # The permission dialog authorizes the allow-rule change, not unrelated
        # repo-controlled hooks/MCP/sandbox relaxations. Never launder those into trust.
        if set(subset) == {"permissions.allow"}:
            TrustStore().record(path.resolve().parent, fingerprint(subset))
    except (OSError, ValueError):
        # The permission rule itself is already safely persisted. Trust recording is
        # best effort and the normal TOFU layer will fail closed on the next load.
        return


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass
