"""Sandbox guest invocation translation for plugin executables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from agent_core.sandbox import GuestCapabilityUnavailable, SandboxInvocation
from agent_core.tools.base import ExecutionScope

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
