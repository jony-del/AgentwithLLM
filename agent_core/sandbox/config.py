"""Configuration for the OS-level sandbox — the *enforcement layer*.

The shape mirrors Open-ClaudeCode's ``settings.sandbox`` (``entrypoints/sandboxTypes.ts``)
so the mental model carries over. These are plain dataclasses hydrated from the
``[sandbox]`` toml table via :func:`agent_core.config.resolve_sandbox_config`; nothing
here touches the OS (that's :mod:`agent_core.sandbox.manager`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SandboxNetworkConfig:
    """Network egress policy handed to the OS backend (bwrap/socat, seatbelt)."""

    # When the sandbox is active, network is default-deny; these hosts are re-allowed.
    allowed_domains: list[str] = field(default_factory=list)
    # Permit binding local ports inside the sandbox (servers/tests that listen).
    allow_local_binding: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SandboxNetworkConfig":
        config = cls()
        if not data:
            return config
        if isinstance(data.get("allowed_domains"), list):
            config.allowed_domains = [str(d) for d in data["allowed_domains"]]
        if "allow_local_binding" in data:
            config.allow_local_binding = _as_bool(data["allow_local_binding"])
        return config


@dataclass(slots=True)
class SandboxFilesystemConfig:
    """Filesystem policy: default-deny writes; the workspace is always writable.

    ``deny_read`` hides sensitive regions (``~/.ssh``); ``allow_read`` punches holes
    back through a deny (``~/.ssh/known_hosts``). Paths are passed to the OS backend
    as-is (``~`` expansion happens in the manager).
    """

    allow_write: list[str] = field(default_factory=list)
    deny_write: list[str] = field(default_factory=list)
    deny_read: list[str] = field(default_factory=list)
    allow_read: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SandboxFilesystemConfig":
        config = cls()
        if not data:
            return config
        for key in ("allow_write", "deny_write", "deny_read", "allow_read"):
            value = data.get(key)
            if isinstance(value, list):
                setattr(config, key, [str(v) for v in value])
        return config


@dataclass(slots=True)
class SandboxConfig:
    """Top-level sandbox settings.

    ``enabled`` is the master switch. On an unsupported platform (Windows) or with
    missing dependencies the manager degrades to a no-op *unless* ``fail_if_unavailable``
    is set, which turns "can't sandbox" into a hard startup error (for managed
    deployments that must not run commands unsandboxed).
    """

    enabled: bool = False
    fail_if_unavailable: bool = False
    # Skip the interactive permission prompt for a command that *will* actually be
    # sandboxed — the OS sandbox is the real boundary (reference: autoAllowBashIfSandboxed).
    auto_allow_command_if_sandboxed: bool = True
    # Honor a per-call ``dangerously_disable_sandbox`` request (run a command unsandboxed).
    allow_unsandboxed_commands: bool = True
    # Commands that run OUTSIDE the sandbox (build tools that break under isolation).
    # NOT a security boundary — excluded commands still go through normal permissions.
    excluded_commands: list[str] = field(default_factory=list)
    network: SandboxNetworkConfig = field(default_factory=SandboxNetworkConfig)
    filesystem: SandboxFilesystemConfig = field(default_factory=SandboxFilesystemConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SandboxConfig":
        config = cls()
        if not data:
            return config
        if "enabled" in data:
            config.enabled = _as_bool(data["enabled"])
        if "fail_if_unavailable" in data:
            config.fail_if_unavailable = _as_bool(data["fail_if_unavailable"])
        if "auto_allow_command_if_sandboxed" in data:
            config.auto_allow_command_if_sandboxed = _as_bool(data["auto_allow_command_if_sandboxed"])
        if "allow_unsandboxed_commands" in data:
            config.allow_unsandboxed_commands = _as_bool(data["allow_unsandboxed_commands"])
        if isinstance(data.get("excluded_commands"), list):
            config.excluded_commands = [str(c) for c in data["excluded_commands"]]
        config.network = SandboxNetworkConfig.from_dict(_as_table(data.get("network")))
        config.filesystem = SandboxFilesystemConfig.from_dict(_as_table(data.get("filesystem")))
        return config


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_table(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None
