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
class SandboxContainerConfig:
    """Container tier settings (podman/docker/nerdctl). Hardened by default.

    Defaults mirror the project's Linux research: default-deny network, read-only rootfs,
    drop all capabilities, no new privileges, and optional cpu/mem/pid limits. The
    workspace is bind-mounted at the *same absolute path* so no in-command path
    translation is needed (on Windows the drive is mapped to a ``/mnt/x`` WSL2 path).
    """

    # Runtime binary: "auto" probes podman → docker → nerdctl in order.
    runtime: str = "auto"
    # Image the command runs inside. Must be present (or auto_pull) at prepare() time.
    image: str = "docker.io/library/debian:stable-slim"
    # Pull the image at startup if it is missing.
    auto_pull: bool = False
    # OCI runtime override for VM-grade isolation reused by the container launcher
    # (e.g. "kata-runtime", "runsc"). Empty → the runtime's default (runc/crun).
    oci_runtime: str = ""
    read_only_rootfs: bool = True
    drop_all_capabilities: bool = True
    no_new_privileges: bool = True
    # Resource limits (empty string → not passed). Strings so "512m"/"1.5" pass through.
    memory: str = ""
    cpus: str = ""
    pids_limit: str = ""
    # Windows only: "wsl2" (Linux containers, default) or "hyperv" (Windows containers).
    windows_isolation: str = "wsl2"
    # Extra raw flags appended verbatim to the `run` invocation (escape hatch).
    extra_run_args: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SandboxContainerConfig":
        config = cls()
        if not data:
            return config
        for key in ("runtime", "image", "oci_runtime", "memory", "cpus", "pids_limit",
                    "windows_isolation"):
            if key in data:
                setattr(config, key, str(data[key]))
        for key in ("auto_pull", "read_only_rootfs", "drop_all_capabilities",
                    "no_new_privileges"):
            if key in data:
                setattr(config, key, _as_bool(data[key]))
        if isinstance(data.get("extra_run_args"), list):
            config.extra_run_args = [str(a) for a in data["extra_run_args"]]
        return config


@dataclass(slots=True)
class SandboxVmConfig:
    """VM tier settings — dedicated VM with snapshot/rollback (strict path).

    ``provider`` selects the platform strategy: "auto" (Kata via container runtime on
    Linux, Hyper-V on Windows, Lima on macOS), or an explicit "kata"/"hyperv"/"lima".
    """

    provider: str = "auto"
    # Base image / VM template the strategy boots or runs.
    base_image: str = "docker.io/library/debian:stable-slim"
    # Long-lived guest VM name (Hyper-V/Lima) the strategy manages.
    vm_name: str = "polaris-sandbox"
    # Snapshot restored on reset() (Hyper-V checkpoint / Lima).
    snapshot_name: str = "polaris-base"
    # SSH target for command exec inside a managed VM (host or user@host).
    guest_host: str = ""
    # Restore the base snapshot before every run/task (strict isolation between tasks).
    reset_each_task: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SandboxVmConfig":
        config = cls()
        if not data:
            return config
        for key in ("provider", "base_image", "vm_name", "snapshot_name", "guest_host"):
            if key in data:
                setattr(config, key, str(data[key]))
        if "reset_each_task" in data:
            config.reset_each_task = _as_bool(data["reset_each_task"])
        return config


@dataclass(slots=True)
class SandboxConfig:
    """Top-level sandbox settings.

    ``enabled`` is the master switch; ``backend`` picks the isolation tier
    ("auto" | "native" | "container" | "vm"). On an unsupported platform / with missing
    dependencies the manager degrades down the tier chain to a no-op *unless*
    ``fail_if_unavailable`` is set, which turns "can't sandbox" into a hard startup error
    (for managed deployments that must not run commands unsandboxed).
    """

    enabled: bool = False
    # Isolation tier: "auto" prefers container → native → noop (never auto-selects the
    # heavyweight vm tier); an explicit tier degrades to the next weaker available one.
    backend: str = "auto"
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
    container: SandboxContainerConfig = field(default_factory=SandboxContainerConfig)
    vm: SandboxVmConfig = field(default_factory=SandboxVmConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SandboxConfig":
        config = cls()
        if not data:
            return config
        if "enabled" in data:
            config.enabled = _as_bool(data["enabled"])
        if "backend" in data:
            config.backend = _normalize_backend(data["backend"])
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
        config.container = SandboxContainerConfig.from_dict(_as_table(data.get("container")))
        config.vm = SandboxVmConfig.from_dict(_as_table(data.get("vm")))
        return config


# Recognised isolation tiers; an unknown value degrades to "auto" (parse-failure-degrade).
_VALID_BACKENDS = frozenset({"auto", "native", "container", "vm"})


def _normalize_backend(value: Any) -> str:
    text = str(value).strip().lower()
    return text if text in _VALID_BACKENDS else "auto"


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_table(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None
