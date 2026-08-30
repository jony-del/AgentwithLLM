"""Hardened OCI Linux-guest backend (Podman/Docker/nerdctl).

Preparation is deliberately stronger than finding a client binary: one runtime must
pass daemon, immutable-image, guest-protocol, bind-mount, capability, and network-deny
probes.  Windows clients are never asked to execute a host ``.exe`` in the container;
all command argv comes from :mod:`agent_core.sandbox.invocation`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path, PureWindowsPath

from agent_core.sandbox.backends.base import SandboxBackend, SandboxTier, expand_paths, to_argv
from agent_core.sandbox.config import SandboxConfig, validate_image_reference
from agent_core.sandbox.invocation import (
    GuestCapabilityUnavailable,
    GuestRuntimeManifest,
    REQUIRED_GUEST_TOOLS,
)

_RUNTIME_PREFERENCE = ("podman", "docker", "nerdctl")
_PREPARE_TIMEOUT = 120
_PROBE = "/opt/polaris/bin/sandbox-probe"
_FALLBACK_CONTAINER_USER = "65532:65532"
_TMPFS = "/tmp:rw,nosuid,nodev,size=256m"


class ContainerUnavailable(RuntimeError):
    """The complete OCI preparation contract could not be satisfied."""


class ContainerBackend(SandboxBackend):
    name = "container"
    tier = SandboxTier.CONTAINER
    uses_guest = True

    def __init__(
        self, config: SandboxConfig | None = None, *, workspace: str | Path | None = None
    ) -> None:
        self._container = (config or SandboxConfig()).container
        self._config = config or SandboxConfig()
        self._workspace = Path(workspace or Path.cwd()).resolve()
        self._requested_runtime = self._container.runtime
        self._runtime: str | None = _resolve_runtime(self._container.runtime)
        self._manifest: GuestRuntimeManifest | None = None
        self._failure_reason = ""

    @property
    def runtime(self) -> str | None:
        return self._runtime

    @property
    def guest_manifest(self) -> GuestRuntimeManifest | None:
        return self._manifest

    @property
    def failure_reason(self) -> str:
        return self._failure_reason

    def missing_dependencies(self) -> list[str]:
        if self._runtime is not None:
            return []
        requested = self._container.runtime
        return [requested] if requested != "auto" else [_RUNTIME_PREFERENCE[0]]

    def available(self) -> bool:
        return self._runtime is not None

    def translate_path(self, path: Path) -> str:
        return _map_workspace_path(path, self._container.windows_isolation)

    def prepare(self) -> None:
        """Select a runtime only after every Linux-guest safety probe succeeds."""

        cfg = self._container
        try:
            validate_image_reference(cfg.image)
            _validate_container_policy(self._config)
            guest_workspace = self.translate_path(self._workspace)
        except (ValueError, GuestCapabilityUnavailable) as exc:
            self._failure_reason = str(exc)
            raise ContainerUnavailable(str(exc)) from exc

        candidates = _runtime_candidates(self._requested_runtime)
        if not candidates:
            reason = "no container runtime on PATH"
            if sys.platform == "win32":
                reason += "; install Podman and initialize it with 'podman machine init'"
            self._failure_reason = reason
            raise ContainerUnavailable(reason)

        failures: list[str] = []
        for runtime in candidates:
            if not _runtime_ready(runtime):
                hint = ""
                if runtime == "podman" and sys.platform == "win32":
                    hint = " (Podman Machine is stopped; run 'podman machine start')"
                failures.append(f"{runtime}: runtime unavailable{hint}")
                continue
            if not _image_exists(runtime, cfg.image):
                if not cfg.auto_pull or not _pull_image(runtime, cfg.image):
                    failures.append(f"{runtime}: immutable image is not present")
                    continue
            try:
                manifest = _probe_manifest(runtime, cfg)
            except GuestCapabilityUnavailable as exc:
                failures.append(f"{runtime}: {exc}")
                continue
            missing = REQUIRED_GUEST_TOOLS - manifest.capabilities
            if missing:
                failures.append(
                    f"{runtime}: guest image is missing required tools: {', '.join(sorted(missing))}"
                )
                continue
            if not _probe_mount(runtime, cfg, self._workspace, guest_workspace):
                failures.append(f"{runtime}: workspace bind-mount probe failed")
                continue
            if not _probe_security(runtime, cfg):
                failures.append(f"{runtime}: read-only/non-root security canary failed")
                continue
            if not _probe_network_denied(runtime, cfg):
                failures.append(f"{runtime}: network-deny canary failed")
                continue
            self._runtime = runtime
            self._manifest = manifest
            self._failure_reason = ""
            return

        self._runtime = None
        self._manifest = None
        details = "; ".join(failures) or "no runtime passed the sandbox probes"
        self._failure_reason = details
        raise ContainerUnavailable(details)

    def wrap(
        self, spec, shell: bool, *, config: SandboxConfig, workspace: Path
    ) -> tuple[object, bool]:
        if self._runtime is None or self._manifest is None:
            raise ContainerUnavailable("container backend has not passed preparation probes")
        _validate_container_policy(config)
        argv = to_argv(spec, shell)
        cfg = config.container
        ws_guest = _map_workspace_path(workspace, cfg.windows_isolation)
        ws_host = _host_mount_path(workspace, cfg.windows_isolation)

        prefix = _hardened_run_prefix(self._runtime, cfg)
        workspace_read_only = str(workspace) in expand_paths(
            config.filesystem.deny_write, workspace
        )
        suffix = ":ro" if workspace_read_only else ""
        prefix += ["-v", f"{ws_host}:{ws_guest}{suffix}", "-w", ws_guest]

        for raw in expand_paths(config.filesystem.allow_write, workspace):
            path = Path(raw).resolve()
            if path == workspace.resolve():
                continue
            prefix += [
                "-v",
                f"{_host_mount_path(path, cfg.windows_isolation)}:"
                f"{_map_workspace_path(path, cfg.windows_isolation)}",
            ]

        writable = set(expand_paths(config.filesystem.allow_write, workspace))
        for raw in expand_paths(config.filesystem.allow_read, workspace):
            if raw in writable:
                continue
            path = Path(raw).resolve()
            if path == workspace.resolve():
                continue
            prefix += [
                "-v",
                f"{_host_mount_path(path, cfg.windows_isolation)}:"
                f"{_map_workspace_path(path, cfg.windows_isolation)}:ro",
            ]

        prefix += [cfg.image]
        return prefix + argv, False


def _validate_container_policy(config: SandboxConfig) -> None:
    cfg = config.container
    if cfg.windows_isolation.casefold() != "wsl2":
        raise GuestCapabilityUnavailable(
            "guest_capability_unavailable: container backend only supports WSL2/Linux; "
            "use backend='vm' for Hyper-V"
        )
    if config.network.allowed_domains or config.network.allow_local_binding:
        raise GuestCapabilityUnavailable(
            "guest_capability_unavailable: container sandbox supports network=deny only"
        )
    if not (cfg.read_only_rootfs and cfg.drop_all_capabilities and cfg.no_new_privileges):
        raise GuestCapabilityUnavailable(
            "guest_capability_unavailable: read-only root, cap-drop ALL, and "
            "no-new-privileges are mandatory"
        )


def _resolve_runtime(requested: str) -> str | None:
    candidates = _runtime_candidates(requested)
    return candidates[0] if candidates else None


def _runtime_candidates(requested: str) -> list[str]:
    if requested and requested != "auto":
        return [requested] if shutil.which(requested) else []
    return [candidate for candidate in _RUNTIME_PREFERENCE if shutil.which(candidate)]


def _runtime_ready(runtime: str) -> bool:
    return _run_probe([runtime, "info"])


def _image_exists(runtime: str, image: str) -> bool:
    return _run_probe([runtime, "image", "inspect", image])


def _pull_image(runtime: str, image: str) -> bool:
    return _run_probe([runtime, "pull", image])


def _probe_manifest(runtime: str, cfg) -> GuestRuntimeManifest:
    proc = _run_capture([*_probe_run_prefix(runtime, cfg), cfg.image, _PROBE, "manifest"])
    if proc is None or proc.returncode:
        raise GuestCapabilityUnavailable(
            "guest_capability_unavailable: image protocol probe failed"
        )
    return GuestRuntimeManifest.from_json(proc.stdout or b"")


def _probe_mount(runtime: str, cfg, workspace: Path, guest_workspace: str) -> bool:
    host = _host_mount_path(workspace, cfg.windows_isolation)
    argv = [
        *_probe_run_prefix(runtime, cfg),
        "-v", f"{host}:{guest_workspace}", "-w", guest_workspace,
        cfg.image, _PROBE, "mount", guest_workspace,
    ]
    return _run_probe(argv)


def _probe_network_denied(runtime: str, cfg) -> bool:
    return _run_probe(
        [*_probe_run_prefix(runtime, cfg), cfg.image, _PROBE, "network-denied"]
    )


def _probe_security(runtime: str, cfg) -> bool:
    return _run_probe([*_probe_run_prefix(runtime, cfg), cfg.image, _PROBE, "security"])


def _probe_run_prefix(runtime: str, cfg) -> list[str]:
    return _hardened_run_prefix(runtime, cfg)


def _hardened_run_prefix(runtime: str, cfg) -> list[str]:
    user, keep_id = _container_identity(runtime)
    prefix = [
        runtime, "run", "--rm", "--init", "--network", "none",
        "--user", user,
        "--env", "HOME=/tmp/polaris-home",
        "--env", "XDG_CACHE_HOME=/tmp/polaris-cache",
        "--env", "TMPDIR=/tmp",
    ]
    if keep_id:
        prefix += ["--userns", "keep-id"]
    if cfg.read_only_rootfs:
        prefix += ["--read-only", "--tmpfs", _TMPFS]
    if cfg.drop_all_capabilities:
        prefix += ["--cap-drop", "ALL"]
    if cfg.no_new_privileges:
        prefix += ["--security-opt", "no-new-privileges"]
    if cfg.memory:
        prefix += ["--memory", cfg.memory]
    if cfg.cpus:
        prefix += ["--cpus", cfg.cpus]
    if cfg.pids_limit:
        prefix += ["--pids-limit", cfg.pids_limit]
    if cfg.oci_runtime:
        prefix += ["--runtime", cfg.oci_runtime]
    return prefix


def _container_identity(runtime: str) -> tuple[str, bool]:
    """Choose a non-root uid that can write the host workspace bind mount."""

    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if sys.platform != "win32" and getuid is not None and getgid is not None:
        uid, gid = int(getuid()), int(getgid())
        if uid != 0:
            is_podman = Path(runtime).name.casefold() in {"podman", "podman.exe"}
            return f"{uid}:{gid}", is_podman
    return _FALLBACK_CONTAINER_USER, False


def _run_capture(argv: list[str]) -> subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_PREPARE_TIMEOUT,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None


def _run_probe(argv: list[str]) -> bool:
    proc = _run_capture(argv)
    return proc is not None and proc.returncode == 0


def _host_mount_path(path: Path, windows_isolation: str) -> str:
    # Podman Machine and the accepted Docker/nerdctl configurations consume WSL-visible
    # paths. The preparation mount probe proves this spelling works before selection.
    if sys.platform == "win32":
        return _map_workspace_path(path, windows_isolation)
    return str(path)


def _map_workspace_path(path: Path, windows_isolation: str = "wsl2") -> str:
    """Map a local drive path to the Linux guest and reject UNC/network locations."""

    raw = str(path)
    if windows_isolation.casefold() != "wsl2":
        raise GuestCapabilityUnavailable(
            "guest_capability_unavailable: Windows container isolation is unsupported"
        )
    if sys.platform != "win32":
        return str(path)
    normalized = raw.replace("\\", "/")
    if normalized.startswith("//"):
        raise GuestCapabilityUnavailable(
            "guest_capability_unavailable: UNC/network workspaces are unsupported"
        )
    drive = PureWindowsPath(raw).drive.rstrip(":")
    if len(drive) != 1 or not drive.isalpha():
        raise GuestCapabilityUnavailable(
            f"guest_capability_unavailable: Windows workspace must use a local drive: {raw!r}"
        )
    tail = normalized[2:].lstrip("/")
    return f"/mnt/{drive.casefold()}/{tail}"


def _split_windows_drive(path: str) -> tuple[str | None, str]:
    """Compatibility helper retained for callers/tests of the old path seam."""

    if len(path) >= 2 and path[1] == ":" and path[0].isalpha():
        return path[0], path[2:]
    return None, path
