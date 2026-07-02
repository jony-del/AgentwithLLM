"""Container tier: run each command inside an OCI container (podman/docker/nerdctl).

The launcher-prefix model generalises cleanly here — ``<rt> run --rm … img sh -c "cmd"``
*is* a launcher prefix — so this backend slots into the same :meth:`wrap` seam the native
backends use. On top of that it adds a **lifecycle** (:meth:`prepare`) that verifies the
runtime and image at startup (eager-loading invariant).

Hardening defaults follow the project's Linux research: default-deny network, read-only
rootfs, ``--cap-drop ALL``, ``--security-opt no-new-privileges``, optional cpu/mem/pid
limits, and *only* the workspace bind-mounted. The workspace is mounted at the **same
absolute path** so no in-command path translation is needed; on Windows the drive is
mapped to a WSL2 ``/mnt/x`` path.

Not exercised on the project's primary (Windows) environment without a runtime; unit-tested
by mocking ``shutil.which`` / ``subprocess`` and asserting the generated ``run`` argv.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from agent_core.sandbox.backends.base import (
    SandboxBackend,
    SandboxTier,
    expand_paths,
    to_argv,
)
from agent_core.sandbox.config import SandboxConfig

# Probe order for `runtime = "auto"`: rootless Podman preferred, Docker next, then nerdctl.
_RUNTIME_PREFERENCE = ("podman", "docker", "nerdctl")

# Single, non-stacked timeout for every runtime probe (image inspect / pull).
_PREPARE_TIMEOUT = 120


class ContainerUnavailable(RuntimeError):
    """Internal signal that prepare() could not ready the container backend."""


class ContainerBackend(SandboxBackend):
    name = "container"
    tier = SandboxTier.CONTAINER

    def __init__(self, config: SandboxConfig | None = None) -> None:
        # A copy of the container settings is enough to resolve the runtime; the full
        # config is still passed to wrap() by the manager.
        self._container = (config or SandboxConfig()).container
        self._runtime: str | None = _resolve_runtime(self._container.runtime)

    # -- capability ------------------------------------------------------------------

    @property
    def runtime(self) -> str | None:
        return self._runtime

    def missing_dependencies(self) -> list[str]:
        if self._runtime is not None:
            return []
        # Report the preferred runtime (or the explicitly requested one) as missing.
        requested = self._container.runtime
        return [requested] if requested != "auto" else [_RUNTIME_PREFERENCE[0]]

    def available(self) -> bool:
        return self._runtime is not None

    # -- lifecycle -------------------------------------------------------------------

    def prepare(self) -> None:
        """Verify the runtime is usable and the image is present (pull if configured).

        Degrades to :class:`ContainerUnavailable` (the manager decides fail-vs-fallback);
        never raises a raw subprocess error into the run.
        """
        if self._runtime is None:
            raise ContainerUnavailable("no container runtime on PATH")
        image = self._container.image
        if _image_exists(self._runtime, image):
            return
        if not self._container.auto_pull:
            raise ContainerUnavailable(
                f"image {image!r} not present (set [sandbox.container].auto_pull = true "
                f"or pre-pull it)"
            )
        if not _pull_image(self._runtime, image):
            raise ContainerUnavailable(f"failed to pull image {image!r}")

    # -- the wrap seam ---------------------------------------------------------------

    def wrap(
        self, spec, shell: bool, *, config: SandboxConfig, workspace: Path
    ) -> tuple[object, bool]:
        if self._runtime is None:  # defensive: manager only wraps when available
            return spec, shell
        argv = to_argv(spec, shell)
        cfg = config.container
        ws_host = str(workspace)
        ws_guest = _map_workspace_path(workspace, cfg.windows_isolation)

        prefix: list[str] = [self._runtime, "run", "--rm", "--init"]
        prefix += ["-v", f"{ws_host}:{ws_guest}", "-w", ws_guest]

        # Extra writable mounts (same-path where possible).
        for path in expand_paths(config.filesystem.allow_write, workspace):
            guest = _map_workspace_path(Path(path), cfg.windows_isolation)
            prefix += ["-v", f"{path}:{guest}"]

        # Network: default-deny; opt back in with a bridge when domains/binding requested.
        if config.network.allowed_domains or config.network.allow_local_binding:
            prefix += ["--network", "bridge"]
        else:
            prefix += ["--network", "none"]

        if cfg.read_only_rootfs:
            prefix += ["--read-only", "--tmpfs", "/tmp"]
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
        if _wants_hyperv(cfg):
            prefix += ["--isolation", "hyperv"]
        prefix += list(cfg.extra_run_args)

        prefix += [cfg.image]
        return prefix + argv, False


# -- helpers -------------------------------------------------------------------------


def _resolve_runtime(requested: str) -> str | None:
    if requested and requested != "auto":
        return requested if shutil.which(requested) else None
    for candidate in _RUNTIME_PREFERENCE:
        if shutil.which(candidate):
            return candidate
    return None


def _image_exists(runtime: str, image: str) -> bool:
    return _run_probe([runtime, "image", "inspect", image])


def _pull_image(runtime: str, image: str) -> bool:
    return _run_probe([runtime, "pull", image])


def _run_probe(argv: list[str]) -> bool:
    """Run a runtime probe with a single bounded timeout; True on exit 0, else False.

    Any failure (non-zero, timeout, OS error) degrades to False — the caller turns that
    into a graceful unavailability, never a raw exception into the run.
    """
    try:
        proc = subprocess.run(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_PREPARE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return False
    except OSError:
        return False
    return proc.returncode == 0


def _wants_hyperv(cfg) -> bool:
    return sys.platform == "win32" and cfg.windows_isolation == "hyperv"


def _map_workspace_path(path: Path, windows_isolation: str) -> str:
    """Host path → the path seen inside the container.

    On Linux/macOS the mount uses the same absolute path. On Windows with WSL2 Linux
    containers a drive path ``E:\\proj`` maps to ``/mnt/e/proj``; with Hyper-V Windows
    containers the Windows path is used as-is.
    """
    if sys.platform != "win32" or windows_isolation == "hyperv":
        return str(path)
    drive, rest = _split_windows_drive(str(path))
    if drive is None:
        return str(path).replace("\\", "/")
    tail = rest.replace("\\", "/").lstrip("/")
    return f"/mnt/{drive.lower()}/{tail}"


def _split_windows_drive(path: str) -> tuple[str | None, str]:
    if len(path) >= 2 and path[1] == ":" and path[0].isalpha():
        return path[0], path[2:]
    return None, path
