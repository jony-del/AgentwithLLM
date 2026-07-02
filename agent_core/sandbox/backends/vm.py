"""VM tier: strongest isolation — a dedicated VM with snapshot/rollback (strict path).

``VmBackend`` dispatches to a platform strategy, each implementing the full lifecycle
(:meth:`prepare` boots/ensures the VM + a base snapshot, :meth:`reset` restores that
snapshot per task, :meth:`teardown` stops it) plus the :meth:`wrap` launcher:

- **Linux — Kata** (``KataStrategy``): reuses the container launcher with an OCI runtime
  override (``--runtime=<kata>``); each container is its own microVM, so ``reset`` is a
  no-op. This is the pragmatic, unit-testable "real VM isolation".
- **Windows — Hyper-V** (``HyperVStrategy``): a long-lived Linux guest managed via
  PowerShell ``Hyper-V`` cmdlets; ``prepare`` ensures a base ``Checkpoint-VM``, ``reset``
  runs ``Restore-VMSnapshot``, ``wrap`` exec's the command over SSH into the guest.
- **macOS — Lima** (``LimaStrategy``): a ``limactl`` Linux VM; ``wrap`` runs via
  ``limactl shell``.

VM strategies are heavily host-dependent; like the native backends they are **not run on
the project's primary environment** and are unit-tested by mocking ``shutil.which`` /
``subprocess`` and asserting the generated argv + lifecycle call order.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from agent_core.sandbox.backends.base import SandboxBackend, SandboxTier, to_argv
from agent_core.sandbox.backends.container import ContainerBackend, _map_workspace_path
from agent_core.sandbox.config import SandboxConfig

# Single, non-stacked timeout for every VM lifecycle call (checkpoint/restore/boot).
_VM_TIMEOUT = 600


class VmUnavailable(RuntimeError):
    """Internal signal that a VM strategy could not ready itself."""


class VmStrategy:
    """Platform strategy interface for the VM tier."""

    name = "vm"
    required_binaries: tuple[str, ...] = ()

    def missing_dependencies(self) -> list[str]:
        return [b for b in self.required_binaries if shutil.which(b) is None]

    def available(self) -> bool:
        return not self.missing_dependencies()

    def prepare(self, config: SandboxConfig) -> None:  # noqa: ARG002
        ...

    def reset(self, config: SandboxConfig) -> None:  # noqa: ARG002
        ...

    def teardown(self, config: SandboxConfig) -> None:  # noqa: ARG002
        ...

    def wrap(self, argv: list[str], *, config: SandboxConfig, workspace: Path) -> list[str]:
        raise NotImplementedError


class VmBackend(SandboxBackend):
    name = "vm"
    tier = SandboxTier.VM

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self._config = config or SandboxConfig()
        self._strategy = _select_vm_strategy(self._config)

    @property
    def strategy_name(self) -> str:
        return self._strategy.name

    def missing_dependencies(self) -> list[str]:
        return self._strategy.missing_dependencies()

    def available(self) -> bool:
        return self._strategy.available()

    def prepare(self) -> None:
        self._strategy.prepare(self._config)

    def reset(self) -> None:
        self._strategy.reset(self._config)

    def teardown(self) -> None:
        self._strategy.teardown(self._config)

    def wrap(
        self, spec, shell: bool, *, config: SandboxConfig, workspace: Path
    ) -> tuple[object, bool]:
        argv = to_argv(spec, shell)
        wrapped = self._strategy.wrap(argv, config=config, workspace=workspace)
        return wrapped, False


# -- strategies ----------------------------------------------------------------------


class KataStrategy(VmStrategy):
    """Linux: microVM-per-container via a Kata OCI runtime, reusing the container launcher."""

    name = "kata"

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config
        self._kata_runtime = config.vm.provider if config.vm.provider not in {"auto", "kata"} else "kata-runtime"
        self._container = ContainerBackend(config)

    def missing_dependencies(self) -> list[str]:
        missing = self._container.missing_dependencies()
        if shutil.which(self._kata_runtime) is None:
            missing = [*missing, self._kata_runtime]
        return missing

    def prepare(self, config: SandboxConfig) -> None:
        self._container.prepare()

    def wrap(self, argv: list[str], *, config: SandboxConfig, workspace: Path) -> list[str]:
        # Force the container to run under the Kata OCI runtime for microVM isolation.
        forced = _with_oci_runtime(config, self._kata_runtime)
        wrapped, _ = self._container.wrap(argv, False, config=forced, workspace=workspace)
        return list(wrapped)  # type: ignore[arg-type]


class HyperVStrategy(VmStrategy):
    """Windows: a long-lived Linux guest under Hyper-V with checkpoint-based rollback."""

    name = "hyperv"
    required_binaries = ("powershell", "ssh")

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config

    def prepare(self, config: SandboxConfig) -> None:
        vm = config.vm
        if not vm.guest_host:
            raise VmUnavailable("[sandbox.vm].guest_host is required for the Hyper-V provider")
        # Ensure a base checkpoint exists to roll back to (idempotent create).
        _powershell(
            f"if (-not (Get-VMSnapshot -VMName '{vm.vm_name}' "
            f"-Name '{vm.snapshot_name}' -ErrorAction SilentlyContinue)) "
            f"{{ Checkpoint-VM -Name '{vm.vm_name}' -SnapshotName '{vm.snapshot_name}' }}"
        )

    def reset(self, config: SandboxConfig) -> None:
        vm = config.vm
        _powershell(f"Restore-VMSnapshot -VMName '{vm.vm_name}' -Name '{vm.snapshot_name}' -Confirm:$false")

    def wrap(self, argv: list[str], *, config: SandboxConfig, workspace: Path) -> list[str]:
        vm = config.vm
        # Guest is a Linux VM; run inside the (shared) workspace over SSH.
        guest_ws = _map_workspace_path(workspace, "wsl2")
        remote = f"cd {shlex.quote(guest_ws)} && {shlex.join(argv)}"
        return ["ssh", vm.guest_host, remote]


class LimaStrategy(VmStrategy):
    """macOS: a limactl-managed Linux VM."""

    name = "lima"
    required_binaries = ("limactl",)

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config

    def prepare(self, config: SandboxConfig) -> None:
        vm = config.vm
        # Start the VM if it is not already running (idempotent, bounded by the VM timeout).
        if not _run_vm_command(["limactl", "start", "--tty=false", vm.vm_name]):
            raise VmUnavailable(f"could not start lima VM {vm.vm_name!r}")

    def wrap(self, argv: list[str], *, config: SandboxConfig, workspace: Path) -> list[str]:
        vm = config.vm
        remote = f"cd {shlex.quote(str(workspace))} && {shlex.join(argv)}"
        return ["limactl", "shell", vm.vm_name, "sh", "-c", remote]


def _select_vm_strategy(config: SandboxConfig) -> VmStrategy:
    provider = config.vm.provider
    if provider == "hyperv" or (provider == "auto" and sys.platform == "win32"):
        return HyperVStrategy(config)
    if provider == "lima" or (provider == "auto" and sys.platform == "darwin"):
        return LimaStrategy(config)
    # Default / "kata" / Linux-auto → Kata microVMs.
    return KataStrategy(config)


# -- helpers -------------------------------------------------------------------------


def _with_oci_runtime(config: SandboxConfig, runtime: str) -> SandboxConfig:
    """Return a shallow copy of config whose container.oci_runtime is forced to *runtime*."""
    from dataclasses import replace

    container = replace(config.container, oci_runtime=runtime)
    return replace(config, container=container)


def _powershell(script: str) -> bool:
    return _run_vm_command(["powershell", "-NoProfile", "-NonInteractive", "-Command", script])


def _run_vm_command(argv: list[str]) -> bool:
    """Run a VM lifecycle command with a single bounded timeout; degrade to False."""
    try:
        proc = subprocess.run(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_VM_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return False
    except OSError:
        return False
    return proc.returncode == 0
