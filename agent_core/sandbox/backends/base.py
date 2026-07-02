"""Sandbox backend interface + shared helpers.

A backend turns a command spec into a *wrapped* spec that runs under OS isolation.
The manager picks one backend (by isolation **tier**, not by platform) and asks it to
:meth:`wrap` each command.

Backends are organised by *isolation tier* (:class:`SandboxTier`):

- ``NATIVE``   — OS primitives on the same kernel (``bwrap`` / ``sandbox-exec`` / no-op);
- ``CONTAINER``— an OCI runtime (``podman`` / ``docker`` / ``nerdctl``);
- ``VM``       — a dedicated VM with snapshot/rollback (Hyper-V / Firecracker·Kata / Lima).

A command spec follows the project's existing convention (see ``builtin._run_subprocess``):

- ``shell=True``  → ``spec`` is a POSIX shell string (run via ``/bin/sh -c``).
- ``shell=False`` → ``spec`` is an ``argv`` list.

Every real backend prefixes a launcher (``bwrap`` / ``sandbox-exec`` / ``podman run`` …)
and returns ``shell=False`` (the launcher itself is exec'd with an explicit argv),
normalizing a shell string into ``["/bin/sh", "-c", <string>]`` first.

Container/VM backends also implement a **lifecycle** (:meth:`prepare` / :meth:`teardown`
/ :meth:`reset`); native backends inherit the no-op defaults.
"""

from __future__ import annotations

import shutil
from enum import Enum
from pathlib import Path

from agent_core.sandbox.config import SandboxConfig

# A command spec is either a shell string (shell=True) or an argv list (shell=False).
Spec = "str | list[str]"


class SandboxTier(str, Enum):
    """Isolation strength, weakest → strongest. Used for backend selection/downgrade."""

    NATIVE = "native"
    CONTAINER = "container"
    VM = "vm"


class SandboxBackend:
    """Base class for sandbox backends. Subclasses implement :meth:`wrap`.

    Container/VM subclasses additionally override the lifecycle hooks; the no-op
    defaults here let native backends ignore them entirely.
    """

    name = "base"
    #: Isolation tier this backend belongs to.
    tier: SandboxTier = SandboxTier.NATIVE
    #: External executables this backend needs on PATH to function.
    required_binaries: tuple[str, ...] = ()

    def missing_dependencies(self) -> list[str]:
        """Names of required binaries that are not on PATH."""
        return [binary for binary in self.required_binaries if shutil.which(binary) is None]

    def available(self) -> bool:
        return not self.missing_dependencies()

    def isolates(self) -> bool:
        """True when this backend does real OS isolation (False only for the no-op path).

        Selection/``is_enabled`` use this instead of an ``isinstance`` check so that a
        higher-tier backend that internally degrades to no-op (e.g. ``NativeBackend`` on
        Windows) is correctly treated as *not* isolating.
        """
        return True

    # -- lifecycle (eager per the eager-loading invariant; no-op for native) ----------

    def prepare(self) -> None:
        """Startup: verify the runtime, pull/verify an image, boot a VM + base snapshot.

        MUST degrade (raise :class:`SandboxUnavailableError` only when the manager's
        ``fail_if_unavailable`` gate decides to; otherwise log + fall back), never sink a
        run. The default is a no-op (native backends have nothing to prepare).
        """

    def teardown(self) -> None:
        """Shutdown: stop/remove a container, power off a VM. Default no-op."""

    def reset(self) -> None:
        """Per-task: restore a VM to its base snapshot. Meaningful only for VM. No-op."""

    # -- the wrap seam called by command tools ---------------------------------------

    def wrap(
        self, spec, shell: bool, *, config: SandboxConfig, workspace: Path
    ) -> tuple[object, bool]:
        """Return ``(wrapped_spec, shell)`` running ``spec`` under this backend."""
        raise NotImplementedError


def to_argv(spec, shell: bool) -> list[str]:
    """Normalize a command spec into an explicit argv list for a launcher to prefix.

    A shell string becomes ``["/bin/sh", "-c", <string>]`` so the OS launcher can exec
    it directly; an argv list is returned as-is (copied).
    """
    if shell:
        return ["/bin/sh", "-c", str(spec)]
    if isinstance(spec, str):
        return ["/bin/sh", "-c", spec]
    return list(spec)


def expand_paths(paths: list[str], workspace: Path) -> list[str]:
    """Expand ``~`` and make workspace-relative paths absolute, for OS bind/deny rules."""
    resolved: list[str] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (workspace / path)
        resolved.append(str(path))
    return resolved
