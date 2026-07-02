"""Sandbox backend interface + shared helpers.

A backend turns a command spec into a *wrapped* spec that runs under OS isolation.
The manager picks one backend per platform and asks it to :meth:`wrap` each command.

A command spec follows the project's existing convention (see ``builtin._run_subprocess``):

- ``shell=True``  → ``spec`` is a POSIX shell string (run via ``/bin/sh -c``).
- ``shell=False`` → ``spec`` is an ``argv`` list.

Every OS backend prefixes a launcher (``bwrap``/``sandbox-exec``) and returns
``shell=False`` (the launcher itself is exec'd with an explicit argv), normalizing a
shell string into ``["/bin/sh", "-c", <string>]`` first.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from agent_core.sandbox.config import SandboxConfig

# A command spec is either a shell string (shell=True) or an argv list (shell=False).
Spec = "str | list[str]"


class SandboxBackend:
    """Base class for OS sandbox backends. Subclasses implement :meth:`wrap`."""

    name = "base"
    #: External executables this backend needs on PATH to function.
    required_binaries: tuple[str, ...] = ()

    def missing_dependencies(self) -> list[str]:
        """Names of required binaries that are not on PATH."""
        return [binary for binary in self.required_binaries if shutil.which(binary) is None]

    def available(self) -> bool:
        return not self.missing_dependencies()

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
