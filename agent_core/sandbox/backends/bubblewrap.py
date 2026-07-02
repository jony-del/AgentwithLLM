"""Linux backend: wrap commands with ``bwrap`` (bubblewrap).

Strategy mirrors Open-ClaudeCode's Linux path: bind the whole filesystem **read-only**,
then re-bind the workspace (and any configured ``allow_write`` paths) read-write, mask
denied-read regions with an empty ``tmpfs``, and drop the network namespace unless the
config opts back in. This is a pragmatic subset — it does not reproduce the reference's
socat proxy or seccomp ``AF_UNIX`` filter (those need vendored BPF blobs) — but it
delivers the core write/read/network containment.

Not exercised on the project's primary (Windows) environment; unit-tested by mocking
``shutil.which`` and asserting the generated argv prefix.
"""

from __future__ import annotations

from pathlib import Path

from agent_core.sandbox.backends.base import SandboxBackend, expand_paths, to_argv
from agent_core.sandbox.config import SandboxConfig


class BubblewrapBackend(SandboxBackend):
    name = "bubblewrap"
    required_binaries = ("bwrap",)

    def wrap(
        self, spec, shell: bool, *, config: SandboxConfig, workspace: Path
    ) -> tuple[object, bool]:
        argv = to_argv(spec, shell)
        prefix: list[str] = ["bwrap", "--die-with-parent", "--ro-bind", "/", "/"]

        # Ephemeral, writable temp + proc/dev so most tools still work.
        prefix += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]

        # Workspace is always writable; then any explicitly allowed write paths.
        writable = [str(workspace)] + expand_paths(config.filesystem.allow_write, workspace)
        for path in _dedupe(writable):
            prefix += ["--bind", path, path]

        # Deny writes: re-bind read-only (overrides a broader writable bind above).
        for path in _dedupe(expand_paths(config.filesystem.deny_write, workspace)):
            prefix += ["--ro-bind", path, path]

        # Deny reads: mask with an empty tmpfs so the contents are invisible.
        for path in _dedupe(expand_paths(config.filesystem.deny_read, workspace)):
            prefix += ["--tmpfs", path]

        # Network: default-deny (unshare). Any allowlist/local-binding needs a proxy we
        # don't ship, so opting into network today means "share the host network".
        if not config.network.allowed_domains and not config.network.allow_local_binding:
            prefix += ["--unshare-net"]

        return prefix + argv, False


def _dedupe(paths: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for path in paths:
        seen.setdefault(path, None)
    return list(seen)
