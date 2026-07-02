"""macOS backend: wrap commands with ``sandbox-exec`` + a generated SBPL profile.

Mirrors Open-ClaudeCode's macOS path: a Seatbelt profile that allows everything by
default, then **denies all file writes** except the workspace and configured
``allow_write`` paths, denies reads of ``deny_read`` regions, and denies network unless
opted in. ``sandbox-exec`` ships with macOS, so this backend has no external deps.

Not exercised on the project's primary (Windows) environment; unit-tested by asserting
the generated ``sandbox-exec -p <profile>`` argv and profile text.
"""

from __future__ import annotations

from pathlib import Path

from agent_core.sandbox.backends.base import SandboxBackend, expand_paths, to_argv
from agent_core.sandbox.config import SandboxConfig


class SeatbeltBackend(SandboxBackend):
    name = "seatbelt"
    required_binaries = ("sandbox-exec",)

    def wrap(
        self, spec, shell: bool, *, config: SandboxConfig, workspace: Path
    ) -> tuple[object, bool]:
        argv = to_argv(spec, shell)
        profile = self._build_profile(config, workspace)
        return ["sandbox-exec", "-p", profile, *argv], False

    def _build_profile(self, config: SandboxConfig, workspace: Path) -> str:
        lines = ["(version 1)", "(allow default)"]

        # Default-deny writes, then re-allow the workspace + configured paths.
        lines.append("(deny file-write*)")
        writable = [str(workspace)] + expand_paths(config.filesystem.allow_write, workspace)
        for path in writable:
            lines.append(f'(allow file-write* (subpath "{_escape(path)}"))')
        # Explicit write denies win (listed after the allows).
        for path in expand_paths(config.filesystem.deny_write, workspace):
            lines.append(f'(deny file-write* (subpath "{_escape(path)}"))')

        # Read denies.
        for path in expand_paths(config.filesystem.deny_read, workspace):
            lines.append(f'(deny file-read* (subpath "{_escape(path)}"))')
        for path in expand_paths(config.filesystem.allow_read, workspace):
            lines.append(f'(allow file-read* (subpath "{_escape(path)}"))')

        # Network: default-deny unless an allowlist / local binding is configured.
        if not config.network.allowed_domains and not config.network.allow_local_binding:
            lines.append("(deny network*)")

        return "\n".join(lines)


def _escape(path: str) -> str:
    return path.replace("\\", "\\\\").replace('"', '\\"')
