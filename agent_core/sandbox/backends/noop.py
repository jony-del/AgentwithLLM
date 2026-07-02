"""No-op backend — the graceful-degradation path (Windows / unsupported / disabled).

Returns the command spec untouched, so the command runs exactly as it does today.
This is what keeps the sandbox architecture *present* on Windows (the project's primary
environment) while honestly doing no OS isolation there, mirroring the reference's
"disable sandboxing on Windows, fall back to permission rules" behavior.
"""

from __future__ import annotations

from pathlib import Path

from agent_core.sandbox.backends.base import SandboxBackend
from agent_core.sandbox.config import SandboxConfig


class NoopBackend(SandboxBackend):
    name = "noop"

    def available(self) -> bool:
        return True

    def isolates(self) -> bool:
        return False

    def wrap(
        self, spec, shell: bool, *, config: SandboxConfig, workspace: Path
    ) -> tuple[object, bool]:
        return spec, shell
