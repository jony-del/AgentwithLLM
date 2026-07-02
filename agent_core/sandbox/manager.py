"""SandboxManager — decides *whether* to sandbox and *wraps* the command if so.

This is the adapter layer between project config and the OS backends, adapted from
Open-ClaudeCode's ``sandbox-adapter.ts`` + ``shouldUseSandbox.ts``. It answers three
questions the rest of the system asks:

- :meth:`is_enabled` — is sandboxing actually active here (enabled + supported + deps)?
- :meth:`should_sandbox` — should *this* command run sandboxed (not excluded)?
- :meth:`wrap` — return the (possibly) OS-wrapped command spec.

On an unsupported platform (Windows) or with missing dependencies it **degrades to a
no-op** — commands run exactly as before — unless ``fail_if_unavailable`` is set, in
which case construction raises :class:`SandboxUnavailableError` (a hard gate for
deployments that must never run commands unsandboxed).
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent_core.permission_rules import PermissionBehavior, _match_shell_command
from agent_core.sandbox.backends import (
    BubblewrapBackend,
    NoopBackend,
    SandboxBackend,
    SeatbeltBackend,
)
from agent_core.sandbox.config import SandboxConfig


class SandboxUnavailableError(RuntimeError):
    """Raised at construction when ``enabled + fail_if_unavailable`` but can't sandbox."""


class SandboxManager:
    def __init__(self, config: SandboxConfig | None = None, workspace: str | Path | None = None) -> None:
        self.config = config or SandboxConfig()
        self.workspace = Path(workspace or Path.cwd()).resolve()
        self._backend = self._select_backend()
        # Fail-fast gate: an operator asked for sandboxing AND declared it mandatory,
        # but this environment can't provide it — refuse to start rather than silently
        # running commands unsandboxed. (Actionable-error invariant.)
        if self.config.enabled and self.config.fail_if_unavailable and not self.is_enabled():
            reason = self.unavailable_reason() or "sandbox is unavailable"
            raise SandboxUnavailableError(
                f"sandbox.enabled + fail_if_unavailable set, but {reason}"
            )

    # -- backend selection / capability ---------------------------------------------

    def _select_backend(self) -> SandboxBackend:
        if sys.platform == "darwin":
            return SeatbeltBackend()
        if sys.platform.startswith("linux"):
            return BubblewrapBackend()
        return NoopBackend()

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def is_supported_platform(self) -> bool:
        """True on a platform with a real OS backend (macOS / Linux incl. WSL2)."""
        return not isinstance(self._backend, NoopBackend)

    def is_enabled(self) -> bool:
        """True when sandboxing is switched on AND can actually run here."""
        return self.config.enabled and self.is_supported_platform() and self._backend.available()

    def should_sandbox(self, command: str | None) -> bool:
        """Whether *this* command should be wrapped: enabled and not excluded."""
        if not self.is_enabled():
            return False
        if command and self._is_excluded(command):
            return False
        return True

    def _is_excluded(self, command: str) -> bool:
        patterns = self.config.excluded_commands
        if not patterns:
            return False
        # Reuse the shell decomposition/matching: any sub-command hitting an excluded
        # pattern means the command runs outside the sandbox.
        return _match_shell_command(command, patterns, PermissionBehavior.DENY)

    # -- the wrap seam called by command tools ---------------------------------------

    def wrap(self, spec, shell: bool, *, command: str | None = None) -> tuple[object, bool]:
        """Return ``(spec, shell)`` wrapped for OS isolation, or unchanged if not sandboxing.

        ``command`` is the raw shell command line (for the exclusion check); pass ``None``
        for argv-based tools (e.g. the test runner) to sandbox purely on ``is_enabled``.
        """
        if command is not None:
            if not self.should_sandbox(command):
                return spec, shell
        elif not self.is_enabled():
            return spec, shell
        return self._backend.wrap(spec, shell, config=self.config, workspace=self.workspace)

    # -- diagnostics -----------------------------------------------------------------

    def unavailable_reason(self) -> str | None:
        """A user-facing reason when ``enabled`` is set but sandboxing can't run; else None."""
        if not self.config.enabled or self.is_enabled():
            return None
        if not self.is_supported_platform():
            return (
                f"{sys.platform} is not a supported sandbox platform "
                "(requires macOS or Linux/WSL2); commands run unsandboxed"
            )
        missing = self._backend.missing_dependencies()
        if missing:
            return (
                f"missing sandbox dependencies: {', '.join(missing)} "
                f"(install them, e.g. apt install {' '.join(missing)})"
            )
        return "sandbox is unavailable"


# A shared disabled manager used as the default for unbound command tools (before the
# agent rebinds the real one). Passthrough on every platform.
NOOP_SANDBOX = SandboxManager(SandboxConfig())
