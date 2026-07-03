"""SandboxManager — selects a backend by isolation *tier*, wraps commands, owns lifecycle.

This is the adapter layer between project config and the pluggable backends, adapted from
Open-ClaudeCode's ``sandbox-adapter.ts`` + ``shouldUseSandbox.ts``. It answers the
questions the rest of the system asks:

- :meth:`is_enabled` — is sandboxing actually active here (enabled + a real backend)?
- :meth:`should_sandbox` — should *this* command run sandboxed (not excluded)?
- :meth:`wrap` — return the (possibly) wrapped command spec.
- :meth:`prepare` / :meth:`reset` / :meth:`teardown` — backend lifecycle (eager per the
  eager-loading invariant; ``reset`` runs per task for the VM tier).

Backend selection is by **tier** (``config.backend``): an explicit ``native``/``container``
/``vm`` starts at that tier and **degrades to the next weaker available** one; ``auto``
prefers ``container → native → noop`` (never auto-selecting the heavyweight VM tier). On
an unsupported environment it degrades to a no-op — commands run exactly as before —
unless ``fail_if_unavailable`` is set, which raises :class:`SandboxUnavailableError` (a
hard gate for deployments that must never run commands unsandboxed).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from agent_core.permission_rules import PermissionBehavior, _match_shell_command
from agent_core.sandbox.backends import (
    ContainerBackend,
    NativeBackend,
    NoopBackend,
    SandboxBackend,
    SandboxTier,
    VmBackend,
)
from agent_core.sandbox.backends.container import ContainerUnavailable
from agent_core.sandbox.backends.vm import VmUnavailable
from agent_core.sandbox.config import SandboxConfig

logger = logging.getLogger(__name__)

# Tier order, weakest → strongest. Downgrade walks this list *downward* from the request.
_TIER_ORDER = (SandboxTier.NATIVE, SandboxTier.CONTAINER, SandboxTier.VM)


class SandboxUnavailableError(RuntimeError):
    """Raised at construction when ``enabled + fail_if_unavailable`` but can't sandbox."""


class SandboxRequiredError(SandboxUnavailableError):
    """An unattended permission mode (auto/dontask/bypass) requires a working sandbox.

    Decision D3: modes that execute commands without per-call confirmation must not run
    with no isolation at all. Raised at agent construction; the interactive path offers
    a "continue unsandboxed?" prompt instead, and
    ``sandbox.allow_unattended_unsandboxed`` / ``AGENT_SANDBOX_ALLOW_UNATTENDED`` is the
    explicit, audited opt-out.
    """


class SandboxManager:
    def __init__(self, config: SandboxConfig | None = None, workspace: str | Path | None = None) -> None:
        self.config = config or SandboxConfig()
        self.workspace = Path(workspace or Path.cwd()).resolve()
        # (backend name, missing deps) for each candidate tried — powers a still-actionable
        # unavailable_reason() after a degrade-to-noop.
        self._selection_diagnostics: list[tuple[str, list[str]]] = []
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
        """Pick the strongest *available* backend at-or-below the requested tier.

        Disabled config short-circuits to a no-op. For ``auto`` the candidate chain is
        ``container → native → noop`` (VM is opt-in only); for an explicit tier it is that
        tier then every weaker one. The first candidate whose backend reports
        :meth:`available` wins; if none do, a :class:`NoopBackend` (passthrough).
        """
        if not self.config.enabled:
            return NoopBackend()
        for factory in self._candidate_backends():
            backend = factory()
            self._selection_diagnostics.append((backend.name, backend.missing_dependencies()))
            # A candidate must both have its deps AND actually isolate (native-on-Windows
            # is "available" but degrades to no-op, so it is skipped here).
            if backend.available() and backend.isolates():
                return backend
        return NoopBackend()

    def _candidate_backends(self):
        """Ordered backend factories to try, strongest requested tier first."""
        requested = self.config.backend
        builders = {
            SandboxTier.NATIVE: NativeBackend,
            SandboxTier.CONTAINER: lambda: ContainerBackend(self.config),
            SandboxTier.VM: lambda: VmBackend(self.config),
        }
        if requested == "auto":
            # Container preferred; native fallback. VM is never auto-selected (too heavy).
            tiers = [SandboxTier.CONTAINER, SandboxTier.NATIVE]
        else:
            start = SandboxTier(requested)
            # Start at the requested tier, then walk *down* to weaker tiers.
            idx = _TIER_ORDER.index(start)
            tiers = list(reversed(_TIER_ORDER[: idx + 1]))
        return [builders[tier] for tier in tiers]

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def backend_tier(self) -> SandboxTier:
        return self._backend.tier

    def is_supported_platform(self) -> bool:
        """True when the selected backend does real isolation (i.e. is not the no-op)."""
        return self._backend.isolates()

    def is_enabled(self) -> bool:
        """True when sandboxing is switched on AND a real backend is active here."""
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

    # -- lifecycle -------------------------------------------------------------------

    def prepare(self) -> None:
        """Ready the active backend (verify runtime, pull image, boot VM + base snapshot).

        Degrades gracefully: a backend that cannot ready itself becomes a no-op passthrough
        (so the run continues unsandboxed) unless ``fail_if_unavailable`` is set, in which
        case the failure is escalated to :class:`SandboxUnavailableError`.
        """
        if not self.is_enabled():
            return
        try:
            self._backend.prepare()
        except (ContainerUnavailable, VmUnavailable) as exc:
            if self.config.fail_if_unavailable:
                raise SandboxUnavailableError(
                    f"sandbox backend {self._backend.name!r} could not start: {exc}"
                ) from exc
            # Fall back to passthrough — never sink a run on a prepare failure.
            self._backend = NoopBackend()

    def reset(self) -> None:
        """Restore the VM to its base snapshot before a task.

        Self-gating so the call site (``react.run``) stays trivial: only the VM tier with
        ``[sandbox.vm].reset_each_task`` set actually does anything; native/container are
        no-ops. Failures degrade silently (never sink a run on a snapshot restore).
        """
        if not self.is_enabled():
            return
        if self.backend_tier is not SandboxTier.VM or not self.config.vm.reset_each_task:
            return
        try:
            self._backend.reset()
        except Exception as exc:  # noqa: BLE001 - a failed snapshot restore must not sink the run
            logger.warning(
                "sandbox VM reset failed; run continues on the previous guest state: %s: %s",
                type(exc).__name__, exc,
            )

    def teardown(self) -> None:
        """Release backend resources (stop/remove container, power off VM)."""
        try:
            self._backend.teardown()
        except Exception as exc:  # noqa: BLE001 - teardown must never raise into shutdown
            logger.warning(
                "sandbox teardown failed (resources may be left behind): %s: %s",
                type(exc).__name__, exc,
            )

    # -- the wrap seam called by command tools ---------------------------------------

    def wrap(self, spec, shell: bool, *, command: str | None = None) -> tuple[object, bool]:
        """Return ``(spec, shell)`` wrapped for isolation, or unchanged if not sandboxing.

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
        """A user-facing reason when ``enabled`` is set but sandboxing can't run; else None.

        Actionable even after a degrade-to-noop: it surfaces the missing dependencies of the
        backend(s) that were tried (recorded during selection), so an explicit
        ``backend = "container"`` with no runtime says *which* runtime to install.
        """
        if not self.config.enabled or self.is_enabled():
            return None
        # Prefer the most-capable candidate we tried that had missing deps.
        for name, missing in self._selection_diagnostics:
            if missing:
                return (
                    f"missing sandbox dependencies for backend {name!r}: "
                    f"{', '.join(missing)}; commands run unsandboxed"
                )
        return (
            f"no sandbox backend is available for {self.config.backend!r} on "
            f"{sys.platform} (unsupported platform or no runtime); commands run unsandboxed"
        )


# A shared disabled manager used as the default for unbound command tools (before the
# agent rebinds the real one). Passthrough on every platform.
NOOP_SANDBOX = SandboxManager(SandboxConfig())
