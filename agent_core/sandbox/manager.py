"""SandboxManager — selects a backend by isolation *tier*, wraps commands, owns lifecycle.

This is the adapter layer between project config and the pluggable backends, adapted from
Open-ClaudeCode's ``sandbox-adapter.ts`` + ``shouldUseSandbox.ts``. It answers the
questions the rest of the system asks:

- :meth:`is_enabled` — is sandboxing actually active here (enabled + a real backend)?
- :meth:`should_sandbox` — is the requested backend prepared for this command?
- :meth:`wrap_invocation` — choose the kernel-specific argv and wrap it.
- :meth:`prepare` / :meth:`reset` / :meth:`teardown` — backend lifecycle (eager per the
  eager-loading invariant; ``reset`` runs per task for the VM tier).

Backend selection is by **tier** (``config.backend``): an explicit ``native``/``container``
/``vm`` may select a weaker *real* backend; ``auto`` prefers ``container → native`` and
never auto-selects the heavyweight VM tier. An enabled sandbox never degrades to host
execution. Selection is not effectiveness: only a successfully prepared backend makes
:meth:`is_enabled` true.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from enum import Enum
import sys
import threading
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
from agent_core.sandbox.invocation import (
    GuestCapabilityUnavailable,
    GuestRuntimeManifest,
    REQUIRED_GUEST_TOOLS,
    SandboxInvocation,
)

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


class SandboxPreparationState(str, Enum):
    SELECTED = "selected"
    PREPARED = "prepared"
    UNAVAILABLE = "unavailable"


class SandboxManager:
    def __init__(self, config: SandboxConfig | None = None, workspace: str | Path | None = None) -> None:
        self.config = config or SandboxConfig()
        self.workspace = Path(workspace or Path.cwd()).resolve()
        # (backend name, missing deps) for each candidate tried — powers an actionable
        # unavailable_reason() when no real backend can be selected.
        self._selection_diagnostics: list[tuple[str, list[str]]] = []
        self._reconfigure_lock = threading.RLock()
        self._retired_backends: list[SandboxBackend] = []
        # prepare() is idempotent (heavyweight readying runs once per manager); teardown
        # resets this so a torn-down manager could be re-readied.
        self._prepared = False
        self._preparation_error = ""
        self._backend = self._select_backend()
        self._state = (
            SandboxPreparationState.SELECTED
            if self.config.enabled and self._backend.isolates()
            else SandboxPreparationState.UNAVAILABLE
        )
        # Fail-fast gate: an operator asked for sandboxing AND declared it mandatory,
        # but this environment can't provide it — refuse to start rather than silently
        # running commands unsandboxed. (Actionable-error invariant.)
        if (
            self.config.enabled
            and self.config.fail_if_unavailable
            and not self.is_supported_platform()
        ):
            reason = self.unavailable_reason() or "sandbox is unavailable"
            raise SandboxUnavailableError(
                f"sandbox.enabled + fail_if_unavailable set, but {reason}"
            )

    # -- backend selection / capability ---------------------------------------------

    def _select_backend(self) -> SandboxBackend:
        """Pick the strongest *available* backend at-or-below the requested tier.

        Disabled config short-circuits to a no-op. For ``auto`` the candidate chain is
        ``container → native → unavailable`` (VM is opt-in only); for an explicit tier it is that
        tier then every weaker one. The first candidate whose backend reports
        :meth:`available` wins; if none do, an internal :class:`NoopBackend` records
        the unavailable state but is never used to execute an enabled invocation.
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
            SandboxTier.CONTAINER: lambda: ContainerBackend(
                self.config, workspace=self.workspace
            ),
            SandboxTier.VM: lambda: VmBackend(self.config, workspace=self.workspace),
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

    @property
    def preparation_state(self) -> SandboxPreparationState:
        return self._state

    @property
    def prepared(self) -> bool:
        return self._state is SandboxPreparationState.PREPARED

    @property
    def requested(self) -> bool:
        return self.config.enabled

    @property
    def uses_guest(self) -> bool:
        return self.config.enabled and self._backend.uses_guest

    @property
    def guest_manifest(self) -> GuestRuntimeManifest | None:
        return self._backend.guest_manifest

    @property
    def capabilities(self) -> frozenset[str]:
        manifest = self.guest_manifest
        return manifest.capabilities if manifest is not None else frozenset()

    @property
    def runtime(self) -> str | None:
        return getattr(self._backend, "runtime", None)

    @property
    def image(self) -> str:
        if self.backend_tier is SandboxTier.VM:
            return self.config.vm.base_image
        return self.config.container.image if self._backend.uses_guest else ""

    def is_supported_platform(self) -> bool:
        """True when the selected backend does real isolation (i.e. is not the no-op)."""
        return self._backend.isolates()

    def is_enabled(self) -> bool:
        """True when sandboxing is switched on AND a real backend is active here."""
        return (
            self.config.enabled
            and self.is_supported_platform()
            and self._backend.available()
            and self._state is SandboxPreparationState.PREPARED
        )

    def should_sandbox(self, command: str | None) -> bool:
        """Whether *this* command should be wrapped: enabled and not excluded."""
        if not self.is_enabled():
            return False
        if command and self._is_excluded(command):
            return False
        return True

    def translate_path(self, path: str | Path) -> str:
        return self._backend.translate_path(Path(path).resolve())

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

        Idempotent: the heavyweight readying runs at most once per manager, so agents
        (and sub-agents) sharing one manager can all call ``prepare()`` cheaply. A backend
        that cannot ready itself becomes unavailable and raises
        :class:`SandboxUnavailableError`; host execution is never a fallback.
        """
        if self._prepared or not self.config.enabled:
            return
        if not self.is_supported_platform() or not self._backend.available():
            self._state = SandboxPreparationState.UNAVAILABLE
            reason = self.unavailable_reason() or "sandbox is unavailable"
            raise SandboxUnavailableError(reason)
        if self.config.excluded_commands:
            self._state = SandboxPreparationState.UNAVAILABLE
            self._preparation_error = (
                "excluded_commands are forbidden while sandbox.enabled=true; use the "
                "explicit --no-sandbox session option instead"
            )
            raise SandboxUnavailableError(self._preparation_error)
        try:
            self._backend.prepare()
            if self._backend.uses_guest and self._backend.guest_manifest is None:
                raise GuestCapabilityUnavailable(
                    "guest_capability_unavailable: backend did not provide a guest manifest"
                )
            if self._backend.uses_guest:
                assert self._backend.guest_manifest is not None
                missing = REQUIRED_GUEST_TOOLS - self._backend.guest_manifest.capabilities
                if missing:
                    raise GuestCapabilityUnavailable(
                        "guest_capability_unavailable: backend is missing required tools: "
                        + ", ".join(sorted(missing))
                    )
        except (ContainerUnavailable, VmUnavailable, GuestCapabilityUnavailable) as exc:
            self._state = SandboxPreparationState.UNAVAILABLE
            self._preparation_error = str(exc)
            raise SandboxUnavailableError(
                f"sandbox backend {self._backend.name!r} could not prepare: {exc}"
            ) from exc
        self._prepared = True
        self._state = SandboxPreparationState.PREPARED
        self._preparation_error = ""

    def reset(self) -> None:
        """Restore the VM to its base snapshot before a task.

        Self-gating so the call site (``react.run``) stays trivial: only the VM tier with
        ``[sandbox.vm].reset_each_task`` set actually does anything; native/container are
        no-ops. A reset failure makes the backend unavailable and aborts the run.
        """
        if not self.is_enabled():
            return
        if self.backend_tier is not SandboxTier.VM or not self.config.vm.reset_each_task:
            return
        try:
            self._backend.reset()
        except Exception as exc:
            self._state = SandboxPreparationState.UNAVAILABLE
            self._prepared = False
            self._preparation_error = f"VM reset failed: {type(exc).__name__}: {exc}"
            raise SandboxUnavailableError(self._preparation_error) from exc

    def reconfigure(self, config: SandboxConfig) -> None:
        """Prepare a replacement backend, then atomically publish it.

        Tools that already wrapped/launched a command keep using the retired backend.
        Retired backend resources are released at session teardown, while all future
        permission checks and wraps observe the replacement immediately.
        """

        candidate = SandboxManager(config, workspace=self.workspace)
        candidate.prepare()
        with self._reconfigure_lock:
            self._retired_backends.append(self._backend)
            self.config = candidate.config
            self._backend = candidate._backend
            self._selection_diagnostics = candidate._selection_diagnostics
            self._prepared = candidate._prepared
            self._state = candidate._state
            self._preparation_error = candidate._preparation_error

    def teardown(self) -> None:
        """Release backend resources (stop/remove container, power off VM)."""
        try:
            self._backend.teardown()
            for backend in self._retired_backends:
                try:
                    backend.teardown()
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001 - teardown must never raise into shutdown
            logger.warning(
                "sandbox teardown failed (resources may be left behind): %s: %s",
                type(exc).__name__, exc,
            )
        finally:
            self._prepared = False
            self._state = (
                SandboxPreparationState.SELECTED
                if self.config.enabled and self._backend.isolates()
                else SandboxPreparationState.UNAVAILABLE
            )
            self._retired_backends.clear()

    # -- the wrap seam called by command tools ---------------------------------------

    def wrap_invocation(
        self, invocation: SandboxInvocation, *, command: str | None = None
    ) -> tuple[object, bool]:
        """Select the host or Linux-guest argv and apply the active backend."""

        if not self.config.enabled:
            return list(invocation.host_argv), False
        if not self.is_enabled():
            reason = self.unavailable_reason() or "sandbox has not completed preparation"
            raise SandboxUnavailableError(reason)
        if command is not None and self._is_excluded(command):
            raise SandboxUnavailableError(
                "excluded commands are forbidden while sandboxing is enabled"
            )
        config, workspace = self._config_for_scope(invocation.scope)
        if self._backend.uses_guest:
            manifest = self.guest_manifest
            if manifest is None:
                raise GuestCapabilityUnavailable(
                    "guest_capability_unavailable: prepared backend has no guest manifest"
                )
            argv = invocation.guest_command(manifest)
        else:
            argv = list(invocation.host_argv)
        return self._backend.wrap(argv, False, config=config, workspace=workspace)

    def wrap(self, spec, shell: bool, *, command: str | None = None, scope=None) -> tuple[object, bool]:
        """Return ``(spec, shell)`` wrapped for isolation, or unchanged if not sandboxing.

        ``command`` is the raw shell command line (for the exclusion check); pass ``None``
        for argv-based tools (e.g. the test runner) to sandbox purely on ``is_enabled``.
        """
        if isinstance(spec, SandboxInvocation):
            return self.wrap_invocation(spec, command=command)
        if not self.config.enabled:
            return spec, shell
        if self._backend.uses_guest:
            raise GuestCapabilityUnavailable(
                "guest_capability_unavailable: cross-kernel execution requires a SandboxInvocation"
            )
        if not self.is_enabled():
            reason = self.unavailable_reason() or "sandbox has not completed preparation"
            raise SandboxUnavailableError(reason)
        if command is not None and self._is_excluded(command):
            raise SandboxUnavailableError(
                "excluded commands are forbidden while sandboxing is enabled"
            )
        config, workspace = self._config_for_scope(scope)
        return self._backend.wrap(spec, shell, config=config, workspace=workspace)

    def _config_for_scope(self, scope) -> tuple[SandboxConfig, Path]:
        workspace = getattr(scope, "workspace", self.workspace)
        config = self.config
        if scope is not None:
            config = deepcopy(self.config)
            writable = [str(path) for path in getattr(scope, "writable_roots", ())]
            private_temp = getattr(scope, "private_temp", None)
            if private_temp is not None:
                writable.append(str(private_temp))
            config.filesystem.allow_write.extend(writable)
            read_only = [str(path) for path in getattr(scope, "read_only_roots", ())]
            git_common = getattr(scope, "git_common_dir", None)
            if git_common is not None:
                read_only.append(str(git_common))
            if not getattr(scope, "workspace_writable", True):
                read_only.append(str(workspace))
            config.filesystem.deny_write.extend(read_only)
            config.filesystem.allow_read.extend(read_only)
            if getattr(scope, "network", "deny") == "deny":
                config.network.allowed_domains.clear()
                config.network.allow_local_binding = False
            elif not self._backend.uses_guest:
                config.network.allow_local_binding = True
            else:
                raise GuestCapabilityUnavailable(
                    "guest_capability_unavailable: sandbox execution only supports network=deny"
                )
        return config, Path(workspace).resolve()

    # -- diagnostics -----------------------------------------------------------------

    def unavailable_reason(self) -> str | None:
        """A user-facing reason when ``enabled`` is set but sandboxing can't run; else None.

        It surfaces the missing dependencies of the backend(s) that were tried during
        selection, so an explicit
        ``backend = "container"`` with no runtime says *which* runtime to install.
        """
        if not self.config.enabled or self.is_enabled():
            return None
        if self._preparation_error:
            return self._preparation_error
        if self._state is SandboxPreparationState.SELECTED:
            return (
                f"sandbox backend {self._backend.name!r} is selected but preparation "
                "probes have not completed"
            )
        # Prefer the most-capable candidate we tried that had missing deps.
        for name, missing in self._selection_diagnostics:
            if missing:
                return (
                    f"missing sandbox dependencies for backend {name!r}: "
                    f"{', '.join(missing)}"
                )
        return (
            f"no sandbox backend is available for {self.config.backend!r} on "
            f"{sys.platform} (unsupported platform or no runtime)"
        )


# A shared disabled manager used as the default for unbound command tools (before the
# agent rebinds the real one). Passthrough on every platform.
NOOP_SANDBOX = SandboxManager(SandboxConfig())


# --- process-level sharing (§5.6 / E4) --------------------------------------------
#
# Constructing an agent must not repeat heavyweight sandbox side effects (verify the
# container runtime, pull an image, boot a VM) per instance: managers are cached per
# (config fingerprint, workspace) so every agent in this process with the same settings
# reuses one prepared manager, and sub-agents are handed their parent's directly.

_shared_lock = threading.Lock()
_shared_managers: dict[tuple[str, str], SandboxManager] = {}


def get_shared_manager(config: SandboxConfig | None, workspace: str | Path | None = None) -> SandboxManager:
    """The process-shared :class:`SandboxManager` for this (config, workspace) pair.

    The first request constructs it (which may raise ``SandboxUnavailableError`` when
    ``fail_if_unavailable`` is set — failures are never cached) and logs the choice once;
    later requests return the same instance, whose idempotent :meth:`~SandboxManager.prepare`
    makes repeated readying free.
    """
    resolved_config = config or SandboxConfig()
    key = (repr(resolved_config), str(Path(workspace or Path.cwd()).resolve()))
    with _shared_lock:
        manager = _shared_managers.get(key)
        if manager is None:
            manager = SandboxManager(resolved_config, workspace=workspace)
            _shared_managers[key] = manager
            logger.info(
                "sandbox manager created for this process: backend=%s enabled=%s workspace=%s",
                manager.backend_name, manager.is_enabled(), manager.workspace,
            )
        return manager


def reset_shared_managers() -> None:
    """Drop the process-level manager cache (test isolation seam)."""
    with _shared_lock:
        _shared_managers.clear()
