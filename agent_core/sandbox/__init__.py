"""OS-level sandbox subsystem (enforcement layer).

Two-layer design:

- the cross-platform *policy layer* lives in :mod:`agent_core.permission_rules` and
  :mod:`agent_core.permissions` (fine-grained allow/deny/ask rules);
- this package is the OS *enforcement layer* — it wraps dangerous command execution in
``bwrap`` (Linux), ``sandbox-exec`` (macOS), or a probed OCI/WSL2 Linux guest.

Command tools reach the active manager via :class:`SandboxAwareMixin`, which
``ReActAgent`` rebinds at startup (parallel to ``SessionAwareMixin``).
"""

from __future__ import annotations

from agent_core.sandbox.backends import SandboxTier
from agent_core.sandbox.config import (
    SandboxConfig,
    SandboxContainerConfig,
    SandboxFilesystemConfig,
    SandboxNetworkConfig,
    SandboxVmConfig,
)
from agent_core.sandbox.manager import (
    NOOP_SANDBOX,
    SandboxManager,
    SandboxRequiredError,
    SandboxUnavailableError,
    get_shared_manager,
    reset_shared_managers,
    SandboxPreparationState,
)
from agent_core.sandbox.invocation import (
    GuestCapabilityUnavailable,
    GuestRuntimeManifest,
    SandboxInvocation,
)


class SandboxAwareMixin:
    """Mixin giving a tool access to the active :class:`SandboxManager`.

    Deliberately defines *no* ``__init__`` so it composes cleanly with
    ``WorkspacePathMixin`` (whose ``__init__`` sets ``self.workspace``): the sandbox
    starts as the shared disabled :data:`NOOP_SANDBOX` (a class default) and
    ``ReActAgent`` calls :meth:`bind_sandbox` to point it at the live manager, exactly
    like ``SessionAwareMixin.bind_session``.
    """

    needs_sandbox = True
    sandbox: SandboxManager = NOOP_SANDBOX

    def bind_sandbox(self, sandbox: SandboxManager) -> None:
        self.sandbox = sandbox


__all__ = [
    "SandboxConfig",
    "SandboxContainerConfig",
    "SandboxVmConfig",
    "SandboxFilesystemConfig",
    "SandboxNetworkConfig",
    "SandboxTier",
    "SandboxManager",
    "SandboxRequiredError",
    "SandboxUnavailableError",
    "SandboxPreparationState",
    "SandboxInvocation",
    "GuestRuntimeManifest",
    "GuestCapabilityUnavailable",
    "SandboxAwareMixin",
    "NOOP_SANDBOX",
    "get_shared_manager",
    "reset_shared_managers",
]
