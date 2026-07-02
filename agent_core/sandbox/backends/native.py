"""Native tier: OS primitives on the same kernel — the light/fast isolation path.

``NativeBackend`` is a thin per-platform dispatcher over the existing launcher-prefix
strategies (``bwrap`` on Linux, ``sandbox-exec`` on macOS, no-op elsewhere). It carries
no lifecycle of its own — the strategies just prefix a launcher onto each command.

Per the project's research this tier is a *fast path*, not a mature security boundary on
its own: prefer the Container/VM tiers for real containment, keep this for low-risk work
and for graceful degradation on Windows.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent_core.sandbox.backends.base import SandboxBackend, SandboxTier
from agent_core.sandbox.backends.bubblewrap import BubblewrapBackend
from agent_core.sandbox.backends.noop import NoopBackend
from agent_core.sandbox.backends.seatbelt import SeatbeltBackend
from agent_core.sandbox.config import SandboxConfig


class NativeBackend(SandboxBackend):
    """Dispatches to the platform-appropriate native launcher strategy."""

    name = "native"
    tier = SandboxTier.NATIVE

    def __init__(self) -> None:
        self._strategy: SandboxBackend = _select_native_strategy()

    @property
    def strategy(self) -> SandboxBackend:
        return self._strategy

    @property
    def strategy_name(self) -> str:
        return self._strategy.name

    def missing_dependencies(self) -> list[str]:
        return self._strategy.missing_dependencies()

    def available(self) -> bool:
        return self._strategy.available()

    def isolates(self) -> bool:
        # On Windows/unsupported the strategy is the no-op; native tier does not isolate.
        return self._strategy.isolates()

    def wrap(
        self, spec, shell: bool, *, config: SandboxConfig, workspace: Path
    ) -> tuple[object, bool]:
        return self._strategy.wrap(spec, shell, config=config, workspace=workspace)


def _select_native_strategy() -> SandboxBackend:
    if sys.platform == "darwin":
        return SeatbeltBackend()
    if sys.platform.startswith("linux"):
        return BubblewrapBackend()
    return NoopBackend()
