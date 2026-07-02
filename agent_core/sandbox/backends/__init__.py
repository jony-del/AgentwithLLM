"""Sandbox backends, organised by isolation tier.

- Native tier: :class:`NativeBackend` (dispatches to the ``bwrap`` / ``sandbox-exec`` /
  no-op launcher strategies).
- Container tier: :class:`ContainerBackend` (podman/docker/nerdctl).
- VM tier: :class:`VmBackend` (Kata / Hyper-V / Lima strategies).
"""

from __future__ import annotations

from agent_core.sandbox.backends.base import (
    SandboxBackend,
    SandboxTier,
    expand_paths,
    to_argv,
)
from agent_core.sandbox.backends.bubblewrap import BubblewrapBackend
from agent_core.sandbox.backends.container import ContainerBackend
from agent_core.sandbox.backends.native import NativeBackend
from agent_core.sandbox.backends.noop import NoopBackend
from agent_core.sandbox.backends.seatbelt import SeatbeltBackend
from agent_core.sandbox.backends.vm import VmBackend

__all__ = [
    "SandboxBackend",
    "SandboxTier",
    "NativeBackend",
    "ContainerBackend",
    "VmBackend",
    "BubblewrapBackend",
    "NoopBackend",
    "SeatbeltBackend",
    "to_argv",
    "expand_paths",
]
