"""OS sandbox backends (one per platform) + the no-op degradation path."""

from __future__ import annotations

from agent_core.sandbox.backends.base import SandboxBackend
from agent_core.sandbox.backends.bubblewrap import BubblewrapBackend
from agent_core.sandbox.backends.noop import NoopBackend
from agent_core.sandbox.backends.seatbelt import SeatbeltBackend

__all__ = ["SandboxBackend", "BubblewrapBackend", "NoopBackend", "SeatbeltBackend"]
