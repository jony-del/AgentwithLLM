"""Compatibility exports for the deprecated JSONL memory implementation."""

from agent_core.memory.models import MemoryRecord
from agent_core.memory.store import MemoryStore

__all__ = ["MemoryRecord", "MemoryStore"]
