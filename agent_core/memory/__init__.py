"""Cross-conversation memory: persistence, recall, extraction, and dreaming."""

from agent_core.memory.config import MemoryConfig
from agent_core.memory.dreaming import Dreamer
from agent_core.memory.extraction import MEMORY_EXTRACTION_MARKER, MemoryExtractor
from agent_core.memory.models import DreamReport, MemoryRecord
from agent_core.memory.retrieval import MemoryRetriever
from agent_core.memory.store import MemoryStore

__all__ = [
    "MemoryConfig",
    "MemoryRecord",
    "DreamReport",
    "MemoryStore",
    "MemoryRetriever",
    "MemoryExtractor",
    "MEMORY_EXTRACTION_MARKER",
    "Dreamer",
]
