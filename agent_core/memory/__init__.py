"""Cross-conversation memory: persistence, recall, extraction, and dreaming."""

from agent_core.memory.config import MemoryConfig
from agent_core.memory.dreaming import Dreamer
from agent_core.memory.extraction import MEMORY_EXTRACTION_MARKER, MemoryExtractor
from agent_core.memory.lifecycle import DreamEligibility, MemoryLifecycle
from agent_core.memory.models import DreamReport, MemoryChange, MemoryDocument, MemoryRecord
from agent_core.memory.paths import MemoryPathResolver, UnsafeMemoryPathError, canonical_agent_key
from agent_core.memory.repository import MemoryRepository, MigrationReport, ValidationReport
from agent_core.memory.retrieval import (
    MEMORY_SELECTION_MARKER,
    MemoryRetriever,
    RepositoryMemoryRetriever,
    SemanticMemorySelector,
)
from agent_core.memory.security import SecretDetectedError, scan_secrets
from agent_core.memory.snapshots import LocalMemorySnapshot, SnapshotStatus
from agent_core.memory.store import MemoryStore, RepositoryMemoryStore

__all__ = [
    "MemoryConfig",
    "MemoryRecord",
    "MemoryDocument",
    "MemoryChange",
    "DreamReport",
    "MemoryStore",
    "RepositoryMemoryStore",
    "MemoryRepository",
    "MemoryPathResolver",
    "UnsafeMemoryPathError",
    "canonical_agent_key",
    "MigrationReport",
    "ValidationReport",
    "MemoryRetriever",
    "RepositoryMemoryRetriever",
    "SemanticMemorySelector",
    "MEMORY_SELECTION_MARKER",
    "MemoryExtractor",
    "MEMORY_EXTRACTION_MARKER",
    "Dreamer",
    "MemoryLifecycle",
    "DreamEligibility",
    "SecretDetectedError",
    "scan_secrets",
    "LocalMemorySnapshot",
    "SnapshotStatus",
]
