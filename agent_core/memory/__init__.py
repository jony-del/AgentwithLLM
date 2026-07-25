"""Cross-conversation memory: persistence, recall, extraction, and dreaming."""

from agent_core.memory.config import (
    MemoryConfig,
    MemoryRetrievalConfig,
    RemovedMemoryConfigWarning,
)
from agent_core.memory.dreaming import Dreamer
from agent_core.memory.extraction import MEMORY_EXTRACTION_MARKER, MemoryExtractor
from agent_core.memory.lifecycle import DreamEligibility, MemoryLifecycle
from agent_core.memory.models import (
    DreamReport,
    EmbeddingBackend,
    MemoryChange,
    MemoryDocument,
    MemoryPassage,
    MemoryRecord,
    MemorySearchHit,
    MemorySearchRequest,
    RerankerBackend,
    RetrievalTrace,
)
from agent_core.memory.paths import MemoryPathResolver, UnsafeMemoryPathError, canonical_agent_key
from agent_core.memory.repository import MemoryRepository, MigrationReport, ValidationReport
from agent_core.memory.retrieval import HybridMemoryRetriever, MemoryRetriever, RepositoryMemoryRetriever
from agent_core.memory.security import SecretDetectedError, scan_secrets
from agent_core.memory.snapshots import LocalMemorySnapshot, SnapshotStatus
from agent_core.memory.store import MemoryStore, RepositoryMemoryStore

__all__ = [
    "MemoryConfig",
    "MemoryRetrievalConfig",
    "RemovedMemoryConfigWarning",
    "MemoryRecord",
    "MemoryDocument",
    "MemoryChange",
    "MemorySearchRequest",
    "MemorySearchHit",
    "MemoryPassage",
    "RetrievalTrace",
    "EmbeddingBackend",
    "RerankerBackend",
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
    "HybridMemoryRetriever",
    "RepositoryMemoryRetriever",
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
