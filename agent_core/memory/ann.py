from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from agent_core.file_lock import FileLock
from agent_core.memory.indexing import ANN_INDEX_VERSION

if TYPE_CHECKING:
    from agent_core.memory.indexing import MemoryIndex


_ANN_CACHE_MAX_INDEXES = 2
_ANN_CACHE_LOCK = threading.Lock()
_ANN_CACHE: OrderedDict[str, tuple[tuple[int, int], Any]] = OrderedDict()
_BUILD_LOCK = threading.Lock()
_BUILDING: set[str] = set()


@dataclass(frozen=True, slots=True)
class AnnManifest:
    schema: int
    engine: str
    index_version: str
    generation: str
    dense_epoch: str
    base_revision: int
    vector_count: int
    dimension: int
    embedding_fingerprint: str
    metric: str
    dtype: str
    connectivity: int
    expansion_add: int
    expansion_search: int
    index_file: str
    index_size_bytes: int
    created_at: float

    @classmethod
    def from_dict(cls, value: object) -> "AnnManifest | None":
        if not isinstance(value, dict):
            return None
        try:
            manifest = cls(
                schema=int(value["schema"]),
                engine=str(value["engine"]),
                index_version=str(value["index_version"]),
                generation=str(value["generation"]),
                dense_epoch=str(value["dense_epoch"]),
                base_revision=int(value["base_revision"]),
                vector_count=int(value["vector_count"]),
                dimension=int(value["dimension"]),
                embedding_fingerprint=str(value["embedding_fingerprint"]),
                metric=str(value["metric"]),
                dtype=str(value["dtype"]),
                connectivity=int(value["connectivity"]),
                expansion_add=int(value["expansion_add"]),
                expansion_search=int(value["expansion_search"]),
                index_file=str(value["index_file"]),
                index_size_bytes=int(value["index_size_bytes"]),
                created_at=float(value["created_at"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
        if (
            manifest.schema != 1
            or manifest.engine != "usearch"
            or manifest.index_version != ANN_INDEX_VERSION
            or not manifest.generation
            or manifest.vector_count < 0
            or manifest.dimension <= 0
            or manifest.index_size_bytes <= 0
            or Path(manifest.index_file).name != manifest.index_file
        ):
            return None
        return manifest


@dataclass(frozen=True, slots=True)
class AnnStatus:
    state: str = "missing"
    vectors: int = 0
    coverage: float = 0.0
    generation: str = ""
    base_revision: int = 0
    dimension: int = 0
    detail: str = ""


def _usearch_index_class():
    from usearch.index import Index

    return Index


def usearch_available() -> bool:
    try:
        _usearch_index_class()
    except (ImportError, OSError):
        return False
    return True


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}-{os.getpid()}-{time.time_ns()}.tmp")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class DenseAnnIndex:
    """Disposable immutable USearch generations over SQLite-owned embeddings."""

    def __init__(self, memory_index: "MemoryIndex") -> None:
        self.memory_index = memory_index
        self.directory = memory_index.directory / "dense"
        self.active_path = self.directory / "active.json"
        self.lock_path = memory_index.directory / ".ann.lock"

    def _manifest(self) -> AnnManifest | None:
        try:
            value = json.loads(self.active_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return AnnManifest.from_dict(value)

    def status(self) -> AnnStatus:
        snapshot = self.memory_index.dense_snapshot()
        total = int(snapshot["count"])
        if self.memory_index.config.dense_strategy == "exact":
            return AnnStatus(state="disabled", coverage=1.0)
        if (
            self.memory_index.config.dense_strategy == "auto"
            and total < self.memory_index.config.ann_min_vectors
        ):
            return AnnStatus(state="not_needed", coverage=1.0)
        manifest = self._manifest()
        if not usearch_available():
            return AnnStatus(
                state="backend_unavailable",
                detail="install the [memory] extra to enable USearch",
            )
        if manifest is None:
            return AnnStatus(state="missing")
        path = self.directory / manifest.index_file
        try:
            index_size = path.stat().st_size
        except OSError:
            index_size = -1
        if (
            manifest.dense_epoch != str(snapshot["epoch"])
            or manifest.embedding_fingerprint != self.memory_index.embedding_fingerprint
            or int(snapshot["min_dimension"]) != int(snapshot["max_dimension"])
            or manifest.dimension != int(snapshot["min_dimension"])
            or index_size != manifest.index_size_bytes
        ):
            return AnnStatus(state="stale", generation=manifest.generation)
        tombstones = self.memory_index.tombstones_after(manifest.base_revision)
        valid_base = max(0, manifest.vector_count - tombstones)
        coverage = min(1.0, valid_base / total) if total else 1.0
        delta = self.memory_index.embedded_count(
            {},
            min_revision=manifest.base_revision,
        )
        threshold = max(1000, math.ceil(max(1, manifest.vector_count) * 0.05))
        state = "stale" if delta + tombstones >= threshold else "ready"
        return AnnStatus(
            state=state,
            vectors=manifest.vector_count,
            coverage=coverage,
            generation=manifest.generation,
            base_revision=manifest.base_revision,
            dimension=manifest.dimension,
            detail=f"delta={delta} tombstones={tombstones}",
        )

    def search(self, query_vector: Any, *, count: int) -> tuple[list[int], AnnStatus]:
        status = self.status()
        if status.state not in {"ready", "stale"} or count <= 0:
            return [], status
        manifest = self._manifest()
        if manifest is None:
            return [], AnnStatus(state="missing")
        path = self.directory / manifest.index_file
        try:
            signature = (int(path.stat().st_size), int(path.stat().st_mtime_ns))
            cache_key = str(path)
            with _ANN_CACHE_LOCK:
                cached = _ANN_CACHE.get(cache_key)
                if cached is None or cached[0] != signature:
                    Index = _usearch_index_class()
                    restored = Index.restore(
                        str(path),
                        view=True,
                        expansion_search=max(
                            manifest.expansion_search,
                            min(8192, int(count)),
                        ),
                    )
                    if restored is None:
                        raise RuntimeError("USearch returned no restored index")
                    _ANN_CACHE[cache_key] = (signature, restored)
                    _ANN_CACHE.move_to_end(cache_key)
                    while len(_ANN_CACHE) > _ANN_CACHE_MAX_INDEXES:
                        _ANN_CACHE.popitem(last=False)
                else:
                    restored = cached[1]
                    _ANN_CACHE.move_to_end(cache_key)
            requested = min(max(1, int(count)), int(restored.size))
            matches = restored.search(query_vector, requested)
            return [int(key) for key in matches.keys.tolist()], status
        except Exception as exc:
            return [], AnnStatus(
                state="corrupt",
                generation=manifest.generation,
                detail=type(exc).__name__,
            )

    def build(self) -> AnnStatus:
        if not usearch_available():
            return AnnStatus(state="backend_unavailable")
        snapshot = self.memory_index.dense_snapshot()
        epoch = str(snapshot["epoch"])
        revision = int(snapshot["revision"])
        if int(snapshot["count"]) != int(snapshot["chunks"]):
            return AnnStatus(state="missing", detail="embedding coverage is incomplete")
        count = self.memory_index.embedded_count({}, max_revision=revision)
        min_dimension = int(snapshot["min_dimension"])
        max_dimension = int(snapshot["max_dimension"])
        if count <= 0 or min_dimension <= 0 or min_dimension != max_dimension:
            return AnnStatus(state="missing", detail="no consistent embeddings")

        self.directory.mkdir(parents=True, exist_ok=True)
        generation = f"{epoch[:12]}-{revision}-{time.time_ns()}"
        final_name = f"{generation}.usearch"
        final_path = self.directory / final_name
        temporary = self.directory / f".{final_name}-{os.getpid()}.tmp"
        actual = 0
        with FileLock(self.lock_path, timeout=self.memory_index.config.timeout_seconds):
            try:
                Index = _usearch_index_class()
                index = Index(
                    ndim=min_dimension,
                    metric="cos",
                    dtype="f16",
                    connectivity=32,
                    expansion_add=256,
                    expansion_search=self.memory_index.config.ann_expansion_search,
                )
                import numpy as np

                for rows in self.memory_index.iter_embedded_batches(
                    {},
                    batch_size=4096,
                    max_revision=revision,
                ):
                    dimensions = {int(row["embedding_dim"]) for row in rows}
                    if dimensions != {min_dimension}:
                        raise ValueError("index contains mixed embedding dimensions")
                    keys = np.asarray([int(row["ann_label"]) for row in rows], dtype=np.uint64)
                    vectors = np.empty((len(rows), min_dimension), dtype=np.float32)
                    for offset, row in enumerate(rows):
                        vectors[offset] = np.frombuffer(
                            row["embedding"],
                            dtype=np.float32,
                            count=min_dimension,
                        )
                    index.add(keys, vectors, threads=self.memory_index.config.model_threads)
                    actual += len(rows)
                if actual <= 0:
                    return AnnStatus(state="missing")
                if int(index.size) != actual:
                    raise RuntimeError("USearch indexed an unexpected vector count")
                index.save(str(temporary))
                os.replace(temporary, final_path)
                current = self.memory_index.dense_snapshot()
                if str(current["epoch"]) != epoch:
                    final_path.unlink(missing_ok=True)
                    return AnnStatus(state="stale", detail="dense epoch changed during build")
                manifest = AnnManifest(
                    schema=1,
                    engine="usearch",
                    index_version=ANN_INDEX_VERSION,
                    generation=generation,
                    dense_epoch=epoch,
                    base_revision=revision,
                    vector_count=actual,
                    dimension=min_dimension,
                    embedding_fingerprint=self.memory_index.embedding_fingerprint,
                    metric="cos",
                    dtype="f16",
                    connectivity=32,
                    expansion_add=256,
                    expansion_search=self.memory_index.config.ann_expansion_search,
                    index_file=final_name,
                    index_size_bytes=final_path.stat().st_size,
                    created_at=time.time(),
                )
                _atomic_write_json(self.active_path, asdict(manifest))
                self.memory_index.prune_tombstones_through(
                    revision,
                    expected_epoch=epoch,
                )
                self._prune_generations(keep={final_name})
                return self.status()
            finally:
                temporary.unlink(missing_ok=True)

    def schedule_build(self) -> None:
        key = str(self.active_path)
        with _BUILD_LOCK:
            if key in _BUILDING:
                return
            _BUILDING.add(key)

        def worker() -> None:
            try:
                self.build()
            except Exception:
                return
            finally:
                with _BUILD_LOCK:
                    _BUILDING.discard(key)

        threading.Thread(
            target=worker,
            name="polaris-memory-ann",
            daemon=True,
        ).start()

    def _prune_generations(self, *, keep: set[str]) -> None:
        candidates = sorted(
            (
                path
                for path in self.directory.glob("*.usearch")
                if path.name not in keep
            ),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for path in candidates[1:]:
            try:
                path.unlink()
            except OSError:
                continue
