from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent_core.memory.models import EmbeddingBackend, RerankerBackend
from agent_core.memory.models_manager import MemoryModelManager

_RUNTIME_LOCK = threading.Lock()
_RUNTIMES: dict[tuple[str, str, int], object] = {}


def _deadline_timeout(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("memory model deadline exceeded")


class _OnnxBase:
    def __init__(
        self,
        root: Path,
        manifest: dict[str, Any],
        *,
        model_threads: int,
        role: str,
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.role = role
        self._fingerprint = str(manifest["fingerprint"])
        self.model_path = root / str(manifest["model"])
        self.tokenizer_path = root / str(manifest["tokenizer"])
        self.max_length = int(manifest.get("max_length", 8192))
        self.model_threads = max(1, int(model_threads))
        self._session: Any | None = None
        self._tokenizer: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_gate = threading.BoundedSemaphore(
            max(1, int(manifest.get("max_concurrency", 1)))
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def _load(self) -> tuple[Any, Any]:
        if self._session is not None and self._tokenizer is not None:
            return self._session, self._tokenizer
        with self._load_lock:
            if self._session is not None and self._tokenizer is not None:
                return self._session, self._tokenizer
            # Optional dependencies stay behind this runtime-only boundary.
            import onnxruntime as ort
            from tokenizers import Tokenizer

            options = ort.SessionOptions()
            options.intra_op_num_threads = self.model_threads
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(
                str(self.model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
            tokenizer.enable_truncation(max_length=self.max_length)
            padding = self.manifest.get("padding", {})
            if isinstance(padding, dict):
                tokenizer.enable_padding(
                    pad_id=int(padding.get("pad_id", 1)),
                    pad_token=str(padding.get("pad_token", "<pad>")),
                )
            else:
                tokenizer.enable_padding()
            self._session = session
            self._tokenizer = tokenizer
            return session, tokenizer

    @staticmethod
    def _inputs(session: Any, encodings: Sequence[Any]) -> dict[str, Any]:
        import numpy as np

        names = {item.name for item in session.get_inputs()}
        values: dict[str, Any] = {}
        if "input_ids" in names:
            values["input_ids"] = np.asarray(
                [encoding.ids for encoding in encodings], dtype=np.int64
            )
        if "attention_mask" in names:
            values["attention_mask"] = np.asarray(
                [encoding.attention_mask for encoding in encodings], dtype=np.int64
            )
        if "token_type_ids" in names:
            values["token_type_ids"] = np.asarray(
                [encoding.type_ids for encoding in encodings], dtype=np.int64
            )
        missing = names.difference(values)
        if missing:
            raise RuntimeError("unsupported ONNX model inputs: " + ", ".join(sorted(missing)))
        return values


class OnnxEmbeddingBackend(_OnnxBase):
    def __init__(
        self,
        root: Path,
        manifest: dict[str, Any],
        *,
        model_threads: int = 4,
    ) -> None:
        super().__init__(
            root,
            manifest,
            model_threads=model_threads,
            role="embedding",
        )
        self._dimension = int(manifest.get("dimension", 1024))

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(
        self,
        texts: Sequence[str],
        *,
        deadline: float | None = None,
    ) -> Sequence[Sequence[float]]:
        _deadline_timeout(deadline)
        if not texts:
            return []
        import numpy as np

        session, tokenizer = self._load()
        encodings = tokenizer.encode_batch([str(text) for text in texts])
        with self._inference_gate:
            _deadline_timeout(deadline)
            outputs = session.run(None, self._inputs(session, encodings))
        if not outputs:
            raise RuntimeError("embedding ONNX model returned no outputs")
        vectors = np.asarray(outputs[0], dtype=np.float32)
        if vectors.ndim == 3:
            pooling = str(self.manifest.get("pooling", "cls"))
            if pooling == "mean":
                mask = np.asarray(
                    [encoding.attention_mask for encoding in encodings],
                    dtype=np.float32,
                )[:, :, None]
                vectors = (vectors * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
            else:
                vectors = vectors[:, 0, :]
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise RuntimeError("embedding ONNX model returned an unexpected shape")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
            raise RuntimeError("embedding ONNX model returned invalid vectors")
        normalized = vectors / norms
        return normalized.tolist()


class OnnxRerankerBackend(_OnnxBase):
    outputs_logits = True

    def __init__(
        self,
        root: Path,
        manifest: dict[str, Any],
        *,
        model_threads: int = 4,
    ) -> None:
        super().__init__(
            root,
            manifest,
            model_threads=model_threads,
            role="reranker",
        )

    def rerank(
        self,
        query: str,
        passages: Sequence[str],
        *,
        deadline: float | None = None,
    ) -> Sequence[float]:
        _deadline_timeout(deadline)
        if not passages:
            return []
        import numpy as np

        session, tokenizer = self._load()
        encodings = tokenizer.encode_batch(
            [(str(query), str(passage)) for passage in passages]
        )
        with self._inference_gate:
            _deadline_timeout(deadline)
            outputs = session.run(None, self._inputs(session, encodings))
        if not outputs:
            raise RuntimeError("reranker ONNX model returned no outputs")
        logits = np.asarray(outputs[0], dtype=np.float32)
        if logits.ndim == 2:
            if logits.shape[1] == 1:
                logits = logits[:, 0]
            else:
                logits = logits[:, -1]
        logits = logits.reshape(-1)
        if len(logits) != len(passages) or not np.all(np.isfinite(logits)):
            raise RuntimeError("reranker ONNX model returned an unexpected shape")
        return logits.tolist()


def load_installed_backends(
    *,
    model_threads: int = 4,
    manager: MemoryModelManager | None = None,
) -> tuple[EmbeddingBackend | None, RerankerBackend | None]:
    manager = manager or MemoryModelManager()
    # Hash every installed artifact before first use in a process. The manager
    # caches a size/mtime signature and revalidates if any file changes.
    active = manager.active_manifest(verify=True)
    if active is None:
        return None, None
    root, manifest = active
    models = manifest.get("models")
    if not isinstance(models, dict):
        return None, None
    embedding_manifest = models.get("embedding")
    reranker_manifest = models.get("reranker")
    if not isinstance(embedding_manifest, dict) or not isinstance(reranker_manifest, dict):
        return None, None
    bundle_id = str(manifest.get("bundle_id", root.name))
    embedding_key = (bundle_id, str(embedding_manifest.get("fingerprint", "")), model_threads)
    reranker_key = (bundle_id, str(reranker_manifest.get("fingerprint", "")), model_threads)
    with _RUNTIME_LOCK:
        embedding = _RUNTIMES.get(embedding_key)
        if embedding is None:
            embedding = OnnxEmbeddingBackend(
                root,
                embedding_manifest,
                model_threads=model_threads,
            )
            _RUNTIMES[embedding_key] = embedding
        reranker = _RUNTIMES.get(reranker_key)
        if reranker is None:
            reranker = OnnxRerankerBackend(
                root,
                reranker_manifest,
                model_threads=model_threads,
            )
            _RUNTIMES[reranker_key] = reranker
    return embedding, reranker  # type: ignore[return-value]


def golden_inference_check(
    *,
    manager: MemoryModelManager | None = None,
    model_threads: int = 1,
) -> tuple[bool, str]:
    manager = manager or MemoryModelManager()
    try:
        active = manager.active_manifest(verify=True)
    except Exception as exc:
        return False, f"memory model verification failed: {type(exc).__name__}: {exc}"
    if active is None:
        return False, "memory models are not installed"
    root, manifest = active
    golden_path = root / str(manifest["golden_vectors"])
    try:
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"golden vectors are invalid: {type(exc).__name__}"
    try:
        embedding, reranker = load_installed_backends(
            manager=manager,
            model_threads=model_threads,
        )
    except Exception as exc:
        return False, f"model runtime could not be constructed: {type(exc).__name__}: {exc}"
    if embedding is None or reranker is None:
        return False, "model runtime could not be constructed"
    try:
        embedding_case = golden["embedding"]
        actual_vectors = embedding.embed(list(embedding_case["texts"]))
        expected_dimension = embedding_case.get("dimension")
        if expected_dimension is not None and any(
            len(vector) != int(expected_dimension) for vector in actual_vectors
        ):
            return False, "embedding golden dimension mismatch"
        if bool(embedding_case.get("unit_norm", False)):
            norm_tolerance = float(embedding_case.get("tolerance", 0.001))
            for vector in actual_vectors:
                norm = math.sqrt(sum(float(value) ** 2 for value in vector))
                if not math.isfinite(norm) or abs(norm - 1.0) > norm_tolerance:
                    return False, "embedding golden normalization mismatch"
        expected_vectors = embedding_case.get("vectors")
        if expected_vectors is not None:
            tolerance = float(embedding_case.get("tolerance", 0.02))
            for actual, expected in zip(actual_vectors, expected_vectors, strict=True):
                if len(actual) != len(expected):
                    return False, "embedding golden dimension mismatch"
                maximum = max(abs(float(a) - float(b)) for a, b in zip(actual, expected, strict=True))
                if not math.isfinite(maximum) or maximum > tolerance:
                    return False, "embedding golden inference mismatch"
        reranker_case = golden["reranker"]
        actual_scores = reranker.rerank(
            str(reranker_case["query"]),
            list(reranker_case["passages"]),
        )
        expected_scores = reranker_case.get("logits")
        if expected_scores is not None:
            tolerance = float(reranker_case.get("tolerance", 0.05))
            if len(actual_scores) != len(expected_scores):
                return False, "reranker golden count mismatch"
            if any(
                abs(float(actual) - float(expected)) > tolerance
                for actual, expected in zip(actual_scores, expected_scores, strict=True)
            ):
                return False, "reranker golden inference mismatch"
        expected_order = reranker_case.get("expected_order")
        if expected_order is not None:
            order = [int(index) for index in expected_order]
            if any(index < 0 or index >= len(actual_scores) for index in order):
                return False, "reranker golden order is invalid"
            observed = sorted(
                range(len(actual_scores)),
                key=lambda index: (-float(actual_scores[index]), index),
            )
            if observed[: len(order)] != order:
                return False, "reranker golden ordering mismatch"
            minimum_margin = float(reranker_case.get("minimum_margin", 0.0))
            if len(order) >= 2 and (
                float(actual_scores[order[0]]) - float(actual_scores[order[1]])
                < minimum_margin
            ):
                return False, "reranker golden margin mismatch"
    except Exception as exc:
        return False, f"golden inference failed: {type(exc).__name__}: {exc}"
    return True, "checksums and golden inference passed"
