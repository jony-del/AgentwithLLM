"""Compare exhaustive dense search with USearch ANN on a Polaris v4 index.

Example:
    python benchmarks/memory_ann.py ~/.polaris/indexes/.../memory.sqlite3
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
import time
from pathlib import Path


def _timed(call):
    started = time.perf_counter()
    value = call()
    return value, (time.perf_counter() - started) * 1000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("--max-vectors", type=int, default=100_000)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    import numpy as np
    from usearch.index import Index

    with sqlite3.connect(args.index) as connection:
        rows = list(
            connection.execute(
                """
                SELECT ann_label, embedding, embedding_dim
                FROM chunks
                WHERE embedding IS NOT NULL
                ORDER BY ann_label LIMIT ?
                """,
                (max(1, args.max_vectors),),
            )
        )
    if not rows:
        raise SystemExit("index contains no embeddings")
    dimensions = {int(row[2]) for row in rows}
    if len(dimensions) != 1:
        raise SystemExit("index contains mixed dimensions")
    dimension = dimensions.pop()
    labels = np.asarray([int(row[0]) for row in rows], dtype=np.uint64)
    matrix = np.empty((len(rows), dimension), dtype=np.float32)
    for offset, row in enumerate(rows):
        matrix[offset] = np.frombuffer(row[1], dtype=np.float32, count=dimension)

    rng = np.random.default_rng(args.seed)
    query_count = min(max(1, args.queries), len(rows))
    query_rows = rng.choice(len(rows), size=query_count, replace=False)
    queries = matrix[query_rows] + rng.normal(0.0, 0.01, (query_count, dimension))
    queries = queries.astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    k = min(max(1, args.k), len(rows))

    exact_results: list[set[int]] = []
    exact_ms: list[float] = []
    for query in queries:
        def exact():
            scores = matrix @ query
            selected = np.argpartition(scores, -k)[-k:]
            selected = selected[np.argsort(-scores[selected], kind="stable")]
            return {int(labels[index]) for index in selected}

        result, elapsed = _timed(exact)
        exact_results.append(result)
        exact_ms.append(elapsed)

    with tempfile.TemporaryDirectory(prefix="polaris-ann-benchmark-") as temporary:
        path = Path(temporary) / "benchmark.usearch"
        ann = Index(
            ndim=dimension,
            metric="cos",
            dtype="f16",
            connectivity=32,
            expansion_add=256,
            expansion_search=256,
        )
        _, build_ms = _timed(lambda: ann.add(labels, matrix, threads=0))
        ann.save(str(path))
        ann.reset()
        viewed = Index.restore(str(path), view=True, expansion_search=256)
        if viewed is None:
            raise SystemExit("failed to restore benchmark ANN index")

        candidate_count = min(
            len(rows),
            max(256, k * max(1, args.candidate_multiplier)),
        )
        label_to_row = {int(label): index for index, label in enumerate(labels.tolist())}
        ann_ms: list[float] = []
        recalls: list[float] = []
        for query, expected in zip(queries, exact_results, strict=True):
            def approximate():
                keys = viewed.search(query, candidate_count).keys.tolist()
                positions = np.asarray(
                    [label_to_row[int(key)] for key in keys],
                    dtype=np.int64,
                )
                scores = matrix[positions] @ query
                selected = positions[np.argsort(-scores, kind="stable")[:k]]
                return {int(labels[index]) for index in selected}

            actual, elapsed = _timed(approximate)
            ann_ms.append(elapsed)
            recalls.append(len(actual.intersection(expected)) / k)
        viewed.reset()

    payload = {
        "vectors": len(rows),
        "dimension": dimension,
        "queries": query_count,
        "k": k,
        "candidate_count": candidate_count,
        "build_ms": round(build_ms, 3),
        "exact": {
            "p50_ms": round(float(np.percentile(exact_ms, 50)), 3),
            "p95_ms": round(float(np.percentile(exact_ms, 95)), 3),
        },
        "ann_exact_rescore": {
            "p50_ms": round(float(np.percentile(ann_ms, 50)), 3),
            "p95_ms": round(float(np.percentile(ann_ms, 95)), 3),
            "mean_recall_at_k": round(float(np.mean(recalls)), 6),
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
