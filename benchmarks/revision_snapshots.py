"""Offline synthetic snapshot comparison. Final certification always rehashes."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_core.revisions import RevisionTracker, git_capture
from agent_core.task_runtime import capture_revision


async def benchmark(root: Path, files: int, size: int, samples: int) -> dict:
    await git_capture(root, "init", "-q")
    for index in range(files):
        path = root / f"module_{index % 16}" / f"file_{index}.txt"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes((f"value_{index}\n".encode() * (size // 4 + 1))[:size])
    await git_capture(root, "add", "--", ".")
    tracker = RevisionTracker(root)
    walk_tracker = RevisionTracker(root, git_aware=False)
    timings = {"walk_full": [], "git_strict": [], "git_warm_hint": [], "walk_strict": [], "walk_warm_hint": []}
    counters = {}
    for _ in range(samples):
        started = time.perf_counter()
        baseline = await asyncio.to_thread(capture_revision, root)
        timings["walk_full"].append(time.perf_counter() - started)
        for key, strict, candidate in (("git_strict", True, tracker), ("git_warm_hint", False, tracker),
                                       ("walk_strict", True, walk_tracker), ("walk_warm_hint", False, walk_tracker)):
            started = time.perf_counter()
            actual = await candidate.capture(strict=strict)
            timings[key].append(time.perf_counter() - started)
            assert actual == baseline
            counters[key] = dict(candidate.last_metrics)
    (root / "module_0" / "file_0.txt").write_bytes(b"changed\n")
    await tracker.capture(strict=False)
    incremental = dict(tracker.last_metrics)
    actual = await tracker.capture()
    assert actual == await asyncio.to_thread(capture_revision, root)
    return {"schema_version": 1, "environment": {"python": platform.python_version(), "platform": platform.platform()},
            "corpus": {"files": files, "bytes_per_file": size, "shape": "synthetic indexed UTF-8 text"},
            "samples": samples, "median_seconds": {key: statistics.median(value) for key, value in timings.items()},
            "last_metrics": counters, "single_file_hint": incremental, "final_strict": tracker.last_metrics,
            "coverage": "same eligible content and matching hashes across all modes",
            "limitations": ["Planning hints may reuse metadata; completion never does", "Git queries have fixed process overhead",
                            "Synthetic local corpus; not a real-world task-success claim"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=int, default=512)
    parser.add_argument("--size", type=int, default=65536)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.files <= 5000 or not 1 <= args.size <= 1024 * 1024 or args.files * args.size > 256 * 1024 * 1024 or not 1 <= args.samples <= 20:
        parser.error("corpus/sample budget exceeded")
    with tempfile.TemporaryDirectory(prefix="polaris-snapshots-") as temporary:
        report = asyncio.run(benchmark(Path(temporary), args.files, args.size, args.samples))
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["median_seconds"]))


if __name__ == "__main__":
    main()
