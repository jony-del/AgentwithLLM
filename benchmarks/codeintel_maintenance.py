"""Offline synthetic index-maintenance measurements; no model or network calls."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_core.codeintel.config import CodeIntelConfig
from agent_core.codeintel.models import ChangeSet, SearchRequest
from agent_core.codeintel.service import CodeIntelligenceService
import agent_core.codeintel.indexer as indexer
from provenance import provenance


def distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    return {"samples": values, "count": len(values), "p50": statistics.median(values),
            "p95": ordered[math.ceil(len(ordered) * 0.95) - 1], "unit": "seconds"}


async def benchmark(base: Path, files: int, lines: int, samples: int) -> dict:
    import tracemalloc
    tracemalloc.start()
    peak: dict[str, int] = {}

    def mark(phase: str) -> None:
        _, peak[phase] = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()

    root = base / "repo"
    root.mkdir()
    for number in range(files):
        name = "stable_probe_symbol" if number == 0 else f"function_{number:06d}"
        body = [f"def {name}():", *[f"    value_{i} = {number + i}" for i in range(lines - 2)],
                f"    return value_{lines - 3}"]
        (root / f"f{number:06d}.py").write_text("\n".join(body) + "\n", encoding="utf-8")
    config = CodeIntelConfig(maintain=False, watch=False)
    service = CodeIntelligenceService(root, config, database=base / "index" / "code.sqlite3")
    original = indexer.python_structure
    count = 0
    guard = threading.Lock()

    def counted(text):
        nonlocal count
        with guard:
            count += 1
        return original(text)

    async def drain(*, audit=False):
        for iteration in range(10000):
            status = await service.ensure_index(audit=audit and iteration == 0)
            if status["catalog_complete"] and not status["pending"]:
                return iteration + 1
            if status.get("stop_reason") == "index_disk_quota":
                raise RuntimeError("benchmark index exceeded disk quota")
        raise RuntimeError("index did not finish within 10000 bounded maintenance slices")

    async def truth():
        page = await service.search(SearchRequest("stable_probe_symbol", kind="symbol"))
        assert [(hit.path, hit.start_line) for hit in page.hits] == [("f000000.py", 1)], page.to_dict()

    indexer.python_structure = counted
    try:
        start = time.perf_counter()
        slices = await drain()
        build = {"seconds": time.perf_counter() - start, "ast_parse_count": count, "slices": slices}
        mark("build")
        await truth()
        before, start = count, time.perf_counter()
        slices = await drain(audit=True)
        audit = {"seconds": time.perf_counter() - start, "ast_parse_count": count - before, "slices": slices}
        mark("audit")
        await truth()
        increment_times, increment_counts, increment_slices = [], [], []
        for sample in range(samples):
            path = f"f{1 + sample % (files - 1):06d}.py"
            source = root / path
            source.write_text(source.read_text(encoding="utf-8") + f"change_{sample} = {sample}\n", encoding="utf-8")
            before, start = count, time.perf_counter()
            await service.publish_changes(ChangeSet(f"increment-{sample}", (path,), "benchmark"))
            increment_slices.append(await drain())
            increment_times.append(time.perf_counter() - start)
            increment_counts.append(count - before)
        await truth()
        query_times, maintenance_times, overlap = [], [], []
        for sample in range(samples):
            path = f"f{files - 1:06d}.py"
            source = root / path
            source.write_text(source.read_text(encoding="utf-8") + f"concurrent_{sample} = {sample}\n", encoding="utf-8")
            await service.publish_changes(ChangeSet(f"concurrent-{sample}", (path,), "benchmark"))
            window = {}
            async def maintain():
                window["start"] = time.perf_counter()
                await drain()
                window["end"] = time.perf_counter()
            task = asyncio.create_task(maintain())
            await asyncio.sleep(0)
            query_start = time.perf_counter()
            try:
                await truth()
            finally:
                query_end = time.perf_counter()
                await task
            query_times.append(query_end - query_start)
            maintenance_times.append(window["end"] - window["start"])
            overlap.append(query_start < window["end"] and query_end > window["start"])
        status = await service.status()
        mark("queries")
        index_dir = base / "index"
        return {"schema_version": 1, "captured_at": datetime.now(timezone.utc).isoformat(),
                "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                "processor": platform.processor(), "cpu_count": os.cpu_count()},
                "provenance": provenance(Path(__file__).resolve().parents[1], asdict(config), seed=None),
                "config": asdict(config), "corpus": {"files": files, "initial_lines_per_file": lines,
                    "initial_loc": files * lines, "samples": samples},
                "build": build, "unchanged_audit": audit,
                "single_file_publish_and_drain": {**distribution(increment_times),
                    "ast_parse_counts": increment_counts, "maintenance_slices": increment_slices},
                "query_during_maintenance": {**distribution(query_times), "overlap_observed": overlap,
                    "maintenance_seconds": maintenance_times},
                "peak_memory_bytes": peak,
                "index_disk_bytes": sum(p.stat().st_size for p in index_dir.iterdir() if p.is_file()),
                "final_status": status,
                "limitations": ["synthetic Python-only workload; no LLM, network or real coding-task execution",
                    "P95 uses nearest-rank; sample count is explicit; OS and SQLite caches are not flushed",
                    "concurrency overlap describes async operation lifetimes, not simultaneous CPU execution",
                    "AST calls are counted by temporarily wrapping indexer.python_structure, then restoring it",
                    "query truth checks one unchanged file and current source locations, not corpus-wide Recall@K",
                    "peak_memory_bytes is tracemalloc Python allocation per phase, not process RSS"]}
    finally:
        indexer.python_structure = original
        await service.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=int, default=128)
    parser.add_argument("--lines-per-file", type=int, default=200)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--allow-large", action="store_true",
                        help="explicitly permit corpora beyond 20000 files / 2M initial lines")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    max_files, max_loc = (60000, 12_000_000) if args.allow_large else (20000, 2_000_000)
    if not 2 <= args.files <= max_files or not 3 <= args.lines_per_file <= 10000 or args.files * args.lines_per_file > max_loc:
        parser.error(f"corpus must have 2..{max_files} files, 3..10000 lines/file, at most {max_loc} initial lines"
                     + ("" if args.allow_large else "; pass --allow-large to extend the quota explicitly"))
    if not 10 <= args.samples <= 100:
        parser.error("--samples must be between 10 and 100")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="maintenance-bench-", dir=output.parent))
    try:
        report = asyncio.run(benchmark(Path(directory), args.files, args.lines_per_file, args.samples))
    finally:
        # Windows may briefly hold SQLite handles; never let cleanup eat the report.
        import shutil
        for attempt in range(5):
            try:
                shutil.rmtree(directory)
                break
            except OSError:
                time.sleep(1)
        else:
            print(f"warning: scratch cleanup deferred by a lingering file lock: {directory}", file=sys.stderr)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "build": report["build"], "unchanged_audit": report["unchanged_audit"],
                      "increment_p95": report["single_file_publish_and_drain"]["p95"],
                      "concurrent_query_p95": report["query_during_maintenance"]["p95"]}))


if __name__ == "__main__":
    main()
