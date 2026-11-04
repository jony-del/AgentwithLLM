"""Reproducible local code-index benchmark. No model, network or downloaded repo.

python benchmarks/codeintel.py --lines 10000000 --output result.json
The temporary corpus is retained only with --keep; the output records corpus shape.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_core.codeintel.config import CodeIntelConfig
from agent_core.codeintel.models import ChangeSet, SearchRequest
from agent_core.codeintel.service import CodeIntelligenceService
from provenance import provenance


async def benchmark(root: Path, lines: int, samples: int, index_timeout: float = 1800) -> dict:
    import tracemalloc
    tracemalloc.start()
    workspace = root / "repository"
    workspace.mkdir(parents=True, exist_ok=True)
    generation_start = time.perf_counter()
    file_count = (lines + 999) // 1000
    actual_lines = 0
    for i in range(file_count):
        count = min(1000, lines - actual_lines)
        directory = workspace / f"module_{i % 32:02d}"
        directory.mkdir(exist_ok=True)
        # Real Python syntax: one function + assignments. Unique identifiers allow exact tests.
        body = [f"def unique_function_{i}():", "    return 42"]
        body.extend(f"value_{j} = {j}" for j in range(max(0, count - 2)))
        (directory / f"source_{i:06d}.py").write_text("\n".join(body) + "\n", encoding="utf-8")
        actual_lines += len(body)
    config = replace(CodeIntelConfig(), watch=False, query_seconds=2, max_files=500, max_candidates=2000)
    service = CodeIntelligenceService(workspace, config, database=root / "index" / "code.sqlite3")
    started = time.perf_counter()
    slices = 0
    while True:
        status = await service.ensure_index()
        slices += 1
        if slices % 10 == 0:
            print(json.dumps({"phase": "index", "files": status["files"], "pending": status["pending"],
                              "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
        if status["catalog_complete"] and status["pending"] == 0:
            break
        if time.perf_counter() - started > index_timeout:
            raise RuntimeError(f"benchmark indexing exceeded {index_timeout:.0f} seconds")
    build_seconds = time.perf_counter() - started
    _, peak_build = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    timings: dict[str, list[float]] = {"symbol": [], "text": [], "no_match": []}
    last_pages = {}
    for _ in range(samples):
        for label, request in (
            ("symbol", SearchRequest("unique_function_7", kind="symbol", modules=("module_07",))),
            ("text", SearchRequest("unique_function_7", kind="text", modules=("module_07",))),
            ("no_match", SearchRequest("nonexistent_identifier_aaaaaaaa", kind="text", modules=("module_07",))),
        ):
            tick = time.perf_counter()
            page = await service.search(request)
            timings[label].append(time.perf_counter() - tick)
            last_pages[label] = {"hits": len(page.hits), "usage": page.usage, "coverage": page.to_dict()["coverage"]}
    target = workspace / "module_07" / "source_000007.py"
    target.write_text("def changed_function():\n    return 43\n", encoding="utf-8")
    tick = time.perf_counter()
    await service.publish_changes(ChangeSet("benchmark-edit", ("module_07/source_000007.py",)))
    incremental = await service.ensure_index()
    incremental_seconds = time.perf_counter() - tick
    assert (await service.search(SearchRequest("changed_function", kind="symbol"))).hits
    assert not (await service.search(SearchRequest("unique_function_7", kind="symbol"))).hits
    result = {"schema_version": 1, "python": sys.version, "platform": sys.platform,
              "corpus": {"lines": actual_lines, "files": file_count, "modules": 32,
                         "language": "python", "shape": "function + integer assignments; synthetic, not a real monorepo"},
              "generation_seconds": started - generation_start, "build_seconds": build_seconds,
              "index_bytes": sum(p.stat().st_size for p in (root / "index").iterdir() if p.is_file()),
              "peak_memory_bytes": {"build": peak_build},
              "query_seconds": {kind: {"median": statistics.median(values),
                "p95": sorted(values)[min(len(values) - 1, int(len(values) * .95))]} for kind, values in timings.items()},
              "queries": last_pages, "incremental_seconds": incremental_seconds,
              "incremental_usage": incremental["usage"], "config": {"query_seconds": config.query_seconds,
               "max_files": config.max_files, "max_bytes": config.max_bytes},
              "provenance": provenance(Path(__file__).resolve().parents[1],
                  {"query_seconds": config.query_seconds, "max_files": config.max_files,
                   "max_bytes": config.max_bytes, "max_candidates": config.max_candidates,
                   "watch": config.watch}, seed=None)}
    await service.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lines", type=int, default=100000)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep", type=Path)
    parser.add_argument("--index-timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.lines < 8000 or args.samples < 1:
        parser.error("at least 8000 lines and one sample are required")
    if args.keep:
        result = asyncio.run(benchmark(args.keep, args.lines, args.samples, args.index_timeout))
        directory = None
    else:
        directory = Path(tempfile.mkdtemp(prefix="polaris-codeintel-benchmark-"))
        result = asyncio.run(benchmark(directory, args.lines, args.samples, args.index_timeout))
    # Persist the measurement before scratch cleanup so a Windows file lock
    # can never lose a completed benchmark.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if directory is not None:
        import shutil
        for attempt in range(5):
            try:
                shutil.rmtree(directory)
                break
            except OSError:
                time.sleep(1)
        else:
            print(f"warning: scratch cleanup deferred by a lingering file lock: {directory}", file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
