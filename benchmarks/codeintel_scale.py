"""Synthetic large-repository scale matrix: 1M/10M LOC and high-file-count cells.

No model, network or downloaded repository. Every cell records raw query samples,
P50/P95 latency, Recall/Precision/MRR against generation-known gold, peak Python
allocation (tracemalloc, not RSS), index disk bytes and provenance (source SHA,
config hash, seed). Token/tool-call and patch/test metrics are not measurable
offline and are reported as null rather than inferred.

Example: python benchmarks/codeintel_scale.py --cells 1m-low,1m-high --samples 50 --rebuilds 3 --output tmp/scale.json
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import random
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

K = 20

CELLS = {
    # name: (lines, files, content mode, deep modules, multilingual manifests, allowed-policy)
    "1m-low": (1_000_000, 1000, "mixed", False, True, False),
    "1m-high": (1_000_000, 21000, "mixed", True, True, False),
    "1m-duplicate": (1_000_000, 1000, "duplicate", False, False, False),
    "1m-acl": (1_000_000, 1000, "mixed", False, True, True),
    "10m-low": (10_000_000, 10000, "mixed", True, True, False),
    "10m-high": (10_000_000, 21000, "mixed", True, True, False),
}


def distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    return {"samples": [round(v, 6) for v in values], "count": len(values),
            "p50": statistics.median(values) if values else None,
            "p95": ordered[math.ceil(len(ordered) * 0.95) - 1] if values else None, "unit": "seconds"}


def ranked_metrics(ranked: list[str], gold: set[str], k: int = K) -> dict[str, float]:
    top = list(dict.fromkeys(ranked))[:k]
    hits = [i + 1 for i, path in enumerate(top) if path in gold]
    retrieved = len({p for p in ranked if p in gold})
    return {"recall_at_k": len(hits) / len(gold) if gold else None,
            "recall_retrieved": retrieved / len(gold) if gold else None,
            "precision_at_k": len(hits) / k, "mrr": 1 / hits[0] if hits else 0}


def generate(root: Path, lines: int, files: int, mode: str, deep: bool, manifests: bool,
             rng: random.Random) -> dict:
    """Seeded synthetic corpus with generation-known gold for truth checks."""
    per_file = max(3, lines // files)
    gold_symbols: dict[str, list[str]] = {}
    gold_text: dict[str, list[str]] = {}
    shared_pool = [f"shared_handler_{i}" for i in range(40)]
    marker_pool = [f"needle_marker_{i}" for i in range(40)]
    for i in range(files):
        if deep:
            directory = root / f"pkg_{i % 64:02d}" / f"sub_{(i // 64) % 16:02d}" / f"leaf_{(i // 1024) % 8}"
        else:
            directory = root / f"module_{i % 32:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"source_{i:06d}.py"
        if mode == "unique" or (mode == "mixed" and i % 2 == 0):
            function = f"unique_function_{i}"
            marker = f"unique_marker_{i} = {i}"
        else:
            function = shared_pool[i % len(shared_pool)]
            marker = f"{marker_pool[i % len(marker_pool)]} = {i}"
        body = [f"def {function}():", f"    {marker}", f"    return {i}"]
        body.extend(f"value_{j} = {j}" for j in range(per_file - len(body)))
        path.write_text("\n".join(body) + "\n", encoding="utf-8")
        relative = path.relative_to(root).as_posix()
        gold_symbols.setdefault(function, []).append(relative)
        gold_text.setdefault(marker.split(" =")[0], []).append(relative)
    if manifests:
        for i, manifest in enumerate(("package.json", "pyproject.toml", "Cargo.toml", "go.mod")):
            directory = root / f"pkg_{i:02d}"
            directory.mkdir(exist_ok=True)
            if manifest == "package.json":
                text = json.dumps({"name": f"pkg-{i}", "dependencies": {"left-pad": "1.0.0"}})
            elif manifest == "pyproject.toml":
                text = f'[project]\nname = "pkg-{i}"\ndependencies = ["requests>=2"]\n'
            elif manifest == "Cargo.toml":
                text = f'[package]\nname = "pkg-{i}"\n[dependencies]\nserde = "1"\n'
            else:
                text = f"module example.com/pkg{i}\n\ngo 1.22\n\nrequire github.com/pkg/errors v0.9.1\n"
            (directory / manifest).write_text(text, encoding="utf-8")
    return {"symbols": gold_symbols, "text": gold_text}


async def drain(service: CodeIntelligenceService, timeout: float = 3600) -> dict:
    started = time.perf_counter()
    slices = 0
    while True:
        status = await service.ensure_index()
        slices += 1
        if status["catalog_complete"] and not status["pending"] and not status.get("replay_pending"):
            return {"seconds": time.perf_counter() - started, "slices": slices, "final": status}
        if time.perf_counter() - started > timeout:
            raise RuntimeError("scale benchmark indexing exceeded its deadline")


async def run_queries(service: CodeIntelligenceService, gold: dict, samples: int,
                      rng: random.Random) -> dict:
    symbol_names = sorted(gold["symbols"])
    text_markers = sorted(gold["text"])
    workload = []
    for i in range(samples):
        kind = ("symbol", "text", "no_match")[i % 3]
        if kind == "symbol":
            query = symbol_names[rng.randrange(len(symbol_names))]
        elif kind == "text":
            query = text_markers[rng.randrange(len(text_markers))]
        else:
            query = f"absent_identifier_{i}"
        workload.append((kind, query))
    results = {"symbol": [], "text": [], "no_match": []}
    metric_rows = []
    for kind, query in workload:
        tick = time.perf_counter()
        ranked: list[str] = []
        pages = 0
        cursor = None
        first_page_recall = None
        while True:
            page = await service.search(SearchRequest(query, kind="text" if kind == "no_match" else kind,
                                                      cursor=cursor, limit=100))
            ranked.extend(hit.path for hit in page.hits)
            pages += 1
            if pages == 1 and kind != "no_match":
                wanted = set(gold["symbols" if kind == "symbol" else "text"].get(query, []))
                first_page_recall = ranked_metrics([h.path for h in page.hits], wanted)["recall_at_k"]
            cursor = page.cursor
            if not cursor or pages >= 4:
                break
        elapsed = time.perf_counter() - tick
        results[kind].append(elapsed)
        wanted = set(gold["symbols" if kind == "symbol" else "text"].get(query, []))
        if kind != "no_match":
            metric_rows.append({"kind": kind, "query": query, "elapsed": elapsed, "pages": pages,
                                "first_page_recall_at_k": first_page_recall,
                                **ranked_metrics(ranked, wanted)})
        elif page.hits:
            metric_rows.append({"kind": kind, "query": query, "elapsed": elapsed, "error": "unexpected_hits"})
    aggregated = {}
    for kind in ("symbol", "text"):
        rows = [r for r in metric_rows if r["kind"] == kind and "recall_at_k" in r]
        if rows:
            aggregated[kind] = {key: sum(r[key] for r in rows) / len(rows)
                                for key in ("recall_at_k", "recall_retrieved", "precision_at_k", "mrr")}
            aggregated[kind]["first_page_recall_at_k"] = sum(r["first_page_recall_at_k"] for r in rows) / len(rows)
            aggregated[kind]["pages_p95"] = sorted(r["pages"] for r in rows)[max(0, math.ceil(len(rows) * 0.95) - 1)]
    return {"latency": {kind: distribution(values) for kind, values in results.items()},
            "metrics": aggregated, "rows": metric_rows,
            "unexpected_no_match_hits": [r for r in metric_rows if "error" in r],
            "samples": len(workload),
            "note": "metrics paginate up to 4 bounded pages; first_page_recall_at_k shows the single-page budget-truncated view"}


async def run_cell(base: Path, name: str, spec: tuple, samples: int, rebuilds: int, seed: int) -> dict:
    lines, files, mode, deep, manifests, acl = spec
    import tracemalloc
    tracemalloc.start()
    cell: dict = {"cell": name, "spec": {"lines": lines, "files": files, "content": mode,
                  "deep_modules": deep, "manifests": manifests, "restricted_policy": acl},
                  "rebuilds": [], "samples_requested": samples}
    for rebuild in range(rebuilds):
        rng = random.Random(seed + rebuild)
        root = base / f"repo-{name}-{rebuild}"
        root.mkdir()
        tick = time.perf_counter()
        gold = generate(root, lines, files, mode, deep, manifests, rng)
        generation_seconds = time.perf_counter() - tick
        mark_config = replace(CodeIntelConfig(), watch=False, maintain=False)
        database = base / f"index-{name}-{rebuild}" / "code.sqlite3"
        allowed = (lambda path: not path.startswith("pkg_00/")) if acl else None
        service = CodeIntelligenceService(root, mark_config, database=database, allowed=allowed)
        try:
            build = await drain(service)
            _, peak_build = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            queries = await run_queries(service, gold, samples, rng)
            _, peak_query = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            # Mutations: modify, delete and timestamp-preserving rewrites, then a publish burst.
            mutated = sorted(root.rglob("source_*.py"))[:8]
            for path in mutated[:3]:
                path.write_text(path.read_text(encoding="utf-8") + "\nchanged = 1\n", encoding="utf-8")
            for path in mutated[3:5]:
                path.unlink()
            for path in mutated[5:]:
                before = path.stat()
                path.write_text(path.read_text(encoding="utf-8") + "\nstealth = 1\n", encoding="utf-8")
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            tick = time.perf_counter()
            for burst in range(8):
                await service.publish_changes(ChangeSet(f"burst-{rebuild}-{burst}",
                    tuple(p.relative_to(root).as_posix() for p in mutated), "benchmark"))
            incremental = await drain(service)
            incremental_seconds = time.perf_counter() - tick
            # Concurrent queries during maintenance, then cancel/close and resume checks.
            for path in mutated:
                if path.exists():
                    path.write_text(path.read_text(encoding="utf-8") + "\nagain = 1\n", encoding="utf-8")
                    await service.publish_changes(ChangeSet(f"c-{rebuild}", (path.relative_to(root).as_posix(),), "benchmark"))
                    break
            task = asyncio.create_task(drain(service))
            await asyncio.sleep(0)
            concurrent_query = await service.search(SearchRequest("value_0", kind="text", limit=5))
            await task
            maintenance = asyncio.create_task(service.ensure_index())
            await asyncio.sleep(0)
            maintenance.cancel()
            try:
                await maintenance
            except asyncio.CancelledError:
                pass
            recovered = await service.ensure_index()
            cell["rebuilds"].append({
                "rebuild": rebuild, "generation_seconds": generation_seconds,
                "build": build, "queries": queries, "incremental_seconds": incremental_seconds,
                "incremental": incremental, "concurrent_hits": len(concurrent_query.hits),
                "post_cancel_stop_reason": recovered.get("stop_reason"),
                "peak_memory_bytes": {"build": peak_build, "queries": peak_query},
                "index_disk_bytes": sum(p.stat().st_size for p in database.parent.iterdir() if p.is_file())})
        finally:
            await service.close()
        # Resume: a fresh service instance on the same database must answer immediately.
        resumed = CodeIntelligenceService(root, mark_config, database=database, allowed=allowed)
        try:
            page = await resumed.search(SearchRequest("value_0", kind="text", limit=5))
            cell["rebuilds"][-1]["resume_hits"] = len(page.hits)
        finally:
            await resumed.close()
    return cell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", default="1m-low",
                        help="comma-separated cells from: " + ",".join(CELLS))
    parser.add_argument("--samples", type=int, default=51)
    parser.add_argument("--rebuilds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 50 or not 1 <= args.rebuilds <= 5:
        parser.error("the matrix requires at least 50 query samples per cell and 1..5 rebuilds")
    selected = []
    for name in args.cells.split(","):
        if name.strip() not in CELLS:
            parser.error(f"unknown cell {name!r}; choose from {sorted(CELLS)}")
        selected.append(name.strip())
    started = time.perf_counter()
    cells = []
    scratch = Path(tempfile.mkdtemp(prefix="codeintel-scale-", dir=args.output.parent))
    for name in selected:
        print(json.dumps({"phase": "cell", "cell": name}), flush=True)
        cells.append(asyncio.run(run_cell(Path(scratch), name, CELLS[name], args.samples, args.rebuilds, args.seed)))
    report = {"schema_version": 1, "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "total_seconds": time.perf_counter() - started, "cells": cells,
              "provenance": provenance(Path(__file__).resolve().parents[1],
                  {"samples": args.samples, "rebuilds": args.rebuilds, "cells": selected}, seed=args.seed),
              "costs": {"input_tokens": None, "output_tokens": None, "tool_calls": None, "llm_calls": None,
                        "patch_applied": None, "tests_passed": None},
              "limitations": [
                  "synthetic seeded corpora with generation-known gold, not a heterogeneous real monorepo",
                  "token/tool-call/patch/test metrics are not measurable offline and stay null",
                  "peak memory is tracemalloc Python allocation per phase, not process RSS",
                  "transaction-overlay and verifier-workspace costs are covered by their own benchmarks",
                  "P95 uses nearest-rank over the recorded per-query samples"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    import shutil
    for _attempt in range(5):
        try:
            shutil.rmtree(scratch)
            break
        except OSError:
            time.sleep(1)
    else:
        print(f"warning: scratch cleanup deferred by a lingering file lock: {scratch}", file=sys.stderr)
    print(json.dumps({"report": str(args.output), "cells": selected,
                      "total_seconds": report["total_seconds"]}))


if __name__ == "__main__":
    main()
