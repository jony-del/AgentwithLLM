"""Reproducible local snapshot/checkpoint/copy benchmark; no LLM or network calls."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_core.checkpoints import CheckpointStore
from agent_core.revisions import RevisionTracker
from agent_core.source_snapshots import SnapshotConfig, disk_path
from agent_core.task_runtime import TaskContract, TaskRun, TaskStore
from agent_core.tools.transaction import JournalStorage
from agent_core.verifier import _copy_source, source_unchanged
from agent_core.verifier_workspace import VerifierWorkspace


def peak_memory_bytes() -> int | None:
    if os.name != "nt":
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
                    *[(name, ctypes.c_size_t) for name in ("peak_working", "working", "peak_paged", "paged",
                                                         "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    return int(counters.peak_working) if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb) else None


def create_corpus(root: Path, count: int, size: int, unique: bool = False) -> None:
    root.mkdir()
    # Eight contents across many paths exercise object deduplication explicitly.
    contents = [(f"value_{i}\n".encode() * (size // 8 + 1))[:size] for i in range(8)]
    for index in range(count):
        path = root / f"module_{index % 16}" / f"f{index:06d}.txt"
        path.parent.mkdir(exist_ok=True)
        if unique:
            pattern = f"value_{index:06d}\n".encode()
            path.write_bytes((pattern * (size // len(pattern) + 1))[:size])
        else:
            path.write_bytes(contents[index % 8])


async def benchmark(base: Path, count: int, size: int, unique: bool = False, progress: bool = False) -> dict:
    root = base / "repo"
    await asyncio.to_thread(create_corpus, root, count, size, unique)
    storage = JournalStorage.local(base / "state", workspace=root)
    checkpoints = CheckpointStore(storage, config=SnapshotConfig())
    tracker = RevisionTracker(root, git_aware=False, snapshots=checkpoints.snapshots)
    phases = {}
    def completed(name):
        if progress:
            print(json.dumps({"phase": name, **phases[name]}), flush=True)
    start = time.perf_counter()
    baseline = await tracker.capture()
    phases["cold_capture"] = {"seconds": time.perf_counter() - start, **tracker.last_metrics}
    completed("cold_capture")
    task = TaskRun(TaskContract("synthetic snapshot benchmark"), baseline)
    state = TaskStore(storage)
    await asyncio.to_thread(state.save, task)
    start = time.perf_counter()
    checkpoint = await asyncio.to_thread(checkpoints.capture, task, baseline)
    phases["checkpoint"] = {"seconds": time.perf_counter() - start}
    completed("checkpoint")
    start = time.perf_counter()
    repeated = await tracker.capture()
    phases["unchanged_strict"] = {"seconds": time.perf_counter() - start, **tracker.last_metrics}
    completed("unchanged_strict")
    assert repeated.snapshot_id == baseline.snapshot_id and tracker.last_metrics["bytes_stored"] == 0
    changed_path = root / "module_0" / "f000000.txt"
    await asyncio.to_thread(changed_path.write_bytes, b"single file changed\n")
    start = time.perf_counter()
    changed = await tracker.capture()
    phases["single_file_change"] = {"seconds": time.perf_counter() - start, **tracker.last_metrics}
    completed("single_file_change")
    assert tracker.last_metrics["new_blobs"] == 1
    copy = base / "copy"
    copy.mkdir()
    start = time.perf_counter()
    materialized = await asyncio.to_thread(checkpoints.snapshots.materialize, changed.snapshot_id,
        changed.workspace, changed.digest, copy, allowed=lambda _: True)
    assert await asyncio.to_thread(source_unchanged, changed, copy)
    phases["materialize_and_verify"] = {"seconds": time.perf_counter() - start, **materialized}
    completed("materialize_and_verify")
    session = SimpleNamespace(workspace=root, task_run=task, checkpoint_store=checkpoints, code_permission_rules=None)
    pool = VerifierWorkspace()
    try:
        for phase in ("lease_cold", "lease_clean_reuse"):
            start = time.perf_counter()
            async with pool.lease(session, changed, _copy_source) as (_, metrics):
                phases[phase] = {"seconds": time.perf_counter() - start, **metrics}
            completed(phase)
        assert phases["lease_clean_reuse"]["files_reused"] == count
    finally:
        await pool.close()
    assert (await asyncio.to_thread(state.load)).baseline.files == baseline.files
    return {
        "schema_version": 1, "captured_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "storage_kind": "JournalStorage.local (no per-object private-root ACL checks)"},
        "corpus": {"files": count, "bytes_per_file": size, "source_bytes": count * size,
                   "unique_initial_contents": count if unique else min(8, count)},
        "phases": phases, "peak_process_memory_bytes": peak_memory_bytes(),
        "task_state_bytes": state.path.stat().st_size,
        "checkpoint_record_bytes": (checkpoints.root / (checkpoint.id + ".json")).stat().st_size,
        "snapshot_manifest_bytes": disk_path(checkpoints.snapshots._manifest_path(baseline.snapshot_id)).stat().st_size,
        "limitations": ["synthetic corpus; content uniqueness specified in corpus metadata", "strict capture still hashes all source bytes",
                        "bytes_stored counts retained new object bodies, excluding staging writes and manifest/pointer overhead",
                        "peak process memory includes corpus generation and Python/runtime metadata",
                        "single sample per phase; not a P95 or end-to-end coding success claim",
                        "lease_cold means first lease; CAS and OS caches may already be warm",
                        "local journal backend; partitioned user-state ACL validation overhead is not measured",
                        "verification-copy measurements include hash checking; no real container/VM measured"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=int, default=512)
    parser.add_argument("--bytes-per-file", type=int, default=4096)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--unique", action="store_true", help="Use distinct contents; requires at least 16 bytes per file")
    parser.add_argument("--progress", action="store_true", help="Print each measured phase as it completes")
    args = parser.parse_args()
    if not 1 <= args.files <= 100_000 or not 1 <= args.bytes_per_file <= 32 * 1024**2 or args.files * args.bytes_per_file > 1024**3:
        parser.error("benchmark corpus exceeds its 100,000-file / 1 GiB safety budget")
    if args.unique and args.bytes_per_file < 16:
        parser.error("--unique requires at least 16 bytes per file")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # All generated source, state and copies are confined to this owned scratch dir.
    with tempfile.TemporaryDirectory(prefix="ss-bench-", dir=output.parent) as directory:
        report = asyncio.run(benchmark(Path(directory), args.files, args.bytes_per_file, args.unique, args.progress))
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "corpus": report["corpus"], "phases": report["phases"],
                      "peak_process_memory_bytes": report["peak_process_memory_bytes"]}))


if __name__ == "__main__":
    main()
