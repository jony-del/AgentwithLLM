from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agent_core.process_tree import (
    terminate_pid_tree,
    terminate_process_tree,
    windows_descendant_pids,
)

_TREE_SCRIPT = (
    "import json, os, pathlib, subprocess, sys, time\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
    "pathlib.Path(sys.argv[1]).write_text("
    "json.dumps({'parent': os.getpid(), 'child': child.pid}), encoding='utf-8')\n"
    "time.sleep(60)\n"
)


def _pid_exists(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _write_tree_script(tmp_path: Path) -> tuple[Path, Path]:
    script = tmp_path / "tree.py"
    pid_file = tmp_path / "tree-pids.json"
    script.write_text(_TREE_SCRIPT, encoding="utf-8")
    return script, pid_file


def _read_pid_file(pid_file: Path) -> tuple[int, int] | None:
    if not pid_file.exists():
        return None
    values = json.loads(pid_file.read_text(encoding="utf-8"))
    return int(values["parent"]), int(values["child"])


async def _await_pid_file(pid_file: Path) -> tuple[int, int]:
    for _ in range(150):
        values = _read_pid_file(pid_file)
        if values is not None:
            return values
        await asyncio.sleep(0.02)
    raise AssertionError("tree process did not publish its pid file")


async def _await_processes_to_exit(*pids: int) -> None:
    for _ in range(250):
        if not any(_pid_exists(pid) for pid in pids):
            return
        await asyncio.sleep(0.02)
    assert not any(_pid_exists(pid) for pid in pids)


def _sync_spawn_options() -> dict[str, object]:
    # Match the production spawn discipline so POSIX killpg hits the right group.
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


async def test_terminate_process_tree_kills_parent_and_child(tmp_path: Path) -> None:
    script, pid_file = _write_tree_script(tmp_path)
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(script), str(pid_file), **_sync_spawn_options()
    )
    parent, child = await _await_pid_file(pid_file)

    await terminate_process_tree(proc)

    assert proc.returncode is not None
    await _await_processes_to_exit(parent, child)


async def test_terminate_process_tree_on_dead_process_does_not_raise() -> None:
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "pass", **_sync_spawn_options()
    )
    await proc.wait()

    await terminate_process_tree(proc)
    await terminate_process_tree(proc, grace=0.1)


def test_terminate_pid_tree_kills_parent_and_child(tmp_path: Path) -> None:
    script, pid_file = _write_tree_script(tmp_path)
    proc = subprocess.Popen(
        [sys.executable, str(script), str(pid_file)], **_sync_spawn_options()
    )
    try:
        deadline = time.monotonic() + 5
        pids = _read_pid_file(pid_file)
        while pids is None and time.monotonic() < deadline:
            time.sleep(0.02)
            pids = _read_pid_file(pid_file)
        assert pids is not None, "tree process did not publish its pid file"
        parent, child = pids

        terminate_pid_tree(proc.pid)

        assert proc.wait(timeout=10) is not None
        deadline = time.monotonic() + 5
        while any(_pid_exists(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not any(_pid_exists(pid) for pid in pids)
    finally:
        if proc.returncode is None:
            proc.kill()
            proc.wait()


def test_terminate_pid_tree_on_dead_pid_does_not_raise() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "pass"], **_sync_spawn_options())
    proc.wait()

    terminate_pid_tree(proc.pid)
    terminate_pid_tree(proc.pid, grace=0.01)


@pytest.mark.skipif(os.name != "nt", reason="Toolhelp32 enumeration is Windows-only")
def test_windows_descendant_pids_finds_spawned_child(tmp_path: Path) -> None:
    script, pid_file = _write_tree_script(tmp_path)
    proc = subprocess.Popen(
        [sys.executable, str(script), str(pid_file)], **_sync_spawn_options()
    )
    try:
        deadline = time.monotonic() + 5
        pids = _read_pid_file(pid_file)
        while pids is None and time.monotonic() < deadline:
            time.sleep(0.02)
            pids = _read_pid_file(pid_file)
        assert pids is not None, "tree process did not publish its pid file"
        parent, child = pids

        assert child in windows_descendant_pids(parent)
    finally:
        terminate_pid_tree(proc.pid)
        proc.wait(timeout=10)


@pytest.mark.skipif(os.name == "nt", reason="process-group signalling is POSIX-only")
def test_terminate_pid_tree_posix_signals_process_group(tmp_path: Path) -> None:
    script, pid_file = _write_tree_script(tmp_path)
    proc = subprocess.Popen(
        [sys.executable, str(script), str(pid_file)], start_new_session=True
    )
    try:
        deadline = time.monotonic() + 5
        pids = _read_pid_file(pid_file)
        while pids is None and time.monotonic() < deadline:
            time.sleep(0.02)
            pids = _read_pid_file(pid_file)
        assert pids is not None, "tree process did not publish its pid file"

        terminate_pid_tree(proc.pid, grace=0.1)

        assert proc.wait(timeout=10) is not None
        deadline = time.monotonic() + 5
        while any(_pid_exists(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not any(_pid_exists(pid) for pid in pids)
    finally:
        if proc.returncode is None:
            proc.kill()
            proc.wait()
