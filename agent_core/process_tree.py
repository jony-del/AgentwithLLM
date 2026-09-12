"""Shared process-tree termination for hooks, supervisors, and helper subprocesses.

Windows walks a Toolhelp32 snapshot so descendants die leaves-first through native
``TerminateProcess`` handles; ``taskkill /PID /T /F`` is only a fallback, because
plain ``taskkill /T`` races the root's exit and can orphan grandchildren that hold
inherited pipe handles open. POSIX targets the child's process group (the spawn
sites use ``start_new_session``): SIGTERM, a grace window, then SIGKILL.

Invariants (aligned with the project's timeout / degrade discipline):

* Termination is **best-effort and never raises** for an already-dead PID — a
  cleanup path must not mask the error it is cleaning up after.
* Every kill is followed by a bounded reap so no zombie or inherited pipe handle
  holds the caller's timeout path open.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import time


def windows_descendant_pids(root_pid: int) -> list[int]:
    """Snapshot descendant PIDs before the root exits (``taskkill /T`` race guard)."""

    if os.name != "nt":
        return []
    try:
        import ctypes
        from ctypes import wintypes

        class ProcessEntry(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", wintypes.LONG),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", wintypes.WCHAR * 260),
            ]

        kernel32 = ctypes.windll.kernel32
        kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
        kernel32.Process32FirstW.restype = wintypes.BOOL
        kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
        kernel32.Process32NextW.restype = wintypes.BOOL
        snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
        if snapshot in {None, wintypes.HANDLE(-1).value}:
            return []
        children: dict[int, list[int]] = {}
        entry = ProcessEntry()
        entry.dwSize = ctypes.sizeof(entry)
        try:
            present = bool(kernel32.Process32FirstW(snapshot, ctypes.byref(entry)))
            while present:
                children.setdefault(int(entry.th32ParentProcessID), []).append(
                    int(entry.th32ProcessID)
                )
                present = bool(kernel32.Process32NextW(snapshot, ctypes.byref(entry)))
        finally:
            kernel32.CloseHandle(snapshot)
        descendants: list[int] = []
        pending = list(children.get(root_pid, []))
        while pending:
            pid = pending.pop()
            descendants.append(pid)
            pending.extend(children.get(pid, []))
        return descendants
    except (AttributeError, OSError, ValueError):
        return []


def windows_terminate_pid(pid: int) -> bool:
    """Terminate one same-user process through a native handle and wait for exit."""

    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateProcess.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x0001 | 0x00100000, False, pid)
        if not handle:
            return False
        try:
            terminated = bool(kernel32.TerminateProcess(handle, 1))
            if terminated:
                kernel32.WaitForSingleObject(handle, 3000)
            return terminated
        finally:
            kernel32.CloseHandle(handle)
    except (AttributeError, OSError, ValueError):
        return False


def terminate_pid_tree(pid: int, *, grace: float = 0.0) -> None:
    """Synchronously kill the tree rooted at ``pid``; safe to call on a dead PID."""

    if os.name == "nt":
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            descendants = windows_descendant_pids(pid)
            # Kill leaves first. If taskkill terminates the root before walking
            # its children, the saved PID list still closes inherited pipe handles.
            for item in [*reversed(descendants), pid]:
                if windows_terminate_pid(item):
                    continue
                subprocess.run(
                    ["taskkill", "/PID", str(item), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=3,
                )
    else:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            getattr(os, "killpg")(pid, signal.SIGTERM)
        if grace > 0:
            time.sleep(grace)
        with contextlib.suppress(ProcessLookupError, PermissionError):
            getattr(os, "killpg")(pid, getattr(signal, "SIGKILL", 9))


async def terminate_process_tree(proc: asyncio.subprocess.Process, *, grace: float = 0.0) -> None:
    """Best-effort cleanup on success, timeout, cancellation, and pipe failure."""

    if os.name == "nt":
        if proc.returncode is None:
            await asyncio.to_thread(terminate_pid_tree, proc.pid)
    else:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            getattr(os, "killpg")(proc.pid, signal.SIGTERM)
        if grace > 0:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=grace)
        else:
            # One scheduler tick: lets the group shut down before SIGKILL lands.
            await asyncio.sleep(0)
        with contextlib.suppress(ProcessLookupError, PermissionError):
            getattr(os, "killpg")(proc.pid, getattr(signal, "SIGKILL", 9))
    if proc.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), timeout=1)
