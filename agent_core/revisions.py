"""Git-aware revision inventory and hint caching; completion always rehashes content."""
from __future__ import annotations

import asyncio
from contextlib import nullcontext
from dataclasses import replace
import logging
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import stat
import time
import threading
from typing import Any, Callable

from agent_core.execution import current_execution_scope
from agent_core.permission_safety import is_secret_path
from agent_core.process_supervisor import safe_process_environment
from agent_core.process_tree import terminate_process_tree
from agent_core.task_runtime import WorkspaceRevision, _IGNORED, _ROOT_STATE
from agent_core.source_snapshots import SnapshotConfig, SnapshotStore, content_digest, stream_hash


async def git_capture(workspace: Path, *arguments: str, limit: int = 4 * 1024 * 1024, timeout: float = 10) -> bytes:
    """Internal metadata queries: bounded pipes, shared deadline and kill-and-await."""
    executable = shutil.which("git")
    if executable is None:
        raise RuntimeError("git unavailable")
    scope = current_execution_scope()
    env = safe_process_environment()
    env.update(LC_ALL="C", GIT_TERMINAL_PROMPT="0")
    process = await asyncio.create_subprocess_exec(executable, "-c", "core.fsmonitor=false", "-c", "core.hooksPath=" + os.devnull, *arguments,
        cwd=workspace, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env,
        start_new_session=os.name != "nt")

    async def read(pipe, maximum):
        result = bytearray()
        while block := await pipe.read(65536):
            result.extend(block)
            if len(result) > maximum:
                raise ValueError("git metadata output budget exceeded")
        return bytes(result)

    async def collect():
        readers = [asyncio.create_task(read(process.stdout, limit)), asyncio.create_task(read(process.stderr, 65536))]
        try:
            output, error = await asyncio.gather(*readers)
            await process.wait()
            if process.returncode:
                raise RuntimeError("git metadata query failed: " + error.decode("utf-8", errors="replace")[:512])
            return output
        finally:
            for reader in readers:
                reader.cancel()
            await asyncio.gather(*readers, return_exceptions=True)

    try:
        if scope is not None:
            return await scope.run_awaitable(collect(), timeout=timeout)
        return await asyncio.wait_for(collect(), timeout)
    except BaseException:
        await terminate_process_tree(process)
        await process.wait()
        raise


class RevisionTracker:
    def __init__(self, workspace: Path, *, excluded: tuple[Path, ...] = (), git_aware: bool = True,
                 config: SnapshotConfig | None = None, snapshots: SnapshotStore | None = None,
                 allowed: Callable[[str], bool] | None = None) -> None:
        self.workspace = workspace.resolve()
        self.excluded = set(path.resolve() for path in excluded)
        self.git_aware = git_aware
        self.config = config or (snapshots.config if snapshots is not None else SnapshotConfig())
        self.snapshots = snapshots if self.config.enabled else None
        self.allowed = allowed
        self._object_hints: dict[str, str] | None = None
        self._cancelled = threading.Event()
        self.cache: dict[str, tuple[tuple[int, ...], str]] = {}
        self.last_metrics: dict[str, Any] = {}
        self.totals = {"captures": 0, "bytes_hashed": 0, "files_reused": 0}
        self._lock = asyncio.Lock()

    def _eligible(self, relative: str, *, tracked: bool = False) -> bool:
        path = PurePosixPath(relative)
        if not path.parts or path.is_absolute() or PureWindowsPath(relative).drive or ".." in path.parts or ".." in PureWindowsPath(relative).parts:
            raise ValueError("unsafe revision inventory path")
        if is_secret_path(relative) or any(part in _IGNORED for part in path.parts) or path.parts[0] in _ROOT_STATE:
            return False
        candidate = self.workspace / relative
        if any(candidate == root or candidate.is_relative_to(root) for root in self.excluded):
            return False
        if tracked:
            return True
        return not any(part in _IGNORED for part in path.parts) and path.parts[0] not in _ROOT_STATE

    def _walk(self) -> list[str]:
        paths = []
        def failed(error):
            raise error
        for directory, dirs, names in os.walk(self.workspace, onerror=failed, followlinks=False):
            self._check()
            dirs[:] = sorted(name for name in dirs if self._eligible((Path(directory) / name).relative_to(self.workspace).as_posix()) and
                             not (Path(directory) / name).is_symlink() and
                             not (hasattr(Path, "is_junction") and (Path(directory) / name).is_junction()))
            for name in sorted(names):
                self._check()
                relative = (Path(directory) / name).relative_to(self.workspace).as_posix()
                if self._eligible(relative):
                    paths.append(relative)
                    if len(paths) > self.config.max_files:
                        raise ValueError("revision file budget exceeded")
        return sorted(paths)

    async def _inventory(self) -> tuple[list[str], str]:
        if self.git_aware:
            try:
                tracked = await git_capture(self.workspace, "ls-files", "--cached", "-z", "--", limit=self.config.max_git_output_bytes)
                others = await git_capture(self.workspace, "ls-files", "--others", "--exclude-standard", "-z", "--", limit=self.config.max_git_output_bytes)
            except (OSError, RuntimeError) as exc:
                # A walk is a conservative superset, never an empty success result.
                logging.getLogger(__name__).info("Git inventory unavailable; using bounded walk: %s", exc)
                return await self._worker(self._walk), "walk-fallback"
            paths = set()
            for payload, is_tracked in ((tracked, True), (others, False)):
                for raw in payload.split(b"\0"):
                    if raw:
                        relative = os.fsdecode(raw)
                        if self._eligible(relative, tracked=is_tracked):
                            paths.add(relative)
            if len(paths) > self.config.max_files:
                raise ValueError("revision file budget exceeded")
            # Deleted tracked files remain in git's inventory, but not in the snapshot.
            def existing():
                result = []
                for path in sorted(paths):
                    self._check()
                    candidate = self.workspace / path
                    if candidate.exists() or candidate.is_symlink():
                        result.append(path)
                return result
            return await self._worker(existing), "git"
        return await self._worker(self._walk), "walk"

    def _check(self) -> None:
        if self._cancelled.is_set():
            raise RuntimeError("revision capture cancelled")
        scope = current_execution_scope()
        if scope is not None:
            scope.raise_if_cancelled()

    def _hash(self, paths: list[str], strict: bool) -> tuple[WorkspaceRevision, dict, tuple[dict, dict, dict, dict]]:
        files: dict[str, str] = {}
        updated, stats, directories, modes = {}, {}, {}, {}
        total = hashed = reused = 0
        store = self.snapshots if strict else None
        if store is not None and self._object_hints is None:
            self._object_hints = store.latest_files(str(self.workspace))
        with store.batch(check=self._check) if store is not None else nullcontext() as batch:
            for relative in paths:
                self._check()
                path = self.workspace / relative
                for parent in (self.workspace, *reversed(path.relative_to(self.workspace).parents)):
                    directory = parent if parent == self.workspace else self.workspace / parent
                    if directory in directories:
                        continue
                    info = directory.lstat()
                    if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_reparse_tag", False):
                        raise ValueError("redirected revision directory")
                    directories[directory] = (info.st_ino, info.st_mode, info.st_mtime_ns, info.st_ctime_ns)
                before = path.lstat()
                if not stat.S_ISREG(before.st_mode) or getattr(before, "st_reparse_tag", False):
                    raise ValueError("revision inventory contains a non-regular file")
                signature = (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino)
                if total + before.st_size > self.config.max_source_bytes:
                    raise ValueError("revision byte budget exceeded")
                cached = self.cache.get(relative)
                if not strict and cached is not None and cached[0] == signature:
                    digest, size = cached[1], before.st_size
                    reused += 1
                elif batch is not None and (self.allowed is None or self.allowed(relative)):
                    hint = cached[1] if cached else (self._object_hints or {}).get(relative)
                    read_before = batch.metrics["bytes_read"]
                    digest, size = batch.capture_file(path, maximum=self.config.max_source_bytes - total,
                                                      hint=hint, check=self._check)
                    hashed += batch.metrics["bytes_read"] - read_before
                else:
                    digest, size = stream_hash(path, maximum=self.config.max_source_bytes - total, check=self._check)
                    hashed += size
                total += size
                after = path.lstat()
                if signature != (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
                    raise ValueError("workspace changed during revision capture")
                files[relative] = digest
                modes[relative] = stat.S_IMODE(after.st_mode) & 0o777
                stats[relative] = signature
                updated[relative] = (signature, digest)
            metrics = {"files": len(files), "source_bytes": total, "bytes_hashed": hashed, "files_reused": reused,
                       **(batch.metrics if batch is not None else {})}
        revision = WorkspaceRevision(str(self.workspace), content_digest(files), files)
        return revision, metrics, (updated, stats, directories, modes)

    def _validate(self, signatures: dict, directories: dict) -> None:
        for relative, signature in signatures.items():
            self._check()
            info = (self.workspace / relative).lstat()
            if (not stat.S_ISREG(info.st_mode) or getattr(info, "st_reparse_tag", False) or
                signature != (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino)):
                raise ValueError("workspace changed during revision capture")
        for directory, signature in directories.items():
            self._check()
            info = directory.lstat()
            if signature != (info.st_ino, info.st_mode, info.st_mtime_ns, info.st_ctime_ns) or getattr(info, "st_reparse_tag", False):
                raise ValueError("workspace directory changed during revision capture")

    async def capture(self, *, strict: bool = True) -> WorkspaceRevision:
        async with self._lock:
            self._cancelled.clear()
            started = time.monotonic()
            paths, inventory = await self._inventory()
            inventoried = time.monotonic()
            revision, metrics, details = await self._worker(self._hash, paths, strict)
            hashed_at = time.monotonic()
            updated, signatures, directories, modes = details
            validated_at = hashed_at
            if strict:
                final_paths, final_inventory = await self._inventory()
                if final_paths != paths or final_inventory != inventory:
                    raise ValueError("workspace membership changed during revision capture")
                await self._worker(self._validate, signatures, directories)
                validated_at = time.monotonic()
                if self.snapshots is not None:
                    key = await self._worker(self.snapshots.publish, revision.workspace, revision.digest, revision.files, modes,
                                             check=self._check)
                    revision = replace(revision, snapshot_id=key)
                    self._object_hints = revision.files
            self.cache = updated
            self.last_metrics = {**metrics, "inventory": inventory, "strict": strict,
                                 "inventory_seconds": inventoried - started, "hash_seconds": hashed_at - inventoried,
                                 "stability_seconds": validated_at - hashed_at, "publish_seconds": time.monotonic() - validated_at,
                                 "seconds": time.monotonic() - started}
            self.totals["captures"] += 1
            self.totals["bytes_hashed"] += metrics["bytes_hashed"]
            self.totals["files_reused"] += metrics["files_reused"]
            return revision

    async def _worker(self, function, *args, **kwargs):
        pending = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            self._cancelled.set()
            # Repeated structural cancellation must not detach the filesystem
            # worker. Keep the capture lock until it observes cancellation/exits.
            while not pending.done():
                try:
                    await asyncio.shield(pending)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not pending.cancelled() and pending.exception() is not None:
                logging.getLogger(__name__).debug("Snapshot worker exited during cancellation: %s", pending.exception())
            raise

    def capture_walk(self) -> WorkspaceRevision:
        """Synchronous compatibility path; uses the same strict capture checks."""
        self._cancelled.clear()
        paths = self._walk()
        revision, _, (_, signatures, directories, _) = self._hash(paths, True)
        if self._walk() != paths:
            raise ValueError("workspace membership changed during revision capture")
        self._validate(signatures, directories)
        return revision
