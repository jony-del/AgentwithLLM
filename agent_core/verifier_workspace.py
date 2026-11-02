"""Session-local clean-copy reuse, with exclusive leases and full revalidation."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile

from agent_core.source_snapshots import SourceTreePaths, check_cancelled, disk_path, stream_hash
from agent_core.tools.codeintel import path_allowed


class VerifierWorkspace:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.directory = None
        self.path = None
        self.key = None
        self.file_metadata = {}

    def _discard(self):
        if self.directory is not None:
            # Only the TemporaryDirectory instance we allocated owns cleanup.
            self.directory.cleanup()
        self.directory = self.path = self.key = None
        self.file_metadata = {}

    def _metadata(self, current):
        assert self.path is not None
        tree = SourceTreePaths(self.path)
        metadata = {}
        for raw in current.files:
            check_cancelled()
            info = disk_path(tree.path(raw)).lstat()
            metadata[raw] = (info.st_mode, info.st_size)
        tree.validate()
        return metadata

    def _clean(self, session, current):
        if self.path is None:
            return False
        try:
            tree = SourceTreePaths(self.path)
            expected_dirs = {str(p) for raw in current.files for p in PurePosixPath(raw).parents}
            pending = [self.path]
            seen = set()
            while pending:
                directory = pending.pop()
                with os.scandir(disk_path(directory)) as entries:
                    for entry in entries:
                        check_cancelled()
                        raw = (directory / entry.name).relative_to(self.path).as_posix()
                        info = disk_path(directory / entry.name).lstat()
                        if getattr(info, "st_reparse_tag", False) or stat.S_ISLNK(info.st_mode):
                            return False
                        if stat.S_ISDIR(info.st_mode):
                            if raw not in expected_dirs:
                                return False
                            tree._directory(directory / entry.name)
                            pending.append(directory / entry.name)
                        elif raw in current.files and stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                            if not path_allowed(session, raw, "run_verifier"):
                                raise PermissionError("verifier source is restricted by read policy")
                            if self.file_metadata.get(raw) != (info.st_mode, info.st_size):
                                return False
                            actual, _ = stream_hash(tree.path(raw), maximum=info.st_size, private=True)
                            if actual != current.files[raw]:
                                return False
                            seen.add(raw)
                        else:
                            return False
            tree.validate()
            return seen == current.files.keys()
        except PermissionError:
            raise
        except (ValueError, OSError):
            return False

    @asynccontextmanager
    async def lease(self, session, current, copy_source, *, reusable=True):
        from agent_core.verifier import bounded_thread
        async with self.lock:
            key = (session.task_run.id, current.workspace, current.digest, current.snapshot_id)
            # Legacy revisions omit mode-bearing manifests: keep their old fresh-copy behavior.
            try:
                reused = bool(reusable and current.snapshot_id and key == self.key and
                              await bounded_thread(self._clean, session, current))
            except BaseException:
                await bounded_thread(self._discard)
                raise
            if not reused:
                await bounded_thread(self._discard)
                self.directory = tempfile.TemporaryDirectory(prefix="polaris-verifier-")
                self.path = Path(self.directory.name) / "workspace"
                self.path.mkdir()
                try:
                    await bounded_thread(copy_source, session, current, self.path)
                    if reusable and current.snapshot_id:
                        self.file_metadata = await bounded_thread(self._metadata, current)
                    self.key = key
                except BaseException:
                    await bounded_thread(self._discard)
                    raise
            try:
                yield self.path, {"reused": reused, "files_reused": len(current.files) if reused else 0}
            except BaseException:
                await bounded_thread(self._discard)
                raise

    async def close(self):
        from agent_core.verifier import bounded_thread
        async with self.lock:
            await bounded_thread(self._discard)


@asynccontextmanager
async def verification_copy(session, current, copy_source, *, reuse):
    if reuse:
        pool = getattr(session, "verifier_workspace", None)
        if pool is None:
            pool = VerifierWorkspace()
            session.verifier_workspace = pool
        async with pool.lease(session, current, copy_source) as lease:
            yield lease
    else:
        pool = VerifierWorkspace()
        try:
            async with pool.lease(session, current, copy_source, reusable=False) as lease:
                yield lease
        finally:
            await pool.close()
