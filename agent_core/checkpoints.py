"""Private content-addressed source checkpoints; restoring is a separate action."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid
from collections.abc import Callable

from agent_core.codeintel.snapshots import contained
from agent_core.task_runtime import TaskRun, WorkspaceRevision
from agent_core.tools.transaction import JournalStorage, _open_state, _secure_mkdir


@dataclass(frozen=True)
class SourceCheckpoint:
    id: str
    task_id: str
    workspace: str
    revision: str
    files: dict[str, str]
    plan_revision: int
    created_at: float
    label: str = ""


class CheckpointStore:
    MAX_BYTES = 512 * 1024 * 1024
    MAX_STORE_BYTES = 1024 * 1024 * 1024
    MAX_MANIFESTS = 128

    def __init__(self, storage: JournalStorage, *, isolated: bool = False) -> None:
        self.storage = storage
        self.root = (storage.run_root if isolated else storage.recovery_root) / "checkpoints"

    def _write(self, path: Path, data: bytes) -> None:
        self.storage.validate()
        _secure_mkdir(path.parent)
        temporary = path.with_name("." + uuid.uuid4().hex + ".tmp")
        try:
            with _open_state(self.storage, temporary, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            self.storage.validate(path)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def capture(self, task: TaskRun, revision: WorkspaceRevision, label: str = "", *,
                allowed: Callable[[str], bool] | None = None) -> SourceCheckpoint:
        self.storage.validate()
        if revision.workspace != str(Path(task.execution_workspace or task.baseline.workspace).resolve()):
            raise ValueError("checkpoint workspace does not match task")
        if len(revision.files) > 20_000:
            raise ValueError("checkpoint file budget exceeded")
        _secure_mkdir(self.root / "blobs")
        if len(list(self.root.glob("*.json"))) >= self.MAX_MANIFESTS:
            raise ValueError("checkpoint retention budget exceeded; archive the session before adding checkpoints")
        stored = sum(path.stat().st_size for path in (self.root / "blobs").iterdir())
        total = 0
        from agent_core.execution import current_execution_scope
        scope = current_execution_scope()
        for relative, expected in revision.files.items():
            if scope is not None:
                scope.raise_if_cancelled()
            if allowed is not None and not allowed(relative):
                continue  # Manifest hashes remain; restricted source bytes are never copied.
            path = contained(Path(revision.workspace), relative)
            if path.stat().st_size + total > self.MAX_BYTES:
                raise ValueError("checkpoint byte budget exceeded")
            data = path.read_bytes()
            total += len(data)
            if total > self.MAX_BYTES or hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("workspace changed while capturing checkpoint")
            blob = self.root / "blobs" / expected
            self.storage.validate(blob)
            if not blob.exists():
                stored += len(data)
                if stored > self.MAX_STORE_BYTES:
                    raise ValueError("checkpoint storage budget exceeded")
                self._write(blob, data)
        checkpoint = SourceCheckpoint(uuid.uuid4().hex, task.id, revision.workspace, revision.digest,
                                      dict(revision.files), task.plan_revision, time.time(), label[:256])
        record = json.dumps({"v": 1, "checkpoint": asdict(checkpoint)}, ensure_ascii=False).encode("utf-8")
        if len(record) > 8 * 1024 * 1024:
            raise ValueError("checkpoint manifest budget exceeded")
        self._write(self.root / (checkpoint.id + ".json"), record)
        task.checkpoints.append({"id": checkpoint.id, "revision": checkpoint.revision,
                                 "workspace": checkpoint.workspace, "label": checkpoint.label,
                                 "plan_revision": checkpoint.plan_revision})
        task.checkpoints = task.checkpoints[-32:]
        return checkpoint

    def load(self, key: str, task: TaskRun) -> SourceCheckpoint:
        if re.fullmatch(r"[0-9a-f]{32}", key) is None:
            raise ValueError("invalid checkpoint id")
        path = self.root / (key + ".json")
        self.storage.validate(path)
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("checkpoint manifest budget exceeded")
        with _open_state(self.storage, path, "r") as handle:
            record = json.load(handle)
        if record.get("v") != 1:
            raise ValueError("unsupported checkpoint schema")
        checkpoint = SourceCheckpoint(**record["checkpoint"])
        if not isinstance(checkpoint.files, dict) or len(checkpoint.files) > 20_000:
            raise ValueError("invalid checkpoint file inventory")
        if checkpoint.id != key or checkpoint.task_id != task.id:
            raise ValueError("checkpoint is not owned by this task")
        if checkpoint.workspace not in {str(Path(task.execution_workspace or task.baseline.workspace).resolve()), task.baseline.workspace}:
            raise ValueError("checkpoint belongs to another workspace")
        for relative, digest in checkpoint.files.items():
            contained(Path(checkpoint.workspace), relative)
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("invalid checkpoint blob hash")
        return checkpoint

    def read(self, checkpoint: SourceCheckpoint, relative: str, *, max_bytes: int | None = None) -> bytes | None:
        digest = checkpoint.files.get(relative)
        if digest is None:
            return None
        path = self.root / "blobs" / digest
        self.storage.validate(path)
        limit = self.MAX_BYTES if max_bytes is None else min(self.MAX_BYTES, max_bytes)
        if limit < 0 or path.stat().st_size > limit:
            raise ValueError("checkpoint blob budget exceeded")
        with _open_state(self.storage, path, "rb") as handle:
            data = handle.read(limit + 1)
        if len(data) > limit or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("checkpoint blob checksum mismatch")
        return data
