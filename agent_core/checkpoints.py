"""Private content-addressed source checkpoints; restoring is a separate action."""
from __future__ import annotations

from dataclasses import dataclass
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
from agent_core.source_snapshots import (SnapshotConfig, SnapshotStore, content_digest, disk_path, children,
                                        validate_relative_path, validate_object_id)
from agent_core.recovery_paths import native_path
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
    snapshot_id: str = ""


class CheckpointStore:
    MAX_BYTES = 512 * 1024 * 1024  # Individual restore operation, not total snapshot size.
    MAX_MANIFESTS = 128

    def __init__(self, storage: JournalStorage, *, isolated: bool = False, config: SnapshotConfig | None = None) -> None:
        self.storage = storage
        self.root = (storage.run_root if isolated else storage.recovery_root) / "checkpoints"
        self.snapshots = SnapshotStore(storage, root=self.root, config=config)

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
            os.replace(native_path(temporary), native_path(path))
        finally:
            disk_path(temporary).unlink(missing_ok=True)

    def capture(self, task: TaskRun, revision: WorkspaceRevision, label: str = "", *,
                allowed: Callable[[str], bool] | None = None) -> SourceCheckpoint:
        self.storage.validate()
        if revision.workspace != str(Path(task.execution_workspace or task.baseline.workspace).resolve()):
            raise ValueError("checkpoint workspace does not match task")
        config = self.snapshots.config
        if len(revision.files) > config.max_files:
            raise ValueError("checkpoint file budget exceeded")
        _secure_mkdir(self.root)
        if sum(1 for entry in children(self.root) if entry.suffix == ".json" and not entry.name.startswith("latest-")) >= self.MAX_MANIFESTS:
            raise ValueError("checkpoint retention budget exceeded; archive the session before adding checkpoints")
        if content_digest(revision.files) != revision.digest:
            raise ValueError("checkpoint revision checksum mismatch")
        manifest = None
        if revision.snapshot_id:
            try:
                manifest = self.snapshots.load(revision.snapshot_id, workspace=revision.workspace, revision=revision.digest)
            except FileNotFoundError:
                pass  # A revision from another store must be copied and hash-verified.
        modes = dict(manifest["modes"]) if manifest is not None else {}
        total = 0
        with self.snapshots.batch() as batch:
            for relative, expected in revision.files.items():
                from agent_core.source_snapshots import check_cancelled
                check_cancelled()
                if allowed is not None and not allowed(relative):
                    modes.setdefault(relative, 0o644)
                    continue  # Hashes remain; restricted source bytes are never copied.
                if manifest is not None and expected in batch.objects:
                    total += batch.objects[expected]
                    batch.metrics["reused_blobs"] += 1
                else:
                    path = contained(Path(revision.workspace), relative)
                    digest, size = batch.capture_file(path, maximum=config.max_source_bytes - total,
                                                      hint=expected)
                    if digest != expected:
                        raise ValueError("workspace changed while capturing checkpoint")
                    total += size
                    modes[relative] = path.stat().st_mode & 0o777
                if total > config.max_source_bytes:
                    raise ValueError("checkpoint byte budget exceeded")
        key = revision.snapshot_id if manifest is not None else self.snapshots.publish(
            revision.workspace, revision.digest, revision.files, modes)
        checkpoint = SourceCheckpoint(uuid.uuid4().hex, task.id, revision.workspace, revision.digest,
                                      revision.files, task.plan_revision, time.time(), label[:256], key)
        # Small owner record; full inventories live once in immutable manifests.
        record = json.dumps({"v": 2, "checkpoint": {name: value for name, value in vars(checkpoint).items()
                                                    if name != "files"}}, ensure_ascii=False).encode("utf-8")
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
        if disk_path(path).stat().st_size > 8 * 1024 * 1024:
            raise ValueError("checkpoint manifest budget exceeded")
        with _open_state(self.storage, path, "r") as handle:
            record = json.load(handle)
        if record.get("v") not in {1, 2}:
            raise ValueError("unsupported checkpoint schema")
        data = record["checkpoint"]
        if record["v"] == 2:
            manifest = self.snapshots.load(data["snapshot_id"], workspace=data["workspace"], revision=data["revision"])
            data["files"] = manifest["files"]
        checkpoint = SourceCheckpoint(**data)
        if not isinstance(checkpoint.files, dict) or len(checkpoint.files) > self.snapshots.config.max_files:
            raise ValueError("invalid checkpoint file inventory")
        if checkpoint.id != key or checkpoint.task_id != task.id:
            raise ValueError("checkpoint is not owned by this task")
        if checkpoint.workspace not in {str(Path(task.execution_workspace or task.baseline.workspace).resolve()), task.baseline.workspace}:
            raise ValueError("checkpoint belongs to another workspace")
        for relative, digest in checkpoint.files.items():
            validate_relative_path(relative)
            validate_object_id(digest)
        return checkpoint

    def read(self, checkpoint: SourceCheckpoint, relative: str, *, max_bytes: int | None = None) -> bytes | None:
        digest = checkpoint.files.get(relative)
        if digest is None:
            return None
        path = self.root / "blobs" / digest
        self.storage.validate(path)
        limit = self.MAX_BYTES if max_bytes is None else min(self.MAX_BYTES, max_bytes)
        if limit < 0 or disk_path(path).stat().st_size > limit:
            raise ValueError("checkpoint blob budget exceeded")
        with _open_state(self.storage, path, "rb") as handle:
            data = handle.read(limit + 1)
        if len(data) > limit or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("checkpoint blob checksum mismatch")
        return data
