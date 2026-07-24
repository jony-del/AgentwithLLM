from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from agent_core.file_lock import FileLock
from agent_core.memory.paths import validate_memory_root
from agent_core.memory.repository import _atomic_write

SnapshotDecision = Literal["replace", "keep", "mark-synced"]


@dataclass(frozen=True, slots=True)
class SnapshotStatus:
    initialized: bool
    changed: bool
    local_checksum: str
    snapshot_checksum: str
    synced_checksum: str | None


def _checksum(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(root.rglob("*.md")):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class LocalMemorySnapshot:
    """Explicit synchronization between local named-agent memory and a Git snapshot."""

    def __init__(self, local_root: str | Path, snapshot_root: str | Path) -> None:
        self.local_root = validate_memory_root(local_root)
        self.snapshot_root = validate_memory_root(snapshot_root)
        self.marker = self.snapshot_root.parent / f".{self.snapshot_root.name}.snapshot.json"
        self.lock = self.snapshot_root.parent / f".{self.snapshot_root.name}.snapshot.lock"

    def status(self) -> SnapshotStatus:
        local = _checksum(self.local_root)
        snapshot = _checksum(self.snapshot_root)
        try:
            marker = json.loads(self.marker.read_text(encoding="utf-8"))
            synced_local = str(marker.get("local_checksum")) if marker.get("local_checksum") else None
            synced_snapshot = (
                str(marker.get("snapshot_checksum")) if marker.get("snapshot_checksum") else None
            )
            synced = f"{synced_local}:{synced_snapshot}" if synced_local and synced_snapshot else None
        except (OSError, json.JSONDecodeError):
            synced_local = None
            synced_snapshot = None
            synced = None
        initialized = self.snapshot_root.exists()
        return SnapshotStatus(
            initialized=initialized,
            changed=initialized and (
                synced is None or local != synced_local or snapshot != synced_snapshot
            ),
            local_checksum=local,
            snapshot_checksum=snapshot,
            synced_checksum=synced,
        )

    def initialize(self) -> SnapshotStatus:
        with FileLock(self.lock):
            if not self.snapshot_root.exists():
                self._replace_tree(self.local_root, self.snapshot_root)
            self._mark()
        return self.status()

    def resolve(self, decision: SnapshotDecision) -> SnapshotStatus:
        if decision not in {"replace", "keep", "mark-synced"}:
            raise ValueError(f"unsupported snapshot decision: {decision}")
        with FileLock(self.lock):
            if decision == "replace":
                if not self.snapshot_root.exists():
                    raise FileNotFoundError("snapshot has not been initialized")
                self._replace_tree(self.snapshot_root, self.local_root)
            elif decision == "keep":
                self._replace_tree(self.local_root, self.snapshot_root)
            self._mark()
        return self.status()

    def _replace_tree(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
        try:
            if source.exists():
                for path in source.rglob("*.md"):
                    target = stage / path.relative_to(source)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
            if destination.exists():
                stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
                backup = destination.parent / ".snapshot-backups" / f"{destination.name}-{stamp}"
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, backup)
            os.replace(stage, destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)

    def _mark(self) -> None:
        _atomic_write(
            self.marker,
            json.dumps(
                {
                    "schema_version": 1,
                    "local_checksum": _checksum(self.local_root),
                    "snapshot_checksum": _checksum(self.snapshot_root),
                    "synced_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                },
                indent=2,
            )
            + "\n",
        )
