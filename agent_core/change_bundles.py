"""Private, versioned change bundles exported by owned worktrees."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from agent_core.task_runtime import WorkspaceRevision, capture_revision
from agent_core.tools.transaction import JournalStorage, _open_state, _secure_mkdir


@dataclass(frozen=True)
class ChangeBundle:
    id: str
    parent_workspace: str
    source_workspace: str
    base_revision: str
    expected_hashes: dict[str, str | None]
    contents: dict[str, str | None]
    baseline_contents: dict[str, str | None] = field(default_factory=dict)


class BundleStore:
    def __init__(self, storage: JournalStorage) -> None:
        self.storage = storage
        self.root = storage.recovery_root / "change-bundles"

    def capture_baseline(self, source: Path, baseline: WorkspaceRevision) -> None:
        """Bounded UTF-8 base blobs. Uncaptured files remain strict-conflict-only."""
        self.storage.validate()
        from agent_core.codeintel.snapshots import contained
        from agent_core.execution import current_execution_scope
        scope = current_execution_scope()
        root = self.root / "bases"
        _secure_mkdir(root)
        total = 0
        stored = sum(path.stat().st_size for path in root.iterdir())
        for relative, digest in baseline.files.items():
            if scope is not None:
                scope.raise_if_cancelled()
            path = contained(source, relative)
            if path.stat().st_size > 2 * 1024 * 1024:
                continue
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError("worktree changed during merge baseline capture")
            try:
                data.decode("utf-8")
            except UnicodeError:
                continue
            total += len(data)
            if total > 64 * 1024 * 1024:
                break
            blob = root / digest
            self.storage.validate(blob)
            if blob.exists():
                continue
            stored += len(data)
            if stored > 512 * 1024 * 1024:
                raise ValueError("merge baseline storage budget exceeded")
            with _open_state(self.storage, blob, "xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())

    def export(self, source: Path, baseline: WorkspaceRevision, parent: Path) -> ChangeBundle | None:
        current = capture_revision(source)
        changed = sorted(path for path in baseline.files.keys() | current.files.keys()
                         if baseline.files.get(path) != current.files.get(path))
        if not changed:
            return None
        contents: dict[str, str | None] = {}
        bases: dict[str, str | None] = {}
        total = 0
        for path in changed:
            expected = baseline.files.get(path)
            if expected is None:
                bases[path] = None
            else:
                blob = self.root / "bases" / expected
                self.storage.validate(blob)
                if blob.exists():
                    if blob.stat().st_size > 2 * 1024 * 1024:
                        raise ValueError("merge baseline byte budget exceeded")
                    with _open_state(self.storage, blob, "rb") as handle:
                        data = handle.read(2 * 1024 * 1024 + 1)
                    total += len(data)
                    if hashlib.sha256(data).hexdigest() != expected:
                        raise ValueError("merge baseline checksum mismatch")
                    bases[path] = data.decode("utf-8")
            if total > 16 * 1024 * 1024:
                raise ValueError("change bundle exceeds its budget")
            if path not in current.files:
                contents[path] = None
                continue
            if (source / path).stat().st_size + total > 16 * 1024 * 1024:
                raise ValueError("change bundle exceeds its budget")
            data = (source / path).read_bytes()
            total += len(data)
            if total > 16 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != current.files[path]:
                raise ValueError("change bundle exceeds its budget or source changed during export")
            contents[path] = data.decode("utf-8")
        bundle = ChangeBundle(uuid.uuid4().hex, str(parent.resolve()), str(source.resolve()), baseline.digest,
                              {path: baseline.files.get(path) for path in changed}, contents, bases)
        _secure_mkdir(self.root)
        temporary = self.root / f".{bundle.id}.tmp"
        final = self.root / f"{bundle.id}.json"
        try:
            with _open_state(self.storage, temporary, "w") as handle:
                json.dump({"v": 2, "bundle": asdict(bundle)}, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            self.storage.validate(final)
            os.replace(temporary, final)
        finally:
            temporary.unlink(missing_ok=True)
        return bundle

    def load(self, key: str) -> ChangeBundle:
        if re.fullmatch(r"[0-9a-f]{32}", key) is None:
            raise ValueError("invalid change bundle id")
        path = self.root / f"{key}.json"
        self.storage.validate(path)
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("change bundle exceeds its budget")
        with _open_state(self.storage, path, "r") as handle:
            record = json.load(handle)
        if record.get("v") not in {1, 2}:
            raise ValueError("unsupported change bundle schema")
        bundle = ChangeBundle(**record["bundle"])
        if bundle.id != key or bundle.contents.keys() != bundle.expected_hashes.keys():
            raise ValueError("invalid change bundle record")
        if not bundle.baseline_contents.keys() <= bundle.contents.keys():
            raise ValueError("invalid merge baseline paths")
        for relative, value in bundle.baseline_contents.items():
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest() if value is not None else None
            if digest != bundle.expected_hashes[relative]:
                raise ValueError("merge baseline does not match expected hash")
        return bundle
