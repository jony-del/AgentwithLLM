"""Private immutable manifests and streaming, content-addressed source objects.

Objects are never hard-linked into a writable verifier workspace. Manifests are
published only after a caller has validated the complete source observation.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections import OrderedDict
from dataclasses import dataclass, fields
import hashlib
import io
import json
import os
from pathlib import Path, PureWindowsPath
import re
import stat
import time
from typing import Any, Callable, Iterator, BinaryIO
import uuid

from agent_core.execution import current_execution_scope
from agent_core.file_lock import FileLock
from agent_core.permission_safety import is_secret_path
from agent_core.recovery_paths import native_path, private_path
from agent_core.tools.transaction import JournalStorage, _open_state, _secure_mkdir

BLOCK_SIZE = 256 * 1024
COPY_CACHE_BYTES = 8 * 1024 * 1024
_OBJECT_ID = re.compile(r"[0-9a-f]{64}")


def validate_relative_path(value: str) -> None:
    """Validate a manifest name without touching the mutable live workspace.

    Metadata is not an I/O capability: consumers still check actual paths and
    permissions at the read/write boundary. Require canonical portable names.
    """
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value or "\0" in value
            or any(part in {"", ".", ".."} or part != part.rstrip(" .") for part in value.split("/"))
            or PureWindowsPath(value).is_reserved() or is_secret_path(value)):
        raise ValueError("invalid snapshot relative path")


def validate_object_id(value: str) -> None:
    if not isinstance(value, str) or _OBJECT_ID.fullmatch(value) is None:
        raise ValueError("invalid snapshot object id")


class SourceTreePaths:
    """Check every accessed ancestor without repeatedly resolving the whole path.

    Only canonical relative manifest names are accepted. Directory identities are
    observations, not cached authorization: lstat/reparse checks run on every
    access and all observed directories are checked again at the end.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(os.path.abspath(root))
        if self.root.resolve() != self.root:
            raise ValueError("redirected snapshot destination root")
        self.identities: dict[Path, tuple[int, int, int]] = {}
        self.parents: dict[Path, tuple[Path, ...]] = {}
        self._directory(self.root)

    def _directory(self, directory: Path, *, create: bool = False) -> None:
        try:
            info = disk_path(directory).lstat()
        except FileNotFoundError:
            if not create:
                raise
            disk_path(directory).mkdir()
            info = disk_path(directory).lstat()
        if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_reparse_tag", False):
            raise ValueError("redirected snapshot directory")
        identity = (info.st_dev, info.st_ino, info.st_mode)
        if directory in self.identities and self.identities[directory] != identity:
            raise ValueError("snapshot directory identity changed")
        self.identities[directory] = identity

    def path(self, relative: str, *, create: bool = False) -> Path:
        validate_relative_path(relative)
        target = self.root / relative
        parent = target.parent
        if parent not in self.parents:
            self.parents[parent] = tuple(self.root / p for p in reversed(target.relative_to(self.root).parents))
        for directory in self.parents[parent]:
            self._directory(directory, create=create)
        return target

    def validate(self) -> None:
        for directory in tuple(self.identities):
            check_cancelled()
            self._directory(directory)


def disk_path(path: Path) -> Path:
    return Path(native_path(path))


def children(path: Path) -> Iterator[Path]:
    # Return canonical logical names, never extended OS spellings as trust roots.
    for entry in disk_path(path).iterdir():
        yield path / entry.name


@dataclass(frozen=True)
class SnapshotConfig:
    enabled: bool = True
    max_files: int = 500_000
    max_source_bytes: int = 16 * 1024**3
    max_store_bytes: int = 32 * 1024**3
    max_manifest_bytes: int = 128 * 1024**2
    max_manifests: int = 1024
    max_git_output_bytes: int = 128 * 1024**2

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> SnapshotConfig:
        values = {field.name: (raw or {})[field.name] for field in fields(cls) if field.name in (raw or {})}
        return cls(**values)

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("snapshots.enabled must be boolean")
        for field in fields(self):
            if field.name != "enabled" and (type(getattr(self, field.name)) is not int or getattr(self, field.name) <= 0):
                raise ValueError(f"snapshots.{field.name} must be a positive integer")


def check_cancelled() -> None:
    scope = current_execution_scope()
    if scope is not None:
        scope.raise_if_cancelled()


def content_digest(files: dict[str, str]) -> str:
    """Same digest as legacy json.dumps(sort_keys=True), without a giant string."""
    digest = hashlib.sha256()
    for part in json.JSONEncoder(sort_keys=True).iterencode(files):
        digest.update(part.encode())
    return digest.hexdigest()


def stream_hash(path: Path, *, sink: BinaryIO | None = None, maximum: int,
                check: Callable[[], None] = check_cancelled, private: bool = False) -> tuple[str, int]:
    """Read one regular file in bounded blocks; reject moving/redirected content."""
    check()
    before = disk_path(path).lstat()
    if not stat.S_ISREG(before.st_mode) or getattr(before, "st_reparse_tag", False) or (private and before.st_nlink != 1):
        raise ValueError("source snapshot contains a non-regular file")
    signature = (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino, before.st_mode)
    if before.st_size > maximum:
        raise ValueError("source snapshot byte budget exceeded")
    digest, size = hashlib.sha256(), 0
    with disk_path(path).open("rb") as source:
        opened = os.fstat(source.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino) or (private and opened.st_nlink != 1):
            raise ValueError("source changed while opening snapshot")
        for block in iter(lambda: source.read(BLOCK_SIZE), b""):
            check()
            size += len(block)
            if size > maximum:
                raise ValueError("source snapshot byte budget exceeded")
            digest.update(block)
            if sink is not None:
                sink.write(block)
    after = disk_path(path).lstat()
    if signature != (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino, after.st_mode) or (private and after.st_nlink != 1):
        raise ValueError("source changed while capturing snapshot")
    return digest.hexdigest(), size


class SnapshotStore:
    def __init__(self, storage: JournalStorage, *, root: Path | None = None,
                 config: SnapshotConfig | None = None) -> None:
        self.storage = storage
        self.root = root or storage.recovery_root / "checkpoints"
        self.config = config or SnapshotConfig()
        self.last_metrics: dict[str, int] = {}

    def blob_path(self, digest: str) -> Path:
        validate_object_id(digest)
        path = self.root / "blobs" / digest
        self.storage.validate(path)
        return path

    def _manifest_path(self, key: str) -> Path:
        if re.fullmatch(r"[0-9a-f]{64}", key) is None:
            raise ValueError("invalid snapshot manifest id")
        path = self.root / "manifests" / (key + ".json")
        self.storage.validate(path)
        return path

    @contextmanager
    def batch(self, *, check: Callable[[], None] = check_cancelled) -> Iterator[SnapshotBatch]:
        check()
        self.storage.validate()
        _secure_mkdir(self.root / "blobs")
        _secure_mkdir(self.root / "manifests")
        lock = self.root / "snapshot.lock"
        self.storage.validate(lock)
        with _open_state(self.storage, lock, "ab"):
            pass  # Create with the private state file mode before FileLock opens it.
        scope = current_execution_scope()
        timeout = min(30.0, max(0.01, scope.deadline - time.monotonic())) if scope is not None and scope.deadline is not None else 30.0
        with FileLock(disk_path(lock), timeout=timeout):
            self.storage.validate(lock)
            check()
            # Account once per batch, including crash-orphaned staging objects.
            stored, objects = 0, {}
            for directory in (self.root / "blobs", self.root / "manifests"):
                # This inventory is a quota/dedup hint, never permission to open
                # an object. Validate common ancestors once and every entry's
                # type/link/ACL, then recheck the directory identity afterwards.
                self.storage.validate(directory)
                before = disk_path(directory).stat()
                with os.scandir(native_path(directory)) as entries:
                    for entry in entries:
                        check()
                        # Windows DirEntry caches omit link count/inode; a real
                        # lstat is required to retain hard-link rejection.
                        info = os.stat(entry.path, follow_symlinks=False)
                        if not stat.S_ISREG(info.st_mode) or getattr(info, "st_reparse_tag", False) or info.st_nlink != 1:
                            raise ValueError("snapshot storage contains a redirected or linked object")
                        if self.storage.private_root is not None:
                            private_path(directory / entry.name)
                        stored += info.st_size
                        if directory.name == "blobs" and _OBJECT_ID.fullmatch(entry.name):
                            objects[entry.name] = info.st_size
                self.storage.validate(directory)
                after = disk_path(directory).stat()
                if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                    raise ValueError("snapshot inventory directory changed")
            batch = SnapshotBatch(self, stored, objects)
            try:
                yield batch
            finally:
                self.last_metrics = dict(batch.metrics)

    def publish(self, workspace: str, revision: str, files: dict[str, str], modes: dict[str, int], *,
                check: Callable[[], None] = check_cancelled) -> str:
        if len(files) > self.config.max_files or set(modes) != set(files):
            raise ValueError("snapshot inventory budget or modes mismatch")
        record = {"v": 1, "workspace": workspace, "revision": revision, "files": files, "modes": modes}
        encoder = json.JSONEncoder(sort_keys=True)
        digest, size = hashlib.sha256(), 0
        for part in encoder.iterencode(record):
            check()
            data = part.encode()
            size += len(data)
            if size > self.config.max_manifest_bytes:
                raise ValueError("snapshot manifest budget exceeded")
            digest.update(data)
        key = digest.hexdigest()
        with self.batch(check=check) as batch:
            target = self._manifest_path(key)
            if not disk_path(target).exists():
                if sum(1 for entry in children(target.parent) if entry.suffix == ".json") >= self.config.max_manifests:
                    raise ValueError("snapshot manifest retention budget exceeded; archive unused session state")
                batch.reserve(size)
                temporary = self.root / "manifests" / ("." + uuid.uuid4().hex + ".tmp")
                try:
                    with _open_state(self.storage, temporary, "wb") as handle:
                        for part in encoder.iterencode(record):
                            check()
                            handle.write(part.encode())
                        handle.flush()
                        os.fsync(handle.fileno())
                    check()
                    os.replace(native_path(temporary), native_path(target))
                finally:
                    disk_path(temporary).unlink(missing_ok=True)
            pointer = self.root / ("latest-" + hashlib.sha256(workspace.encode()).hexdigest() + ".json")
            self.storage.validate(pointer)
            if disk_path(pointer).exists():
                with _open_state(self.storage, pointer, "r") as handle:
                    if handle.read(257) == json.dumps({"v": 1, "snapshot": key}):
                        return key
            staging = self.root / ("." + uuid.uuid4().hex + ".tmp")
            try:
                with _open_state(self.storage, staging, "w") as handle:
                    json.dump({"v": 1, "snapshot": key}, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                self.storage.validate(pointer)
                check()
                os.replace(native_path(staging), native_path(pointer))
            finally:
                disk_path(staging).unlink(missing_ok=True)
        return key

    def latest_files(self, workspace: str) -> dict[str, str]:
        path = self.root / ("latest-" + hashlib.sha256(workspace.encode()).hexdigest() + ".json")
        self.storage.validate(path)
        if not disk_path(path).exists():
            return {}
        if disk_path(path).stat().st_size > 256:
            raise ValueError("invalid latest snapshot pointer")
        with _open_state(self.storage, path, "r") as handle:
            pointer = json.load(handle)
        if pointer.get("v") != 1:
            raise ValueError("unsupported snapshot pointer schema")
        return self.load(pointer["snapshot"], workspace=workspace)["files"]

    def load(self, key: str, *, workspace: str | None = None, revision: str | None = None) -> dict[str, Any]:
        path = self._manifest_path(key)
        if disk_path(path).stat().st_size > self.config.max_manifest_bytes:
            raise ValueError("snapshot manifest budget exceeded")
        with _open_state(self.storage, path, "rb") as handle:
            data = handle.read(self.config.max_manifest_bytes + 1)
        if len(data) > self.config.max_manifest_bytes or hashlib.sha256(data).hexdigest() != key:
            raise ValueError("snapshot manifest checksum mismatch")
        record = json.loads(data)
        if record.get("v") != 1 or (workspace is not None and record.get("workspace") != workspace) or (
                revision is not None and record.get("revision") != revision):
            raise ValueError("snapshot manifest identity mismatch")
        files, modes = record.get("files"), record.get("modes")
        if (not isinstance(files, dict) or len(files) > self.config.max_files or not isinstance(modes, dict)
                or set(files) != set(modes) or not isinstance(record.get("workspace"), str)
                or not Path(record["workspace"]).is_absolute()):
            raise ValueError("invalid snapshot file inventory")
        for relative, digest in files.items():
            check_cancelled()
            validate_relative_path(relative)
            validate_object_id(digest)
            if type(modes.get(relative)) is not int or not 0 <= modes[relative] <= 0o777:
                raise ValueError("invalid snapshot file mode")
        if content_digest(files) != record["revision"]:
            raise ValueError("snapshot revision checksum mismatch")
        return record

    def materialize(self, key: str, workspace: str, revision: str, destination: Path, *,
                    allowed: Callable[[str], bool]) -> dict[str, int]:
        record = self.load(key, workspace=workspace, revision=revision)
        tree = SourceTreePaths(destination)
        reader = SnapshotReader(self)
        total = 0
        cached: OrderedDict[str, bytes] = OrderedDict()
        cache_size = 0
        for relative, digest in record["files"].items():
            check_cancelled()
            if not allowed(relative):
                raise PermissionError("snapshot source is restricted by read policy: " + relative)
            target = tree.path(relative, create=True)
            # A new independent file: no writable link to the CAS or live source.
            with disk_path(target).open("xb") as handle:
                data = cached.pop(digest, None)
                if data is not None:
                    cached[digest] = data
                    actual, size = digest, len(data)
                    if size > self.config.max_source_bytes - total:
                        raise ValueError("source snapshot byte budget exceeded")
                    handle.write(data)
                else:
                    blob = reader.path(digest)  # Revalidate every ancestor before an actual CAS read.
                    if disk_path(blob).stat().st_size <= BLOCK_SIZE:
                        buffer = io.BytesIO()
                        actual, size = stream_hash(blob, sink=buffer, maximum=min(BLOCK_SIZE, self.config.max_source_bytes - total), private=True)
                        if actual != digest:
                            raise ValueError("snapshot blob checksum mismatch")
                        data = buffer.getvalue()
                        handle.write(data)
                        while cached and (cache_size + size > COPY_CACHE_BYTES or len(cached) >= 256):
                            cache_size -= len(cached.popitem(last=False)[1])
                        cached[digest] = data
                        cache_size += size
                    else:
                        actual, size = stream_hash(blob, sink=handle, maximum=self.config.max_source_bytes - total, private=True)
            if actual != digest:
                raise ValueError("snapshot blob checksum mismatch")
            total += size
            disk_path(target).chmod(record["modes"][relative])
        tree.validate()
        reader.validate()
        self.storage.validate(reader.directory)
        return {"files": len(record["files"]), "bytes_copied": total}


class SnapshotReader:
    """A bounded read batch over canonical, flat, digest-named private objects.

    Canonical resolution runs at batch boundaries. Every read still checks all
    ancestor identities/reparse flags/ACLs and the object's ACL; stream_hash checks
    the opened regular file, link count and digest. No cached check grants access.
    """

    def __init__(self, store: SnapshotStore) -> None:
        self.store = store
        self.directory = store.root / "blobs"
        store.storage.validate(self.directory)
        self.identities: dict[Path, tuple[int, int, int]] = {}
        current = self.directory
        while True:
            info = disk_path(current).lstat()
            self.identities[current] = (info.st_dev, info.st_ino, info.st_mode)
            if current == store.storage.anchor:
                break
            if not current.is_relative_to(store.storage.anchor):
                raise ValueError("snapshot object directory escapes storage anchor")
            current = current.parent
        self.validate()

    def validate(self) -> None:
        for directory, identity in self.identities.items():
            check_cancelled()
            info = disk_path(directory).lstat()
            if (not stat.S_ISDIR(info.st_mode) or getattr(info, "st_reparse_tag", False)
                    or (info.st_dev, info.st_ino, info.st_mode) != identity):
                raise ValueError("snapshot object directory identity changed or redirected")
            private_root = self.store.storage.private_root
            if private_root is not None and (directory == private_root or private_root in directory.parents):
                private_path(directory)

    def path(self, digest: str) -> Path:
        validate_object_id(digest)
        self.validate()
        # A lowercase SHA-256 basename cannot encode traversal, an alternate data
        # stream, Windows device/8.3 spelling, or a noncanonical relative component.
        path = self.directory / digest
        if self.store.storage.private_root is not None:
            private_path(path)
        return path


class SnapshotBatch:
    def __init__(self, store: SnapshotStore, stored: int, objects: dict[str, int]) -> None:
        self.store, self.stored = store, stored
        self.objects = objects
        self.metrics = {"new_blobs": 0, "reused_blobs": 0, "bytes_stored": 0, "bytes_read": 0, "bytes_staged": 0}

    def reserve(self, size: int) -> None:
        if self.stored + size > self.store.config.max_store_bytes:
            raise ValueError("snapshot storage budget exceeded; archive unused session state")
        self.stored += size

    def capture_file(self, source: Path, *, maximum: int, hint: str | None = None,
                     check: Callable[[], None] = check_cancelled) -> tuple[str, int]:
        if disk_path(source).stat().st_size <= BLOCK_SIZE:
            buffer = io.BytesIO()
            digest, size = stream_hash(source, sink=buffer, maximum=min(maximum, BLOCK_SIZE), check=check)
            self.metrics["bytes_read"] += size
            if digest in self.objects:
                self.metrics["reused_blobs"] += 1
                return digest, size
            self.reserve(size)
            temporary = self.store.root / "blobs" / ("." + uuid.uuid4().hex + ".tmp")
            try:
                with _open_state(self.store.storage, temporary, "wb") as handle:
                    handle.write(buffer.getvalue())
                    self.metrics["bytes_staged"] += size
                    handle.flush()
                    os.fsync(handle.fileno())
                target = self.store.blob_path(digest)
                check()
                os.replace(native_path(temporary), native_path(target))
                self.objects[digest] = size
                self.metrics["new_blobs"] += 1
                self.metrics["bytes_stored"] += size
                return digest, size
            finally:
                disk_path(temporary).unlink(missing_ok=True)
        if hint and hint in self.objects:
            digest, size = stream_hash(source, maximum=maximum, check=check)
            self.metrics["bytes_read"] += size
            if digest == hint:
                self.metrics["reused_blobs"] += 1
                return digest, size
            # The strict read discovered a change. Copy only this file again and
            # require identical bytes; a moving source never publishes a snapshot.
            actual, count = self.capture_file(source, maximum=maximum, check=check)
            if actual != digest:
                raise ValueError("source changed during snapshot object capture")
            return actual, count
        temporary = self.store.root / "blobs" / ("." + uuid.uuid4().hex + ".tmp")
        try:
            remaining = min(maximum, max(0, self.store.config.max_store_bytes - self.stored))
            with _open_state(self.store.storage, temporary, "wb") as handle:
                digest, size = stream_hash(source, sink=handle, maximum=remaining, check=check)
                self.metrics["bytes_read"] += size
                self.metrics["bytes_staged"] += size
                handle.flush()
                os.fsync(handle.fileno())
            target = self.store.blob_path(digest)
            if disk_path(target).exists():
                self.metrics["reused_blobs"] += 1
            else:
                self.reserve(size)
                os.replace(native_path(temporary), native_path(target))
                self.metrics["new_blobs"] += 1
                self.metrics["bytes_stored"] += size
                self.objects[digest] = size
            return digest, size
        finally:
            disk_path(temporary).unlink(missing_ok=True)
