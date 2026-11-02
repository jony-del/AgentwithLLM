from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import json
from pathlib import Path
import threading

import pytest

from agent_core.checkpoints import CheckpointStore
from agent_core.config import resolve_snapshot_config
from agent_core.revisions import RevisionTracker
from agent_core.recovery_paths import native_path
from agent_core.session import SessionContext
from agent_core.source_snapshots import SnapshotConfig, SnapshotStore
from agent_core.task_runtime import TaskContract, TaskRun, TaskStore, capture_revision
from agent_core.tools.transaction import JournalStorage
from agent_core.verifier import _copy_source, source_unchanged


def setup(tmp_path, **limits):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "a.py").write_bytes(b"value = 1\r\n")
    (workspace / "b.dat").write_bytes(b"binary\x00content")
    storage = JournalStorage.local(tmp_path / "state", workspace=workspace)
    checkpoints = CheckpointStore(storage, config=SnapshotConfig(**limits))
    tracker = RevisionTracker(workspace, git_aware=False, config=checkpoints.snapshots.config,
                              snapshots=checkpoints.snapshots)
    return workspace, storage, checkpoints, tracker


async def test_persistent_objects_reuse_content_but_strict_capture_rehashes(tmp_path):
    root, storage, checkpoints, tracker = setup(tmp_path)
    first = await tracker.capture()
    assert first.snapshot_id and first == capture_revision(root)
    assert tracker.last_metrics["new_blobs"] == 2
    assert tracker.last_metrics["bytes_stored"] == sum(p.stat().st_size for p in root.iterdir())
    repeated = await tracker.capture()
    assert repeated.snapshot_id == first.snapshot_id
    assert tracker.last_metrics["new_blobs"] == tracker.last_metrics["bytes_stored"] == 0
    assert tracker.last_metrics["files_reused"] == 0 and tracker.last_metrics["bytes_hashed"] > 0
    (root / "a.py").write_bytes(b"value = 2\r\n")
    changed = await tracker.capture()
    assert changed.snapshot_id != first.snapshot_id and tracker.last_metrics["new_blobs"] == 1
    assert tracker.last_metrics["bytes_stored"] == (root / "a.py").stat().st_size
    # A new tracker loads persisted content hints; no new object is written.
    restarted = RevisionTracker(root, git_aware=False, snapshots=SnapshotStore(storage))
    assert (await restarted.capture()).snapshot_id == changed.snapshot_id
    assert restarted.last_metrics["new_blobs"] == 0 and restarted.last_metrics["bytes_hashed"] > 0
    (root / "b.dat").unlink()
    removed = await tracker.capture()
    assert "b.dat" not in checkpoints.snapshots.load(removed.snapshot_id)["files"]
    assert "b.dat" in checkpoints.snapshots.load(first.snapshot_id)["files"]


async def test_checkpoint_and_verifier_use_pinned_objects_not_live_source(tmp_path, monkeypatch):
    root, _, checkpoints, tracker = setup(tmp_path)
    revision = await tracker.capture()
    task = TaskRun(TaskContract("experiment"), revision)
    (root / "a.py").write_bytes(b"later live change")
    original = Path.open

    def no_live_read(path, mode="r", *args, **kwargs):
        if (path.is_relative_to(root) or path.is_relative_to(Path(native_path(root)))) and "r" in mode:
            raise AssertionError("immutable snapshot consumer reopened live source")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_live_read)
    checkpoint = checkpoints.capture(task, revision)
    assert checkpoints.read(checkpoint, "a.py") == b"value = 1\r\n"
    session = SessionContext(workspace=root, checkpoint_store=checkpoints)
    copy = tmp_path / "verify"
    copy.mkdir()
    _copy_source(session, revision, copy)
    assert (copy / "a.py").read_bytes() == b"value = 1\r\n"
    blob = checkpoints.snapshots.blob_path(revision.files["a.py"])
    assert blob.stat().st_ino != (copy / "a.py").stat().st_ino
    assert source_unchanged(revision, copy)
    (copy / "a.py").write_bytes(b"probe mutation")
    assert not source_unchanged(revision, copy)
    assert checkpoints.read(checkpoint, "a.py") == b"value = 1\r\n"


async def test_task_state_references_manifest_and_v3_remains_readable(tmp_path):
    root, storage, checkpoints, tracker = setup(tmp_path)
    revision = await tracker.capture()
    task = TaskRun(TaskContract("persist"), revision)
    store = TaskStore(storage)
    store.save(task)
    record = json.loads(store.path.read_text())
    assert record["v"] == 4 and "files" not in record["task"]["baseline"]
    assert store.load().baseline.files == revision.files
    assert store.load().baseline.snapshot_id == revision.snapshot_id
    legacy = replace(task, baseline=replace(revision, snapshot_id=""))
    store.save(legacy)
    assert json.loads(store.path.read_text())["v"] == 3
    assert store.load().baseline == revision
    # Legacy checkpoint owner manifests continue to use existing CAS blobs.
    saved = checkpoints.capture(task, revision)
    old = asdict(saved)
    old.pop("snapshot_id")
    checkpoints._write(checkpoints.root / (saved.id + ".json"), json.dumps({"v": 1, "checkpoint": old}).encode())
    assert checkpoints.load(saved.id, task).files == revision.files
    assert checkpoints.read(checkpoints.load(saved.id, task), "a.py") == (root / "a.py").read_bytes()


async def test_permissions_checked_before_object_capture_and_materialization(tmp_path):
    root, _, checkpoints, tracker = setup(tmp_path)
    tracker.allowed = lambda path: path != "b.dat"
    revision = await tracker.capture()
    assert "b.dat" in revision.files  # Metadata coverage does not confer read permission.
    assert not checkpoints.snapshots.blob_path(revision.files["b.dat"]).exists()
    task = TaskRun(TaskContract("restricted"), revision)
    checkpoint = checkpoints.capture(task, revision, allowed=tracker.allowed)
    assert not checkpoints.snapshots.blob_path(revision.files["b.dat"]).exists()
    copy = tmp_path / "verify"
    copy.mkdir()
    with pytest.raises(PermissionError, match="restricted"):
        checkpoints.snapshots.materialize(revision.snapshot_id, revision.workspace, revision.digest, copy,
                                          allowed=lambda path: False)
    assert not list(copy.iterdir())
    assert checkpoints.read(checkpoint, "a.py") == (root / "a.py").read_bytes()


async def test_corrupted_blob_and_manifest_fail_closed(tmp_path):
    _, storage, checkpoints, tracker = setup(tmp_path)
    revision = await tracker.capture()
    task = TaskRun(TaskContract("integrity"), revision)
    saved = checkpoints.capture(task, revision)
    checkpoints.snapshots.blob_path(revision.files["a.py"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        checkpoints.read(saved, "a.py")
    copy = tmp_path / "verify"
    copy.mkdir()
    with pytest.raises(ValueError, match="checksum"):
        checkpoints.snapshots.materialize(revision.snapshot_id, revision.workspace, revision.digest, copy, allowed=lambda _: True)
    store = TaskStore(storage)
    store.save(task)
    checkpoints.snapshots._manifest_path(revision.snapshot_id).write_bytes(b"{}")
    with pytest.raises(ValueError, match="checksum"):
        store.load()


async def test_failed_publication_leaves_previous_snapshot_usable(tmp_path, monkeypatch):
    import os
    root, _, checkpoints, tracker = setup(tmp_path)
    revision = await tracker.capture()
    (root / "a.py").write_text("changed", encoding="utf-8")
    replace_file = os.replace

    def crash(source, target):
        if Path(target).parent.name == "manifests":
            raise OSError("injected publication failure")
        return replace_file(source, target)

    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(OSError, match="publication failure"):
        await tracker.capture()
    assert checkpoints.snapshots.latest_files(str(root)) == revision.files
    assert not list(checkpoints.root.rglob("*.tmp"))
    assert checkpoints.snapshots.load(revision.snapshot_id)["files"] == revision.files


@pytest.mark.parametrize("phase", ["inventory", "hash", "publish"])
async def test_cancellation_drains_capture_before_returning(tmp_path, monkeypatch, phase):
    _, _, checkpoints, tracker = setup(tmp_path)
    entered, release = threading.Event(), threading.Event()
    owner, name = (checkpoints.snapshots, "publish") if phase == "publish" else (tracker, "_walk" if phase == "inventory" else "_hash")
    original = getattr(owner, name)

    def slow(*args, **kwargs):
        entered.set()
        release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, slow)
    pending = asyncio.create_task(tracker.capture())
    assert await asyncio.to_thread(entered.wait, 5)
    pending.cancel()
    await asyncio.sleep(0)
    assert not pending.done()
    pending.cancel()  # A second cancellation must not abandon the worker either.
    await asyncio.sleep(0)
    assert not pending.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert not list(checkpoints.root.glob("manifests/*.json"))


@pytest.mark.parametrize("limits", [{"max_files": 1}, {"max_source_bytes": 3}, {"max_store_bytes": 3}, {"max_manifest_bytes": 10}])
async def test_explicit_quotas_never_publish_partial_success(tmp_path, limits):
    _, _, checkpoints, tracker = setup(tmp_path, **limits)
    with pytest.raises(ValueError, match="budget"):
        await tracker.capture()
    assert not list(checkpoints.root.glob("latest-*.json"))
    assert not list(checkpoints.root.glob("manifests/*.json"))


def test_snapshot_config_precedence_and_validation(tmp_path, monkeypatch):
    config = tmp_path / "agent.toml"
    config.write_text('[snapshots]\nmax_files = 25000\nfuture_option = "ignored"\n', encoding="utf-8")
    assert resolve_snapshot_config(config).max_files == 25000
    monkeypatch.setenv("AGENT_SNAPSHOTS_MAX_FILES", "30000")
    assert resolve_snapshot_config(config).max_files == 30000
    monkeypatch.setenv("AGENT_SNAPSHOTS", "false")
    assert not resolve_snapshot_config(config).enabled
    for invalid in (0, -1, True, "100"):
        with pytest.raises(ValueError):
            SnapshotConfig.from_dict({"max_files": invalid})


async def test_disabled_store_preserves_strict_versions_without_persisting_source(tmp_path):
    root, _, checkpoints, tracker = setup(tmp_path, enabled=False)
    # Embedders supplying a configured store need not pass the same limits twice.
    tracker = RevisionTracker(root, git_aware=False, snapshots=checkpoints.snapshots)
    revision = await tracker.capture()
    assert revision == capture_revision(root) and not revision.snapshot_id
    assert not checkpoints.root.exists()


async def test_deep_private_paths_preserve_full_hashes_and_integrity(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_bytes(b"deep path content")
    state = tmp_path / ("nested-" + "x" * 70) / ("state-" + "y" * 30)
    storage = JournalStorage.local(state, workspace=root)
    checkpoints = CheckpointStore(storage)
    tracker = RevisionTracker(root, git_aware=False, snapshots=checkpoints.snapshots)
    revision = await tracker.capture()
    assert len(str(checkpoints.snapshots.blob_path(revision.files["a.py"]))) > 260
    assert checkpoints.snapshots.load(revision.snapshot_id)["files"] == revision.files
    saved = checkpoints.capture(TaskRun(TaskContract("deep paths"), revision), revision)
    assert checkpoints.read(saved, "a.py") == b"deep path content"


async def test_small_duplicate_content_is_deduplicated_before_staging(tmp_path):
    root, _, checkpoints, tracker = setup(tmp_path)
    for index in range(100):
        (root / f"duplicate_{index}.py").write_bytes(b"value = 1\r\n")
    revision = await tracker.capture()
    assert tracker.last_metrics["new_blobs"] == 2
    assert tracker.last_metrics["bytes_staged"] == tracker.last_metrics["bytes_stored"] == 25
    assert tracker.last_metrics["bytes_hashed"] == sum(p.stat().st_size for p in root.iterdir())
    await tracker.capture()
    assert tracker.last_metrics["bytes_staged"] == 0
    assert checkpoints.snapshots.load(revision.snapshot_id)["files"] == revision.files


async def test_manifest_metadata_does_not_access_live_source(tmp_path, monkeypatch):
    root, _, checkpoints, tracker = setup(tmp_path)
    revision = await tracker.capture()
    original = Path.resolve
    def no_live_resolution(path, *args, **kwargs):
        if path.is_relative_to(root):
            raise AssertionError("manifest loading traversed live source")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", no_live_resolution)
    assert checkpoints.snapshots.load(revision.snapshot_id)["files"] == revision.files


@pytest.mark.parametrize("path", ["../escape", "/absolute", "C:/escape", "x\\y", "a//b", "./a", "a/../b", "x:stream", ".env", "nul", "a. "])
def test_manifest_paths_reject_noncanonical_and_secret_names(path):
    from agent_core.source_snapshots import validate_relative_path
    with pytest.raises(ValueError, match="relative path"):
        validate_relative_path(path)


async def test_batch_inventory_rejects_hardlinked_objects(tmp_path):
    import os
    root, _, checkpoints, tracker = setup(tmp_path)
    revision = await tracker.capture()
    blob = checkpoints.snapshots.blob_path(revision.files["a.py"])
    victim = tmp_path / "victim"
    victim.write_bytes(b"do not change")
    blob.unlink()
    os.link(victim, blob)
    with pytest.raises(ValueError, match="linked object"):
        await tracker.capture()
    assert victim.read_bytes() == b"do not change"


async def test_copy_cache_rechecks_permission_for_every_path(tmp_path):
    root, _, checkpoints, tracker = setup(tmp_path)
    (root / "b.dat").write_bytes((root / "a.py").read_bytes())
    revision = await tracker.capture()
    destination = tmp_path / "copy"
    destination.mkdir()
    with pytest.raises(PermissionError, match="restricted"):
        checkpoints.snapshots.materialize(revision.snapshot_id, revision.workspace, revision.digest, destination,
                                          allowed=lambda relative: relative == "a.py")
    assert (destination / "a.py").read_bytes() == (root / "a.py").read_bytes()
    assert not (destination / "b.dat").exists()


async def test_copy_rejects_directory_redirect_after_first_cached_path(tmp_path, directory_redirect):
    root, _, checkpoints, tracker = setup(tmp_path)
    (root / "nested").mkdir()
    (root / "nested/a.py").write_bytes(b"same")
    (root / "nested/b.py").write_bytes(b"same")
    revision = await tracker.capture()
    destination, victim = tmp_path / "copy", tmp_path / "victim"
    destination.mkdir()
    victim.mkdir()
    def allowed(relative):
        if relative == "nested/b.py":
            (destination / "nested").rename(destination / "saved")
            directory_redirect(destination / "nested", victim)
        return True
    with pytest.raises(ValueError, match="redirected"):
        checkpoints.snapshots.materialize(revision.snapshot_id, revision.workspace, revision.digest, destination, allowed=allowed)
    assert not list(victim.iterdir())


def test_copy_verification_rejects_missing_changed_and_redirected_paths(tmp_path, directory_redirect):
    from agent_core.source_snapshots import SourceTreePaths
    root = tmp_path / "copy"
    root.mkdir()
    (root / "nested").mkdir()
    (root / "nested/a.py").write_bytes(b"original")
    revision = capture_revision(root)
    assert source_unchanged(revision, root)
    tree = SourceTreePaths(root)
    tree.path("nested/a.py")
    (root / "nested").rename(root / "saved")
    directory_redirect(root / "nested", root / "saved")
    with pytest.raises(ValueError, match="redirected"):
        tree.path("nested/a.py")
    assert not source_unchanged(revision, root)


async def test_object_reader_rejects_redirect_between_distinct_objects(tmp_path, directory_redirect):
    _, _, checkpoints, tracker = setup(tmp_path)
    revision = await tracker.capture()
    destination = tmp_path / "copy"
    destination.mkdir()
    def allowed(relative):
        if relative == "b.dat":
            blobs = checkpoints.root / "blobs"
            moved = checkpoints.root / "saved-blobs"
            blobs.rename(moved)
            directory_redirect(blobs, moved)
        return True
    with pytest.raises(ValueError, match="directory.*redirected"):
        checkpoints.snapshots.materialize(revision.snapshot_id, revision.workspace, revision.digest, destination, allowed=allowed)


async def test_object_reader_rejects_hardlink_and_checks_leaf_acl(tmp_path, monkeypatch):
    import os
    _, storage, checkpoints, tracker = setup(tmp_path)
    revision = await tracker.capture()
    destination = tmp_path / "copy"
    destination.mkdir()
    digest = revision.files["a.py"]
    checkpoints.snapshots.storage = replace(storage, private_root=storage.recovery_root)
    def denied(path):
        if path.name == digest:
            raise PermissionError("injected object ACL denial")
    monkeypatch.setattr("agent_core.source_snapshots.private_path", denied)
    with pytest.raises(PermissionError, match="ACL denial"):
        checkpoints.snapshots.materialize(revision.snapshot_id, revision.workspace, revision.digest, destination, allowed=lambda _: True)
    assert (destination / "a.py").read_bytes() == b""
    (destination / "a.py").unlink()
    monkeypatch.setattr("agent_core.source_snapshots.private_path", lambda _: None)
    victim = tmp_path / "victim"
    victim.write_bytes(b"value = 1\r\n")
    blob = checkpoints.snapshots.blob_path(digest)
    blob.unlink()
    os.link(victim, blob)
    with pytest.raises(ValueError, match="non-regular"):
        checkpoints.snapshots.materialize(revision.snapshot_id, revision.workspace, revision.digest, destination, allowed=lambda _: True)
