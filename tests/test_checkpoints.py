import hashlib
import asyncio
import json
from dataclasses import replace
import threading

import pytest

from agent_core.checkpoints import CheckpointStore
from agent_core.models import ToolCall
from agent_core.permissions import PermissionMode, PermissionPolicy
from agent_core.session import SessionContext
from agent_core.task_runtime import PlanStep, TaskContract, TaskRun, TaskStore, capture_revision
from agent_core.tools.checkpoints import CreateCheckpointTool, RollbackCheckpointTool
from agent_core.tools.executor import ToolExecutor
from agent_core.tools.registry import ToolRegistry
from agent_core.tools.transaction import JournalStorage


async def setup_checkpoint(tmp_path, *, approve=True):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_bytes(b"value=1\r\n")
    (root / "binary.dat").write_bytes(b"\xff\x00")
    storage = JournalStorage.local(tmp_path / "state", workspace=root)
    task = TaskRun(TaskContract("experiment"), capture_revision(root))
    task.replace_plan([PlanStep("edit", "experiment", status="completed")])
    session = SessionContext(workspace=root, task_run=task, checkpoint_store=CheckpointStore(storage), task_store=TaskStore(storage))
    registry = ToolRegistry()
    registry.register(CreateCheckpointTool(session))
    tool = RollbackCheckpointTool(root)
    tool.bind_session(session)
    registry.register(tool)
    policy = PermissionPolicy(PermissionMode.BYPASS, prompter=(lambda request: "once") if approve else None, workspace=root)
    executor = ToolExecutor(registry, policy, journal_storage=storage)
    result = (await executor.execute_many([ToolCall("create_checkpoint", {})]))[0]
    assert result.ok, result.content
    return root, session, executor, result.metadata["checkpoint_id"]


async def test_checkpoint_restores_binary_exact_bytes_creation_and_deletion(tmp_path):
    root, session, executor, key = await setup_checkpoint(tmp_path)
    (root / "a.py").write_text("value=2\n")
    (root / "binary.dat").unlink()
    (root / "new.txt").write_text("new")
    session.task_run.reviews = [{"status": "passed"}]
    arguments = {"checkpoint_id": key, "paths": ["a.py", "binary.dat", "new.txt"]}
    preview = (await executor.execute_many([ToolCall("rollback_checkpoint", arguments)]))[0]
    assert preview.ok, preview.content
    assert (root / "a.py").read_text() == "value=2\n"
    result = (await executor.execute_many([ToolCall("rollback_checkpoint", {
        **arguments, "preview": False, "expected_revision": preview.metadata["restore_preview"]["expected_revision"]})]))[0]
    assert result.ok, result.content
    assert (root / "a.py").read_bytes() == b"value=1\r\n"
    assert (root / "binary.dat").read_bytes() == b"\xff\x00"
    assert not (root / "new.txt").exists()
    assert not session.task_run.reviews and session.task_run.plan[0].status == "pending"
    assert session.task_store.load().plan[0].status == "pending"


async def test_stale_preview_and_noninteractive_bypass_cannot_restore(tmp_path):
    root, _, executor, key = await setup_checkpoint(tmp_path, approve=False)
    result = (await executor.execute_many([ToolCall("rollback_checkpoint", {
        "checkpoint_id": key, "paths": ["a.py"], "preview": False, "expected_revision": capture_revision(root).digest})]))[0]
    assert not result.ok
    assert (root / "a.py").read_bytes() == b"value=1\r\n"
    executor.permissions.prompter = lambda request: "once"
    executor.permissions.interactive = True
    result = (await executor.execute_many([ToolCall("rollback_checkpoint", {
        "checkpoint_id": key, "paths": ["a.py"], "preview": False, "expected_revision": "stale"})]))[0]
    assert not result.ok and result.metadata["error_type"] == "StaleRestorePreview"


@pytest.mark.parametrize("path", ["../escape", ".env", "missing.py"])
async def test_restore_rejects_paths_outside_coverage_and_secrets(tmp_path, path):
    root, _, executor, key = await setup_checkpoint(tmp_path)
    result = (await executor.execute_many([ToolCall("rollback_checkpoint", {"checkpoint_id": key, "paths": [path]})]))[0]
    assert not result.ok
    assert (root / "a.py").read_bytes() == b"value=1\r\n"


async def test_checkpoint_ownership_integrity_retention_and_read_policy(tmp_path):
    root, session, _, key = await setup_checkpoint(tmp_path)
    store = session.checkpoint_store
    with pytest.raises(ValueError, match="owned"):
        store.load(key, replace(session.task_run, id="foreign"))
    checkpoint = store.load(key, session.task_run)
    blob = store.root / "blobs" / checkpoint.files["a.py"]
    blob.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        store.read(checkpoint, "a.py")
    store.MAX_MANIFESTS = 1
    with pytest.raises(ValueError, match="retention"):
        store.capture(session.task_run, capture_revision(root))
    other = CheckpointStore(JournalStorage.local(tmp_path / "other", workspace=root))
    saved = other.capture(session.task_run, capture_revision(root), allowed=lambda path: path != "a.py")
    assert saved.files["a.py"] == hashlib.sha256(b"value=1\r\n").hexdigest()
    assert not (other.root / "blobs" / saved.files["a.py"]).exists()
    assert json.loads(session.task_store.path.read_text())["v"] == 3


async def test_private_state_write_is_drained_before_cancellation_returns(tmp_path):
    _, session, _, _ = await setup_checkpoint(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = session.task_store.save
    def slow_save(task):
        entered.set()
        release.wait(5)
        original(task)
    session.task_store.save = slow_save
    pending = asyncio.create_task(session.persist_task_async())
    await asyncio.to_thread(entered.wait, 5)
    pending.cancel()
    await asyncio.sleep(0)
    assert not pending.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    session.task_run.status = "cancelled"
    await session.persist_task_async()
    assert session.task_store.load().status == "cancelled"
