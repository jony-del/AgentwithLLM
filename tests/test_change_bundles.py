import asyncio
import shutil
import threading
import uuid

import pytest

from agent_core.change_bundles import BundleStore
from agent_core.models import ToolCall
from agent_core.permissions import PermissionMode, PermissionPolicy
from agent_core.session import SessionContext
from agent_core.task_runtime import capture_revision
from agent_core.tools.change_bundles import MergeChangeBundleTool
from agent_core.tools.executor import ToolExecutor
from agent_core.tools.registry import ToolRegistry
from agent_core.tools.transaction import JournalStorage, TurnExecutionJournal, WorkspaceTransaction


def _bundle(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "f.py").write_bytes(b"x=1\r\n")
    (parent / "g.py").write_text("old\n")
    source = tmp_path / "child"
    shutil.copytree(parent, source)
    baseline = capture_revision(source)
    (source / "f.py").write_bytes(b"x=2\r\n")
    (source / "g.py").unlink()
    (source / "new.py").write_text("new\n")
    storage = JournalStorage.local(tmp_path / "state", workspace=parent)
    store = BundleStore(storage)
    bundle = store.export(source, baseline, parent)
    session = SessionContext(workspace=parent, bundle_store=store)
    tool = MergeChangeBundleTool(parent)
    registry = ToolRegistry()
    registry.register(tool)
    registry.rebind_workspace(str(parent))
    registry.bind_runtime(session=session)
    executor = ToolExecutor(registry, PermissionPolicy(PermissionMode.ACCEPTEDITS, workspace=parent), journal_storage=storage)
    return parent, source, store, bundle, executor


async def test_bundle_merges_all_files_and_preserves_exact_bytes(tmp_path):
    parent, source, store, bundle, executor = _bundle(tmp_path)
    assert store.load(bundle.id) == bundle
    result = (await executor.execute_many([ToolCall("merge_change_bundle", {"bundle_id": bundle.id})]))[0]
    assert result.ok, result.content
    assert (parent / "f.py").read_bytes() == b"x=2\r\n"
    assert not (parent / "g.py").exists()
    assert (parent / "new.py").read_bytes() == (source / "new.py").read_bytes()


async def test_bundle_conflict_rejects_every_write(tmp_path):
    parent, _, _, bundle, executor = _bundle(tmp_path)
    (parent / "f.py").write_text("concurrent\n")
    result = (await executor.execute_many([ToolCall("merge_change_bundle", {"bundle_id": bundle.id})]))[0]
    assert not result.ok
    assert result.metadata["error_type"] == "MergeConflict"
    assert (parent / "f.py").read_text() == "concurrent\n"
    assert (parent / "g.py").read_text() == "old\n"


@pytest.mark.parametrize("conflict", [False, True])
async def test_three_way_bundle_uses_owned_base_and_remains_atomic(tmp_path, conflict):
    parent, source = tmp_path / "parent", tmp_path / "child"
    parent.mkdir()
    base = "def a():\r\n    return 1\r\n\r\ndef b():\r\n    return 2\r\n"
    (parent / "f.py").write_bytes(base.encode())
    (parent / "other.txt").write_text("unchanged")
    shutil.copytree(parent, source)
    storage = JournalStorage.local(tmp_path / "state", workspace=parent)
    store = BundleStore(storage)
    baseline = capture_revision(source)
    store.capture_baseline(source, baseline)
    (source / "f.py").write_bytes(base.replace("return 2", "return 20").encode())
    (source / "other.txt").write_text("child changed")
    bundle = store.export(source, baseline, parent)
    current = base.replace("return 2" if conflict else "return 1", "return 30" if conflict else "return 10")
    (parent / "f.py").write_bytes(current.encode())
    session = SessionContext(workspace=parent, bundle_store=store)
    registry = ToolRegistry()
    registry.register(MergeChangeBundleTool(parent))
    registry.bind_runtime(session=session)
    executor = ToolExecutor(registry, PermissionPolicy(PermissionMode.ACCEPTEDITS, workspace=parent), journal_storage=storage)
    result = (await executor.execute_many([ToolCall("merge_change_bundle", {"bundle_id": bundle.id, "strategy": "three_way"})]))[0]
    assert result.ok is not conflict, result.content
    assert (parent / "other.txt").read_text() == ("unchanged" if conflict else "child changed")
    assert (parent / "f.py").read_bytes() == (current if conflict else current.replace("return 2", "return 20")).encode()
    assert not (parent / "new.py").exists()


async def test_bundle_protected_file_cannot_bypass_permission(tmp_path):
    parent, source, store, _, executor = _bundle(tmp_path)
    baseline = capture_revision(source)
    (source / "agent.toml").write_text("new configuration")
    bundle = store.export(source, baseline, parent)
    result = (await executor.execute_many([ToolCall("merge_change_bundle", {"bundle_id": bundle.id})]))[0]
    assert not result.ok
    assert not (parent / "agent.toml").exists()


async def test_bundle_wrong_workspace_and_missing_record_fail_closed(tmp_path):
    parent, _, _, bundle, executor = _bundle(tmp_path)
    executor.permissions.workspace = tmp_path
    wrong = (await executor.execute_many([ToolCall("merge_change_bundle", {"bundle_id": bundle.id})]))[0]
    missing = (await executor.execute_many([ToolCall("merge_change_bundle", {"bundle_id": uuid.uuid4().hex})]))[0]
    assert not wrong.ok and not missing.ok
    assert (parent / "f.py").read_bytes() == b"x=1\r\n"


async def test_bundle_write_failure_rolls_back_every_staged_change(tmp_path, monkeypatch):
    from agent_core.tools import change_bundles
    parent, _, _, bundle, executor = _bundle(tmp_path)
    original = change_bundles.write_text_exact

    def fail_new_file(path, content):
        if path.name == "new.py":
            raise OSError("simulated write failure")
        original(path, content)

    monkeypatch.setattr(change_bundles, "write_text_exact", fail_new_file)
    result = (await executor.execute_many([ToolCall("merge_change_bundle", {"bundle_id": bundle.id})]))[0]
    assert not result.ok
    assert (parent / "f.py").read_bytes() == b"x=1\r\n"
    assert (parent / "g.py").read_text() == "old\n"
    assert not (parent / "new.py").exists()


@pytest.mark.parametrize("partitioned", [False, True])
async def test_concurrent_transactions_validate_and_replace_under_one_lock(tmp_path, monkeypatch, partitioned):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "f.py").write_text("old")
    monkeypatch.setenv("POLARIS_HOME", str(tmp_path / "user-state"))
    storages = [JournalStorage.user_state(workspace, f"session-{i}", f"run-{i}") if partitioned else
                JournalStorage.local(tmp_path / f"state-{i}", workspace=workspace) for i in range(2)]
    journals = [TurnExecutionJournal(storage) for storage in storages]
    transactions = [WorkspaceTransaction(workspace, journal.turn_id, journal=journal) for journal in journals]
    barrier = threading.Barrier(2)
    try:
        for i, transaction in enumerate(transactions):
            transaction.ensure_paths([workspace / "f.py"])
            (transaction.overlay / "f.py").write_text(str(i))

        def commit(transaction):
            barrier.wait(timeout=3)
            try:
                return transaction.commit()
            except RuntimeError as error:
                return error

        outcomes = await asyncio.gather(*(asyncio.to_thread(commit, item) for item in transactions))
        assert sum(isinstance(outcome, list) for outcome in outcomes) == 1
        assert sum(isinstance(outcome, RuntimeError) for outcome in outcomes) == 1
        assert (workspace / "f.py").read_text() in {"0", "1"}
    finally:
        for transaction in transactions:
            transaction.rollback("test cleanup")
        for journal in journals:
            journal.close()


def test_bundle_invalid_id_cannot_escape_private_store(tmp_path):
    store = BundleStore(JournalStorage.local(tmp_path / "state"))
    with pytest.raises(ValueError):
        store.load("../other")
