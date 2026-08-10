from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from agent_core.hooks import HookOutcome, HookPipeline
from agent_core.models import ToolCall, ToolResult, ToolRisk
from agent_core.permissions import PermissionMode, PermissionPolicy
from agent_core.permission_types import PermissionResult
from agent_core.tool_config import ToolSuiteConfig
from agent_core.tools.base import (
    ConcurrencySpec,
    ExecutionSafety,
    ResourceLock,
    Tool,
    WorkspacePathMixin,
)
from agent_core.tools.builtin import ReadTextFileTool, WriteTextFileTool
from agent_core.tools.executor import ToolExecutor
from agent_core.tools.registry import ToolRegistry
from agent_core.tools.transaction import TurnExecutionJournal, WorkspaceTransaction


class _FinalOnlyCounter(Tool):
    name = "final_counter"
    description = "record execution"
    input_schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    risk = ToolRisk.READ

    def __init__(self) -> None:
        self.values: list[str] = []

    def _invoke(self, arguments: dict) -> ToolResult:
        self.values.append(str(arguments["value"]))
        return ToolResult(self.name, self.values[-1])


class _DagGate(Tool):
    name = "dag_gate"
    description = "hold a resource lock"
    input_schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "resource": {"type": "string"},
            "mode": {"type": "string"},
        },
        "required": ["label", "resource", "mode"],
    }
    risk = ToolRisk.READ
    execution_safety = ExecutionSafety.SPECULATIVE_SAFE

    def __init__(self) -> None:
        self.started: dict[str, asyncio.Event] = {}
        self.release: dict[str, asyncio.Event] = {}

    def concurrency_spec(self, arguments: dict) -> ConcurrencySpec:
        return ConcurrencySpec(
            (ResourceLock("dag", arguments["resource"], arguments["mode"]),)
        )

    async def run(self, arguments: dict) -> ToolResult:
        label = arguments["label"]
        self.started.setdefault(label, asyncio.Event()).set()
        await self.release.setdefault(label, asyncio.Event()).wait()
        return ToolResult(self.name, label)


class _FailingTransaction(Tool):
    name = "failing_transaction"
    description = "fail after transactional admission"
    input_schema = {"type": "object", "properties": {}}
    risk = ToolRisk.WRITE
    accept_edits_safe = True
    execution_safety = ExecutionSafety.TRANSACTIONAL
    transaction_backend = "workspace"

    def concurrency_spec(self, arguments: dict) -> ConcurrencySpec:
        return ConcurrencySpec((ResourceLock("logical", "failure", "write"),))

    def _invoke(self, arguments: dict) -> ToolResult:
        return ToolResult(self.name, "failed", ok=False)


class _PostCounter:
    def __init__(self) -> None:
        self.calls = 0

    async def on_post_tool(self, _context) -> HookOutcome:
        self.calls += 1
        return HookOutcome()


class _OutOfScopeWriter(WorkspacePathMixin, Tool):
    name = "out_of_scope_writer"
    description = "attempt a write beyond the declared path"
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    risk = ToolRisk.WRITE
    accept_edits_safe = True
    execution_safety = ExecutionSafety.TRANSACTIONAL
    transaction_backend = "workspace"

    async def check_permissions(self, arguments: dict, context) -> PermissionResult:
        return PermissionResult.allow("test fixture")

    def concurrency_spec(self, arguments: dict) -> ConcurrencySpec:
        return ConcurrencySpec((self.workspace_lock(arguments["path"], "write"),))

    def _invoke(self, arguments: dict) -> ToolResult:
        target = self.workspace / "outside.txt"
        target.write_text("should never commit", encoding="utf-8")
        return ToolResult(self.name, "wrote outside lock")


class _UnsafeTimedTool(Tool):
    name = "unsafe_timed"
    description = "wait until the scheduler timeout supervises cleanup"
    input_schema = {"type": "object", "properties": {}}
    risk = ToolRisk.READ
    execution_safety = ExecutionSafety.SPECULATIVE_SAFE
    execution_timeout = 0.05

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    def concurrency_spec(self, arguments: dict) -> ConcurrencySpec:
        return ConcurrencySpec()

    async def run(self, arguments: dict) -> ToolResult:
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cancelled.set()
        return ToolResult(self.name, "unexpected")


def test_trusted_policy_override_and_template_failure_are_fail_closed() -> None:
    registry = ToolRegistry()
    tool = _FinalOnlyCounter()
    registry.register(tool)
    suite = ToolSuiteConfig.from_dict(
        {
            "execution_policies": {
                tool.name: {
                    "safety": "speculative_safe",
                    "exclusive": False,
                    "resources": [
                        {"namespace": "logical", "argument": "value", "mode": "read"}
                    ],
                }
            }
        }
    )
    registry.set_policy_overrides(suite.execution_policies, trusted=True)

    policy = registry.execution_policy(tool, {"value": "key"})
    degraded = registry.execution_policy(tool, {})

    assert policy.safety is ExecutionSafety.SPECULATIVE_SAFE
    assert policy.concurrency.locks == (ResourceLock("logical", "key", "read"),)
    assert degraded.safety is ExecutionSafety.FINAL_ONLY
    assert degraded.concurrency.exclusive


async def test_unannotated_read_is_final_only_even_with_stable_ordinal(tmp_path: Path) -> None:
    registry = ToolRegistry()
    tool = _FinalOnlyCounter()
    registry.register(tool)
    executor = ToolExecutor(
        registry,
        PermissionPolicy(PermissionMode.AUTO),
        journal_dir=tmp_path / "journals",
    )
    batch = executor.begin_batch()
    call = ToolCall(tool.name, {"value": "one"}, id="t1")

    assert batch.submit_streamed(call, ordinal=0)
    await asyncio.sleep(0.02)
    assert tool.values == []

    results = await batch.finish([call])
    assert results[0].ok
    assert tool.values == ["one"]
    assert batch.journal._closed is True


async def test_schema_and_same_turn_result_references_fail_before_execution(tmp_path: Path) -> None:
    registry = ToolRegistry()
    tool = _FinalOnlyCounter()
    registry.register(tool)
    executor = ToolExecutor(
        registry,
        PermissionPolicy(PermissionMode.AUTO),
        journal_dir=tmp_path / "journals",
    )

    missing = (await executor.execute_many([ToolCall(tool.name, {})]))[0]
    dependency = (
        await executor.execute_many(
            [ToolCall(tool.name, {"value": {"$tool_result": "t0"}})]
        )
    )[0]

    assert missing.metadata["error_type"] == "SchemaValidationError"
    assert dependency.metadata["error_type"] == "ResultDependencyRequiresNextTurn"
    assert tool.values == []


async def test_dag_allows_independent_call_to_bypass_blocked_call(tmp_path: Path) -> None:
    registry = ToolRegistry()
    tool = _DagGate()
    registry.register(tool)
    executor = ToolExecutor(
        registry,
        PermissionPolicy(PermissionMode.AUTO),
        max_workers=3,
        journal_dir=tmp_path / "journals",
    )
    batch = executor.begin_batch()
    calls = [
        ToolCall(tool.name, {"label": "a", "resource": "x", "mode": "write"}, id="a"),
        ToolCall(tool.name, {"label": "b", "resource": "x", "mode": "read"}, id="b"),
        ToolCall(tool.name, {"label": "c", "resource": "y", "mode": "read"}, id="c"),
    ]
    for ordinal, call in enumerate(calls):
        assert batch.submit_streamed(call, ordinal=ordinal)

    await asyncio.wait_for(tool.started.setdefault("a", asyncio.Event()).wait(), 1)
    await asyncio.wait_for(tool.started.setdefault("c", asyncio.Event()).wait(), 1)
    assert not tool.started.setdefault("b", asyncio.Event()).is_set()
    tool.release.setdefault("c", asyncio.Event()).set()
    tool.release.setdefault("a", asyncio.Event()).set()
    await asyncio.wait_for(tool.started["b"].wait(), 1)
    tool.release.setdefault("b", asyncio.Event()).set()

    results = await batch.finish(calls)
    assert [item.content for item in results] == ["a", "b", "c"]


async def test_transactional_write_is_staged_and_read_sees_overlay(tmp_path: Path) -> None:
    target = tmp_path / "value.txt"
    target.write_text("old", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(WriteTextFileTool(tmp_path))
    registry.register(ReadTextFileTool(tmp_path))
    registry.rebind_workspace(str(tmp_path))
    executor = ToolExecutor(
        registry,
        PermissionPolicy(PermissionMode.ACCEPTEDITS),
        journal_dir=tmp_path / "journals",
    )
    batch = executor.begin_batch()
    write = ToolCall("write_text_file", {"path": "value.txt", "content": "new"}, id="w")
    read = ToolCall("read_text_file", {"path": "value.txt"}, id="r")
    assert batch.submit_streamed(write, ordinal=0)
    assert batch.submit_streamed(read, ordinal=1)

    await asyncio.wait_for(batch._by_id["w"].done.wait(), 2)
    await asyncio.sleep(0)
    await asyncio.wait_for(batch._by_id["r"].done.wait(), 2)
    assert target.read_text(encoding="utf-8") == "old"
    assert batch._by_id["r"].result is not None
    assert batch._by_id["r"].result.content == "new"

    results = await batch.finish([write, read])
    assert all(item.ok for item in results)
    assert target.read_text(encoding="utf-8") == "new"
    assert results[0].metadata["execution_status"] == "finalized"


async def test_authoritative_argument_mismatch_rolls_back_staged_write(tmp_path: Path) -> None:
    target = tmp_path / "value.txt"
    target.write_text("old", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(WriteTextFileTool(tmp_path))
    registry.rebind_workspace(str(tmp_path))
    executor = ToolExecutor(
        registry,
        PermissionPolicy(PermissionMode.ACCEPTEDITS),
        journal_dir=tmp_path / "journals",
    )
    batch = executor.begin_batch()
    streamed = ToolCall(
        "write_text_file", {"path": "value.txt", "content": "staged"}, id="w"
    )
    assert batch.submit_streamed(streamed, ordinal=0)
    await asyncio.wait_for(batch._by_id["w"].done.wait(), 2)
    assert target.read_text(encoding="utf-8") == "old"

    authoritative = ToolCall(
        "write_text_file", {"path": "value.txt", "content": "different"}, id="w"
    )
    results = await batch.finish([authoritative])

    assert not results[0].ok
    assert results[0].metadata["error_type"] == "StreamingToolProtocolMismatch"
    assert target.read_text(encoding="utf-8") == "old"


async def test_any_transactional_failure_rolls_back_successful_staged_results(tmp_path: Path) -> None:
    target = tmp_path / "value.txt"
    target.write_text("old", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(WriteTextFileTool(tmp_path))
    registry.register(_FailingTransaction())
    registry.rebind_workspace(str(tmp_path))
    executor = ToolExecutor(
        registry,
        PermissionPolicy(PermissionMode.ACCEPTEDITS),
        journal_dir=tmp_path / "journals",
    )
    calls = [
        ToolCall("write_text_file", {"path": "value.txt", "content": "new"}, id="w"),
        ToolCall("failing_transaction", {}, id="f"),
    ]

    results = await executor.execute_many(calls)

    assert not results[0].ok
    assert results[0].metadata["error_type"] == "RolledBack"
    assert not results[1].ok
    assert target.read_text(encoding="utf-8") == "old"


async def test_unproven_termination_rolls_back_without_post_hook(tmp_path: Path) -> None:
    target = tmp_path / "value.txt"
    target.write_text("old", encoding="utf-8")
    post = _PostCounter()
    registry = ToolRegistry()
    registry.register(WriteTextFileTool(tmp_path))
    registry.rebind_workspace(str(tmp_path))
    executor = ToolExecutor(
        registry,
        PermissionPolicy(PermissionMode.ACCEPTEDITS),
        hooks=HookPipeline(external_post_tool_hooks=[post]),
        journal_dir=tmp_path / "journals",
    )
    batch = executor.begin_batch()
    call = ToolCall("write_text_file", {"path": "value.txt", "content": "new"}, id="w")
    assert batch.submit_streamed(call, ordinal=0)
    await asyncio.wait_for(batch._by_id["w"].done.wait(), 2)

    results = await batch.finish([call], termination_proven=False)

    assert results[0].metadata["error_type"] == "ProviderTerminationUnproven"
    assert target.read_text(encoding="utf-8") == "old"
    assert post.calls == 0


async def test_post_hook_runs_once_only_after_successful_commit(tmp_path: Path) -> None:
    target = tmp_path / "value.txt"
    target.write_text("old", encoding="utf-8")
    post = _PostCounter()
    registry = ToolRegistry()
    registry.register(WriteTextFileTool(tmp_path))
    registry.rebind_workspace(str(tmp_path))
    executor = ToolExecutor(
        registry,
        PermissionPolicy(PermissionMode.ACCEPTEDITS),
        hooks=HookPipeline(external_post_tool_hooks=[post]),
        journal_dir=tmp_path / "journals",
    )
    batch = executor.begin_batch()
    call = ToolCall("write_text_file", {"path": "value.txt", "content": "new"}, id="w")
    assert batch.submit_streamed(call, ordinal=0)
    await asyncio.wait_for(batch._by_id["w"].done.wait(), 2)
    assert post.calls == 0

    results = await batch.finish(
        [call], termination_proven=True, termination_event="message_stop"
    )

    assert results[0].ok
    assert target.read_text(encoding="utf-8") == "new"
    assert post.calls == 1


async def test_transaction_rejects_changes_outside_declared_write_lock(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(_OutOfScopeWriter(tmp_path))
    registry.rebind_workspace(str(tmp_path))
    executor = ToolExecutor(
        registry,
        PermissionPolicy(PermissionMode.ACCEPTEDITS),
        journal_dir=tmp_path / "journals",
    )
    call = ToolCall("out_of_scope_writer", {"path": "allowed.txt"}, id="w")

    result = (await executor.execute_many([call]))[0]

    assert not result.ok
    assert result.metadata["error_type"] == "RolledBack"
    assert not (tmp_path / "allowed.txt").exists()
    assert not (tmp_path / "outside.txt").exists()


def test_transaction_materializes_only_declared_paths_and_checks_conflicts(tmp_path: Path) -> None:
    declared = tmp_path / "declared.txt"
    unrelated = tmp_path / "unrelated.txt"
    declared.write_text("old", encoding="utf-8")
    unrelated.write_text("large unrelated tree", encoding="utf-8")
    journal = TurnExecutionJournal(tmp_path / "journals")
    transaction = WorkspaceTransaction(tmp_path, "scoped", journal=journal)
    transaction.ensure_paths([declared])

    assert (transaction.overlay / "declared.txt").exists()
    assert not (transaction.overlay / "unrelated.txt").exists()
    (transaction.overlay / "declared.txt").write_text("new", encoding="utf-8")
    declared.write_text("external", encoding="utf-8")

    try:
        transaction.commit()
    except RuntimeError as exc:
        assert "workspace changed" in str(exc)
    else:  # pragma: no cover - safety assertion
        raise AssertionError("concurrent workspace modification was accepted")
    transaction.rollback("test cleanup")
    journal.close()


def test_transaction_rejects_concurrent_creation_in_write_subtree(tmp_path: Path) -> None:
    scoped = tmp_path / "scoped"
    scoped.mkdir()
    journal = TurnExecutionJournal(tmp_path / "journals")
    transaction = WorkspaceTransaction(tmp_path, "subtree", journal=journal)
    transaction.declare_resources(
        (ResourceLock("fs", str(scoped), "write", subtree=True),)
    )
    staged = transaction.overlay / "scoped" / "new.txt"
    staged.write_text("tool", encoding="utf-8")
    (scoped / "external.txt").write_text("outside process", encoding="utf-8")

    try:
        transaction.commit()
    except RuntimeError as exc:
        assert "subtree changed" in str(exc)
    else:  # pragma: no cover - safety assertion
        raise AssertionError("concurrent subtree creation was accepted")
    transaction.rollback("test cleanup")
    journal.close()


def test_transaction_rejects_race_on_new_exact_file(tmp_path: Path) -> None:
    target = tmp_path / "new.txt"
    journal = TurnExecutionJournal(tmp_path / "journals")
    transaction = WorkspaceTransaction(tmp_path, "new-file", journal=journal)
    transaction.ensure_paths([target])
    (transaction.overlay / "new.txt").write_text("tool", encoding="utf-8")
    target.write_text("outside process", encoding="utf-8")

    try:
        transaction.commit()
    except RuntimeError as exc:
        assert "workspace changed" in str(exc)
    else:  # pragma: no cover - safety assertion
        raise AssertionError("concurrent file creation was overwritten")
    transaction.rollback("test cleanup")
    journal.close()


def test_recovery_journal_redacts_sensitive_tool_content(tmp_path: Path) -> None:
    journal = TurnExecutionJournal(tmp_path / "journals")
    journal.record(
        "history_ready",
        history_payload={
            "tool_results": [
                {
                    "role": "tool",
                    "content": "api_key=super-secret-value",
                    "metadata": {"sensitive": True},
                }
            ]
        },
    )
    records = TurnExecutionJournal.load(journal.path)
    journal.close()

    serialized = str(records)
    assert "super-secret-value" not in serialized
    assert "<redacted-sensitive-tool-output>" in serialized


def test_committed_history_recovery_is_idempotent(tmp_path: Path) -> None:
    journal = TurnExecutionJournal(tmp_path / "journals")
    journal.record(
        "history_ready",
        history_payload={"assistant": {"role": "assistant"}, "tool_results": []},
    )
    journal.record("committed", changed=[])
    journal._release_for_later_recovery()
    recovered: list[dict[str, object]] = []

    first = TurnExecutionJournal.recover_all(
        tmp_path / "journals",
        history_writer=lambda payload: not recovered.append(payload),
    )
    second = TurnExecutionJournal.recover_all(
        tmp_path / "journals",
        history_writer=lambda payload: not recovered.append(payload),
    )

    assert first == [{"turn_id": journal.turn_id, "status": "history_persisted"}]
    assert second == []
    assert len(recovered) == 1


def test_recovery_restores_partially_applied_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _SimulatedCrash(BaseException):
        pass

    target = tmp_path / "value.txt"
    target.write_text("old", encoding="utf-8")
    journal = TurnExecutionJournal(tmp_path / "journals")
    transaction = WorkspaceTransaction(tmp_path, "crash", journal=journal)
    transaction.ensure_paths([target])
    (transaction.overlay / "value.txt").write_text("new", encoding="utf-8")
    original_replace = os.replace

    def replace_then_crash(source, destination) -> None:
        original_replace(source, destination)
        if Path(destination).resolve() == target.resolve():
            raise _SimulatedCrash()

    monkeypatch.setattr(os, "replace", replace_then_crash)
    with pytest.raises(_SimulatedCrash):
        transaction.commit()
    monkeypatch.setattr(os, "replace", original_replace)
    assert target.read_text(encoding="utf-8") == "new"
    journal._release_for_later_recovery()

    first = TurnExecutionJournal.recover_all(tmp_path / "journals")
    second = TurnExecutionJournal.recover_all(tmp_path / "journals")

    assert target.read_text(encoding="utf-8") == "old"
    assert first == [{"turn_id": journal.turn_id, "status": "rolled_back"}]
    assert second == []


async def test_unsafe_async_tool_times_out_into_supervised_cleanup(tmp_path: Path) -> None:
    registry = ToolRegistry()
    tool = _UnsafeTimedTool()
    registry.register(tool)
    executor = ToolExecutor(
        registry,
        PermissionPolicy(PermissionMode.AUTO),
        journal_dir=tmp_path / "journals",
    )
    batch = executor.begin_batch()
    call = ToolCall(tool.name, {}, id="u")
    assert batch.submit_streamed(call, ordinal=0)
    await asyncio.wait_for(tool.started.wait(), 1)

    await asyncio.wait_for(batch.abort("provider_error"), 1)

    assert tool.cancelled.is_set()
    assert batch._by_id["u"].result is not None
    assert batch._by_id["u"].result.metadata["error_type"] == "IndeterminateToolExecution"
    assert batch._by_id["u"].cleanup_pending is False
