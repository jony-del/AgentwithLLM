"""Pinning tests for the frozen core contracts (see ``docs/core-contracts.md``).

These tests lock in the invariants that later phases build on:

- ``ExecutionScope`` monotonic tightening, shared cancellation/registry, budget
  semantics, and close-time reaping;
- ``RecoveryState`` as the single authority for turn-journal states;
- ``Message`` identity (uuid / origin_id / version / round_id) derivation rules;
- the resume/fork/cross-project product contract.

Already-covered neighbours are referenced, not duplicated: compaction boundary
persistence lives in ``test_transcript_boundary.py``; recovery containment and
ownership live in ``test_recovery_containment.py`` / ``test_recovery_security.py``;
fork chain cloning lives in ``test_transcript.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path

import pytest

from agent_core.execution import ExecutionScope
from agent_core.models import Message
from agent_core.tools.transaction import (
    LEGACY_STATE_ALIASES,
    TERMINAL_RECOVERY_STATES,
    JournalStorage,
    JournalWriteError,
    RecoveryState,
    TurnExecutionJournal,
)

# --------------------------------------------------------------------------
# ExecutionScope contract
# --------------------------------------------------------------------------


def test_scope_child_only_tightens(tmp_path: Path) -> None:
    now = time.monotonic()
    parent = ExecutionScope.for_workspace(
        tmp_path, deadline=now + 100, network="deny", workspace_writable=False
    )
    child = parent.child(deadline=now + 1000, network="allow", workspace_writable=True)
    assert child.deadline == parent.deadline
    assert child.network == "deny"
    assert child.workspace_writable is False
    tighter = parent.child(deadline=now + 10)
    assert tighter.deadline == now + 10


def test_scope_child_shares_cancellation_token_and_task_registry(tmp_path: Path) -> None:
    parent = ExecutionScope.for_workspace(tmp_path)
    child = parent.child()
    grandchild = child.child(deadline=time.monotonic() + 1)
    assert child.cancellation is parent.cancellation
    assert grandchild.cancellation is parent.cancellation
    assert child.tasks is parent.tasks
    assert grandchild.tasks is parent.tasks


def test_scope_remaining_budget_bounds_requested(tmp_path: Path) -> None:
    unbounded = ExecutionScope.for_workspace(tmp_path)
    assert unbounded.remaining_budget(30.0) == 30.0
    assert unbounded.remaining_budget() is None
    scope = ExecutionScope.for_workspace(tmp_path, deadline=time.monotonic() + 50)
    assert scope.remaining_budget(10.0) == 10.0
    assert 0.0 <= scope.remaining_budget(10_000.0) <= 50.0
    expired = ExecutionScope.for_workspace(tmp_path, deadline=time.monotonic() - 1)
    assert expired.remaining_budget(5.0) == 0.0


def test_scope_cancel_probe_folds_into_token(tmp_path: Path) -> None:
    scope = ExecutionScope.for_workspace(tmp_path, cancel_probe=lambda: True)
    with pytest.raises(asyncio.CancelledError):
        scope.raise_if_cancelled()
    assert scope.cancellation.cancelled()


async def test_scope_run_awaitable_enforces_budget(tmp_path: Path) -> None:
    scope = ExecutionScope.for_workspace(tmp_path, deadline=time.monotonic() + 0.05)
    with pytest.raises(TimeoutError):
        await scope.run_awaitable(asyncio.sleep(5))
    expired = ExecutionScope.for_workspace(tmp_path, deadline=time.monotonic() - 0.01)
    with pytest.raises(TimeoutError):
        expired.raise_if_cancelled()


async def test_scope_close_cancels_token_and_reaps_tasks(tmp_path: Path) -> None:
    scope = ExecutionScope.for_workspace(tmp_path)
    started = asyncio.Event()

    async def worker() -> None:
        started.set()
        await asyncio.sleep(60)

    task = scope.create_task(worker())
    await started.wait()
    await scope.close()
    assert scope.cancelled()
    assert task.done()


# --------------------------------------------------------------------------
# RecoveryState contract
# --------------------------------------------------------------------------

EXPECTED_STATES = {
    "discovered",
    "admitted",
    "authorized",
    "staged",
    "cleanup_required",
    "cleanup_complete",
    "external_intent",
    "external_outcome_committed",
    "turn_validated",
    "history_ready",
    "history_persisted",
    "transaction_opened",
    "commit_started",
    "committed",
    "recovery_required",
    "rolled_back",
    "recovery_actions_applied",
    "journal_closed",
    "indeterminate_external_effect",
    "external_effect_history_missing",
}


def test_recovery_state_membership_is_frozen() -> None:
    assert {state.value for state in RecoveryState} == EXPECTED_STATES


def test_terminal_states_derive_from_enum() -> None:
    assert TurnExecutionJournal.TERMINAL_STATES == TERMINAL_RECOVERY_STATES
    assert set(TERMINAL_RECOVERY_STATES) <= EXPECTED_STATES
    assert "journal_closed" in TERMINAL_RECOVERY_STATES


def test_journal_record_rejects_unknown_state(tmp_path: Path) -> None:
    storage = JournalStorage.local(tmp_path / "journals", workspace=tmp_path)
    journal = TurnExecutionJournal(storage)
    try:
        with pytest.raises(JournalWriteError):
            journal.record("not_a_state")
        # The pre-contract test-only name stays rejected (writes must use the enum).
        with pytest.raises(JournalWriteError):
            journal.record_telemetry("turn_opened")
    finally:
        journal.close()


def test_production_journal_writes_use_contract_states() -> None:
    import agent_core.tools.executor as executor_module
    import agent_core.tools.transaction as transaction_module

    literals: set[str] = set()
    for module in (executor_module, transaction_module):
        source = Path(str(module.__file__)).read_text(encoding="utf-8")
        literals |= set(re.findall(r'\.record\(\s*"([a-z_]+)"', source))
        literals |= set(re.findall(r'record_telemetry\(\s*"([a-z_]+)"', source))
    assert literals <= EXPECTED_STATES


def _write_legacy_journal(path: Path, turn_id: str, owner: dict[str, str], states: list[str]) -> None:
    """Write a schema-v3 journal with a valid checksum chain by hand."""

    previous = ""
    lines: list[str] = []
    for sequence, state in enumerate(states):
        record = {
            "v": 3,
            "turn_id": turn_id,
            "sequence": sequence,
            "state": state,
            "ts": 1.0,
            "pid": 1234,
            "process_token": "test",
            "owner": owner,
            "previous_checksum": previous,
        }
        canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        checksum = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        record["checksum"] = checksum
        lines.append(json.dumps(record, ensure_ascii=False))
        previous = checksum
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_legacy_state_aliases_normalize_on_load(tmp_path: Path) -> None:
    turn_id = "a" * 32
    owner = {
        "project_id": "p",
        "session_id": "s",
        "run_id": "r",
        "workspace": str(tmp_path),
        "nonce": "b" * 32,
    }
    path = tmp_path / f"{turn_id}.jsonl"
    _write_legacy_journal(path, turn_id, owner, ["transaction_opened", "external_outcome"])
    records, error = TurnExecutionJournal._load_verified(path)
    assert error is None
    assert LEGACY_STATE_ALIASES == {"external_outcome": "external_outcome_committed"}
    assert [item["state"] for item in records] == [
        "transaction_opened",
        "external_outcome_committed",
    ]


# --------------------------------------------------------------------------
# Message identity contract
# --------------------------------------------------------------------------


def test_evolve_message_preserves_identity_and_bumps_version() -> None:
    from agent_core.compression import _evolve_message

    original = Message("user", "x" * 100)
    evolved = _evolve_message(original, "short", "snip")
    assert evolved.uuid == original.uuid
    assert evolved.origin_id == original.uuid
    assert evolved.version == original.version + 1


def test_message_identity_roundtrip() -> None:
    message = Message(
        "assistant",
        "working",
        metadata={"tool_calls": [{"id": "call_1", "name": "t", "arguments": {}}]},
    )
    restored = Message.from_dict(message.to_dict())
    assert restored.uuid == message.uuid
    assert restored.origin_id == message.origin_id
    assert restored.version == message.version
    assert restored.round_id == message.round_id
    assert restored.round_id == message.uuid  # default: own uuid for tool-call rounds


# --------------------------------------------------------------------------
# resume / fork / cross-project product contract
# --------------------------------------------------------------------------


def _session_args(**overrides) -> argparse.Namespace:
    base = {"resume": None, "continue_": False, "fork_session": False, "session_id": None}
    base.update(overrides)
    return argparse.Namespace(**base)


async def test_cross_project_resume_rejected_but_fork_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_core.cli import _resolve_session
    from agent_core.transcript import TranscriptStore, project_dir

    # Short names keep the sanitized transcript path under the Windows 260-char limit.
    session_root = tmp_path / "s"
    project_a = tmp_path / "a"
    project_b = tmp_path / "b"
    project_a.mkdir()
    project_b.mkdir()
    project_dir(session_root, project_a).mkdir(parents=True)
    source_id = "c" * 32
    store = TranscriptStore(session_root, project_a, source_id)
    try:
        first = Message("user", "hello")
        second = Message("assistant", "hi")
        second.parent_uuid = first.uuid
        assert await store.append_message(first)
        assert await store.append_message(second)
    finally:
        store.close()

    monkeypatch.chdir(project_b)
    with pytest.raises(RuntimeError, match="belongs to project"):
        _resolve_session(_session_args(resume=source_id), str(session_root))

    selection = _resolve_session(_session_args(resume=source_id, fork_session=True), str(session_root))
    assert selection.action == "fork"
    assert selection.descriptor.session_id != source_id
    assert selection.descriptor.workspace == project_b.resolve()
    cloned = list(selection.history)
    assert [message.content for message in cloned] == ["hello", "hi"]
    # Forked chain: every uuid is fresh and parents are re-linked inside the clone.
    assert {message.uuid for message in cloned}.isdisjoint({first.uuid, second.uuid})
    assert cloned[0].parent_uuid is None
    assert cloned[1].parent_uuid == cloned[0].uuid
    # The source transcript still holds the original chain.
    from agent_core.transcript import build_chain, load_transcript

    original = build_chain(load_transcript(store.path))
    assert [message.uuid for message in original] == [first.uuid, second.uuid]
