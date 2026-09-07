from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent_core.memory.config import MemoryConfig
from agent_core.providers.fake import FakeProvider
from agent_core.react import ReActAgent, ReActConfig
from agent_core.tools.transaction import JournalStorage, TurnExecutionJournal


def _unfinished_transaction(
    storage: JournalStorage,
    *,
    changed: object | None = None,
    existed: object | None = None,
) -> tuple[TurnExecutionJournal, Path]:
    journal = TurnExecutionJournal(storage)
    overlay = journal.create_overlay(journal.turn_id)
    journal.record(
        "transaction_opened",
        overlay=str(overlay),
        workspace=str(storage.workspace),
    )
    if changed is not None:
        journal.record(
            "commit_started",
            overlay=str(overlay),
            changed=changed,
            existed={} if existed is None else existed,
        )
    journal._release_for_later_recovery()
    return journal, overlay


def _assert_rejected(outcomes: list[dict[str, str]], reason: str) -> None:
    assert len(outcomes) == 1
    assert outcomes[0]["status"] in {"rejected", "foreign"}
    assert outcomes[0]["reason"] == reason


def test_unsigned_workspace_style_journal_cannot_delete_external_overlay(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("safe", encoding="utf-8")
    storage = JournalStorage.local(tmp_path / "trusted", workspace=workspace)
    storage.run_root.mkdir(parents=True)
    turn_id = "a" * 32
    (storage.run_root / f"{turn_id}.jsonl").write_text(
        json.dumps(
            {
                "state": "transaction_opened",
                "turn_id": turn_id,
                "overlay": str(victim),
                "workspace": str(workspace),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    outcomes = TurnExecutionJournal.recover_all(storage)

    _assert_rejected(outcomes, "unsupported_schema")
    assert (victim / "keep.txt").read_text(encoding="utf-8") == "safe"


def test_checksummed_journal_still_rejects_overlay_outside_controlled_root(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    storage = JournalStorage.local(tmp_path / "journals", workspace=workspace)
    journal = TurnExecutionJournal(storage)
    journal.record(
        "transaction_opened",
        overlay=str(victim),
        workspace=str(workspace),
    )
    journal._release_for_later_recovery()

    outcomes = TurnExecutionJournal.recover_all(storage)

    _assert_rejected(outcomes, "overlay_outside_controlled_root")
    assert victim.is_dir()


@pytest.mark.parametrize(
    "malicious",
    [
        "../../victim.txt",
        "/tmp/victim.txt",
        "C:/victim.txt",
        "C:\\victim.txt",
        "//server/share/victim.txt",
        "\\\\server\\share\\victim.txt",
        "\\\\?\\C:\\victim.txt",
        "\\\\.\\C:\\victim.txt",
        "NUL",
        "dir/CON.txt",
    ],
)
def test_recovery_rejects_noncanonical_and_platform_escape_paths(tmp_path: Path, malicious: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("safe", encoding="utf-8")
    storage = JournalStorage.local(tmp_path / "journals", workspace=workspace)
    _unfinished_transaction(storage, changed=[malicious], existed={malicious: False})

    outcomes = TurnExecutionJournal.recover_all(storage)

    _assert_rejected(outcomes, "unsafe_changed_path")
    assert victim.read_text(encoding="utf-8") == "safe"


def test_recovery_rejects_symlink_or_junction_parent_escape(tmp_path: Path, directory_redirect) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    victim_dir = tmp_path / "victim"
    victim_dir.mkdir()
    victim = victim_dir / "keep.txt"
    victim.write_text("safe", encoding="utf-8")
    directory_redirect(workspace / "redirect", victim_dir)
    storage = JournalStorage.local(tmp_path / "journals", workspace=workspace)
    relative = "redirect/keep.txt"
    _unfinished_transaction(storage, changed=[relative], existed={relative: False})

    outcomes = TurnExecutionJournal.recover_all(storage)

    _assert_rejected(outcomes, "canonical_containment_failed")
    assert victim.read_text(encoding="utf-8") == "safe"


@pytest.mark.parametrize("foreign_kind", ["project", "session", "run"])
def test_foreign_owner_cannot_consume_or_finalize_journal(tmp_path: Path, foreign_kind: str) -> None:
    workspace = tmp_path / "workspace-a"
    workspace.mkdir()
    root = tmp_path / "journals"
    owner = JournalStorage.local(
        root,
        workspace=workspace,
        session_id="session-a",
        run_id="run-a",
    )
    journal, overlay = _unfinished_transaction(owner)
    if foreign_kind == "project":
        other_workspace = tmp_path / "workspace-b"
        other_workspace.mkdir()
        foreign = JournalStorage.local(
            root, workspace=other_workspace, session_id="session-a", run_id="run-a"
        )
        reason = "foreign_project"
    elif foreign_kind == "session":
        foreign = JournalStorage.local(root, workspace=workspace, session_id="session-b", run_id="run-a")
        reason = "foreign_session"
    else:
        foreign = JournalStorage.local(root, workspace=workspace, session_id="session-a", run_id="run-b")
        reason = "foreign_run"

    _assert_rejected(TurnExecutionJournal.recover_all(foreign), reason)
    assert overlay.exists()
    assert TurnExecutionJournal.load(journal.path)[-1]["state"] == "transaction_opened"

    assert TurnExecutionJournal.recover_all(owner, dry_run=False) == [{"turn_id": journal.turn_id, "status": "rolled_back"}]


def test_same_session_can_recover_prior_run_but_foreign_session_cannot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state_home = tmp_path / "state"
    monkeypatch.setenv("POLARIS_HOME", str(state_home))
    first_run = JournalStorage.user_state(workspace, "session-a", "run-a")
    journal, overlay = _unfinished_transaction(first_run)

    foreign_session = JournalStorage.user_state(workspace, "session-b", "run-b")
    assert TurnExecutionJournal.recover_all(foreign_session) == []
    assert overlay.exists()

    next_run = JournalStorage.user_state(workspace, "session-a", "run-b")
    assert TurnExecutionJournal.recover_all(next_run, dry_run=False) == [
        {"turn_id": journal.turn_id, "status": "rolled_back"}
    ]
    assert not overlay.exists()


@pytest.mark.parametrize("corruption", ["checksum", "truncated"])
def test_corrupt_or_truncated_journal_is_report_only(tmp_path: Path, corruption: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    storage = JournalStorage.local(tmp_path / "journals", workspace=workspace)
    journal, overlay = _unfinished_transaction(storage)
    raw = journal.path.read_bytes()
    if corruption == "checksum":
        records = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        records[-1]["checksum"] = "0" * 64
        journal.path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        reason = "checksum_mismatch"
    else:
        journal.path.write_bytes(raw[:-1])
        reason = "journal_truncated"

    outcomes = TurnExecutionJournal.recover_all(storage)

    _assert_rejected(outcomes, reason)
    assert overlay.exists()


def test_checksummed_field_type_error_is_report_only(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    storage = JournalStorage.local(tmp_path / "journals", workspace=workspace)
    journal, overlay = _unfinished_transaction(
        storage,
        changed="not-a-list",
        existed={},
    )

    outcomes = TurnExecutionJournal.recover_all(storage)

    _assert_rejected(outcomes, "invalid_commit_fields")
    assert overlay.exists()
    assert TurnExecutionJournal.load(journal.path)


def test_dry_run_audits_plan_without_mutating_then_recovery_restores(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "value.txt"
    target.write_text("new", encoding="utf-8")
    storage = JournalStorage.local(tmp_path / "journals", workspace=workspace)
    journal, overlay = _unfinished_transaction(
        storage,
        changed=["value.txt"],
        existed={"value.txt": True},
    )
    backup = overlay / "recovery" / "value.txt"
    backup.parent.mkdir(parents=True)
    backup.write_text("old", encoding="utf-8")

    preview = TurnExecutionJournal.recover_all(storage, dry_run=True)

    assert preview == [{"turn_id": journal.turn_id, "status": "would_rollback_workspace"}]
    assert target.read_text(encoding="utf-8") == "new"
    assert backup.exists()
    audit_lines = [
        json.loads(line)
        for line in (storage.recovery_root / "recovery-audit.log").read_text(encoding="utf-8").splitlines()
    ]
    assert audit_lines[-1]["dry_run"] is True
    audit_checksum = audit_lines[-1].pop("checksum")
    canonical = json.dumps(audit_lines[-1], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert audit_checksum == hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    assert TurnExecutionJournal.recover_all(storage, dry_run=False) == [
        {"turn_id": journal.turn_id, "status": "rolled_back"}
    ]
    assert target.read_text(encoding="utf-8") == "old"
    assert not overlay.exists()


def test_agent_startup_ignores_workspace_journal_before_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("safe", encoding="utf-8")
    old_root = workspace / "runs" / ".turn-journals"
    old_root.mkdir(parents=True)
    (old_root / "evil.jsonl").write_text(
        json.dumps(
            {
                "state": "transaction_opened",
                "turn_id": "evil",
                "overlay": str(victim),
                "workspace": str(workspace),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    state_home = tmp_path / "external-state"
    monkeypatch.setenv("POLARIS_HOME", str(state_home))

    agent = ReActAgent(
        FakeProvider(),
        ReActConfig(
            run_dir=str(workspace / "runs"),
            session_dir="",
            memory=MemoryConfig(enabled=False),
            project_instructions=False,
            git_context=False,
        ),
        workspace=workspace,
    )

    assert (victim / "keep.txt").read_text(encoding="utf-8") == "safe"
    assert (old_root / "evil.jsonl").exists()
    with pytest.raises(ValueError):
        agent.executor.journal_dir.resolve().relative_to(workspace.resolve())
    assert agent.executor.journal_storage.session_id == agent.session_id
    assert agent.executor.journal_storage.run_id == agent.logger.run_id
