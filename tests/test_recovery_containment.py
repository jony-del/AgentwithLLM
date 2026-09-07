from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from agent_core.cli import main
from agent_core.memory.config import MemoryConfig
from agent_core.models import Message
from agent_core.providers.fake import FakeProvider
from agent_core.react import ReActAgent, ReActConfig
from agent_core.tools.transaction import (
    JournalStorage, JournalWriteError, RecoveryRequiredError, TurnExecutionJournal,
)
from agent_core.transcript import TranscriptStore, load_transcript


@pytest.fixture
def storage(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("POLARIS_HOME", str(tmp_path / "private-state"))
    monkeypatch.chdir(workspace)
    return JournalStorage.user_state(workspace, "session-a", "run-a")


def pending(storage, *, names=()):
    journal = TurnExecutionJournal(storage)
    overlay = journal.create_overlay(journal.turn_id)
    journal.record("transaction_opened", overlay=str(overlay), workspace=str(storage.workspace))
    for name in names:
        target = storage.workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("new", encoding="utf-8")
        backup = overlay / "recovery" / name
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_text("old", encoding="utf-8")
    if names:
        journal.record("commit_started", overlay=str(overlay), changed=list(names), existed={name: True for name in names})
    journal._release_for_later_recovery()
    return journal, overlay


def snapshot(root):
    return {
        item.relative_to(root).as_posix(): (item.read_bytes(), item.stat().st_mtime_ns, item.stat().st_ino)
        for item in root.rglob("*") if item.is_file() and item.name != "recovery-audit.log"
    }


@pytest.mark.parametrize("index_state", ["missing", "corrupt", "incomplete"])
def test_preview_preserves_all_state_even_with_bad_index(storage, index_state):
    journal, overlay = pending(storage, names=("value.txt",))
    index = storage.recovery_root / ".open-journals.json"
    if index_state == "missing":
        index.unlink(missing_ok=True)
    else:
        index.write_text("{" if index_state == "corrupt" else '{"v":1,"paths":[]}', encoding="utf-8")
    # Preview must not recreate absent journal locks either.
    journal.path.with_suffix(".lock").unlink()
    before = snapshot(storage.anchor)
    report = TurnExecutionJournal.inspect_recovery(storage)
    assert report.blocked
    assert report.plans[0]["actions"] == [
        {"action": "restore_file", "target": str(storage.workspace / "value.txt")},
        {"action": "remove_overlay", "target": str(overlay)},
    ]
    assert snapshot(storage.anchor) == before
    assert TurnExecutionJournal.recover_all(storage)[0]["status"].startswith("would_")
    assert snapshot(storage.anchor) == before


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("sidecar", ["index", "index_lock", "audit", "turn_lock", "journal", "retention", "retention_lock"])
def test_hardlinked_state_files_cannot_write_external_victim(storage, tmp_path, sidecar, dry_run):
    journal, overlay = pending(storage)
    victim = tmp_path / "victim.txt"
    victim.write_text("keep", encoding="utf-8")
    target = {
        "index": storage.recovery_root / ".open-journals.json",
        "index_lock": storage.recovery_root / ".open-journals.lock",
        "audit": storage.recovery_root / "recovery-audit.log",
        "turn_lock": journal.path.with_suffix(".lock"),
        "journal": journal.path,
        "retention": storage.recovery_root / ".retention-state.json",
        "retention_lock": storage.recovery_root / ".retention.lock",
    }[sidecar]
    target.unlink(missing_ok=True)
    os.link(victim, target)
    outcomes = TurnExecutionJournal.recover_all(storage, dry_run=dry_run)
    assert any(item["status"] == "rejected" for item in outcomes)
    assert victim.read_text(encoding="utf-8") == "keep"
    assert overlay.exists()


@pytest.mark.parametrize("redirect", ["state_home", "session", "run", "overlays"])
def test_state_directory_redirects_are_rejected(storage, tmp_path, directory_redirect, redirect):
    journal, overlay = pending(storage)
    path = {
        "state_home": storage.private_root,
        "session": storage.recovery_root,
        "run": storage.run_root,
        "overlays": storage.run_root / "overlays",
    }[redirect]
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("keep", encoding="utf-8")
    moved = path.with_name(path.name + "-saved")
    assert path.is_relative_to(tmp_path) and moved.is_relative_to(tmp_path)
    path.rename(moved)
    directory_redirect(path, victim)
    outcomes = TurnExecutionJournal.recover_all(storage, dry_run=False)
    assert any(item["status"] == "rejected" for item in outcomes)
    assert (victim / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_existing_redirected_user_home_is_not_canonicalized_into_trust(tmp_path, monkeypatch, directory_redirect):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    redirect = tmp_path / "state"
    directory_redirect(redirect, victim)
    monkeypatch.setenv("POLARIS_HOME", str(redirect))
    with pytest.raises(JournalWriteError):
        JournalStorage.user_state(workspace, "s", "r")
    assert list(victim.iterdir()) == []


def test_shared_writable_state_directory_is_rejected(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    if os.name == "nt":
        assert state.resolve().is_relative_to(tmp_path.resolve())
        subprocess.run(["icacls", str(state), "/grant", "*S-1-1-0:(OI)(CI)M"], check=True, capture_output=True, timeout=10)
    else:
        state.chmod(0o777)
    monkeypatch.setenv("POLARIS_HOME", str(state))
    with pytest.raises(JournalWriteError, match="unsafe recovery state"):
        JournalStorage.user_state(workspace, "s", "r")


def test_no_shared_temp_fallback_when_user_home_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.delenv("POLARIS_HOME")
    def unavailable():
        raise RuntimeError("home unavailable")
    monkeypatch.setattr(Path, "home", unavailable)
    with pytest.raises(JournalWriteError, match="cannot establish"):
        JournalStorage.user_state(tmp_path, "s", "r")


def test_apply_rejects_target_changed_by_audit_callback(storage):
    journal, overlay = pending(storage, names=("value.txt",))
    def change_target(event):
        if event["phase"] == "before":
            (storage.workspace / "value.txt").write_text("user update", encoding="utf-8")
    outcomes = TurnExecutionJournal.recover_all(storage, dry_run=False, audit_writer=change_target)
    assert outcomes[0]["status"] == "recovery_failed"
    assert (storage.workspace / "value.txt").read_text(encoding="utf-8") == "user update"
    assert (overlay / "recovery/value.txt").read_text(encoding="utf-8") == "old"
    assert TurnExecutionJournal.load(journal.path)[-1]["state"] == "commit_started"


def test_apply_failure_retains_all_backups_and_releases_ownership(storage, monkeypatch):
    journal, overlay = pending(storage, names=("one.txt", "two.txt"))
    replace = os.replace
    def fail_one(source, destination):
        if Path(destination) == storage.workspace / "one.txt":
            raise OSError("injected replacement failure")
        replace(source, destination)
    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_one)
        outcomes = TurnExecutionJournal.recover_all(storage, dry_run=False)
    assert outcomes[0]["status"] == "recovery_failed"
    assert (storage.workspace / "one.txt").read_text(encoding="utf-8") == "new"
    assert (storage.workspace / "two.txt").read_text(encoding="utf-8") == "old"
    assert all((overlay / "recovery" / name).read_text(encoding="utf-8") == "old" for name in ("one.txt", "two.txt"))
    assert TurnExecutionJournal.recover_all(storage, dry_run=False) == [{"turn_id": journal.turn_id, "status": "rolled_back"}]
    assert TurnExecutionJournal.recover_all(storage, dry_run=False) == []


def test_audit_failure_prevents_workspace_and_journal_mutations(storage, monkeypatch):
    journal, overlay = pending(storage, names=("value.txt",))
    before = snapshot(storage.anchor)
    monkeypatch.setattr(TurnExecutionJournal, "_audit", staticmethod(lambda *args: False))
    assert TurnExecutionJournal.recover_all(storage, dry_run=False) == [{"turn_id": journal.turn_id, "status": "audit_failed"}]
    assert snapshot(storage.anchor) == before
    assert overlay.exists()


def test_busy_journal_reports_busy_without_finalizing(storage):
    journal = TurnExecutionJournal(storage)
    try:
        journal.record("turn_opened")
        raw = journal.path.read_bytes()
        assert TurnExecutionJournal.recover_all(storage, dry_run=False) == [{"turn_id": journal.turn_id, "status": "busy"}]
        assert journal.path.read_bytes() == raw
    finally:
        journal.close()


def make_agent(storage, tmp_path, *, session_id="session-a", session_dir=""):
    return ReActAgent(
        FakeProvider(),
        ReActConfig(run_dir=str(tmp_path / "runs"), session_dir=session_dir,
                    memory=MemoryConfig(enabled=False), project_instructions=False, git_context=False),
        workspace=storage.workspace, session_id=session_id,
    )


async def test_agent_startup_previews_and_run_blocks_until_explicit_recovery(storage, tmp_path, monkeypatch):
    journal, overlay = pending(storage, names=("value.txt",))
    monkeypatch.setattr(TurnExecutionJournal, "prune_terminal", lambda *a, **k: pytest.fail("startup must not prune journals"))
    agent = make_agent(storage, tmp_path)
    try:
        assert overlay.exists()
        with pytest.raises(RecoveryRequiredError):
            await agent.run("say hello")
        assert agent.provider.inner.calls == 0
        assert (storage.workspace / "value.txt").read_text(encoding="utf-8") == "new"
        assert TurnExecutionJournal.recover_all(storage, dry_run=False)[0]["status"] == "rolled_back"
        await agent.run("say hello")
        assert agent.provider.inner.calls > 0
    finally:
        agent.logger.close()


async def test_resume_rejects_pending_target_before_replacing_current_runtime(storage, tmp_path):
    pending(storage)
    root = str(tmp_path / "transcripts")
    transcript = TranscriptStore(root, storage.workspace, storage.session_id)
    await transcript.append_message(Message("user", "old history"))
    transcript.close()
    loaded = load_transcript(transcript.path)
    agent = make_agent(storage, tmp_path, session_id="session-b", session_dir=root)
    previous_runtime = agent.runtime
    try:
        with pytest.raises(RecoveryRequiredError):
            await agent.resume_loaded_session(loaded)
        assert agent.session_id == "session-b"
        assert agent.runtime is previous_runtime
    finally:
        agent.logger.close()


def test_cli_previews_and_applies_without_constructing_an_agent(storage, monkeypatch, capsys):
    journal, overlay = pending(storage, names=("value.txt",))
    monkeypatch.setattr("agent_core.cli.build_agent", lambda *a, **k: pytest.fail("recovery cannot construct an agent"))
    assert main(["recovery", "--session-id", storage.session_id, "--json"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["dry_run"] is True and preview["blocked"] is True
    assert preview["plans"][0]["turn_id"] == journal.turn_id
    assert overlay.exists()
    assert main(["recovery", "--session-id", storage.session_id, "--apply", "--json"]) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["blocked"] is False
    assert applied["outcomes"][0]["status"] == "rolled_back"
    assert not overlay.exists()
    assert main(["recovery", "--session-id", storage.session_id, "--apply", "--json"]) == 0


def test_cli_startup_blocks_before_provider_sandbox_or_mcp(storage, monkeypatch, capsys):
    pending(storage)
    monkeypatch.setattr("agent_core.cli._make_provider", lambda *a: pytest.fail("provider must not be constructed"))
    monkeypatch.setattr("agent_core.cli.get_shared_manager", lambda *a, **k: pytest.fail("sandbox must not start"))
    monkeypatch.setattr("agent_core.cli._start_mcp", lambda *a, **k: pytest.fail("MCP must not start"))
    assert main(["run", "hello", "--provider", "fake", "--session-id", storage.session_id]) == 1
    assert "polaris recovery" in capsys.readouterr().err


@pytest.mark.parametrize("arguments", [[], ["--session-id", "../escape"], ["--session-id", "s", "--apply", "--dry-run"]])
def test_recovery_cli_rejects_invalid_arguments(arguments):
    with pytest.raises(SystemExit) as error:
        main(["recovery", *arguments])
    assert error.value.code == 2


@pytest.mark.parametrize("version", [[], {}, True, 1.0])
def test_invalid_schema_types_are_reported_without_crashing(storage, version):
    storage.run_root.mkdir(parents=True)
    path = storage.run_root / ("a" * 32 + ".jsonl")
    path.write_text(json.dumps({"v": version}) + "\n", encoding="utf-8")
    assert TurnExecutionJournal.recover_all(storage)[0]["reason"] == "unsupported_schema"


@pytest.mark.parametrize("schema_version", [3, 4])
def test_legacy_schema_remains_readable_after_explicit_recovery(storage, schema_version):
    journal = TurnExecutionJournal(storage, _schema_version=schema_version)
    overlay = journal.create_overlay(journal.turn_id)
    journal.record("transaction_opened", overlay=str(overlay), workspace=str(storage.workspace))
    journal._release_for_later_recovery()
    assert TurnExecutionJournal.recover_all(storage, dry_run=False)[0]["status"] == "rolled_back"
    records = TurnExecutionJournal.load(journal.path)
    assert records[-1]["state"] == "journal_closed"
    assert {item["v"] for item in records} == {schema_version}
    assert TurnExecutionJournal.recover_all(storage, dry_run=False) == []


@pytest.mark.parametrize("failure", ["cleanup", "terminal_record"])
def test_completed_actions_are_not_repeated_after_cleanup_or_terminal_failure(storage, monkeypatch, failure):
    journal, overlay = pending(storage, names=("value.txt",))
    original_record = TurnExecutionJournal.record
    with monkeypatch.context() as patch:
        if failure == "cleanup":
            def fail_cleanup(*args):
                raise OSError("cleanup failed")
            patch.setattr(TurnExecutionJournal, "discard_overlay", fail_cleanup)
        else:
            def fail_record(self, state, **payload):
                if state == "rolled_back":
                    raise JournalWriteError("terminal record failed")
                original_record(self, state, **payload)
            patch.setattr(TurnExecutionJournal, "record", fail_record)
        assert TurnExecutionJournal.recover_all(storage, dry_run=False)[0]["status"] == "recovery_failed"
    assert TurnExecutionJournal.load(journal.path)[-1]["state"] == "recovery_actions_applied"
    target = storage.workspace / "value.txt"
    assert target.read_text(encoding="utf-8") == "old"
    target.write_text("later edit", encoding="utf-8")
    preview = TurnExecutionJournal.inspect_recovery(storage)
    assert preview.plans[0]["action"] == "cleanup_overlay"
    assert all(item["action"] != "restore_file" for item in preview.plans[0]["actions"])
    assert TurnExecutionJournal.recover_all(storage, dry_run=False)[0]["status"] == "rolled_back"
    assert target.read_text(encoding="utf-8") == "later edit"
    assert not overlay.exists()


def committed_history(storage, transcript):
    journal = TurnExecutionJournal(storage)
    overlay = journal.create_overlay(journal.turn_id)
    journal.record("transaction_opened", overlay=str(overlay), workspace=str(storage.workspace))
    journal.record("commit_started", overlay=str(overlay), changed=["value.txt"], existed={"value.txt": True})
    journal.record("history_ready", history_payload={
        "session_id": storage.session_id, "transcript_path": str(transcript.path),
        "assistant": Message("assistant", "persisted result").to_dict(),
        "tool_results": [], "execution_manifest": {},
    })
    journal.record("committed", changed=["value.txt"])
    journal.discard_overlay(overlay)  # Normal commit cleanup already removed backups.
    journal._release_for_later_recovery()
    (storage.workspace / "value.txt").write_text("committed", encoding="utf-8")
    return journal


def test_committed_history_can_be_recovered_after_backup_cleanup(storage, tmp_path, capsys):
    transcript = TranscriptStore(tmp_path / "transcripts", storage.workspace, storage.session_id)
    journal = committed_history(storage, transcript)
    command = ["recovery", "--session-id", storage.session_id, "--session-dir", str(tmp_path / "transcripts"), "--json"]
    assert main(command) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["plans"][0]["actions"] == [{"action": "append_history", "target": str(transcript.path)}]
    assert not transcript.path.exists()
    assert main([*command, "--apply"]) == 0
    assert json.loads(capsys.readouterr().out)["outcomes"] == [{"turn_id": journal.turn_id, "status": "history_persisted"}]
    raw = transcript.path.read_bytes()
    assert main([*command, "--apply"]) == 0
    assert transcript.path.read_bytes() == raw
    assert (storage.workspace / "value.txt").read_text(encoding="utf-8") == "committed"


def test_history_target_mismatch_is_rejected_without_writes(storage, tmp_path, capsys):
    transcript = TranscriptStore(tmp_path / "transcripts", storage.workspace, storage.session_id)
    journal = committed_history(storage, transcript)
    raw = journal.path.read_bytes()
    assert main(["recovery", "--session-id", storage.session_id, "--session-dir", str(tmp_path / "wrong"), "--apply", "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["outcomes"][0]["reason"] == "history_target_mismatch"
    assert not transcript.path.exists()
    assert not (tmp_path / "wrong").exists()
    assert journal.path.read_bytes() == raw


def test_history_writer_failure_preserves_pending_journal(storage, tmp_path):
    transcript = TranscriptStore(tmp_path / "transcripts", storage.workspace, storage.session_id)
    journal = committed_history(storage, transcript)
    raw = journal.path.read_bytes()
    assert TurnExecutionJournal.recover_all(
        storage, dry_run=False, history_path=transcript.path, history_writer=lambda payload: False,
    )[0]["status"] == "recovery_failed"
    assert journal.path.read_bytes() == raw
    assert TurnExecutionJournal.recover_all(
        storage, dry_run=False, history_writer=transcript.recover_tool_round,
    )[0]["status"] == "history_persisted"


def test_unknown_external_effect_is_never_replayed_or_terminalized(storage):
    journal = TurnExecutionJournal(storage)
    journal.record("external_intent", ordinal=0, tool_name="remote_write")
    journal._release_for_later_recovery()
    raw = journal.path.read_bytes()
    outcomes = TurnExecutionJournal.recover_all(storage, dry_run=False)
    assert outcomes == [{"turn_id": journal.turn_id, "status": "IndeterminateExternalEffect"}]
    assert journal.path.read_bytes() == raw
    assert TurnExecutionJournal.inspect_recovery(storage).blocked


def test_preview_of_absent_state_does_not_create_directories(storage):
    assert not storage.private_root.exists()
    assert TurnExecutionJournal.inspect_recovery(storage).outcomes == []
    assert not storage.private_root.exists()


def test_all_targets_are_validated_before_any_restore(storage):
    journal, overlay = pending(storage, names=("one.txt", "two.txt"))
    target = storage.workspace / "one.txt"
    target.unlink()
    target.mkdir()
    assert TurnExecutionJournal.recover_all(storage, dry_run=False)[0]["reason"] == "unsafe_recovery_target"
    assert (storage.workspace / "two.txt").read_text(encoding="utf-8") == "new"
    assert overlay.exists()


def test_workspace_parent_redirected_after_preview_cannot_escape(storage, tmp_path, directory_redirect):
    journal, overlay = pending(storage, names=("sub/value.txt",))
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "value.txt").write_text("keep", encoding="utf-8")
    def redirect_parent(event):
        if event["phase"] == "before":
            parent = storage.workspace / "sub"
            saved = storage.workspace / "saved"
            assert parent.is_relative_to(tmp_path) and saved.is_relative_to(tmp_path)
            parent.rename(saved)
            directory_redirect(parent, victim)
    outcomes = TurnExecutionJournal.recover_all(storage, dry_run=False, audit_writer=redirect_parent)
    assert outcomes[0]["status"] == "recovery_failed"
    assert (victim / "value.txt").read_text(encoding="utf-8") == "keep"
    assert (overlay / "recovery/sub/value.txt").read_text(encoding="utf-8") == "old"


def test_only_live_in_memory_owners_are_exempt_from_run_gate(storage):
    journal = TurnExecutionJournal(storage)
    journal.record("turn_opened")
    try:
        # A parent's active journal must not block its own subagents.
        assert not TurnExecutionJournal.require_recovered(storage).blocked
        raw = journal.path.read_bytes()
    finally:
        journal._release_for_later_recovery()
    # PID/process_token/nonce on disk are unchanged but no longer confer liveness.
    assert journal.path.read_bytes() == raw
    with pytest.raises(RecoveryRequiredError):
        TurnExecutionJournal.require_recovered(storage)
