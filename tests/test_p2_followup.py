from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import struct
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from agent_core.compression import CompressionConfig, CompressionPipeline
from agent_core.compression_summary import build_summarizer
from agent_core.hook_adapters import (
    CommandHookAdapter,
    HookFailedError,
    bounded_hook_payload,
)
from agent_core.hooks import (
    ExternalHookSpec,
    HookContext,
    HookEvent,
    HookLimitsConfig,
)
from agent_core.memory.config import MemoryConfig
from agent_core.memory.extraction import MemoryExtractor
from agent_core.memory.repository import MemoryRepository
from agent_core.memory.store import RepositoryMemoryStore
from agent_core.models import LLMResult, Message
from agent_core.process_supervisor import ProcessSupervisor
from agent_core.providers.base import ProviderConfig
from agent_core.providers.openai_compat import OpenAICompatProvider
from agent_core.retention import prune_sessions
from agent_core.react import ReActAgent, ReActConfig
from agent_core.scheduler import SchedulerStore
from agent_core.session import SessionRetentionConfig
from agent_core.storage import JSONLRunLogger
from agent_core.tool_config import ShellToolConfig
from agent_core.tools.transaction import JournalStorage, TurnExecutionJournal
from agent_core.transcript import TranscriptStore, build_chain, load_transcript, project_dir
from agent_core.workflow_runtime import (
    WorkflowError,
    _MAX_WORKFLOW_FRAME_BYTES,
    _encode_frame,
    _read_frame,
)


class _Provider:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    async def complete(self, *args, **kwargs) -> LLMResult:
        self.calls += 1
        return LLMResult(self.content)


async def test_memory_poison_is_dead_lettered_and_cursor_advances(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    store = RepositoryMemoryStore(repository)
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    provider = _Provider(json.dumps([
        {"operation": "create", "content": "prefers compact output", "kind": "preference"},
        {"operation": "create", "content": secret, "kind": "fact"},
    ]))
    message = Message("user", "remember these")
    extractor = MemoryExtractor(provider, store)

    stored = await extractor.extract([message], source_run_id="run", cursor_key="session")

    assert len(stored) == 1
    state_text = (repository.root / ".extraction-state.json").read_text(encoding="utf-8")
    state = json.loads(state_text)
    assert state["cursors"]["session"] == message.uuid
    assert state["dead_letters"][0]["reason"] == "secret_detected"
    assert secret not in state_text
    again = MemoryExtractor(provider, store)
    assert await again.extract([message], cursor_key="session") == []
    assert provider.calls == 1


async def test_memory_concurrent_extractors_cannot_regress_cursor(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    older_started = asyncio.Event()
    release_older = asyncio.Event()

    class DelayedProvider(_Provider):
        async def complete(self, *args, **kwargs) -> LLMResult:
            older_started.set()
            await release_older.wait()
            return await super().complete(*args, **kwargs)

    messages = [Message("user", f"m{index}") for index in range(3)]
    older = MemoryExtractor(
        DelayedProvider("[]"), RepositoryMemoryStore(repository)
    )
    newer = MemoryExtractor(_Provider("[]"), RepositoryMemoryStore(repository))

    older_task = asyncio.create_task(
        older.extract(messages[:2], cursor_key="shared")
    )
    await asyncio.wait_for(older_started.wait(), 1)
    await newer.extract(messages, cursor_key="shared")
    release_older.set()
    await older_task

    state = json.loads(
        (repository.root / ".extraction-state.json").read_text(encoding="utf-8")
    )
    assert state["cursors"]["shared"] == messages[-1].uuid


async def test_memory_legacy_cursor_migration_preserves_all_keys(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    repository.root.mkdir(parents=True, exist_ok=True)
    legacy = repository.root / ".extraction-cursors.json"
    legacy.write_text(
        json.dumps({"active": "old-active", "untouched": "keep-me"}),
        encoding="utf-8",
    )
    message = Message("user", "new")
    extractor = MemoryExtractor(_Provider("[]"), RepositoryMemoryStore(repository))

    await extractor.extract([message], cursor_key="active")

    state = json.loads(
        (repository.root / ".extraction-state.json").read_text(encoding="utf-8")
    )
    assert state["cursors"] == {
        "active": message.uuid,
        "untouched": "keep-me",
    }
    assert state["legacy_cursor_imported"] is True


async def test_memory_infrastructure_failure_does_not_advance_cursor(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")

    class FailingStore(RepositoryMemoryStore):
        async def add(self, *args, **kwargs):
            raise OSError("storage unavailable")

    provider = _Provider(json.dumps([
        {"operation": "create", "content": "valid durable fact", "kind": "fact"}
    ]))
    extractor = MemoryExtractor(provider, FailingStore(repository))

    with pytest.raises(OSError, match="storage unavailable"):
        await extractor.extract([Message("user", "remember")], cursor_key="s")

    assert not (repository.root / ".extraction-state.json").exists()


async def test_memory_all_poison_batch_advances_with_content_free_bounded_dlq(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    secret = "sk-abcdefghijklmnopqrstuvwxyz987654"
    provider = _Provider(json.dumps([
        {"operation": "explode", "content": "invalid operation"},
        {"operation": "create", "content": secret, "kind": "fact"},
    ]))
    message = Message("user", "remember")
    extractor = MemoryExtractor(
        provider,
        RepositoryMemoryStore(repository),
        config=MemoryConfig(extraction_dead_letter_limit=1),
    )

    assert await extractor.extract([message], cursor_key="s") == []

    state_text = (repository.root / ".extraction-state.json").read_text(encoding="utf-8")
    state = json.loads(state_text)
    assert state["cursors"]["s"] == message.uuid
    assert len(state["dead_letters"]) == 1
    assert secret not in state_text and "invalid operation" not in state_text


def test_hook_projection_recursively_redacts_and_hard_caps() -> None:
    limits = HookLimitsConfig(total_bytes=1024, string_bytes=200, max_depth=4, max_items=3)
    value = {
        "prompt": "x" * 5000 + "</system-reminder>",
        "detail": {"api_key": "do-not-leak", "nested": [{"password": "also-secret"}]},
    }
    projected, counters = bounded_hook_payload(value, limits)
    encoded = json.dumps(projected, ensure_ascii=False).encode("utf-8")
    assert len(encoded) <= 1024
    assert b"do-not-leak" not in encoded and b"also-secret" not in encoded
    assert counters["redacted"] == 2
    assert "</system-reminder>" not in encoded.decode("utf-8")


def test_hook_projection_bounds_and_defangs_field_names() -> None:
    long_key = "密" * 500 + "</system-reminder>"
    secret_key = "x" * 500 + "_api_key"
    limits = HookLimitsConfig(total_bytes=700, string_bytes=80, max_depth=3, max_items=4)

    projected, counters = bounded_hook_payload(
        {long_key: "value", secret_key: "must-not-leak", "nested": {"a": {"b": {"c": 1}}}},
        limits,
    )
    encoded = json.dumps(projected, ensure_ascii=False).encode("utf-8")

    assert len(encoded) <= limits.total_bytes
    assert b"must-not-leak" not in encoded
    assert "</system-reminder>" not in encoded.decode("utf-8")
    assert counters["keys_truncated"] >= 2
    assert counters["redacted"] == 1
    assert counters["depth_truncated"] >= 1


def _pid_exists(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _wait_for_pid_file(path: Path) -> tuple[int, int]:
    for _ in range(150):
        if path.exists():
            values = json.loads(path.read_text(encoding="utf-8"))
            return int(values["parent"]), int(values["child"])
        await asyncio.sleep(0.02)
    raise AssertionError("hook process did not publish its pid file")


async def _wait_for_processes_to_exit(*pids: int) -> None:
    for _ in range(150):
        if not any(_pid_exists(pid) for pid in pids):
            return
        await asyncio.sleep(0.02)
    assert not any(_pid_exists(pid) for pid in pids)


def _process_tree_hook(tmp_path: Path, *, timeout: float) -> tuple[CommandHookAdapter, Path]:
    script = tmp_path / "hook_tree.py"
    pid_file = tmp_path / "hook-pids.json"
    script.write_text(
        "import json, os, pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({'parent': os.getpid(), 'child': child.pid}), encoding='utf-8')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    logger = JSONLRunLogger(tmp_path / "logs")
    adapter = CommandHookAdapter(
        ExternalHookSpec(
            event="UserPromptSubmit",
            type="command",
            command_argv=[sys.executable, str(script), str(pid_file)],
            timeout=timeout,
        ),
        logger,
        HookLimitsConfig(output_bytes=1024),
    )
    return adapter, pid_file


async def test_command_hook_timeout_kills_and_reaps_process_tree(tmp_path: Path) -> None:
    adapter, pid_file = _process_tree_hook(tmp_path, timeout=0.5)
    invocation = asyncio.create_task(
        adapter._invoke(HookContext(HookEvent.USER_PROMPT_SUBMIT, []))
    )
    parent, child = await _wait_for_pid_file(pid_file)

    with pytest.raises(HookFailedError, match="timed out"):
        await invocation

    await _wait_for_processes_to_exit(parent, child)
    adapter.logger.close()


async def test_command_hook_cancellation_kills_and_reaps_process_tree(tmp_path: Path) -> None:
    adapter, pid_file = _process_tree_hook(tmp_path, timeout=30)
    invocation = asyncio.create_task(
        adapter._invoke(HookContext(HookEvent.USER_PROMPT_SUBMIT, []))
    )
    parent, child = await _wait_for_pid_file(pid_file)
    invocation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await invocation

    await _wait_for_processes_to_exit(parent, child)
    adapter.logger.close()


async def test_command_hook_caps_stdout_and_stderr_independently(tmp_path: Path) -> None:
    script = tmp_path / "noisy_hook.py"
    script.write_text(
        "import os\nos.write(1, b'x' * 4096)\nos.write(2, b'y' * 4096)\n",
        encoding="utf-8",
    )
    logger = JSONLRunLogger(tmp_path / "logs")
    adapter = CommandHookAdapter(
        ExternalHookSpec(
            event="UserPromptSubmit",
            type="command",
            command_argv=[sys.executable, str(script)],
            timeout=5,
        ),
        logger,
        HookLimitsConfig(output_bytes=1024),
    )

    with pytest.raises(HookFailedError, match="output exceeded"):
        await adapter._invoke(HookContext(HookEvent.USER_PROMPT_SUBMIT, []))

    logger.close()


def test_transcript_reports_schema_orphans_and_cycles(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    records = [
        {"type": "message", "role": "bogus", "content": "bad"},
        {"type": "message", "role": "user", "content": "orphan", "uuid": "a", "parent_uuid": "missing"},
        {"type": "message", "role": "user", "content": "cycle-a", "uuid": "b", "parent_uuid": "c"},
        {"type": "message", "role": "assistant", "content": "cycle-b", "uuid": "c", "parent_uuid": "b"},
        {"type": "message", "role": "user", "content": "valid", "uuid": "root", "parent_uuid": None},
    ]
    path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")

    loaded = load_transcript(path)

    codes = {item.code for item in loaded.diagnostics}
    assert {"invalid_message_role", "orphan_parent", "parent_cycle"} <= codes
    assert [message.content for message in build_chain(loaded)] == ["valid"]


def test_transcript_rejects_foreign_session_records_without_poisoning_chain(
    tmp_path: Path,
) -> None:
    path = tmp_path / "owned.jsonl"
    root = Message("user", "root", uuid="root")
    foreign = Message("assistant", "foreign", uuid="foreign", parent_uuid="root")
    good = Message("assistant", "good", uuid="good", parent_uuid="root")
    records = [
        {"type": "session", "session_id": "owned", "cwd": str(tmp_path)},
        {"type": "message", "session_id": "owned", **root.to_dict()},
        {"type": "message", "session_id": "other", **foreign.to_dict()},
        {"type": "message", "session_id": "owned", **good.to_dict()},
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    loaded = load_transcript(path)

    assert [message.content for message in build_chain(loaded)] == ["root", "good"]
    assert "foreign" not in loaded.messages
    assert any(item.code == "session_conflict" for item in loaded.diagnostics)


async def test_transcript_round_index_avoids_cold_full_scans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = TranscriptStore(tmp_path / "sessions", tmp_path, "indexed")
    assistant = Message("assistant", "call", metadata={"tool_calls": [{"id": "c"}]})
    result = Message("tool", "result", metadata={"tool_call_id": "c"})
    assert await store.append_tool_round(assistant, [result], {"calls": []})
    store.close()
    transcript_path = store.path
    original_open = Path.open
    transcript_reads = 0

    def counting_open(path: Path, *args, **kwargs):
        nonlocal transcript_reads
        mode = str(args[0]) if args else str(kwargs.get("mode", "r"))
        if path == transcript_path and mode.startswith("r"):
            transcript_reads += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    for index in range(10):
        cold = TranscriptStore(tmp_path / "sessions", tmp_path, "indexed")
        next_assistant = Message(
            "assistant", f"call-{index}", metadata={"tool_calls": [{"id": f"c{index}"}]}
        )
        next_result = Message(
            "tool", f"result-{index}", metadata={"tool_call_id": f"c{index}"}
        )
        assert await cold.append_tool_round(next_assistant, [next_result], {"calls": []})
        cold.close()

    assert transcript_reads == 0


def test_process_supervisor_marks_interrupted_history_lost(tmp_path: Path) -> None:
    root = tmp_path / "tasks"
    root.mkdir()
    metadata = root / "old.json"
    metadata.write_text(json.dumps({
        "task_id": "old", "state": "running", "exit_code": None,
        "output_path": str(root / "old.log"),
    }), encoding="utf-8")
    (root / "old.log").write_text("partial", encoding="utf-8")

    supervisor = ProcessSupervisor(ShellToolConfig(), root)

    recovered = json.loads(metadata.read_text(encoding="utf-8"))
    assert recovered["state"] == "lost"
    assert recovered["loss_reason"] == "supervisor_restart"
    assert supervisor.recovered_lost_count == 1


def test_process_supervisor_preserves_all_non_running_terminal_states(tmp_path: Path) -> None:
    root = tmp_path / "tasks"
    root.mkdir()
    states = ["completed", "failed", "timed_out", "stopped", "lost"]
    before: dict[str, dict[str, object]] = {}
    for state in states:
        payload: dict[str, object] = {
            "task_id": state,
            "state": state,
            "exit_code": 0 if state == "completed" else 1,
            "output_path": str(root / f"{state}.log"),
        }
        before[state] = payload
        (root / f"{state}.json").write_text(json.dumps(payload), encoding="utf-8")

    supervisor = ProcessSupervisor(ShellToolConfig(), root)

    assert supervisor.recovered_lost_count == 0
    for state, payload in before.items():
        assert json.loads((root / f"{state}.json").read_text(encoding="utf-8")) == payload


def test_scheduler_retries_then_dead_letters_and_redrives(tmp_path: Path) -> None:
    store = SchedulerStore(
        tmp_path / "scheduler.sqlite3", retry_base_seconds=30,
        delivery_lease_seconds=100,
    )
    job = store.create(
        owner_session="s", owner_agent="a", schedule="* * * * *", timezone="UTC",
        prompt="run", persistent=True, now=0,
    )
    store.heartbeat("s", "a", ttl=1000, now=0)
    store.route_due(now=61)

    first = store.claim_deliveries("s", "a", now=61)[0]
    assert first["attempt_count"] == 1
    assert store.fail_delivery(int(first["id"]), "RuntimeError", now=61) == "retry_wait"
    assert store.claim_deliveries("s", "a", now=90) == []
    second = store.claim_deliveries("s", "a", now=91)[0]
    assert store.fail_delivery(int(second["id"]), "RuntimeError", now=91) == "retry_wait"
    third = store.claim_deliveries("s", "a", now=151)[0]
    assert store.fail_delivery(int(third["id"]), "RuntimeError", now=151) == "dead_letter"
    dead = store.list_dead_letters(owner_session="s", owner_agent="a")
    assert len(dead) == 1 and store.get(str(job["id"]))["inflight"] == 0
    store.redrive_dead_letter(
        int(dead[0]["id"]), owner_session="s", owner_agent="a", now=200
    )
    assert store.claim_deliveries("s", "a", now=200)[0]["attempt_count"] == 1


def test_scheduler_recovers_expired_delivery_lease(tmp_path: Path) -> None:
    store = SchedulerStore(
        tmp_path / "scheduler.sqlite3", retry_base_seconds=0,
        delivery_lease_seconds=10,
    )
    store.create(
        owner_session="s", owner_agent="a", schedule="* * * * *", timezone="UTC",
        prompt="run", persistent=True, now=0,
    )
    store.heartbeat("s", "a", ttl=1000, now=0)
    store.route_due(now=61)
    first = store.claim_deliveries("s", "a", now=61)[0]
    assert first["state"] == "running"
    recovered = store.claim_deliveries("s", "a", now=72)[0]
    assert recovered["attempt_count"] == 2


def test_scheduler_additively_migrates_old_running_delivery(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY, owner_session TEXT NOT NULL, owner_agent TEXT NOT NULL,
                schedule TEXT NOT NULL, timezone TEXT NOT NULL, prompt TEXT NOT NULL,
                persistent INTEGER NOT NULL, one_shot INTEGER NOT NULL DEFAULT 0,
                next_run REAL NOT NULL, last_run REAL, missed_count INTEGER NOT NULL DEFAULT 0,
                inflight INTEGER NOT NULL DEFAULT 0, coalesced INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL
            );
            CREATE TABLE heartbeats (
                owner_session TEXT NOT NULL, owner_agent TEXT NOT NULL, expires_at REAL NOT NULL,
                PRIMARY KEY(owner_session, owner_agent)
            );
            CREATE TABLE deliveries (
                id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                owner_session TEXT NOT NULL, owner_agent TEXT NOT NULL,
                prompt TEXT NOT NULL, due_at REAL NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending', created_at REAL NOT NULL
            );
            INSERT INTO jobs VALUES (
                'j','s','a','* * * * *','UTC','run',1,0,999,NULL,0,1,0,0
            );
            INSERT INTO deliveries(job_id,owner_session,owner_agent,prompt,due_at,state,created_at)
                VALUES ('j','s','a','run',10,'running',10);
            """
        )

    store = SchedulerStore(path, retry_base_seconds=0)
    claimed = store.claim_deliveries("s", "a", now=20)

    assert len(claimed) == 1
    assert claimed[0]["state"] == "running"
    assert claimed[0]["attempt_count"] == 1
    with sqlite3.connect(path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(deliveries)")}
        assert {"attempt_count", "available_at", "lease_until", "finished_at"} <= columns


def test_scheduler_one_shot_dead_letter_releases_inflight_and_checks_owner(
    tmp_path: Path,
) -> None:
    store = SchedulerStore(tmp_path / "scheduler.sqlite3", max_delivery_attempts=1)
    job = store.create(
        owner_session="s",
        owner_agent="a",
        schedule="* * * * *",
        timezone="UTC",
        prompt="run",
        persistent=True,
        one_shot=True,
        now=0,
    )
    store.heartbeat("s", "a", ttl=1000, now=0)
    delivery = store.route_due(now=61)[0]
    claimed = store.claim_deliveries("s", "a", now=61)[0]
    assert claimed["id"] == delivery["delivery_id"]

    assert store.fail_delivery(int(claimed["id"]), "failed", now=62) == "dead_letter"
    assert store.get(str(job["id"]))["inflight"] == 0
    assert store.list_dead_letters(owner_session="other", owner_agent="a") == []
    with pytest.raises(Exception, match="owned dead-letter"):
        store.redrive_dead_letter(
            int(claimed["id"]), owner_session="other", owner_agent="a", now=63
        )
    store.complete_delivery(int(claimed["id"]))
    assert store.list_dead_letters(owner_session="s", owner_agent="a")


async def test_react_scheduler_failure_is_delivered_to_dead_letter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SchedulerStore(tmp_path / "scheduler.sqlite3", max_delivery_attempts=1)
    agent = ReActAgent(
        provider=_Provider("done"),
        config=ReActConfig(
            run_dir=str(tmp_path / "runs"),
            session_dir="",
            project_instructions=False,
            git_context=False,
        ),
        session_id="s",
    )
    agent.session.scheduler_store = store
    store.create(
        owner_session="s",
        owner_agent=agent.session.agent_id,
        schedule="* * * * *",
        timezone="UTC",
        prompt="scheduled failure",
        persistent=True,
        now=0,
    )
    store.heartbeat("s", agent.session.agent_id, ttl=1000, now=0)
    store.route_due(now=61)

    async def fail_run(*args, **kwargs):
        raise RuntimeError("scheduled run failed")

    monkeypatch.setattr(agent, "run_messages", fail_run)
    history, results = await agent.drain_scheduler_deliveries([])

    assert history == [] and results == []
    assert len(store.list_dead_letters(
        owner_session="s", owner_agent=agent.session.agent_id
    )) == 1
    agent.logger.close()


async def test_hierarchical_summary_covers_middle_and_bounds_track_b() -> None:
    pipeline = CompressionPipeline(CompressionConfig(
        summary_input_max_chars=1000,
        summary_total_input_max_chars=4000,
        summary_max_chunks=4,
        summary_output_max_chars=500,
        track_b_max_chars=500,
    ))
    prefix = [Message("user", ("MIDDLE-SENTINEL " if index == 50 else "") + "x" * 900) for index in range(100)]
    seen: list[str] = []

    async def summarize(messages: list[Message]) -> str:
        seen.append("\n".join(message.content for message in messages))
        return "map-or-final"

    block, note, metadata = await pipeline._collapse_prefix(prefix, summarize)

    assert note == "llm_summary"
    assert any("MIDDLE-SENTINEL" in value for value in seen)
    assert metadata["chunks"] == 4 and metadata["omitted_chars"] > 0
    assert len(block.content) < 1000
    fallback, _truncated = pipeline._collapse_prefix_naive(prefix)
    assert len(fallback.content) < 1000


async def test_hierarchical_summary_accounts_every_input_and_output_budget() -> None:
    pipeline = CompressionPipeline(CompressionConfig(
        summary_input_max_chars=1000,
        summary_total_input_max_chars=4000,
        summary_max_chunks=4,
        summary_map_output_tokens=100,
        summary_total_output_tokens=1000,
        compact_summary_start_tokens=400,
        summary_output_max_chars=500,
    ))
    prefix = [Message("user", f"round-{index} " + "x" * 900) for index in range(100)]
    input_sizes: list[int] = []
    output_budgets: list[int] = []

    async def summarize(messages: list[Message]) -> str:
        input_sizes.append(sum(len(message.content) for message in messages))
        output_budgets.append(int(messages[0].metadata["_summary_output_tokens"]))
        return "summary" * 40

    _block, note, metadata = await pipeline._collapse_prefix(prefix, summarize)

    assert note == "llm_summary"
    assert max(input_sizes) <= 1000
    assert sum(input_sizes) <= 4000
    assert sum(output_budgets) <= 1000
    assert metadata["input_chars"] == sum(input_sizes)
    assert metadata["output_budget_tokens"] == sum(output_budgets)
    assert metadata["calls"] == len(input_sizes)
    assert metadata["omitted_chars"] > 0


def test_summary_projects_one_oversized_tool_round_before_call() -> None:
    pipeline = CompressionPipeline(CompressionConfig(
        summary_input_max_chars=1000,
        summary_total_input_max_chars=4000,
        summary_max_chunks=4,
    ))
    assistant = Message(
        "assistant",
        "a" * 5000 + "GIANT-MIDDLE" + "b" * 5000,
        metadata={"tool_calls": [{"id": "c"}]},
    )
    tool = Message(
        "tool",
        "c" * 5000 + "TOOL-MIDDLE" + "d" * 5000,
        metadata={"tool_call_id": "c"},
    )

    chunks, projected, omitted = pipeline._summary_chunks([assistant, tool])

    assert len(chunks) == 1
    assert sum(len(message.content) for message in chunks[0]) <= 1000
    assert "GIANT-MIDDLE" in chunks[0][0].content
    assert "TOOL-MIDDLE" in chunks[0][0].content
    assert projected <= 1000 and omitted > 0


async def test_summary_retry_ladder_shares_injected_output_cap() -> None:
    class TruncatingProvider:
        def __init__(self) -> None:
            self.budgets: list[int] = []

        async def complete(self, messages, tools, config, stream=None) -> LLMResult:
            self.budgets.append(config.max_tokens)
            return LLMResult("<summary>x</summary>", stop_reason="max_tokens")

    provider = TruncatingProvider()
    summarizer = build_summarizer(
        provider,  # type: ignore[arg-type]
        ProviderConfig(model="claude-opus-4-8"),
        CompressionConfig(
            compact_summary_start_tokens=8000,
            compact_max_output_tokens=20000,
            compact_max_output_retries=2,
        ),
    )
    assert summarizer is not None
    message = Message("user", "input", metadata={"_summary_output_tokens": 10_000})

    await summarizer([message])

    assert provider.budgets == [8000, 2000]
    assert sum(provider.budgets) == 10_000


def test_track_b_incremental_fallback_stays_hard_bounded() -> None:
    pipeline = CompressionPipeline(CompressionConfig(track_b_max_chars=257))
    prefix = [Message("user", "x" * 1000) for _ in range(10_000)]

    body, truncated = pipeline._bounded_track_b(prefix)

    assert truncated is True
    assert len(body) <= 257


async def test_workflow_frame_supports_over_64k_and_rejects_oversize() -> None:
    payload = {"type": "result", "result": "x" * 70_000}
    frame = _encode_frame(payload)
    reader = asyncio.StreamReader()
    reader.feed_data(frame[:17])
    reader.feed_data(frame[17:])
    reader.feed_eof()
    assert await _read_frame(reader) == payload
    with pytest.raises(WorkflowError, match="size limit"):
        _encode_frame({"result": "x" * _MAX_WORKFLOW_FRAME_BYTES})
    oversized = asyncio.StreamReader()
    oversized.feed_data(struct.pack(">I", _MAX_WORKFLOW_FRAME_BYTES + 1))
    oversized.feed_eof()
    with pytest.raises(WorkflowError, match="size limit"):
        await _read_frame(oversized)


async def test_workflow_frame_handles_sticky_frames_and_protocol_errors() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(_encode_frame({"type": "one"}) + _encode_frame({"type": "two"}))
    reader.feed_eof()
    assert await _read_frame(reader) == {"type": "one"}
    assert await _read_frame(reader) == {"type": "two"}

    truncated = asyncio.StreamReader()
    truncated.feed_data(struct.pack(">I", 10) + b"{}")
    truncated.feed_eof()
    with pytest.raises(WorkflowError, match="truncated frame body"):
        await _read_frame(truncated)

    invalid = asyncio.StreamReader()
    invalid.feed_data(struct.pack(">I", 1) + b"[")
    invalid.feed_eof()
    with pytest.raises(WorkflowError, match="invalid JSON"):
        await _read_frame(invalid)

    wrong_type = asyncio.StreamReader()
    wrong_type.feed_data(struct.pack(">I", 2) + b"[]")
    wrong_type.feed_eof()
    with pytest.raises(WorkflowError, match="non-object"):
        await _read_frame(wrong_type)


def _node_24_or_newer() -> bool:
    node = shutil.which("node")
    if node is None:
        return False
    result = subprocess.run(
        [node, "--version"], capture_output=True, text=True, timeout=3, check=False
    )
    try:
        return int(result.stdout.strip().lstrip("v").split(".", 1)[0]) >= 24
    except ValueError:
        return False


@pytest.mark.skipif(not _node_24_or_newer(), reason="Node 24 is required")
async def test_workflow_node_round_trip_exceeds_64k() -> None:
    from agent_core.workflow_runtime import WorkflowRuntime

    expected = "界" * 30_000

    async def unused_agent(*args) -> str:
        raise AssertionError("workflow should not call an agent")

    result = await WorkflowRuntime().run(
        "return args.value;", {"value": expected}, unused_agent, timeout=5
    )

    assert result == expected


@pytest.mark.skipif(not _node_24_or_newer(), reason="Node 24 is required")
async def test_workflow_timeout_reaps_runtime_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_core.workflow_runtime as workflow_module

    original_terminate = workflow_module._terminate_workflow_process
    processes: list[asyncio.subprocess.Process] = []

    async def recording_terminate(process: asyncio.subprocess.Process) -> None:
        processes.append(process)
        await original_terminate(process)

    monkeypatch.setattr(workflow_module, "_terminate_workflow_process", recording_terminate)

    async def unused_agent(*args) -> str:
        raise AssertionError("workflow should not call an agent")

    with pytest.raises(TimeoutError):
        await workflow_module.WorkflowRuntime().run(
            "await new Promise(() => {});", {}, unused_agent, timeout=0.1
        )

    assert len(processes) == 1
    assert processes[0].returncode is not None


async def test_openai_stream_usage_negotiation_is_cached() -> None:
    bodies: list[dict] = []
    stream = b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\ndata: [DONE]\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "stream_options" in body:
            return httpx.Response(422, json={"error": {"message": "stream_options unsupported"}})
        return httpx.Response(200, content=stream)

    provider = OpenAICompatProvider(api_key="k", base_url="https://example.test")
    provider._transport = httpx.MockTransport(handler)
    config = ProviderConfig(model="m", stream=True)
    sink = type("Sink", (), {
        "on_text_delta": lambda self, text: None,
        "on_thinking_delta": lambda self, text: None,
        "on_tool_args_delta": lambda self, name, text: None,
    })()
    first = await provider.complete([Message("user", "x")], [], config, stream=sink)
    await provider.complete([Message("user", "y")], [], config, stream=sink)
    assert first.usage is not None and first.usage.total_tokens == 5
    assert len(bodies) == 3
    assert "stream_options" in bodies[0]
    assert all("stream_options" not in body for body in bodies[1:])


async def test_openai_usage_only_stream_chunk_is_parsed() -> None:
    bodies: list[dict] = []
    stream = (
        b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":4}}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=stream)

    provider = OpenAICompatProvider(api_key="k", base_url="https://example.test")
    provider._transport = httpx.MockTransport(handler)
    sink = type("Sink", (), {
        "on_text_delta": lambda self, text: None,
        "on_thinking_delta": lambda self, text: None,
        "on_tool_args_delta": lambda self, name, text: None,
    })()

    result = await provider.complete(
        [Message("user", "x")], [], ProviderConfig(model="m", stream=True), stream=sink
    )

    assert bodies[0]["stream_options"] == {"include_usage": True}
    assert result.content == "ok"
    assert result.usage is not None and result.usage.total_tokens == 15


async def test_openai_unrelated_4xx_does_not_downgrade_or_retry() -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            400, json={"error": {"message": "invalid messages payload"}}
        )

    provider = OpenAICompatProvider(api_key="k", base_url="https://example.test")
    provider._transport = httpx.MockTransport(handler)
    sink = type("Sink", (), {
        "on_text_delta": lambda self, text: None,
        "on_thinking_delta": lambda self, text: None,
        "on_tool_args_delta": lambda self, name, text: None,
    })()

    with pytest.raises(Exception):
        await provider.complete(
            [Message("user", "x")], [], ProviderConfig(model="m", stream=True), stream=sink
        )

    assert len(bodies) == 1
    assert "stream_options" in bodies[0]
    assert provider._stream_usage_supported is None


async def test_openai_concurrent_usage_probe_downgrades_only_once() -> None:
    bodies: list[dict] = []
    stream = (
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "stream_options" in body:
            await asyncio.sleep(0.05)
            return httpx.Response(
                422, json={"error": {"message": "include_usage is not supported"}}
            )
        return httpx.Response(200, content=stream)

    provider = OpenAICompatProvider(api_key="k", base_url="https://example.test")
    provider._transport = httpx.MockTransport(handler)
    sink = type("Sink", (), {
        "on_text_delta": lambda self, text: None,
        "on_thinking_delta": lambda self, text: None,
        "on_tool_args_delta": lambda self, name, text: None,
    })()
    config = ProviderConfig(model="m", stream=True)

    await asyncio.gather(
        provider.complete([Message("user", "x")], [], config, stream=sink),
        provider.complete([Message("user", "y")], [], config, stream=sink),
    )

    assert sum("stream_options" in body for body in bodies) == 1
    assert len(bodies) == 3
    assert provider._stream_usage_supported is False


async def test_retention_protects_tagged_and_invalid_transcripts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    old = TranscriptStore(root, workspace, "old")
    await old.append_message(Message("user", "old"))
    old.close()
    newest = TranscriptStore(root, workspace, "new")
    await newest.append_message(Message("user", "new"))
    newest.close()
    tagged = TranscriptStore(root, workspace, "tagged")
    await tagged.append_message(Message("user", "tagged"))
    await tagged.append_meta("tag", {"tag": "keep"})
    tagged.close()
    project = project_dir(root, workspace)
    invalid = project / "invalid.jsonl"
    invalid.write_text("[]\n", encoding="utf-8")
    old_time = time.time() - 100 * 86_400
    os.utime(old.path, (old_time, old_time))
    sidecar = old.path.with_suffix(old.path.suffix + ".round-index.json")
    sidecar.write_text("{}", encoding="utf-8")
    config = SessionRetentionConfig(transcript_days=90, max_transcripts_per_project=1)

    preview = prune_sessions(root, workspace, config, apply=False)
    report = prune_sessions(root, workspace, config, apply=True)

    assert preview["selected"] == report["selected"]
    assert "old" in report["selected"] and not old.path.exists()
    assert not sidecar.exists()
    assert newest.path.exists() and tagged.path.exists() and invalid.exists()


async def test_retention_protects_active_transcript(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    active = TranscriptStore(root, workspace, "active")
    await active.append_message(Message("user", "still running"))
    old = time.time() - 100 * 86_400
    os.utime(active.path, (old, old))

    report = prune_sessions(
        root,
        workspace,
        SessionRetentionConfig(transcript_days=1, max_transcripts_per_project=1),
        apply=True,
    )

    assert report["deleted"] == 0
    assert active.path.exists()
    active.close()


def test_retention_disabled_skips_transcripts_and_terminal_journals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    config = SessionRetentionConfig(enabled=False, transcript_days=1)
    assert prune_sessions(root, workspace, config, apply=True)["deleted"] == 0

    called = False

    def forbidden_prune(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("journal retention must be disabled")

    monkeypatch.setattr(TurnExecutionJournal, "prune_terminal", forbidden_prune)
    agent = ReActAgent(
        provider=_Provider("done"),
        config=ReActConfig(
            run_dir=str(tmp_path / "runs"),
            session_dir="",
            project_instructions=False,
            git_context=False,
            session_retention=config,
        ),
    )

    agent.recover_turn_journals()

    assert called is False


async def test_transcript_failure_is_sticky_and_keeps_durable_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("POLARIS_HOME", str(tmp_path / "polaris"))
    class FailingTranscript:
        def __init__(self) -> None:
            self.path = tmp_path / "failed.jsonl"
            self.last_error = "OSError: disk full"
            self.calls = 0

        async def append_message(self, message: Message) -> bool:
            self.calls += 1
            return False

        def recover_tool_round(self, payload: dict[str, object]) -> bool:
            return False

        def close(self) -> None:
            pass

    transcript = FailingTranscript()
    agent = ReActAgent(
        provider=_Provider("done"),
        config=ReActConfig(
            run_dir=str(tmp_path / "runs"), session_dir="",
            project_instructions=False, git_context=False,
        ),
        transcript=transcript,  # type: ignore[arg-type]
    )

    result = await agent.run("hello")

    head = agent.runtime.durable_head
    assert transcript.calls == 1
    assert head.persistence_degraded is True
    assert head.durable_head_id is None
    assert head.memory_head_id == result.messages[-1].uuid
    events = [json.loads(line) for line in agent.logger.path.read_text(encoding="utf-8").splitlines()]
    assert sum(item["event"] == "transcript_persistence_degraded" for item in events) == 1


def test_journal_open_index_tracks_only_unfinished(tmp_path: Path) -> None:
    storage = JournalStorage.local(tmp_path / "journals", workspace=tmp_path)
    open_journal = TurnExecutionJournal(storage)
    open_journal.record("discovered")
    assert TurnExecutionJournal._journal_paths(storage) == [open_journal.path]
    open_journal.close()
    assert TurnExecutionJournal._journal_paths(storage) == []


def test_terminal_journal_retention_deletes_only_verified_terminal(tmp_path: Path) -> None:
    storage = JournalStorage.local(tmp_path / "journals", workspace=tmp_path)
    terminal = TurnExecutionJournal(storage)
    terminal.record("discovered")
    terminal.close()
    old = time.time() - 10 * 86_400
    os.utime(terminal.path, (old, old))
    corrupt = storage.run_root / ("f" * 32 + ".jsonl")
    corrupt.write_text("not-json\n", encoding="utf-8")
    unfinished = TurnExecutionJournal(storage)
    unfinished.record("discovered")
    unfinished._release_for_later_recovery()
    os.utime(unfinished.path, (old, old))

    report = TurnExecutionJournal.prune_terminal(
        storage, retention_days=7, scan_interval_seconds=60, now=time.time()
    )

    assert report["deleted"] == 1
    assert not terminal.path.exists()
    assert corrupt.exists()
    assert unfinished.path.exists()
