from __future__ import annotations

import asyncio
import json
import os
import struct
import time
from pathlib import Path

import httpx
import pytest

from agent_core.compression import CompressionConfig, CompressionPipeline
from agent_core.hook_adapters import bounded_hook_payload
from agent_core.hooks import HookLimitsConfig
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
    config = SessionRetentionConfig(transcript_days=90, max_transcripts_per_project=1)

    report = prune_sessions(root, workspace, config, apply=True)

    assert "old" in report["selected"] and not old.path.exists()
    assert newest.path.exists() and tagged.path.exists() and invalid.exists()


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
    open_journal.record("turn_opened")
    assert TurnExecutionJournal._journal_paths(storage) == [open_journal.path]
    open_journal.close()
    assert TurnExecutionJournal._journal_paths(storage) == []


def test_terminal_journal_retention_deletes_only_verified_terminal(tmp_path: Path) -> None:
    storage = JournalStorage.local(tmp_path / "journals", workspace=tmp_path)
    terminal = TurnExecutionJournal(storage)
    terminal.record("turn_opened")
    terminal.close()
    old = time.time() - 10 * 86_400
    os.utime(terminal.path, (old, old))
    corrupt = storage.run_root / ("f" * 32 + ".jsonl")
    corrupt.write_text("not-json\n", encoding="utf-8")

    report = TurnExecutionJournal.prune_terminal(
        storage, retention_days=7, scan_interval_seconds=60, now=time.time()
    )

    assert report["deleted"] == 1
    assert not terminal.path.exists()
    assert corrupt.exists()
