"""Compaction-boundary persistence (phase 2): a fold writes a compact boundary + summary
into the transcript so a resume loads only the compacted state."""

import json
from pathlib import Path

from agent_core.compression import build_summary_user_message
from agent_core.memory import MemoryConfig
from agent_core.models import Message
from agent_core.providers.fake import FakeProvider
from agent_core.react import ReActAgent, ReActConfig
from agent_core import transcript as T
from agent_core.transcript import build_chain, load_transcript


def _agent(tmp_path: Path, *, boundary: bool = True) -> ReActAgent:
    cfg = ReActConfig(
        session_dir=str(tmp_path),
        memory=MemoryConfig(enabled=False),
        persist_compaction_boundary=boundary,
    )
    return ReActAgent(provider=FakeProvider(), config=cfg)


def _in_memory_chain(after: list[Message], leaf: Message) -> list[str]:
    by_uuid = {m.uuid: m for m in after}
    out: list[str] = []
    cur: Message | None = leaf
    seen: set[str] = set()
    while cur is not None and cur.uuid not in seen:
        seen.add(cur.uuid)
        out.append(cur.uuid)
        cur = by_uuid.get(cur.parent_uuid) if cur.parent_uuid else None
    out.reverse()
    return out


async def _emit_conversation(agent: ReActAgent, msgs: list[Message]) -> list[Message]:
    """Persist a real conversation tail (a,b,c,d) the way the loop does."""
    a = Message("user", "a")
    b = Message("assistant", "b")
    c = Message("user", "c")
    d = Message("assistant", "d")
    for m in (a, b, c, d):
        await agent._emit(msgs, m)
    return [a, b, c, d]


# --------------------------------------------------------------------------- write side


async def test_commit_writes_boundary_relink_and_attachment(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    sys = Message("system", "sys")
    pin = Message("user", "ctx", metadata={"pinned": "user_context"})
    msgs: list[Message] = [sys, pin]  # preserved front (not persisted, like a real run)
    a, b, c, d = await _emit_conversation(agent, msgs)
    before = list(msgs)

    # Simulate a context-collapse fold: a,b folded into a summary; c,d kept; +1 attachment.
    summary = build_summary_user_message("SUMMARY", marker="context_collapse", messages_collapsed=2)
    att = Message("user", "FILE CONTEXT", metadata={"post_compact_attachment": True})
    after = [sys, pin, summary, c, d, att]

    await agent._commit_compaction_boundary(before, after)

    loaded = load_transcript(agent.transcript.path)
    # Boundary summary persisted as a root with forensic logical parent.
    assert loaded.messages[summary.uuid].parent_uuid is None
    assert loaded.messages[summary.uuid].metadata["compact_boundary"] is True
    assert loaded.messages[summary.uuid].metadata["logical_parent_uuid"] == d.uuid
    # Relink re-pointed the kept tail's head onto the summary.
    assert loaded.messages[c.uuid].parent_uuid == summary.uuid

    chain = build_chain(loaded)
    chain_uuids = [m.uuid for m in chain]
    # Resume sees only the compacted state: summary, kept tail, attachment — NOT a/b.
    assert chain_uuids == [summary.uuid, c.uuid, d.uuid, att.uuid]
    assert a.uuid not in chain_uuids and b.uuid not in chain_uuids
    # In-memory chain matches the on-disk chain (chat-resume == disk-resume).
    assert _in_memory_chain(after, att) == chain_uuids


async def test_toggle_off_keeps_full_history(tmp_path: Path) -> None:
    agent = _agent(tmp_path, boundary=False)
    msgs: list[Message] = [Message("system", "sys")]
    a, b, c, d = await _emit_conversation(agent, msgs)
    before = list(msgs)
    summary = build_summary_user_message("S", marker="context_collapse", messages_collapsed=2)
    after = [msgs[0], summary, c, d]

    await agent._commit_compaction_boundary(before, after)

    raw = agent.transcript.path.read_text(encoding="utf-8")
    assert "compact_boundary" not in raw
    # Full faithful history remains resumable (phase-1 behavior).
    assert [m.uuid for m in build_chain(load_transcript(agent.transcript.path))] == [
        a.uuid, b.uuid, c.uuid, d.uuid
    ]


async def test_no_fold_writes_nothing(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    msgs: list[Message] = [Message("system", "sys")]
    a, b, c, d = await _emit_conversation(agent, msgs)
    before = list(msgs)
    size_before = agent.transcript.path.stat().st_size
    # snip/microcompact mutate content in place — same uuid set, no new messages.
    await agent._commit_compaction_boundary(before, list(msgs))
    assert agent.transcript.path.stat().st_size == size_before


async def test_multiple_folds_last_boundary_wins(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    sys = Message("system", "sys")
    msgs: list[Message] = [sys]
    a, b, c, d = await _emit_conversation(agent, msgs)

    # First fold: a,b -> summary1; keep c,d.
    s1 = build_summary_user_message("S1", marker="context_collapse", messages_collapsed=2)
    after1 = [sys, s1, c, d]
    await agent._commit_compaction_boundary([sys, a, b, c, d], after1)
    # A later real turn lands after the first boundary.
    e = Message("assistant", "e")
    await agent._emit(msgs, e)

    # Second fold: s1,c folded -> summary2; keep d,e.
    s2 = build_summary_user_message("S2", marker="context_collapse", messages_collapsed=2)
    after2 = [sys, s2, d, e]
    await agent._commit_compaction_boundary([sys, s1, c, d, e], after2)

    chain_uuids = [m.uuid for m in build_chain(load_transcript(agent.transcript.path))]
    assert chain_uuids == [s2.uuid, d.uuid, e.uuid]
    for old in (a.uuid, b.uuid, c.uuid, s1.uuid):
        assert old not in chain_uuids


async def test_reactive_compact_real_fold_writes_boundary(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    # Preserved front + a long conversation the pipeline will actually fold.
    msgs: list[Message] = [
        Message("system", "sys"),
        Message("user", "ctx", metadata={"pinned": "user_context"}),
    ]
    for i in range(16):
        await agent._emit(msgs, Message("user" if i % 2 == 0 else "assistant", f"turn {i}"))
    before = list(msgs)

    after, events = await agent.compression.reactive_compact(
        msgs, model=agent.config.model, summarizer=agent._summarizer
    )
    assert any(e.stage == "context_collapse" for e in events)
    await agent._commit_compaction_boundary(before, after)

    raw = agent.transcript.path.read_text(encoding="utf-8")
    assert '"compact_boundary": true' in raw
    chain = build_chain(load_transcript(agent.transcript.path))
    # The resumed chain is shorter than the pre-fold history and starts at the summary.
    assert len(chain) < len(before)
    assert chain[0].metadata.get("compressed") in {"context_collapse", "llm_summary"}


# --------------------------------------------------------------------------- read side


def _msg_line(uuid: str, role: str, content: str, parent: str | None, **meta) -> str:
    m = Message(role, content, metadata=meta, uuid=uuid, parent_uuid=parent)
    return json.dumps({"type": "message", **m.to_dict(), "session_id": "sess"}, ensure_ascii=False)


def _write_boundary_file(path: Path) -> None:
    lines = [
        _msg_line("m1", "user", "old one", None),
        _msg_line("m2", "assistant", "old two", "m1"),
        json.dumps({"type": "custom-title", "session_id": "sess", "title": "My Title"}),
        _msg_line("S", "user", "SUMMARY", None, compact_boundary=True, compressed="context_collapse"),
        json.dumps({"type": "relink", "uuid": "m3", "parent_uuid": "S"}),
        _msg_line("m3", "user", "kept three", "m2"),  # original parent m2, relinked to S
        _msg_line("m4", "assistant", "kept four", "m3"),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_applies_relink_and_excludes_pre_boundary(tmp_path: Path) -> None:
    f = tmp_path / "sess.jsonl"
    _write_boundary_file(f)
    loaded = load_transcript(f, skip_precompact=False)
    assert loaded.messages["m3"].parent_uuid == "S"  # relink applied
    chain = [m.uuid for m in build_chain(loaded)]
    assert chain == ["S", "m3", "m4"]  # boundary excludes m1/m2 from the resume chain
    assert loaded.title == "My Title"


def test_byte_truncation_skips_pre_boundary(tmp_path: Path, monkeypatch) -> None:
    f = tmp_path / "sess.jsonl"
    _write_boundary_file(f)
    monkeypatch.setattr(T, "SKIP_PRECOMPACT_THRESHOLD", 0)  # force truncation for any file

    truncated = load_transcript(f, skip_precompact=True)
    # Pre-boundary messages were never parsed into memory...
    assert "m1" not in truncated.messages and "m2" not in truncated.messages
    # ...yet title is rescued and the resume chain is identical to the full read.
    assert truncated.title == "My Title"
    assert [m.uuid for m in build_chain(truncated)] == ["S", "m3", "m4"]

    full = load_transcript(f, skip_precompact=False)
    assert "m1" in full.messages  # full read keeps pre-boundary on hand (forensics)
    assert [m.uuid for m in build_chain(full)] == [m.uuid for m in build_chain(truncated)]


def test_read_lite_first_prompt_and_count(tmp_path: Path) -> None:
    f = tmp_path / "sess.jsonl"
    _write_boundary_file(f)
    info = T.read_lite(f)
    assert info is not None
    assert info.message_count == 5  # m1, m2, S, m3, m4 are all "type":"message" lines
    assert info.first_prompt == "old one"  # original first user prompt, near the head
    assert info.title == "My Title"


def test_no_boundary_file_reads_whole_even_when_truncation_forced(tmp_path: Path, monkeypatch) -> None:
    f = tmp_path / "sess.jsonl"
    f.write_text(
        "\n".join([_msg_line("u1", "user", "hello", None), _msg_line("a1", "assistant", "hi", "u1")]) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(T, "SKIP_PRECOMPACT_THRESHOLD", 0)
    loaded = load_transcript(f)  # no boundary -> falls back to whole-file read
    assert [m.uuid for m in build_chain(loaded)] == ["u1", "a1"]
