from pathlib import Path

from agent_core.memory import MemoryConfig
from agent_core.models import LLMResult, Message, ToolCall
from agent_core.providers.fake import FakeProvider
from agent_core.react import ReActAgent, ReActConfig
from agent_core.transcript import (
    build_chain,
    find_session,
    fork_chain,
    latest_session,
    list_sessions,
    load_transcript,
    project_dir,
    sanitize_project,
)


class ToolThenDoneProvider:
    """One tool call (with thinking blocks) then a final answer — exercises the full
    assistant/tool message shapes the transcript must round-trip."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools, config, stream=None, should_cancel=None) -> LLMResult:
        self.calls += 1
        if self.calls == 1:
            return LLMResult(
                "Calling echo",
                tool_calls=[ToolCall("echo", {"text": "hi"}, id="toolu_1")],
                stop_reason="tool_use",
                thinking_blocks=[{"type": "thinking", "thinking": "ponder", "signature": "sig"}],
            )
        return LLMResult("all done", stop_reason="end")


def _config(tmp_path: Path) -> ReActConfig:
    return ReActConfig(session_dir=str(tmp_path), memory=MemoryConfig(enabled=False))


# --------------------------------------------------------------------------- contract


def test_message_round_trip_preserves_identity_and_metadata() -> None:
    msg = Message(
        "assistant",
        "body",
        metadata={"tool_calls": [{"name": "echo"}], "thinking_blocks": [{"signature": "s"}]},
        parent_uuid="parent123",
    )
    restored = Message.from_dict(msg.to_dict())
    assert restored == msg  # equality ignores uuid/parent_uuid (compare=False)
    assert restored.uuid == msg.uuid
    assert restored.parent_uuid == "parent123"
    assert restored.metadata["thinking_blocks"] == [{"signature": "s"}]


def test_message_uuid_excluded_from_equality() -> None:
    a = Message("user", "x")
    b = Message("user", "x")
    assert a == b
    assert a.uuid != b.uuid


def test_sanitize_project_distinguishes_cwds() -> None:
    a = sanitize_project("E:/proj/one")
    b = sanitize_project("E:/proj/two")
    assert a != b
    assert "/" not in a and ":" not in a


# --------------------------------------------------------------------------- write/read


async def test_run_persists_faithful_chain(tmp_path: Path) -> None:
    agent = ReActAgent(provider=ToolThenDoneProvider(), config=_config(tmp_path))
    result = await agent.run("please echo")

    loaded = load_transcript(agent.transcript.path)
    chain = build_chain(loaded)
    # user task, assistant(tool_use), tool result, assistant(final)
    assert [m.role for m in chain] == ["user", "assistant", "tool", "assistant"]
    # parent_uuid forms an unbroken chain rooted at the first user message.
    assert chain[0].parent_uuid is None
    assert all(chain[i].parent_uuid == chain[i - 1].uuid for i in range(1, len(chain)))
    # Thinking blocks and tool linkage survive the round trip (the API replay invariant).
    assert chain[1].metadata["thinking_blocks"][0]["signature"] == "sig"
    assert chain[2].metadata["tool_call_id"] == "toolu_1"
    assert result.answer == "all done"


async def test_disabled_persistence_writes_nothing(tmp_path: Path) -> None:
    cfg = ReActConfig(session_dir="", memory=_config(tmp_path).memory)
    agent = ReActAgent(provider=FakeProvider(), config=cfg)
    assert agent.transcript is None
    await agent.run("hello")
    assert not any(tmp_path.rglob("*.jsonl"))


# --------------------------------------------------------------------------- resume


async def test_resume_continues_same_session_file(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    first = ReActAgent(provider=FakeProvider(), config=cfg)
    sid = first.session_id
    await first.run("turn one")
    count_after_first = load_transcript(first.transcript.path).message_count

    # A fresh agent reusing the id appends to the same file.
    second = ReActAgent(provider=FakeProvider(), config=cfg, session_id=sid)
    history = build_chain(load_transcript(first.transcript.path))
    result = await second.run("turn two", history=history)

    reloaded = load_transcript(first.transcript.path)
    assert reloaded.message_count > count_after_first
    # The second run saw the first turn's content in its context.
    contents = " ".join(m.content for m in result.messages)
    assert "turn one" in contents and "turn two" in contents


async def test_chat_style_history_carries_across_turns(tmp_path: Path) -> None:
    agent = ReActAgent(provider=FakeProvider(), config=_config(tmp_path))
    r1 = await agent.run("remember apples")
    r2 = await agent.run("and oranges", history=r1.messages)
    roles_contents = [(m.role, m.content) for m in r2.messages]
    # The earlier user turn is present exactly once (no duplicated system/userContext).
    assert sum(1 for _r, c in roles_contents if c == "remember apples") == 1
    assert any(c == "and oranges" for _r, c in roles_contents)
    # Exactly one base system prompt at the front each turn.
    assert r2.messages[0].role == "system"
    assert sum(1 for m in r2.messages if m.metadata.get("pinned") == "user_context") == 1


# --------------------------------------------------------------------------- fork / list


async def test_fork_clones_tree_without_touching_source(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    agent = ReActAgent(provider=FakeProvider(), config=cfg)
    await agent.run("original")
    source_bytes = agent.transcript.path.read_bytes()

    loaded = load_transcript(agent.transcript.path)
    new_id, cloned = fork_chain(loaded)
    assert new_id != agent.session_id
    assert len(cloned) == loaded.message_count
    assert cloned[0].parent_uuid is None
    # New uuids, but the relative parent links are preserved.
    assert all(cloned[i].parent_uuid == cloned[i - 1].uuid for i in range(1, len(cloned)))
    assert {m.uuid for m in cloned}.isdisjoint(set(loaded.messages))
    # Source file is byte-for-byte untouched.
    assert agent.transcript.path.read_bytes() == source_bytes


async def test_list_and_latest_and_find(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    a = ReActAgent(provider=FakeProvider(), config=cfg)
    await a.run("first prompt here")
    b = ReActAgent(provider=FakeProvider(), config=cfg)
    await b.run("second prompt here")

    proj = project_dir(str(tmp_path), Path.cwd())
    infos = list_sessions(proj)
    assert len(infos) == 2
    # Newest first.
    assert infos[0].modified >= infos[1].modified
    assert latest_session(proj).session_id == infos[0].session_id
    # find_session locates a known id in the current project.
    assert find_session(str(tmp_path), Path.cwd(), a.session_id) == a.transcript.path
    assert find_session(str(tmp_path), Path.cwd(), "nonexistent") is None
    # first_prompt was captured.
    assert any(i.first_prompt in {"first prompt here", "second prompt here"} for i in infos)


async def test_subagent_transcript_is_sidechain_not_listed(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    parent = ReActAgent(provider=FakeProvider(), config=cfg)
    child = parent._child_transcript("agentABC")
    assert child is not None
    await child.append_message(Message("user", "child work"))
    # Sidechain lives under {session_id}/subagents/, so the project listing ignores it.
    infos = list_sessions(project_dir(str(tmp_path), Path.cwd()))
    assert all(i.session_id != "agentABC" for i in infos)
    assert "subagents" in str(child.path)
