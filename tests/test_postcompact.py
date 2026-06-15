from pathlib import Path

from agent_core.compression import CompressionConfig
from agent_core.models import Message, ToolCall, ToolResult
from agent_core.providers.fake import FakeProvider
from agent_core.react import ReActAgent, ReActConfig


def _agent(tmp_path: Path, **compression_overrides) -> ReActAgent:
    config = ReActConfig(
        run_dir=str(tmp_path / "runs"),
        compression=CompressionConfig(**compression_overrides),
    )
    agent = ReActAgent(FakeProvider(), config)
    agent.session.workspace = tmp_path.resolve()
    return agent


def test_record_read_result_keys_by_resolved_path(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    call = ToolCall("read_text_file", {"path": "sub/f.txt"}, id="t1")
    result = ToolResult(name="read_text_file", content="CONTENT", ok=True)
    agent._record_read_result(call, result)

    key = str((tmp_path / "sub/f.txt").resolve())
    assert agent.session.read_file_state == {key: "CONTENT"}


def test_record_read_result_skips_non_read_and_failed(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent._record_read_result(ToolCall("echo", {"path": "x"}, id="a"), ToolResult(name="echo", content="c", ok=True))
    agent._record_read_result(
        ToolCall("read_text_file", {"path": "x"}, id="b"), ToolResult(name="read_text_file", content="c", ok=False)
    )
    agent._record_read_result(ToolCall("read_text_file", {}, id="c"), ToolResult(name="read_text_file", content="c", ok=True))
    assert agent.session.read_file_state == {}


def test_record_read_result_excludes_claude_md(tmp_path: Path) -> None:
    # CLAUDE.md is already pinned as the userContext reminder, so re-attaching it after a
    # fold is pure duplication — it must never enter the read-state.
    agent = _agent(tmp_path)
    agent._record_read_result(
        ToolCall("read_text_file", {"path": "CLAUDE.md"}, id="t1"),
        ToolResult(name="read_text_file", content="# project rules", ok=True),
    )
    agent._record_read_result(
        ToolCall("read_text_file", {"path": "sub/CLAUDE.md"}, id="t2"),
        ToolResult(name="read_text_file", content="# nested rules", ok=True),
    )
    assert agent.session.read_file_state == {}


def test_build_read_attachments_empty_when_nothing_read(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    assert agent._build_read_attachments() == []


def test_build_read_attachments_framing_and_metadata(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent.session.record_read(str((tmp_path / "a.txt").resolve()), "AAA")
    [msg] = agent._build_read_attachments()
    assert msg.role == "user"
    assert msg.metadata == {"post_compact_attachment": True}
    assert "post_compact_attachment" in msg.metadata
    assert "<system-reminder>" in msg.content and "</system-reminder>" in msg.content
    assert "re-attached after the conversation was compacted" in msg.content
    assert "## a.txt" in msg.content  # workspace-relative heading
    assert "AAA" in msg.content


def test_build_read_attachments_recency_cap_newest_first(tmp_path: Path) -> None:
    agent = _agent(tmp_path, post_compact_max_files=2)
    for name in ("old", "mid", "new"):
        agent.session.record_read(str((tmp_path / f"{name}.txt").resolve()), name.upper())
    [msg] = agent._build_read_attachments()
    # Only the 2 most-recent, newest first.
    assert msg.content.index("## new.txt") < msg.content.index("## mid.txt")
    assert "## old.txt" not in msg.content


def test_build_read_attachments_per_file_truncation(tmp_path: Path) -> None:
    # Token-based: per-file cap of 10 tokens → 40-char ceiling.
    agent = _agent(tmp_path, post_compact_max_tokens_per_file=10)
    agent.session.record_read(str((tmp_path / "big.txt").resolve()), "x" * 100)
    [msg] = agent._build_read_attachments()
    assert "x" * 40 in msg.content  # first 40 (10 tokens) kept
    assert "[truncated 60 chars]" in msg.content


def test_build_read_attachments_total_budget(tmp_path: Path) -> None:
    # Token-based: 7-token total budget → 28-char ceiling, smaller than one file.
    agent = _agent(
        tmp_path,
        post_compact_max_files=5,
        post_compact_max_tokens_per_file=250,
        post_compact_total_budget_tokens=7,
    )
    for name in ("a", "b", "c"):
        agent.session.record_read(str((tmp_path / f"{name}.txt").resolve()), name * 1000)
    [msg] = agent._build_read_attachments()
    assert "[truncated to fit budget]" in msg.content


async def test_forced_compaction_injects_read_attachment(tmp_path: Path) -> None:
    # Drive a real fold via the pipeline and confirm a recorded read re-injects.
    agent = _agent(tmp_path, max_message_chars=1000, collapsed_keep_recent=4)
    agent.session.record_read(str((tmp_path / "src.py").resolve()), "def f(): return 1")

    system = Message("system", "keep these instructions")
    convo = [Message("user", f"{i}: {'x' * 40}") for i in range(10)]
    attachments = agent._build_read_attachments()
    compacted, events = await agent.compression.reactive_compact(
        [system, *convo], attachments=attachments
    )

    assert events
    post = [m for m in compacted if m.metadata.get("post_compact_attachment")]
    assert len(post) == 1
    assert "def f(): return 1" in post[0].content
    assert "## src.py" in post[0].content
