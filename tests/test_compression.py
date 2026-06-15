from typing import Any

from agent_core.compression import CompressionConfig, CompressionEvent, CompressionPipeline
from agent_core.compression_summary import build_summarizer, extract_summary
from agent_core.models import LLMResult, Message
from agent_core.providers.base import LLMProvider, gated_provider
from agent_core.providers.fake import FakeProvider


class _RecordingProvider(LLMProvider):
    """Non-fake provider that records each ``complete`` call and returns a canned reply."""

    def __init__(self, content: str = "<analysis>scratch</analysis><summary>DONE</summary>") -> None:
        self.content = content
        self.calls: list[tuple[list[Message], list[dict[str, Any]], dict[str, Any]]] = []

    async def complete(self, messages, tools, config, stream=None, should_cancel=None) -> LLMResult:
        self.calls.append((messages, tools, config))
        return LLMResult(content=self.content)


def _long_history(count: int, *, prefix: str = "m") -> list[Message]:
    return [Message("user", f"{prefix}{index} {'x' * 5}") for index in range(count)]


async def test_auto_compact_triggers_when_context_is_large() -> None:
    pipeline = CompressionPipeline(CompressionConfig(max_context_chars=100, auto_threshold_ratio=0.5))
    messages = [Message("user", "x" * 80), Message("tool", "y" * 100)]
    compacted, events = await pipeline.auto_compact(messages)
    assert events
    assert sum(len(message.content) for message in compacted) < 180


async def test_auto_compact_skips_below_threshold() -> None:
    pipeline = CompressionPipeline(CompressionConfig(max_context_chars=100000, auto_threshold_ratio=0.8))
    messages = [Message("user", "small"), Message("tool", "also small")]
    compacted, events = await pipeline.auto_compact(messages)
    assert compacted is messages
    assert events == []


async def test_reactive_compact_is_more_aggressive() -> None:
    pipeline = CompressionPipeline(CompressionConfig(max_message_chars=60, collapsed_keep_recent=4))
    messages = [Message("user", str(index) + "x" * 100) for index in range(10)]
    compacted, events = await pipeline.reactive_compact(messages)
    assert events
    assert len(compacted) < len(messages)


async def test_context_collapse_preserves_system_prompt() -> None:
    pipeline = CompressionPipeline(CompressionConfig(max_message_chars=1000, collapsed_keep_recent=4))
    messages = [Message("system", "keep these instructions")]
    messages.extend(Message("user", f"{index}: {'x' * 40}") for index in range(10))

    compacted, events = await pipeline.reactive_compact(messages)

    assert events
    assert compacted[0] == Message("system", "keep these instructions")
    assert any(message.metadata.get("compressed") == "context_collapse" for message in compacted)


async def test_compact_reports_each_stage() -> None:
    pipeline = CompressionPipeline(CompressionConfig(max_message_chars=60, collapsed_keep_recent=4))
    messages = [Message("user", str(index) + "x" * 100) for index in range(10)]
    calls: list[tuple[int, int, str]] = []

    await pipeline.reactive_compact(
        messages, on_stage=lambda done, total, event: calls.append((done, total, event.stage))
    )

    assert [(done, total) for done, total, _ in calls] == [(1, 3), (2, 3), (3, 3)]
    assert [stage for _, _, stage in calls] == ["snip", "microcompact", "context_collapse"]


async def test_microcompact_preserves_pinned() -> None:
    pinned = Message("system", "PINNED " * 2000, metadata={"pinned": "claudemd"})
    pipeline = CompressionPipeline(CompressionConfig(max_message_chars=60, collapsed_keep_recent=4))
    messages = [pinned]
    messages.extend(Message("user", str(index) + "x" * 200) for index in range(10))

    compacted, _ = await pipeline.reactive_compact(messages)

    # The pinned block survives verbatim; ordinary long messages are still compressed.
    assert pinned in compacted
    assert any(message.metadata.get("compressed") for message in compacted)


async def test_collapse_event_carries_detail() -> None:
    pipeline = CompressionPipeline(CompressionConfig(max_message_chars=1000, collapsed_keep_recent=4))
    messages = [Message("user", f"{index}: {'x' * 40}") for index in range(10)]
    events: list[CompressionEvent] = []

    await pipeline.reactive_compact(messages, on_stage=lambda done, total, event: events.append(event))

    collapse = next(event for event in events if event.stage == "context_collapse")
    # 10 messages, keep_recent=4 → fold 6; no summarizer → deterministic Track B.
    assert collapse.detail == "collapsed 6 msgs (track_b)"


async def test_track_b_collapse_output_is_stable() -> None:
    """Byte-exact regression guard for Track B (deterministic) prefix folding.

    The naive collapse must keep producing the exact ``"role: content[:160]"`` join
    so the async refactor — and the later Track A branch — can be proven not to drift
    the no-LLM fallback behavior.
    """
    # keep has a floor of 4 (max(4, ...)); 8 messages → fold the first 4.
    pipeline = CompressionPipeline(CompressionConfig(max_message_chars=10000, collapsed_keep_recent=8))
    messages = [Message("user", f"u{index} {'x' * 5}") for index in range(8)]

    compacted, _ = await pipeline.reactive_compact(messages)

    # [collapsed summary block, last 4 recent messages]
    assert len(compacted) == 5
    summary = compacted[0]
    assert summary.role == "system"
    assert summary.metadata == {"compressed": "context_collapse", "messages_collapsed": 4}
    assert summary.content == (
        "Earlier conversation summary: "
        "user: u0 xxxxx | user: u1 xxxxx | user: u2 xxxxx | user: u3 xxxxx"
    )
    assert compacted[1:] == messages[-4:]


# --- Track A (LLM summary) ----------------------------------------------------


async def test_track_a_folds_prefix_with_summarizer() -> None:
    pipeline = CompressionPipeline(CompressionConfig(max_message_chars=10000, summary_keep_recent=8))
    messages = _long_history(12)

    async def stub(prefix: list[Message]) -> str:
        return f"COMPACTED {len(prefix)} messages"

    compacted, events = await pipeline.reactive_compact(messages, summarizer=stub)

    # aggressive keep = max(4, 8 // 2) = 4 → fold the first 8.
    block = next(m for m in compacted if m.metadata.get("compressed") == "llm_summary")
    assert block.role == "system"
    assert block.content == "Earlier conversation summary: COMPACTED 8 messages"
    assert block.metadata["messages_collapsed"] == 8
    collapse = next(e for e in events if e.stage == "context_collapse")
    assert collapse.detail == "collapsed 8 msgs (llm_summary)"
    # The recent window is preserved verbatim.
    assert compacted[-4:] == messages[-4:]


async def test_track_a_disabled_by_config_uses_track_b() -> None:
    pipeline = CompressionPipeline(
        CompressionConfig(max_message_chars=10000, collapsed_keep_recent=8, use_llm_summary=False)
    )
    messages = _long_history(12)

    async def stub(prefix: list[Message]) -> str:
        return "SHOULD NOT BE USED"

    compacted, _ = await pipeline.reactive_compact(messages, summarizer=stub)

    assert any(m.metadata.get("compressed") == "context_collapse" for m in compacted)
    assert not any(m.metadata.get("compressed") == "llm_summary" for m in compacted)


async def test_track_a_falls_back_when_summarizer_raises() -> None:
    pipeline = CompressionPipeline(CompressionConfig(max_message_chars=10000, summary_keep_recent=8))
    messages = _long_history(12)

    async def boom(prefix: list[Message]) -> str:
        raise RuntimeError("summarizer down")

    compacted, events = await pipeline.reactive_compact(messages, summarizer=boom)

    # Degrades to the deterministic block; the run is unharmed.
    assert any(m.metadata.get("compressed") == "context_collapse" for m in compacted)
    assert not any(m.metadata.get("compressed") == "llm_summary" for m in compacted)
    collapse = next(e for e in events if e.stage == "context_collapse")
    assert collapse.detail == "collapsed 8 msgs (summary_fallback: RuntimeError)"


async def test_track_a_falls_back_when_summary_is_blank() -> None:
    pipeline = CompressionPipeline(CompressionConfig(max_message_chars=10000, summary_keep_recent=8))
    messages = _long_history(12)

    async def blank(prefix: list[Message]) -> str:
        return "   \n  "

    compacted, events = await pipeline.reactive_compact(messages, summarizer=blank)

    assert any(m.metadata.get("compressed") == "context_collapse" for m in compacted)
    collapse = next(e for e in events if e.stage == "context_collapse")
    assert "summary_fallback: ValueError" in collapse.detail


async def test_track_a_keeps_pinned_blocks_out_of_summary() -> None:
    pinned = Message("system", "PROJECT INSTRUCTIONS", metadata={"pinned": "claudemd"})
    pipeline = CompressionPipeline(CompressionConfig(max_message_chars=10000, summary_keep_recent=8))
    messages = [pinned, *_long_history(12)]

    captured: list[list[Message]] = []

    async def stub(prefix: list[Message]) -> str:
        captured.append(prefix)
        return "OK"

    compacted, _ = await pipeline.reactive_compact(messages, summarizer=stub)

    # Pinned block survives verbatim and is never handed to the summarizer.
    assert pinned in compacted
    assert all(pinned not in prefix for prefix in captured)


async def test_repeated_track_a_refolds_prior_summary() -> None:
    pipeline = CompressionPipeline(CompressionConfig(max_message_chars=10000, summary_keep_recent=8))

    async def stub(prefix: list[Message]) -> str:
        return "SUMMARY"

    first, _ = await pipeline.reactive_compact(_long_history(12), summarizer=stub)
    # Feed the already-collapsed output back in with fresh turns; the prior summary
    # block must be foldable, not piling up as a second permanent summary.
    second_input = [*first, *_long_history(8, prefix="n")]
    second, _ = await pipeline.reactive_compact(second_input, summarizer=stub)

    summaries = [m for m in second if m.metadata.get("compressed") == "llm_summary"]
    assert len(summaries) == 1


# --- summarizer seam (build_summarizer / extract_summary) ---------------------


def test_build_summarizer_is_none_for_fake_provider() -> None:
    assert build_summarizer(FakeProvider(), {}, CompressionConfig()) is None
    # Even wrapped in the shared gate, the concrete fake is detected.
    assert build_summarizer(gated_provider(FakeProvider()), {}, CompressionConfig()) is None


def test_build_summarizer_is_none_when_disabled() -> None:
    assert build_summarizer(_RecordingProvider(), {}, CompressionConfig(use_llm_summary=False)) is None


async def test_build_summarizer_issues_no_tools_bounded_call() -> None:
    provider = _RecordingProvider()
    summarizer = build_summarizer(
        provider, {"model": "m", "max_tokens": 2048, "stream": True, "thinking_budget": 4096},
        CompressionConfig(summary_max_tokens=99),
    )
    assert summarizer is not None

    out = await summarizer([Message("user", "hello"), Message("assistant", "hi")])

    assert out == "DONE"  # <summary> extracted, <analysis> stripped
    (_messages, tools, config), = provider.calls
    assert tools == []  # no tool use during summary
    assert config["max_tokens"] == 99
    assert config["stream"] is False
    assert config["thinking_budget"] is None


def test_extract_summary_parses_and_falls_back() -> None:
    assert extract_summary("<analysis>think</analysis><summary>BODY</summary>") == "BODY"
    assert extract_summary("plain text, no tags") == "plain text, no tags"
    assert extract_summary("<analysis>only scratch</analysis>") == ""
