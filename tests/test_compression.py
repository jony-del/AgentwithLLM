import asyncio
from typing import Any

from agent_core.compression import (
    PTL_RETRY_MARKER,
    CompressionConfig,
    CompressionEvent,
    CompressionPipeline,
    build_post_compact_messages,
    build_summary_user_message,
    group_into_rounds,
    is_preserved,
    parse_prompt_too_long_gap,
    shrink_oversize_messages,
    split_on_round_boundary,
    truncate_head_for_ptl_retry,
)
from agent_core.compression_summary import build_summarizer, extract_summary
from agent_core.models import LLMResult, Message
from agent_core.providers.base import LLMProvider, gated_provider
from agent_core.providers.fake import FakeProvider

# The fixed continuation-wrapper fragments the summary USER message is built from.
SUMMARY_HEADER = (
    "This session is being continued from a previous conversation that ran out of "
    "context. The summary below covers the earlier portion of the conversation."
)
SUMMARY_FOOTER = (
    "Continue the conversation from where it left off without asking the user any "
    "further questions. Resume directly — do not acknowledge the summary, do not "
    "recap, do not preface. Pick up the last task as if the break never happened."
)


def _assistant_with_calls(content: str, call_ids: list[str]) -> Message:
    return Message(
        "assistant",
        content,
        metadata={"tool_calls": [{"id": cid, "name": "echo", "arguments": {}} for cid in call_ids]},
    )


def _tool_result(content: str, call_id: str) -> Message:
    return Message("tool", content, name="echo", metadata={"tool_call_id": call_id, "ok": True})


class _RecordingProvider(LLMProvider):
    """Non-fake provider that records each ``complete`` call and returns a canned reply.

    ``stop_reasons`` optionally drives the per-call ``stop_reason`` (e.g.
    ``["max_tokens", "end_turn"]`` to truncate the first attempt then succeed); once
    exhausted the last value repeats. ``None`` returns ``stop_reason=None`` every call.
    """

    def __init__(
        self,
        content: str = "<analysis>scratch</analysis><summary>DONE</summary>",
        stop_reasons: list[str] | None = None,
    ) -> None:
        self.content = content
        self.calls: list[tuple[list[Message], list[dict[str, Any]], dict[str, Any]]] = []
        self.streams: list[Any] = []
        self._stop_reasons = list(stop_reasons) if stop_reasons else None

    async def complete(self, messages, tools, config, stream=None, should_cancel=None) -> LLMResult:
        self.calls.append((messages, tools, config))
        self.streams.append(stream)
        stop = None
        if self._stop_reasons:
            stop = self._stop_reasons[min(len(self.calls) - 1, len(self._stop_reasons) - 1)]
        return LLMResult(content=self.content, stop_reason=stop)


def _long_history(count: int, *, prefix: str = "m") -> list[Message]:
    return [Message("user", f"{prefix}{index} {'x' * 5}") for index in range(count)]


def _gate_config(**overrides: Any) -> CompressionConfig:
    """A config whose token gate is easy to cross: a small window override so the
    derived threshold is a handful of tokens (estimator default = char_count // 4)."""
    base = {
        "context_window_tokens": 600,
        "autocompact_buffer_tokens": 100,
        "reserved_output_tokens_for_summary": 100,
        "max_message_chars": 60,
        "collapsed_keep_recent": 4,
    }
    base.update(overrides)
    return CompressionConfig(**base)


async def test_auto_compact_triggers_when_estimate_at_or_above_threshold() -> None:
    # window 600, reserve min(8192,100)=100 → effective 500, buffer 100 → threshold 400.
    pipeline = CompressionPipeline(_gate_config())
    # 12 messages * ~120 chars = ~1440 chars → ~360... push well over: char/4 must be >= 400.
    messages = [Message("user", f"m{i} " + "x" * 200) for i in range(12)]
    estimate = sum(len(m.content) for m in messages) // 4
    assert estimate >= 400
    compacted, events = await pipeline.auto_compact(messages, model="claude-haiku")
    assert events
    assert sum(len(m.content) for m in compacted) < sum(len(m.content) for m in messages)


async def test_auto_compact_skips_below_threshold() -> None:
    pipeline = CompressionPipeline(_gate_config())
    messages = [Message("user", "small"), Message("tool", "also small")]
    assert sum(len(m.content) for m in messages) // 4 < 400
    compacted, events = await pipeline.auto_compact(messages, model="claude-haiku")
    assert compacted is messages
    assert events == []


async def test_auto_compact_uses_injected_estimator() -> None:
    pipeline = CompressionPipeline(_gate_config())
    messages = [Message("user", "tiny")]
    # Injected estimator reports a huge number → gate trips despite tiny chars.
    compacted, events = await pipeline.auto_compact(
        messages, model="claude-haiku", token_estimator=lambda msgs: 10_000
    )
    # Stages run (events may be empty if nothing shrinks), but the gate was crossed:
    # a single tiny message can't fold, so it returns unchanged — the point is the gate
    # tripped. Verify the inverse: a low estimator never trips on a large history.
    big = [Message("user", "x" * 4000) for _ in range(5)]
    compacted2, events2 = await pipeline.auto_compact(
        big, model="claude-haiku", token_estimator=lambda msgs: 1
    )
    assert compacted2 is big
    assert events2 == []


async def test_auto_compact_pct_override_lowers_threshold() -> None:
    # Without override the gate stays closed; a small pct override trips it.
    messages = [Message("user", "x" * 800) for _ in range(4)]  # ~3200 chars → ~800 tokens
    closed = CompressionPipeline(CompressionConfig(max_message_chars=60, collapsed_keep_recent=4))
    out, events = await closed.auto_compact(messages, model="claude-haiku")
    assert out is messages and events == []

    override = CompressionPipeline(
        CompressionConfig(max_message_chars=60, collapsed_keep_recent=4, autocompact_pct_override=0.1)
    )
    out2, events2 = await override.auto_compact(messages, model="claude-haiku")
    # pct 0.1% of effective (~180000) ≈ 180 tokens → well under ~800 estimate → trips.
    assert events2
    assert sum(len(m.content) for m in out2) < sum(len(m.content) for m in messages)


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

    # [collapsed summary USER message, last 4 recent messages]
    assert len(compacted) == 5
    summary = compacted[0]
    assert summary.role == "user"
    assert summary.metadata == {"compressed": "context_collapse", "messages_collapsed": 4}
    body = "user: u0 xxxxx | user: u1 xxxxx | user: u2 xxxxx | user: u3 xxxxx"
    assert summary.content == f"{SUMMARY_HEADER}\n\n{body}\n\n{SUMMARY_FOOTER}"
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
    assert block.role == "user"
    assert block.content == f"{SUMMARY_HEADER}\n\nCOMPACTED 8 messages\n\n{SUMMARY_FOOTER}"
    assert block.metadata["messages_collapsed"] == 8
    assert block.metadata["is_compact_summary"] is True
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


async def test_build_summarizer_issues_no_tools_streamed_bounded_call() -> None:
    provider = _RecordingProvider()
    summarizer = build_summarizer(
        provider,
        {"model": "claude-opus-4-8", "max_tokens": 2048, "stream": False, "thinking_budget": 4096},
        CompressionConfig(compact_summary_start_tokens=8000, compact_max_output_tokens=20000),
    )
    assert summarizer is not None

    out = await summarizer([Message("user", "hello"), Message("assistant", "hi")])

    assert out == "DONE"  # <summary> extracted, <analysis> stripped
    # stop_reason is None (success) → no escalation, a single attempt.
    (_messages, tools, config), = provider.calls
    assert tools == []  # no tool use during summary
    assert config["max_tokens"] == 8000  # first ladder tier = compact_summary_start_tokens
    assert config["stream"] is True  # summary call streams (dodges non-streaming timeout)
    assert config["thinking_budget"] is None
    assert provider.streams[0] is not None  # a (no-op) StreamHandler sink is passed


async def test_build_summarizer_escalates_budget_on_max_tokens() -> None:
    # Truncate the first two attempts, succeed on the third: 8k → 20k → model ceiling (128k).
    provider = _RecordingProvider(stop_reasons=["max_tokens", "max_tokens", "end_turn"])
    summarizer = build_summarizer(
        provider,
        {"model": "claude-opus-4-8"},
        CompressionConfig(
            compact_summary_start_tokens=8000,
            compact_max_output_tokens=20000,
            compact_max_output_retries=2,
        ),
    )
    assert summarizer is not None

    out = await summarizer([Message("user", "x" * 200)])

    assert out == "DONE"
    budgets = [config["max_tokens"] for (_m, _t, config) in provider.calls]
    assert budgets == [8000, 20000, 128_000]  # Opus hard ceiling from tokens.model_output_tokens


async def test_build_summarizer_stops_escalating_after_retry_budget() -> None:
    # Every attempt truncates; retries=1 caps the ladder at 2 attempts (no third tier).
    provider = _RecordingProvider(stop_reasons=["max_tokens"])
    summarizer = build_summarizer(
        provider,
        {"model": "claude-opus-4-8"},
        CompressionConfig(
            compact_summary_start_tokens=8000,
            compact_max_output_tokens=20000,
            compact_max_output_retries=1,
        ),
    )
    assert summarizer is not None

    await summarizer([Message("user", "x" * 200)])

    budgets = [config["max_tokens"] for (_m, _t, config) in provider.calls]
    assert budgets == [8000, 20000]  # 1 attempt + 1 escalation, ceiling tier not reached


def test_extract_summary_parses_and_falls_back() -> None:
    assert extract_summary("<analysis>think</analysis><summary>BODY</summary>") == "BODY"
    assert extract_summary("plain text, no tags") == "plain text, no tags"
    assert extract_summary("<analysis>only scratch</analysis>") == ""


# --- round grouping (group_into_rounds / split_on_round_boundary) -------------


def test_group_into_rounds_keeps_tool_round_atomic() -> None:
    msgs = [
        Message("user", "do it"),
        _assistant_with_calls("calling", ["c1", "c2"]),
        _tool_result("r1", "c1"),
        _tool_result("r2", "c2"),
        Message("assistant", "done"),
    ]
    rounds = group_into_rounds(msgs)
    # user | (assistant+2 tools) | assistant
    assert [len(g) for g in rounds] == [1, 3, 1]
    assert rounds[1][0] is msgs[1]
    assert rounds[1][1:] == [msgs[2], msgs[3]]


def test_group_into_rounds_round_trips_in_order() -> None:
    msgs = [
        Message("user", "a"),
        _assistant_with_calls("c", ["x"]),
        _tool_result("rx", "x"),
        Message("user", "b"),
        _assistant_with_calls("c2", ["y"]),
        _tool_result("ry", "y"),
    ]
    flat = [m for group in group_into_rounds(msgs) for m in group]
    assert flat == msgs


def test_group_into_rounds_unmatched_tool_is_its_own_group() -> None:
    # A tool result whose id does not match the preceding assistant's calls starts a
    # fresh group rather than being glued to the wrong round.
    msgs = [
        _assistant_with_calls("c", ["x"]),
        _tool_result("ry", "OTHER"),
    ]
    rounds = group_into_rounds(msgs)
    assert [len(g) for g in rounds] == [1, 1]


def test_split_on_round_boundary_never_orphans_a_tool() -> None:
    msgs = [
        Message("user", "u0"),
        _assistant_with_calls("a1", ["c1"]),
        _tool_result("r1", "c1"),
        Message("user", "u1"),
        _assistant_with_calls("a2", ["c2"]),
        _tool_result("r2", "c2"),
    ]
    # keep=2 would land mid-round (orphan tool r2); the split must snap earlier.
    prefix, recent = split_on_round_boundary(msgs, keep=2)
    assert prefix + recent == msgs
    assert recent[0].role != "tool"  # recent never starts with an orphan tool result
    # Each side concatenates from whole rounds.
    for side in (prefix, recent):
        flat = [m for g in group_into_rounds(side) for m in g]
        assert flat == side


async def test_context_collapse_never_splits_a_round() -> None:
    pipeline = CompressionPipeline(CompressionConfig(max_message_chars=10000, collapsed_keep_recent=4))
    msgs: list[Message] = []
    for i in range(6):
        msgs.append(Message("user", f"u{i}"))
        msgs.append(_assistant_with_calls(f"a{i}", [f"c{i}"]))
        msgs.append(_tool_result(f"r{i}", f"c{i}"))

    compacted, events = await pipeline.reactive_compact(msgs)
    assert events
    # The summary is index 0 here (no preserved system block); after it, no message may
    # be an orphan tool result whose matching assistant got folded away.
    recent = compacted[1:]
    assert recent and recent[0].role != "tool"
    # Verify every tool result in the recent tail still has its assistant alongside.
    flat_rounds = group_into_rounds(recent)
    for group in flat_rounds:
        if group[0].role == "tool":
            raise AssertionError("recent tail starts a round with an orphan tool result")


# --- pinned-across-roles preservation -----------------------------------------


async def test_pinned_user_message_survives_collapse() -> None:
    pinned_user = Message(
        "user", "USER CONTEXT", metadata={"pinned": "user_context"}
    )
    pipeline = CompressionPipeline(CompressionConfig(max_message_chars=10000, collapsed_keep_recent=4))
    messages = [
        Message("system", "base prompt"),
        pinned_user,
        *_long_history(10),
    ]

    compacted, events = await pipeline.reactive_compact(messages)

    assert events
    # Both the base system block and the pinned USER message survive verbatim, at front.
    assert compacted[0] == Message("system", "base prompt")
    assert pinned_user in compacted
    assert compacted[1] is pinned_user


async def test_prior_user_summary_is_refolded_once() -> None:
    pipeline = CompressionPipeline(CompressionConfig(max_message_chars=10000, collapsed_keep_recent=4))
    first, _ = await pipeline.reactive_compact(_long_history(12))
    # The folded summary is now a USER message; feeding it back must re-fold it, not
    # accumulate a second permanent summary block.
    second_input = [*first, *_long_history(8, prefix="n")]
    second, _ = await pipeline.reactive_compact(second_input)
    summaries = [m for m in second if m.metadata.get("compressed") in {"context_collapse", "llm_summary"}]
    assert len(summaries) == 1


# --- summary-as-user-message --------------------------------------------------


def test_build_summary_user_message_track_a_and_b() -> None:
    a = build_summary_user_message("BODY A", marker="llm_summary", messages_collapsed=5)
    assert a.role == "user"
    assert a.content == f"{SUMMARY_HEADER}\n\nBODY A\n\n{SUMMARY_FOOTER}"
    assert a.metadata == {"compressed": "llm_summary", "messages_collapsed": 5, "is_compact_summary": True}

    b = build_summary_user_message("BODY B", marker="context_collapse", messages_collapsed=3)
    assert b.role == "user"
    assert b.metadata == {"compressed": "context_collapse", "messages_collapsed": 3}
    assert "is_compact_summary" not in b.metadata


def test_build_post_compact_messages_orders_and_appends_attachments() -> None:
    summary = build_summary_user_message("S", marker="llm_summary", messages_collapsed=1)
    recent = [Message("user", "r1"), Message("assistant", "r2")]
    attach = [Message("user", "file dump", metadata={"attachment": True})]
    out = build_post_compact_messages(summary, recent, attachments=attach)
    assert out == [summary, *recent, *attach]
    # No attachments → just summary + recent.
    assert build_post_compact_messages(summary, recent) == [summary, *recent]


# --- post-compact attachments injected only on a real fold --------------------


def _attachment(text: str = "FILE DUMP") -> Message:
    return Message("user", text, metadata={"post_compact_attachment": True})


async def test_attachments_injected_on_a_real_fold() -> None:
    pipeline = CompressionPipeline(CompressionConfig(max_message_chars=1000, collapsed_keep_recent=4))
    system = Message("system", "keep these instructions")
    convo = [Message("user", f"{index}: {'x' * 40}") for index in range(10)]
    messages = [system, *convo]
    attach = [_attachment()]

    compacted, events = await pipeline.reactive_compact(messages, attachments=attach)

    assert events  # a fold happened
    # The attachment lives in the TAIL (after the summary + recent), exactly once.
    post = [m for m in compacted if m.metadata.get("post_compact_attachment")]
    assert post == attach
    assert compacted[-1] is attach[0]


async def test_attachments_not_injected_on_no_fold_early_return() -> None:
    # Too few conversation messages to fold (<= keep + 1) → early return, no attachment.
    pipeline = CompressionPipeline(CompressionConfig(max_message_chars=10000, collapsed_keep_recent=8))
    messages = [Message("system", "sys"), Message("user", "just one turn")]
    attach = [_attachment()]

    compacted, events = await pipeline.auto_compact(
        messages,
        token_estimator=lambda msgs: 10**9,  # force the gate open
        attachments=attach,
    )

    assert not any(m.metadata.get("post_compact_attachment") for m in compacted)


async def test_attachments_are_not_preserved_and_refold_on_next_pass() -> None:
    # An attachment is foldable conversation (not pinned): a subsequent compaction folds it.
    attach = _attachment()
    assert not is_preserved(attach)

    pipeline = CompressionPipeline(CompressionConfig(max_message_chars=1000, collapsed_keep_recent=4))
    system = Message("system", "keep these instructions")
    convo = [Message("user", f"{index}: {'x' * 40}") for index in range(10)]
    first, _ = await pipeline.reactive_compact([system, *convo], attachments=[attach])
    assert any(m.metadata.get("post_compact_attachment") for m in first)

    # Second pass with more history: the old attachment is now in the foldable prefix.
    more = [Message("user", f"n{index}: {'y' * 40}") for index in range(10)]
    second, events = await pipeline.reactive_compact([*first, *more])
    assert events
    # Preserved front is still exactly the system block (front-prefix invariant holds).
    preserved = [m for m in second if is_preserved(m)]
    assert preserved == [system]
    assert second[: len(preserved)] == preserved


# --- single timeout degrades to Track B ---------------------------------------


async def test_slow_summarizer_times_out_to_track_b() -> None:
    pipeline = CompressionPipeline(
        CompressionConfig(max_message_chars=10000, summary_keep_recent=8, summary_timeout_seconds=0.05)
    )

    async def slow(prefix: list[Message]) -> str:
        await asyncio.sleep(1.0)
        return "TOO LATE"

    compacted, events = await pipeline.reactive_compact(_long_history(12), summarizer=slow)

    # Track A timed out → deterministic Track B block, run unharmed.
    assert any(m.metadata.get("compressed") == "context_collapse" for m in compacted)
    assert not any(m.metadata.get("compressed") == "llm_summary" for m in compacted)
    collapse = next(e for e in events if e.stage == "context_collapse")
    assert "summary_fallback: TimeoutError" in collapse.detail


# --- circuit breaker ----------------------------------------------------------


async def test_circuit_breaker_short_circuits_after_consecutive_failures() -> None:
    # A single message can't fold (conversation <= keep+1) so stages never reduce it,
    # yet a forced over-threshold estimator makes every attempt a "failure".
    pipeline = CompressionPipeline(
        CompressionConfig(
            context_window_tokens=600,
            autocompact_buffer_tokens=100,
            reserved_output_tokens_for_summary=100,
            max_consecutive_autocompact_failures=3,
        )
    )
    # One un-foldable conversation message (conversation <= keep+1) so context_collapse
    # never reduces it below the line; the always-over estimator makes every run a fail.
    messages = [Message("user", "x" * 4000)]
    over = lambda msgs: 10_000  # always over the ~400 threshold

    # First 3 attempts run the stages but stay over → counted as failures.
    for n in range(1, 4):
        out, _ = await pipeline.auto_compact(messages, model="m", token_estimator=over)
        assert pipeline._consecutive_autocompact_failures == n

    # Breaker tripped: subsequent calls short-circuit silently — input returned as-is,
    # no events, counter not incremented past the cap.
    out, events = await pipeline.auto_compact(messages, model="m", token_estimator=over)
    assert out is messages
    assert events == []
    assert pipeline._consecutive_autocompact_failures == 3


async def test_circuit_breaker_resets_on_success() -> None:
    pipeline = CompressionPipeline(
        CompressionConfig(
            context_window_tokens=600,
            autocompact_buffer_tokens=100,
            reserved_output_tokens_for_summary=100,
            max_message_chars=60,
            collapsed_keep_recent=4,
            max_consecutive_autocompact_failures=3,
        )
    )
    failing = [Message("user", "x" * 4000)]  # un-foldable
    over = lambda msgs: 10_000
    for _ in range(2):
        await pipeline.auto_compact(failing, model="m", token_estimator=over)
    assert pipeline._consecutive_autocompact_failures == 2

    # A foldable history that drops below the threshold afterwards resets the counter.
    big = [Message("user", f"m{i} " + "x" * 200) for i in range(12)]

    def estimator(msgs: list[Message]) -> int:
        # Over before folding, under after (so the success branch is taken).
        return 10_000 if len(msgs) > 6 else 10

    out, events = await pipeline.auto_compact(big, model="m", token_estimator=estimator)
    assert events
    assert pipeline._consecutive_autocompact_failures == 0


async def test_gate_below_threshold_resets_breaker() -> None:
    pipeline = CompressionPipeline(
        CompressionConfig(
            context_window_tokens=600,
            autocompact_buffer_tokens=100,
            reserved_output_tokens_for_summary=100,
        )
    )
    pipeline._consecutive_autocompact_failures = 2
    out, events = await pipeline.auto_compact(
        [Message("user", "tiny")], model="m", token_estimator=lambda msgs: 1
    )
    assert out is not None and events == []
    assert pipeline._consecutive_autocompact_failures == 0


# --- Phase 3D: reactive 413 gap parsing + head truncation ---------------------


def test_parse_prompt_too_long_gap_valid() -> None:
    text = "prompt is too long: 219763 tokens > 200000 maximum"
    assert parse_prompt_too_long_gap(text) == 19763


def test_parse_prompt_too_long_gap_case_insensitive_and_wrapped() -> None:
    # JSON/SDK-wrapped, mixed casing, plural/singular "token(s)" all tolerated.
    text = '{"error": {"message": "Prompt Is Too Long: 1500 Tokens > 1000"}}'
    assert parse_prompt_too_long_gap(text) == 500
    assert parse_prompt_too_long_gap("prompt is too long: 1001 token > 1000") == 1


def test_parse_prompt_too_long_gap_missing_returns_none() -> None:
    assert parse_prompt_too_long_gap("some unrelated error") is None
    assert parse_prompt_too_long_gap("") is None


def test_parse_prompt_too_long_gap_reversed_returns_none() -> None:
    # actual <= limit → no positive gap → None.
    assert parse_prompt_too_long_gap("prompt is too long: 1000 tokens > 1000") is None
    assert parse_prompt_too_long_gap("prompt is too long: 900 tokens > 1000") is None


def _pinned(role: str, content: str) -> Message:
    return Message(role, content, metadata={"pinned": True})


def _ptl_history(rounds: int) -> list[Message]:
    """Preserved front (system + pinned userContext) followed by ``rounds`` API rounds,
    each an assistant-with-tool_calls glued to its tool result."""
    messages: list[Message] = [
        Message("system", "base system block"),
        _pinned("user", "<system-reminder>userContext</system-reminder>"),
    ]
    for index in range(rounds):
        call_id = f"toolu_{index}"
        messages.append(_assistant_with_calls(f"call {index}", [call_id]))
        messages.append(_tool_result(f"result {index}", call_id))
    return messages


def _assert_no_orphan_tool_results(messages: list[Message]) -> None:
    """Every tool message must follow an assistant whose tool_calls include its id."""
    live_call_ids: set[str] = set()
    for message in messages:
        if message.role == "assistant":
            live_call_ids = {
                call["id"] for call in message.metadata.get("tool_calls", []) if "id" in call
            }
        elif message.role == "tool":
            assert message.metadata["tool_call_id"] in live_call_ids, "orphaned tool_result"


def test_truncate_head_drops_oldest_whole_rounds_keeps_preserved() -> None:
    messages = _ptl_history(4)  # preserved front + 4 rounds
    out = truncate_head_for_ptl_retry(messages, token_gap=None)
    assert out is not None
    # Preserved front survives intact at the head.
    assert out[0].role == "system"
    assert out[1].metadata.get("pinned")
    # 20% of 4 rounds = floor(0.8) = max(1, 0) = 1 round dropped (round 0).
    assert not any("call 0" in m.content for m in out)
    assert any("call 1" in m.content for m in out)
    _assert_no_orphan_tool_results(out)


def test_truncate_head_returns_none_with_fewer_than_two_rounds() -> None:
    # Only one droppable round → can't make progress → None.
    assert truncate_head_for_ptl_retry(_ptl_history(1)) is None
    # Zero rounds (only the preserved front) → None.
    assert truncate_head_for_ptl_retry(_ptl_history(0)) is None


def test_truncate_head_always_keeps_at_least_one_round() -> None:
    # A gap far larger than the whole history would drop everything, but we cap at
    # len(rounds) - 1 so at least one round survives.
    messages = _ptl_history(3)
    out = truncate_head_for_ptl_retry(messages, token_gap=10_000_000)
    assert out is not None
    rounds_left = group_into_rounds([m for m in out if not is_preserved(m)])
    assert len(rounds_left) == 1
    _assert_no_orphan_tool_results(out)


def test_truncate_head_gap_driven_drop_count() -> None:
    # Each round is ~ "call N"(6) + "result N"(8) = ~14 chars → ~3 tokens (char/4).
    # Use an explicit estimator returning 10 tokens/round so the math is exact.
    messages = _ptl_history(5)
    estimator = lambda group: 10  # noqa: E731 - 10 tokens per round, exact math
    # gap=25 → 10 (1 group, <25) → 20 (2 groups, <25) → 30 (3 groups, >=25) → drop 3.
    out = truncate_head_for_ptl_retry(messages, token_gap=25, token_estimator=estimator)
    assert out is not None
    rounds_left = group_into_rounds([m for m in out if not is_preserved(m)])
    assert len(rounds_left) == 2  # 5 - 3 dropped
    _assert_no_orphan_tool_results(out)


def test_truncate_head_fallback_drop_count_is_twenty_percent() -> None:
    messages = _ptl_history(10)  # floor(10 * 0.2) = 2 rounds dropped
    out = truncate_head_for_ptl_retry(messages, token_gap=None)
    assert out is not None
    rounds_left = group_into_rounds([m for m in out if not is_preserved(m)])
    assert len(rounds_left) == 8
    _assert_no_orphan_tool_results(out)


def test_truncate_head_preserves_round_integrity_under_gap_drop() -> None:
    # Ordering/round-integrity: the kept conversation concatenates whole rounds in order.
    messages = _ptl_history(6)
    out = truncate_head_for_ptl_retry(messages, token_gap=None)  # drops floor(6*0.2)=1
    assert out is not None
    conversation = [m for m in out if not is_preserved(m)]
    # Re-grouping the survivors yields the same intact rounds (no split mid-round).
    for group in group_into_rounds(conversation):
        assert group[0].role == "assistant"
        assert all(m.role == "tool" for m in group[1:])
    _assert_no_orphan_tool_results(out)


def test_truncate_head_does_not_insert_marker_when_userContext_present() -> None:
    # The preserved front ends with the pinned userContext USER message, so the first
    # non-system message is already a user turn → no synthetic marker is prepended.
    out = truncate_head_for_ptl_retry(_ptl_history(4), token_gap=None)
    assert out is not None
    assert not any(m.metadata.get("ptl_retry_marker") for m in out)


def test_truncate_head_inserts_user_marker_when_no_pinned_userContext() -> None:
    # No pinned userContext (preserved front is system-only). After dropping the oldest
    # round, the first conversation message would be an assistant → a PTL_RETRY_MARKER
    # user turn is prepended so the messages array still begins with a user turn.
    messages = [Message("system", "base system")]
    for index in range(3):
        call_id = f"toolu_{index}"
        messages.append(_assistant_with_calls(f"call {index}", [call_id]))
        messages.append(_tool_result(f"result {index}", call_id))
    out = truncate_head_for_ptl_retry(messages, token_gap=None)
    assert out is not None
    # First non-system message is the synthetic user marker.
    first_non_system = next(m for m in out if m.role != "system")
    assert first_non_system.role == "user"
    assert first_non_system.metadata.get("ptl_retry_marker")
    assert first_non_system.content == PTL_RETRY_MARKER
    _assert_no_orphan_tool_results(out)


def test_shrink_oversize_messages_shrinks_largest_and_converges() -> None:
    # A single oversized non-preserved message: whole-round truncation can't help, so
    # shrink head/tail-truncates it with an omission marker until the gap is shed.
    messages = [
        Message("system", "base"),
        _pinned("user", "ctx"),
        Message("user", "x" * 8000),
    ]
    out = shrink_oversize_messages(messages, tokens_to_drop=500, min_keep_chars=200)
    assert out is not None
    # Preserved front untouched (identity preserved).
    assert out[0] is messages[0] and out[1] is messages[1]
    shrunk = out[2]
    assert shrunk.metadata.get("compressed") == "ptl_shrink"
    assert len(shrunk.content) < 8000
    assert "[truncated" in shrunk.content
    # Head and tail are both retained.
    assert shrunk.content.startswith("x" * 200)
    assert shrunk.content.endswith("x" * 200)


def test_shrink_oversize_messages_returns_none_when_nothing_to_shrink() -> None:
    # All messages already at/under the floor → nothing can be shed → None (caller gives up).
    messages = [Message("system", "base"), _pinned("user", "ctx"), Message("user", "tiny")]
    assert shrink_oversize_messages(messages, tokens_to_drop=500) is None
    # Non-positive request → None.
    assert shrink_oversize_messages([Message("user", "x" * 8000)], tokens_to_drop=0) is None


def test_shrink_oversize_messages_never_touches_preserved() -> None:
    # Even a huge preserved (pinned) message is left intact; only conversation shrinks.
    messages = [
        Message("system", "base"),
        _pinned("user", "P" * 8000),
        Message("assistant", "A" * 8000),
    ]
    out = shrink_oversize_messages(messages, tokens_to_drop=100, min_keep_chars=200)
    assert out is not None
    assert out[1].content == "P" * 8000  # pinned untouched
    assert len(out[2].content) < 8000  # conversation shrunk


async def test_context_collapse_then_truncate_keeps_rounds_intact() -> None:
    # Adversarial: run a real fold, THEN head-truncate the folded result. Round integrity
    # and the preserved front must survive both stages in sequence.
    pipeline = CompressionPipeline(CompressionConfig(summary_keep_recent=4, use_llm_summary=False))
    messages = _ptl_history(8)  # system + pinned userContext + 8 tool rounds
    folded, event = await pipeline._context_collapse(messages, aggressive=False, summarizer=None)
    # A summary USER message replaced the old prefix; preserved front still leads.
    assert folded[0].role == "system"
    assert folded[1].metadata.get("pinned")
    _assert_no_orphan_tool_results(folded)
    # Now truncate the folded history.
    truncated = truncate_head_for_ptl_retry(folded, token_gap=None)
    assert truncated is not None
    assert truncated[0].role == "system"
    assert truncated[1].metadata.get("pinned")
    _assert_no_orphan_tool_results(truncated)


def test_summary_timeout_default_is_bounded() -> None:
    # Regression: the Track A summary call must be time-bounded by default so a wedged
    # provider can't hang a run (was previously None / unbounded). 120s bounds the WHOLE
    # streamed escalation ladder under one (non-stacked) timeout.
    assert CompressionConfig().summary_timeout_seconds == 120.0
