from pathlib import Path

import pytest

from agent_core.hooks import HookResult
from agent_core.memory import MemoryConfig
from agent_core.models import LLMContextTooLongError, LLMResult, Message, ToolCall
from agent_core.providers.fake import FakeProvider
from agent_core.react import ReActAgent, ReActConfig
from agent_core.storage import JSONLRunLogger
from agent_core.ui import AgentUI


class RejectEchoHook:
    def before_tool(self, tool_call: ToolCall) -> HookResult:
        if tool_call.name == "echo":
            return HookResult(allowed=False, reason="blocked")
        return HookResult()


class ToolIdProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools, config, stream=None, should_cancel=None) -> LLMResult:
        self.calls += 1
        if self.calls == 1:
            return LLMResult(
                "Calling echo",
                tool_calls=[ToolCall("echo", {"text": "hello"}, id="toolu_123")],
                stop_reason="tool_use",
            )
        return LLMResult("done", stop_reason="end")


class MultiToolProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.second_turn_messages = []

    async def complete(self, messages, tools, config, stream=None, should_cancel=None) -> LLMResult:
        self.calls += 1
        if self.calls == 1:
            return LLMResult(
                "Calling tools",
                tool_calls=[
                    ToolCall("echo", {"text": "first"}, id="toolu_1"),
                    ToolCall("echo", {"text": "second"}, id="toolu_2"),
                ],
                stop_reason="tool_use",
            )
        self.second_turn_messages = list(messages)
        return LLMResult("done", stop_reason="end")


async def test_react_returns_final_answer_without_tools(tmp_path: Path) -> None:
    logger = JSONLRunLogger(tmp_path)
    agent = ReActAgent(FakeProvider(), ReActConfig(run_dir=str(tmp_path)), logger=logger)
    result = await agent.run("hello")
    assert "Final answer" in result.answer
    assert logger.path.exists()


async def test_react_executes_demo_tool(tmp_path: Path) -> None:
    logger = JSONLRunLogger(tmp_path)
    agent = ReActAgent(FakeProvider(), ReActConfig(run_dir=str(tmp_path), permission="auto"), logger=logger)
    result = await agent.run("please use tool: echo")
    assert "observation" in result.answer
    assert "echo:" in result.answer


async def test_run_stops_when_cancelled(tmp_path: Path) -> None:
    logger = JSONLRunLogger(tmp_path)
    agent = ReActAgent(FakeProvider(), ReActConfig(run_dir=str(tmp_path)), logger=logger)
    result = await agent.run("please use tool: echo", should_cancel=lambda: True)
    assert "interrupt" in result.answer.lower()
    assert result.steps == 0


async def test_run_stops_when_provider_gate_sees_cancel(tmp_path: Path) -> None:
    calls = 0

    def cancel_after_loop_guard() -> bool:
        nonlocal calls
        calls += 1
        return calls > 1

    logger = JSONLRunLogger(tmp_path)
    agent = ReActAgent(FakeProvider(), ReActConfig(run_dir=str(tmp_path)), logger=logger)

    result = await agent.run("hello", should_cancel=cancel_after_loop_guard)

    assert "interrupt" in result.answer.lower()
    assert result.steps == 1


class _CancelDuringTurnProvider:
    """Final-answer (no-tool) provider that trips an interrupt flag while 'thinking'.

    Simulates the user pressing Esc *during* the model turn: the flag is still
    False at the loop-top guard and the provider gate (so the call dispatches and
    returns normally), and only becomes True by the time the loop re-polls after
    ``complete()`` returns. Before the post-turn re-poll existed, this Esc was
    silently swallowed and the run completed as if nothing happened.
    """

    def __init__(self) -> None:
        self.interrupted = False

    async def complete(self, messages, tools, config, stream=None, should_cancel=None) -> LLMResult:
        self.interrupted = True
        return LLMResult("Final answer: hi", stop_reason="end")


async def test_run_honors_cancel_pressed_during_final_answer_turn(tmp_path: Path) -> None:
    provider = _CancelDuringTurnProvider()
    logger = JSONLRunLogger(tmp_path)
    config = ReActConfig(run_dir=str(tmp_path), memory=MemoryConfig(enabled=False))
    agent = ReActAgent(provider, config, logger=logger)

    result = await agent.run("hello", should_cancel=lambda: provider.interrupted)

    assert "interrupt" in result.answer.lower()
    assert "Final answer" not in result.answer


async def test_run_injects_pinned_claude_md(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "CLAUDE.md").write_text("PROJECT INSTRUCTIONS HERE", encoding="utf-8")
    logger = JSONLRunLogger(tmp_path)
    config = ReActConfig(run_dir=str(tmp_path), memory=MemoryConfig(enabled=False))
    agent = ReActAgent(FakeProvider(), config, logger=logger)
    agent.session.workspace = workspace

    result = await agent.run("hello")

    # CLAUDE.md now lives in the pinned <system-reminder> userContext user message
    # (claudeMd entry), not a standalone system message.
    meta = [m for m in result.messages if m.metadata.get("pinned") == "user_context"]
    assert len(meta) == 1
    assert meta[0].role == "user"
    assert "# claudeMd\n" in meta[0].content
    assert "PROJECT INSTRUCTIONS HERE" in meta[0].content


async def test_run_skips_claude_md_when_disabled(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "CLAUDE.md").write_text("PROJECT INSTRUCTIONS HERE", encoding="utf-8")
    logger = JSONLRunLogger(tmp_path)
    config = ReActConfig(
        run_dir=str(tmp_path), memory=MemoryConfig(enabled=False), project_instructions=False
    )
    agent = ReActAgent(FakeProvider(), config, logger=logger)
    agent.session.workspace = workspace

    result = await agent.run("hello")

    # With project_instructions off there is no claudeMd entry; the userContext message
    # still carries currentDate but never the CLAUDE.md text.
    assert not any("PROJECT INSTRUCTIONS HERE" in m.content for m in result.messages)
    assert not any("# claudeMd" in m.content for m in result.messages)


async def test_react_preserves_tool_call_id_in_messages(tmp_path: Path) -> None:
    logger = JSONLRunLogger(tmp_path)
    agent = ReActAgent(ToolIdProvider(), ReActConfig(run_dir=str(tmp_path), permission="auto"), logger=logger)

    result = await agent.run("hello")

    assistant_message = next(message for message in result.messages if message.metadata.get("tool_calls"))
    tool_message = next(message for message in result.messages if message.role == "tool")
    assert assistant_message.metadata["tool_calls"][0]["id"] == "toolu_123"
    assert tool_message.metadata["tool_call_id"] == "toolu_123"


async def test_react_appends_multiple_tool_results_in_tool_call_order(tmp_path: Path) -> None:
    provider = MultiToolProvider()
    logger = JSONLRunLogger(tmp_path)
    config = ReActConfig(run_dir=str(tmp_path), permission="auto", memory=MemoryConfig(enabled=False))
    agent = ReActAgent(provider, config, logger=logger)

    result = await agent.run("hello")

    tool_messages = [message for message in result.messages if message.role == "tool"]
    assert [message.content for message in tool_messages] == ["echo: first", "echo: second"]
    assert [message.metadata["tool_call_id"] for message in tool_messages] == ["toolu_1", "toolu_2"]
    assert [message.metadata["tool_call_id"] for message in provider.second_turn_messages if message.role == "tool"] == [
        "toolu_1",
        "toolu_2",
    ]


# --- Phase 3D: bounded reactive 413 recovery ---------------------------------


class _RoundsThen413Provider:
    """Builds two tool rounds, then 413s ``fail_times`` times before succeeding.

    Calls 1 and 2 emit tool calls (so the agent accumulates two API rounds in its live
    history). The next ``fail_times`` calls raise ``LLMContextTooLongError`` — the first
    is the outer-loop 413 that triggers ``reactive_compact``; any further ones drive
    ``truncate_head_for_ptl_retry`` inside the bounded retry loop. After that it returns a
    final answer. The error text carries a parseable token gap.
    """

    def __init__(self, fail_times: int = 2) -> None:
        self.calls = 0
        self.fail_times = fail_times
        self.last_messages: list[Message] = []

    async def complete(self, messages, tools, config, stream=None, should_cancel=None) -> LLMResult:
        self.calls += 1
        self.last_messages = list(messages)
        if self.calls <= 2:
            return LLMResult(
                f"round {self.calls}",
                tool_calls=[ToolCall("echo", {"text": f"r{self.calls}"}, id=f"toolu_{self.calls}")],
                stop_reason="tool_use",
            )
        if self.calls <= 2 + self.fail_times:
            raise LLMContextTooLongError("prompt is too long: 300000 tokens > 200000")
        return LLMResult("done", stop_reason="end")


async def test_reactive_compact_retries_after_context_error(tmp_path: Path) -> None:
    logger = JSONLRunLogger(tmp_path)
    provider = FakeProvider(fail_once_context=True)
    # Disable memory so we count only the compaction retry: with memory on, the
    # post-run extraction makes a third provider call and the count is ambiguous.
    config = ReActConfig(run_dir=str(tmp_path), memory=MemoryConfig(enabled=False))
    agent = ReActAgent(provider, config, logger=logger)
    result = await agent.run("hello")
    assert result.answer
    assert provider.calls == 2


async def test_reactive_recovery_head_truncates_across_multiple_413s(tmp_path: Path) -> None:
    # Two rounds built, then 413 twice: the first 413 → reactive_compact, the second 413
    # → truncate_head_for_ptl_retry, then success. The run still completes via the bounded
    # loop, and the preserved system/userContext front survives.
    provider = _RoundsThen413Provider(fail_times=2)
    logger = JSONLRunLogger(tmp_path)
    config = ReActConfig(run_dir=str(tmp_path), permission="auto", memory=MemoryConfig(enabled=False))
    agent = ReActAgent(provider, config, logger=logger)

    result = await agent.run("kick off the work")

    assert result.answer == "done"
    # Calls: 2 rounds + 2 failed 413 retries + 1 success = 5.
    assert provider.calls == 5
    # The preserved front (base system block + pinned userContext) survived truncation.
    final = provider.last_messages
    assert final[0].role == "system"
    assert any(m.metadata.get("pinned") for m in final)


class _Always413Provider:
    """Raises ``LLMContextTooLongError`` on every call (unrecoverable overflow)."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools, config, stream=None, should_cancel=None) -> LLMResult:
        self.calls += 1
        raise LLMContextTooLongError("prompt is too long: 300000 tokens > 200000")


async def test_reactive_recovery_propagates_when_history_cannot_shrink(tmp_path: Path) -> None:
    # The very first model call 413s with a history that has < 2 droppable rounds AND no
    # message large enough to shrink, so truncate_head returns None, shrink returns None,
    # and the error propagates — bounded, no loop.
    provider = _Always413Provider()
    logger = JSONLRunLogger(tmp_path)
    config = ReActConfig(run_dir=str(tmp_path), permission="auto", memory=MemoryConfig(enabled=False))
    agent = ReActAgent(provider, config, logger=logger)

    with pytest.raises(LLMContextTooLongError):
        await agent.run("hello")
    # Bounded: it gave up well before MAX_PTL_RETRIES could spin (no droppable rounds).
    assert provider.calls <= 2


class _GiantRoundUntilFitsProvider:
    """Seeds one giant assistant round, then 413s while any non-system message is still
    large, then returns a final answer. Exercises the single-oversized-round shrink path:
    whole-round head-truncation can't help (one big round), so the loop falls back to
    ``shrink_oversize_messages`` until the giant message is small enough to fit. The 413
    decision keys off the largest *conversation* message (not the preserved system block),
    so the test is independent of the system-prompt size.
    """

    def __init__(self, big_threshold: int = 1000) -> None:
        self.calls = 0
        self.big_threshold = big_threshold
        self.last_messages: list[Message] = []

    async def complete(self, messages, tools, config, stream=None, should_cancel=None) -> LLMResult:
        self.calls += 1
        self.last_messages = list(messages)
        if self.calls == 1:
            return LLMResult(
                "G" * 8000,
                tool_calls=[ToolCall("echo", {"text": "x"}, id="toolu_1")],
                stop_reason="tool_use",
            )
        # Only react to non-preserved conversation messages — the preserved system block
        # and the pinned userContext (which may carry a large CLAUDE.md) can't be shrunk.
        biggest = max(
            (len(m.content) for m in messages if m.role != "system" and not m.metadata.get("pinned")),
            default=0,
        )
        if biggest > self.big_threshold:
            total = sum(len(m.content) for m in messages)
            # Gap large enough to force shrinking the giant message.
            raise LLMContextTooLongError(f"prompt is too long: {total} tokens > {total - 2000}")
        return LLMResult("done", stop_reason="end")


async def test_reactive_recovery_shrinks_single_oversized_round(tmp_path: Path) -> None:
    provider = _GiantRoundUntilFitsProvider(big_threshold=1000)
    logger = JSONLRunLogger(tmp_path)
    config = ReActConfig(run_dir=str(tmp_path), permission="auto", memory=MemoryConfig(enabled=False))
    agent = ReActAgent(provider, config, logger=logger)

    result = await agent.run("kick off the work")

    # The run converges via the shrink fallback rather than spinning or propagating.
    assert result.answer == "done"
    # The giant assistant message was head/tail-truncated (carries the shrink marker).
    assert any(m.metadata.get("compressed") == "ptl_shrink" for m in provider.last_messages)


class _RecordingUI(AgentUI):
    """Captures the token-usage and recap events the loop emits."""

    def __init__(self) -> None:
        self.usages: list[dict] = []
        self.recap: dict | None = None

    def on_token_usage(self, usage: dict) -> None:
        self.usages.append(usage)

    def on_run_completed(self, stats: dict) -> None:
        self.recap = stats


async def test_run_emits_token_usage_and_recap_tokens(tmp_path: Path) -> None:
    logger = JSONLRunLogger(tmp_path)
    ui = _RecordingUI()
    agent = ReActAgent(FakeProvider(), ReActConfig(run_dir=str(tmp_path)), logger=logger, ui=ui)

    await agent.run("hello")

    # FakeProvider reports usage every turn → at least one live token line was emitted,
    # carrying the running context size, the model window, and cumulative in/out.
    assert ui.usages, "expected at least one on_token_usage emission"
    last = ui.usages[-1]
    assert last["window"] > 0
    assert last["context_tokens"] >= 0
    assert last["input_tokens"] > 0
    # The gauge splits the prompt into conversation vs. the fixed run-start baseline
    # (system prompt + pinned CLAUDE.md/userContext). A short task with no history is
    # dominated by the baseline, so the conversation slice stays a small fraction of
    # the total and never exceeds it.
    assert "conversation_tokens" in last
    assert 0 <= last["conversation_tokens"] <= last["context_tokens"]
    assert last["conversation_tokens"] < last["context_tokens"], (
        "baseline (system + pinned context) should dominate a fresh short task"
    )
    # The recap carries the run's token totals (FakeProvider's fixed +8 output per turn).
    assert ui.recap is not None
    assert ui.recap["output_tokens"] > 0
    assert "input_tokens" in ui.recap and "context_tokens" in ui.recap


# --- _estimate_tokens: anchored real-usage + rough delta ---------------------


def _estimate_agent(tmp_path: Path) -> ReActAgent:
    logger = JSONLRunLogger(tmp_path)
    return ReActAgent(FakeProvider(), ReActConfig(run_dir=str(tmp_path)), logger=logger)


def test_estimate_tokens_no_anchor_is_rough_estimate_of_all(tmp_path: Path) -> None:
    agent = _estimate_agent(tmp_path)
    msgs = [Message("user", "x" * 40), Message("assistant", "y" * 8)]
    # No message carries usage → full rough estimate (per-message len//4: 10 + 2).
    assert agent._estimate_tokens(msgs) == 12


def test_estimate_tokens_anchors_on_usage_plus_delta_since(tmp_path: Path) -> None:
    agent = _estimate_agent(tmp_path)
    msgs = [
        Message("user", "x" * 4000),  # before the anchor — its real cost is folded into usage
        Message("assistant", "done", metadata={"usage_tokens": 5000}),
        Message("tool", "z" * 40),  # added since the anchor → rough estimate (40//4 = 10)
        Message("user", "w" * 8),  # 8//4 = 2
    ]
    # anchor (5000) + rough(messages after it) = 5000 + 10 + 2. The pre-anchor 4000-char
    # message is NOT char-counted — its real footprint is already inside the 5000 anchor.
    assert agent._estimate_tokens(msgs) == 5012


def test_estimate_tokens_uses_the_most_recent_anchor(tmp_path: Path) -> None:
    agent = _estimate_agent(tmp_path)
    msgs = [
        Message("assistant", "a", metadata={"usage_tokens": 1000}),
        Message("tool", "x" * 40),
        Message("assistant", "b", metadata={"usage_tokens": 7000}),  # most recent anchor
        Message("tool", "y" * 8),  # 8//4 = 2
    ]
    assert agent._estimate_tokens(msgs) == 7002


def test_estimate_tokens_falls_back_when_anchor_folded_away(tmp_path: Path) -> None:
    agent = _estimate_agent(tmp_path)
    # After a compaction fold the anchor-bearing assistant turns are gone; the summary is
    # a plain USER message with no usage_tokens → fall back to a full rough estimate.
    msgs = [
        Message("user", "summary " + "s" * 33, metadata={"post_compact": True}),  # 40//4 = 10
        Message("user", "y" * 8),  # 8//4 = 2
    ]
    assert agent._estimate_tokens(msgs) == 12
