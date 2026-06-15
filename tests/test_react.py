from pathlib import Path

import pytest

from agent_core.hooks import HookResult
from agent_core.memory import MemoryConfig
from agent_core.models import LLMContextTooLongError, LLMResult, Message, ToolCall
from agent_core.providers.fake import FakeProvider
from agent_core.react import ReActAgent, ReActConfig
from agent_core.storage import JSONLRunLogger


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
    # The very first model call 413s with a history that has < 2 droppable rounds, so
    # truncate_head_for_ptl_retry returns None and the error propagates — bounded, no loop.
    provider = _Always413Provider()
    logger = JSONLRunLogger(tmp_path)
    config = ReActConfig(run_dir=str(tmp_path), permission="auto", memory=MemoryConfig(enabled=False))
    agent = ReActAgent(provider, config, logger=logger)

    with pytest.raises(LLMContextTooLongError):
        await agent.run("hello")
    # Bounded: it gave up well before MAX_PTL_RETRIES could spin (no droppable rounds).
    assert provider.calls <= 2
