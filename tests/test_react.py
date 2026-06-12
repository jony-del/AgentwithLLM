from pathlib import Path

from agent_core.hooks import HookResult
from agent_core.memory import MemoryConfig
from agent_core.models import LLMResult, ToolCall
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
