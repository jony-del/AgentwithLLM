"""OpenAICompatProvider (D5): chat-completions protocol over httpx.MockTransport.

The acceptance criterion behind these tests is architectural: the provider fits
behind ``LLMProvider.complete`` with zero changes to ``providers/base.py`` or the
core loop. Both response paths (streaming AND non-streaming) are covered per the
provider-change rule in CLAUDE.md.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from agent_core.models import LLMContextTooLongError, LLMTransientError, Message
from agent_core.providers.base import ProviderConfig
from agent_core.providers.openai_compat import OpenAICompatProvider


def _provider(handler, **kwargs) -> OpenAICompatProvider:
    provider = OpenAICompatProvider(api_key="test-key", base_url="https://compat.example", **kwargs)
    provider._transport = httpx.MockTransport(handler)
    return provider


class _Recorder:
    def __init__(self) -> None:
        self.text = ""
        self.tool_args = ""

    def on_text_delta(self, text: str) -> None:
        self.text += text

    def on_thinking_delta(self, text: str) -> None:  # pragma: no cover
        pass

    def on_tool_args_delta(self, tool_name: str, partial_json: str) -> None:
        self.tool_args += partial_json


_CONFIG = ProviderConfig(model="test-model", stream=False)


# --- non-streaming -----------------------------------------------------------


async def test_non_streaming_parses_content_and_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )

    result = await _provider(handler).complete([Message("user", "hi")], [], _CONFIG)
    assert result.content == "Hello"
    assert result.stop_reason == "stop"
    assert result.usage is not None and result.usage.input_tokens == 7
    assert result.thinking_blocks == []  # provider-owned opaque data: this provider owns none


async def test_chat_completions_request_omits_responses_only_fields() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    await _provider(handler).complete(
        [Message("user", "hi")], [], ProviderConfig(model="gpt-5.6", effort="high", thinking_budget=2048)
    )

    assert "reasoning" not in seen
    assert "include" not in seen
    assert "thinking" not in seen
    assert "provider_state" not in seen
    assert seen["max_tokens"] == 1024
    assert "max_output_tokens" not in seen


async def test_tool_schema_and_history_round_trip() -> None:
    """The neutral tool schema and tool-call history project into function shapes."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_9",
                                    "type": "function",
                                    "function": {"name": "echo", "arguments": '{"text": "hi"}'},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )

    messages = [
        Message("system", "be brief"),
        Message("user", "use the tool"),
        Message(
            "assistant",
            "Calling echo",
            metadata={"tool_calls": [{"id": "call_1", "name": "echo", "arguments": {"text": "x"}}]},
        ),
        Message("tool", "echo: x", name="echo", metadata={"tool_call_id": "call_1", "ok": True}),
    ]
    tools = [{"name": "echo", "description": "echo text", "input_schema": {"type": "object"}}]
    result = await _provider(handler).complete(messages, tools, _CONFIG)

    # Request: tools in function shape, history with role=tool + assistant tool_calls.
    assert seen["tools"] == [
        {
            "type": "function",
            "function": {"name": "echo", "description": "echo text", "parameters": {"type": "object"}},
        }
    ]
    roles = [m["role"] for m in seen["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert seen["messages"][2]["tool_calls"][0]["function"]["name"] == "echo"
    assert seen["messages"][3]["tool_call_id"] == "call_1"
    # Response: tool calls come back as ToolCall contract objects.
    (call,) = result.tool_calls
    assert call.name == "echo" and call.arguments == {"text": "hi"} and call.id == "call_9"


async def test_context_overflow_maps_to_contract_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"error": {"code": "context_length_exceeded", "message": "too long"}}
        )

    with pytest.raises(LLMContextTooLongError):
        await _provider(handler).complete([Message("user", "hi")], [], _CONFIG)


async def test_unsupported_chat_parameter_error_is_actionable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "Unsupported parameter: max_output_tokens",
                    "code": "unsupported_parameter",
                    "param": "max_output_tokens",
                    "type": "invalid_request_error",
                }
            },
        )

    with pytest.raises(RuntimeError) as excinfo:
        await _provider(handler).complete([Message("user", "hi")], [], _CONFIG)

    message = str(excinfo.value)
    assert "OpenAI-compatible Chat Completions" in message
    assert "test-model" in message
    assert "max_output_tokens" in message
    assert "does not infer" in message


async def test_retries_on_429_then_succeeds(monkeypatch) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
        )

    sleeps: list[float] = []

    async def no_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    result = await _provider(handler, on_retry=None).complete([Message("user", "hi")], [], _CONFIG)
    assert result.content == "ok"
    assert calls["n"] == 2 and len(sleeps) == 1


async def test_exhausted_retries_raise_transient() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "down"}})

    provider = _provider(handler, max_retries=0, on_retry=None)
    with pytest.raises(LLMTransientError):
        await provider.complete([Message("user", "hi")], [], _CONFIG)


async def test_missing_api_key_is_actionable() -> None:
    provider = OpenAICompatProvider(api_key="placeholder")
    provider.api_key = None
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        await provider.complete([Message("user", "hi")], [], _CONFIG)


async def test_missing_model_is_actionable() -> None:
    provider = OpenAICompatProvider(api_key="k")
    with pytest.raises(RuntimeError, match="model"):
        await provider.complete([Message("user", "hi")], [], ProviderConfig(model=""))


async def test_compat_provider_prefers_new_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "compat-key")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://compat-env.example")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://legacy.example")

    provider = OpenAICompatProvider()

    assert provider.api_key == "compat-key"
    assert provider.base_url == "https://compat-env.example"


async def test_compat_provider_legacy_env_fallback_warns_once(monkeypatch, capsys) -> None:
    from agent_core.providers.openai_compat import _WARNED_DEPRECATED_ENV

    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://legacy.example")
    _WARNED_DEPRECATED_ENV.clear()

    OpenAICompatProvider()

    assert "deprecated" in capsys.readouterr().err


# --- streaming ----------------------------------------------------------------


def _sse(*chunks: dict | str) -> bytes:
    lines = []
    for chunk in chunks:
        data = chunk if isinstance(chunk, str) else json.dumps(chunk)
        lines.append(f"data: {data}\n\n")
    return "".join(lines).encode("utf-8")


async def test_streaming_assembles_text_and_pushes_deltas() -> None:
    body = _sse(
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
        "[DONE]",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, content=body)

    recorder = _Recorder()
    result = await _provider(handler).complete(
        [Message("user", "hi")], [], ProviderConfig(model="test-model", stream=True), stream=recorder
    )
    assert result.content == "Hello" == recorder.text
    assert result.stop_reason == "stop"
    assert result.usage is not None and result.usage.input_tokens == 5


async def test_streaming_assembles_tool_calls() -> None:
    body = _sse(
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_z", "function": {"name": "echo", "arguments": '{"te'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": 'xt": "hi"}'}}]}, "finish_reason": "tool_calls"}]},
        "[DONE]",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    recorder = _Recorder()
    result = await _provider(handler).complete(
        [Message("user", "go")], [], ProviderConfig(model="test-model", stream=True), stream=recorder
    )
    (call,) = result.tool_calls
    assert call.name == "echo" and call.arguments == {"text": "hi"} and call.id == "call_z"
    assert recorder.tool_args == '{"text": "hi"}'


async def test_provider_drives_the_react_loop_end_to_end(tmp_path) -> None:
    """D5 acceptance: the second protocol family runs the UNCHANGED core loop.

    Turn 1 returns a tool call, the loop executes it and feeds the observation
    back, turn 2 returns the final answer — all through ``LLMProvider.complete``
    with zero modifications to providers/base.py or react.py.
    """
    from agent_core.react import ReActAgent, ReActConfig

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [{
                        "message": {
                            "content": "Using the tool.",
                            "tool_calls": [{
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "echo", "arguments": '{"text": "ping"}'},
                            }],
                        },
                        "finish_reason": "tool_calls",
                    }],
                },
            )
        body = json.loads(request.content)
        # The tool observation came back in chat-completions shape.
        assert body["messages"][-1]["role"] == "tool"
        assert "ping" in body["messages"][-1]["content"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Echo done."}, "finish_reason": "stop"}]},
        )

    agent = ReActAgent(
        _provider(handler),
        ReActConfig(
            model="test-model", run_dir=str(tmp_path), session_dir="",
            git_context=False, project_instructions=False, stream=False,
        ),
    )
    result = await agent.run("please echo ping")
    assert result.answer == "Echo done."
    assert calls["n"] == 2


async def test_streaming_cancel_interrupts_promptly() -> None:
    body = _sse(
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
        "[DONE]",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    fired = {"deltas": 0}

    def should_cancel() -> bool:
        return fired["deltas"] >= 1

    recorder = _Recorder()
    original = recorder.on_text_delta

    def counting(text: str) -> None:
        original(text)
        fired["deltas"] += 1

    recorder.on_text_delta = counting  # type: ignore[method-assign]
    with pytest.raises(asyncio.CancelledError):
        await _provider(handler).complete(
            [Message("user", "hi")], [], ProviderConfig(model="test-model", stream=True),
            stream=recorder, should_cancel=should_cancel,
        )
    assert recorder.text == "Hello"  # interrupted before the second delta
