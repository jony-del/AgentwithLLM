from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from agent_core.models import LLMContextTooLongError, LLMTransientError, Message
from agent_core.providers.base import ProviderConfig
from agent_core.providers.openai_responses import OpenAIResponsesProvider


_CONFIG = ProviderConfig(model="gpt-test", stream=False)
_TOOL = {"name": "echo", "description": "echo text", "input_schema": {"type": "object"}}


def _provider(handler, **kwargs) -> OpenAIResponsesProvider:
    provider = OpenAIResponsesProvider(api_key="test-key", base_url="https://api.openai.test", **kwargs)
    provider._transport = httpx.MockTransport(handler)
    return provider


def _response(output, *, status="completed", usage=None, **extra):
    payload = {"id": "resp_1", "status": status, "output": output}
    if usage is not None:
        payload["usage"] = usage
    payload.update(extra)
    return payload


def _message(text: str):
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _call(call_id: str, name: str = "echo", arguments: str = '{"text":"hi"}'):
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}


class _Recorder:
    def __init__(self) -> None:
        self.text = ""
        self.tool_args = ""
        self.tool_names: list[str] = []

    def on_text_delta(self, text: str) -> None:
        self.text += text

    def on_thinking_delta(self, text: str) -> None:  # pragma: no cover
        pass

    def on_tool_args_delta(self, tool_name: str, partial_json: str) -> None:
        self.tool_names.append(tool_name)
        self.tool_args += partial_json


# --- non-streaming request/response -----------------------------------------


async def test_non_streaming_text_request_and_response_shape() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json=_response([_message("Hello")], usage={"input_tokens": 7, "output_tokens": 3}),
        )

    result = await _provider(handler).complete([Message("user", "hi")], [], _CONFIG)

    assert seen["path"] == "/v1/responses"
    body = seen["body"]
    assert body["model"] == "gpt-test"
    assert body["input"] == [{"role": "user", "content": "hi"}]
    assert body["max_output_tokens"] == _CONFIG.max_tokens
    assert "messages" not in body and "max_tokens" not in body
    assert result.content == "Hello"
    assert result.usage is not None and result.usage.input_tokens == 7 and result.usage.output_tokens == 3
    assert result.raw["id"] == "resp_1"


async def test_reasoning_capable_gpt_requests_encrypted_reasoning_and_safe_effort() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_response([_message("ok")]))

    await _provider(handler).complete(
        [Message("user", "hi")], [], ProviderConfig(model="gpt-5.6", effort="high")
    )
    assert seen["store"] is False
    assert seen["include"] == ["reasoning.encrypted_content"]
    assert seen["reasoning"] == {"effort": "high"}
    assert "temperature" not in seen


async def test_non_reasoning_gpt_omits_reasoning_only_fields() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_response([_message("ok")]))

    await _provider(handler).complete(
        [Message("user", "hi")], [], ProviderConfig(model="gpt-4.1-nano", effort="high")
    )
    assert seen["store"] is False
    assert "include" not in seen
    assert "reasoning" not in seen
    assert "temperature" not in seen


async def test_reasoning_gpt_omits_unsupported_effort_but_keeps_include() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_response([_message("ok")]))

    await _provider(handler).complete(
        [Message("user", "hi")], [], ProviderConfig(model="gpt-5.6", effort="xhigh")
    )
    assert seen["include"] == ["reasoning.encrypted_content"]
    assert "reasoning" not in seen


async def test_shared_responses_provider_picks_reasoning_shape_per_call() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_response([_message("ok")]))

    provider = _provider(handler)
    await provider.complete([Message("user", "hi")], [], ProviderConfig(model="gpt-4.1-nano", effort="high"))
    await provider.complete([Message("user", "hi")], [], ProviderConfig(model="gpt-5.6", effort="high"))

    assert "include" not in requests[0]
    assert "reasoning" not in requests[0]
    assert requests[1]["include"] == ["reasoning.encrypted_content"]
    assert requests[1]["reasoning"] == {"effort": "high"}


async def test_flat_tool_schema() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_response([_message("ok")]))

    await _provider(handler).complete([Message("user", "use tool")], [_TOOL], _CONFIG)
    assert seen["tools"] == [
        {"type": "function", "name": "echo", "description": "echo text", "parameters": {"type": "object"}}
    ]
    assert "function" not in seen["tools"][0]


async def test_function_call_maps_to_tool_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_response([_call("call_1", arguments='{"text":"hi"}')]))

    result = await _provider(handler).complete([Message("user", "go")], [_TOOL], _CONFIG)
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "echo"
    assert call.arguments == {"text": "hi"}
    assert result.stop_reason == "tool_calls"


async def test_invalid_function_arguments_are_tolerated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_response([_call("call_bad", arguments="not-json")]))

    result = await _provider(handler).complete([Message("user", "go")], [_TOOL], _CONFIG)
    assert result.tool_calls[0].arguments == {"_raw_arguments": "not-json"}


async def test_tool_result_replays_as_function_call_output_with_matching_call_id() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_response([_message("done")]))

    messages = [
        Message("user", "use echo"),
        Message(
            "assistant",
            "Calling echo",
            metadata={"tool_calls": [{"id": "call_1", "name": "echo", "arguments": {"text": "hi"}}]},
        ),
        Message("tool", "echo: hi", name="echo", metadata={"tool_call_id": "call_1", "ok": True}),
    ]
    await _provider(handler).complete(messages, [_TOOL], _CONFIG)

    assert seen["input"][1]["type"] == "message"
    assert seen["input"][2] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "echo",
        "arguments": '{"text": "hi"}',
    }
    assert seen["input"][3] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "echo: hi",
    }


async def test_multiple_parallel_function_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_response([
                _call("call_1", arguments='{"text":"a"}'),
                _call("call_2", arguments='{"text":"b"}'),
            ]),
        )

    result = await _provider(handler).complete([Message("user", "go")], [_TOOL], _CONFIG)
    assert [(c.id, c.arguments) for c in result.tool_calls] == [
        ("call_1", {"text": "a"}),
        ("call_2", {"text": "b"}),
    ]


async def test_reasoning_and_output_items_are_preserved_and_replayed() -> None:
    requests: list[dict] = []
    output = [
        {"type": "reasoning", "id": "rs_1", "encrypted_content": "cipher"},
        _message("thinking done"),
        _call("call_1", arguments='{"text":"hi"}'),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(200, json=_response(output))
        return httpx.Response(200, json=_response([_message("done")]))

    provider = _provider(handler)
    config = ProviderConfig(model="gpt-5.6", stream=False)
    first = await provider.complete([Message("user", "go")], [_TOOL], config)
    assert first.provider_state == {"output": output}
    messages = [
        Message("user", "go"),
        Message(
            "assistant",
            first.content,
            metadata={
                "provider_state": first.provider_state,
                "tool_calls": [{"id": "call_1", "name": "echo", "arguments": {"text": "hi"}}],
            },
        ),
        Message("tool", "echo: hi", name="echo", metadata={"tool_call_id": "call_1"}),
    ]
    await provider.complete(messages, [_TOOL], config)
    assert requests[1]["input"][1:5] == [*output, {"type": "function_call_output", "call_id": "call_1", "output": "echo: hi"}]


async def test_reasoning_output_is_not_replayed_to_non_reasoning_gpt() -> None:
    seen: dict = {}
    output = [
        {"type": "reasoning", "id": "rs_1", "encrypted_content": "cipher"},
        _message("using tool"),
        _call("call_1", arguments='{"text":"hi"}'),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_response([_message("done")]))

    messages = [
        Message("user", "go"),
        Message(
            "assistant",
            "using tool",
            metadata={
                "provider_state": {"output": output},
                "tool_calls": [{"id": "call_1", "name": "echo", "arguments": {"text": "hi"}}],
            },
        ),
        Message("tool", "echo: hi", name="echo", metadata={"tool_call_id": "call_1"}),
    ]
    await _provider(handler).complete(messages, [_TOOL], ProviderConfig(model="gpt-4.1-nano", effort="high"))

    replayed = seen["input"][1:]
    assert not any(item.get("type") == "reasoning" for item in replayed)
    assert replayed == [
        _message("using tool"),
        _call("call_1", arguments='{"text":"hi"}'),
        {"type": "function_call_output", "call_id": "call_1", "output": "echo: hi"},
    ]


async def test_usage_parsing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_response([_message("ok")], usage={"input_tokens": 12, "output_tokens": 4}))

    result = await _provider(handler).complete([Message("user", "hi")], [], _CONFIG)
    assert result.usage is not None
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 4


async def test_max_output_tokens_incomplete_maps_to_max_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_response(
                [_message("partial")],
                status="incomplete",
                incomplete_details={"reason": "max_output_tokens"},
            ),
        )

    result = await _provider(handler).complete([Message("user", "hi")], [], _CONFIG)
    assert result.stop_reason == "max_tokens"


# --- errors/retries -----------------------------------------------------------


async def test_context_overflow_error_mapping() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"code": "context_length_exceeded", "message": "too long"}})

    with pytest.raises(LLMContextTooLongError):
        await _provider(handler).complete([Message("user", "hi")], [], _CONFIG)


async def test_429_retry_then_success(monkeypatch) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json=_response([_message("ok")]))

    sleeps: list[float] = []

    async def no_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    result = await _provider(handler, on_retry=None).complete([Message("user", "hi")], [], _CONFIG)
    assert result.content == "ok"
    assert calls["n"] == 2 and len(sleeps) == 1


async def test_5xx_exhausted_retry_raises_transient() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "down"}})

    with pytest.raises(LLMTransientError):
        await _provider(handler, max_retries=0, on_retry=None).complete([Message("user", "hi")], [], _CONFIG)


async def test_non_retryable_4xx_raises_runtime_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    with pytest.raises(RuntimeError, match="401"):
        await _provider(handler).complete([Message("user", "hi")], [], _CONFIG)


async def test_missing_api_key_is_actionable() -> None:
    provider = OpenAIResponsesProvider(api_key="placeholder")
    provider.api_key = None
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        await provider.complete([Message("user", "hi")], [], _CONFIG)


async def test_missing_model_is_actionable() -> None:
    with pytest.raises(RuntimeError, match="model"):
        await OpenAIResponsesProvider(api_key="k").complete([Message("user", "hi")], [], ProviderConfig(model=""))


# --- streaming ----------------------------------------------------------------


def _sse(*events: dict) -> bytes:
    lines: list[str] = []
    for event in events:
        lines.append(f"event: {event['type']}\n")
        lines.append(f"data: {json.dumps(event)}\n\n")
    return "".join(lines).encode("utf-8")


async def test_streaming_text() -> None:
    final = _response([_message("Hello")], usage={"input_tokens": 3, "output_tokens": 2})
    body = _sse(
        {"type": "response.created", "response": {"id": "resp_1"}},
        {"type": "response.output_text.delta", "delta": "Hel"},
        {"type": "response.output_text.delta", "delta": "lo"},
        {"type": "response.completed", "response": final},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, content=body)

    recorder = _Recorder()
    result = await _provider(handler).complete(
        [Message("user", "hi")], [], ProviderConfig(model="gpt-test", stream=True), stream=recorder
    )
    assert recorder.text == "Hello"
    assert result.content == "Hello"
    assert result.usage is not None and result.usage.input_tokens == 3


async def test_streaming_function_arguments() -> None:
    final = _response([_call("call_1", arguments='{"text":"hi"}')])
    body = _sse(
        {"type": "response.output_item.added", "output_index": 0, "item": _call("call_1", arguments="")},
        {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"te'},
        {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": 'xt":"hi"}'},
        {"type": "response.output_item.done", "output_index": 0, "item": _call("call_1", arguments='{"text":"hi"}')},
        {"type": "response.completed", "response": final},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    recorder = _Recorder()
    result = await _provider(handler).complete(
        [Message("user", "go")], [_TOOL], ProviderConfig(model="gpt-test", stream=True), stream=recorder
    )
    assert recorder.tool_args == '{"text":"hi"}'
    assert recorder.tool_names == ["echo", "echo"]
    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].arguments == {"text": "hi"}


async def test_streaming_cancel_interrupts_promptly() -> None:
    body = _sse(
        {"type": "response.output_text.delta", "delta": "Hello"},
        {"type": "response.output_text.delta", "delta": " world"},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    recorder = _Recorder()
    original = recorder.on_text_delta
    fired = {"deltas": 0}

    def counting(text: str) -> None:
        original(text)
        fired["deltas"] += 1

    recorder.on_text_delta = counting  # type: ignore[method-assign]

    with pytest.raises(asyncio.CancelledError):
        await _provider(handler).complete(
            [Message("user", "hi")], [], ProviderConfig(model="gpt-test", stream=True),
            stream=recorder, should_cancel=lambda: fired["deltas"] >= 1,
        )
    assert recorder.text == "Hello"


async def test_response_completed_final_response_is_authoritative() -> None:
    body = _sse(
        {"type": "response.output_text.delta", "delta": "draft"},
        {"type": "response.completed", "response": _response([_message("final")])},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    result = await _provider(handler).complete(
        [Message("user", "hi")], [], ProviderConfig(model="gpt-test", stream=True), stream=_Recorder()
    )
    assert result.content == "final"


# --- ReAct/transcript integration --------------------------------------------


async def test_react_end_to_end_tool_call_then_final_answer(tmp_path: Path) -> None:
    from agent_core.react import ReActAgent, ReActConfig

    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(200, json=_response([_message("Using the tool."), _call("call_1")]))
        assert body["input"][-1] == {"type": "function_call_output", "call_id": "call_1", "output": "echo: hi"}
        return httpx.Response(200, json=_response([_message("Echo done.")]))

    agent = ReActAgent(
        _provider(handler),
        ReActConfig(
            provider="openai", model="gpt-test", run_dir=str(tmp_path), session_dir="",
            git_context=False, project_instructions=False, stream=False,
        ),
    )
    result = await agent.run("please echo hi")
    assert result.answer == "Echo done."
    assert len(requests) == 2


async def test_transcript_resume_and_fork_can_continue_responses_tool_call(tmp_path: Path) -> None:
    from agent_core.react import ReActAgent, ReActConfig
    from agent_core.transcript import build_chain, fork_chain, load_transcript

    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json=_response([_message("continued")]))

    provider = _provider(handler)
    cfg = ReActConfig(
        provider="openai", model="gpt-test", run_dir=str(tmp_path), session_dir=str(tmp_path),
        git_context=False, project_instructions=False, stream=False,
    )
    first = ReActAgent(provider, cfg)
    written: list[Message] = []
    await first._emit(written, Message("user", "go"))
    await first._emit(
        written,
        Message(
            "assistant",
            "Using the tool.",
            metadata={
                "provider_state": {"output": [_message("Using the tool."), _call("call_1")]},
                "tool_calls": [{"id": "call_1", "name": "echo", "arguments": {"text": "hi"}}],
            },
        ),
    )
    await first._emit(written, Message("tool", "echo: hi", name="echo", metadata={"tool_call_id": "call_1"}))

    loaded = load_transcript(first.transcript.path)
    history = build_chain(loaded)
    second = ReActAgent(provider, cfg, session_id=first.session_id)
    await second.run("continue", history=history)
    assert any(item.get("type") == "function_call" and item.get("call_id") == "call_1" for item in requests[-1]["input"])
    assert any(item.get("type") == "function_call_output" and item.get("call_id") == "call_1" for item in requests[-1]["input"])

    _new_id, cloned = fork_chain(loaded)
    requests.clear()
    third = ReActAgent(provider, cfg)
    await third.run("continue fork", history=cloned)
    assert any(item.get("type") == "function_call" and item.get("call_id") == "call_1" for item in requests[-1]["input"])
    assert any(item.get("type") == "function_call_output" and item.get("call_id") == "call_1" for item in requests[-1]["input"])
