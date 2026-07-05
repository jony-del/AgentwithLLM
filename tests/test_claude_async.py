"""Transport (httpx) tests for ClaudeProvider.complete."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from agent_core.models import LLMContextTooLongError, LLMResult, LLMTransientError, Message
from agent_core.providers.base import ProviderConfig
from agent_core.providers.claude import ClaudeProvider


def _provider(handler, **kwargs) -> ClaudeProvider:
    provider = ClaudeProvider(api_key="test-key", **kwargs)
    provider._transport = httpx.MockTransport(handler)
    return provider


class _Recorder:
    def __init__(self) -> None:
        self.text = ""

    def on_text_delta(self, text: str) -> None:
        self.text += text

    def on_thinking_delta(self, text: str) -> None:  # pragma: no cover
        pass

    def on_tool_args_delta(self, tool_name: str, partial_json: str) -> None:  # pragma: no cover
        pass


# --- non-streaming -----------------------------------------------------------


async def test_complete_non_streaming_parses_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "test-key"
        body = json.loads(request.content)
        assert body["model"] == "claude-test"
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "Hello there"}],
                "stop_reason": "end_turn",
            },
        )

    provider = _provider(handler)
    result = await provider.complete(
        [Message("user", "hi")], [], ProviderConfig(model="claude-test", stream=False)
    )
    assert isinstance(result, LLMResult)
    assert result.content == "Hello there"
    assert result.stop_reason == "end_turn"


async def test_complete_context_too_long_maps_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "prompt is too long: too many tokens"}})

    provider = _provider(handler)
    with pytest.raises(LLMContextTooLongError):
        await provider.complete([Message("user", "hi")], [], ProviderConfig(stream=False))


# --- retry / backoff ---------------------------------------------------------


async def test_complete_retries_on_429(monkeypatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"})

    provider = _provider(handler, max_retries=2)
    result = await provider.complete([Message("user", "hi")], [], ProviderConfig(stream=False))
    assert result.content == "ok"
    assert calls["n"] == 2
    assert len(sleeps) == 1  # one backoff between the two attempts


async def test_complete_retries_transient_transport_error_then_succeeds(monkeypatch) -> None:
    async def fake_sleep(delay: float) -> None:
        pass

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("connection reset")
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"})

    provider = _provider(handler, max_retries=2)
    result = await provider.complete([Message("user", "hi")], [], ProviderConfig(stream=False))
    assert result.content == "ok"
    assert calls["n"] == 2


async def test_complete_honors_retry_after_header(monkeypatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "2"}, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"})

    provider = _provider(handler, max_retries=2)
    await provider.complete([Message("user", "hi")], [], ProviderConfig(stream=False))
    assert sleeps == [2.0]  # honored the server-supplied delay exactly


async def test_complete_raises_transient_after_retries_exhausted(monkeypatch) -> None:
    async def fake_sleep(delay: float) -> None:
        pass

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(529, json={"error": {"message": "overloaded"}})

    provider = _provider(handler, max_retries=1)
    with pytest.raises(LLMTransientError):
        await provider.complete([Message("user", "hi")], [], ProviderConfig(stream=False))


async def test_complete_non_retryable_status_raises_immediately() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    provider = _provider(handler, max_retries=3)
    with pytest.raises(RuntimeError) as exc_info:
        await provider.complete([Message("user", "hi")], [], ProviderConfig(stream=False))

    assert calls["n"] == 1  # not retried
    assert "401" in str(exc_info.value)
    assert not isinstance(exc_info.value, LLMTransientError)


# --- streaming ---------------------------------------------------------------


_SSE = (
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n'
    "\n"
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n'
    "\n"
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}\n'
    "\n"
    'data: {"type":"content_block_stop","index":0}\n'
    "\n"
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n'
    "\n"
)


async def test_complete_streaming_assembles_and_pushes_deltas() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, content=_SSE.encode("utf-8"))

    provider = _provider(handler)
    recorder = _Recorder()
    result = await provider.complete([Message("user", "hi")], [], ProviderConfig(stream=True), stream=recorder)
    assert result.content == "Hello world"
    assert result.stop_reason == "end_turn"
    assert recorder.text == "Hello world"  # deltas were streamed live


async def test_streaming_cancel_aborts_mid_stream() -> None:
    """A cancel that fires while streaming raises CancelledError and stops early.

    The probe returns True only after the first text delta, so we prove the stream
    is interrupted partway (Esc mid-response) rather than running to completion.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_SSE.encode("utf-8"))

    provider = _provider(handler)
    recorder = _Recorder()
    fire_after = {"deltas": 0}

    def should_cancel() -> bool:
        # Only trip once at least one delta has been delivered to the UI.
        return fire_after["deltas"] >= 1

    original = recorder.on_text_delta

    def counting_delta(text: str) -> None:
        original(text)
        fire_after["deltas"] += 1

    recorder.on_text_delta = counting_delta  # type: ignore[method-assign]

    with pytest.raises(asyncio.CancelledError):
        await provider.complete(
            [Message("user", "hi")], [], ProviderConfig(stream=True), stream=recorder, should_cancel=should_cancel
        )
    assert recorder.text == "Hello"  # interrupted before the " world" delta


async def test_complete_streaming_error_event_is_transient() -> None:
    sse = 'data: {"type":"error","error":{"type":"overloaded_error","message":"boom"}}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse.encode("utf-8"))

    provider = _provider(handler)
    with pytest.raises(LLMTransientError):
        await provider.complete([Message("user", "hi")], [], ProviderConfig(stream=True), stream=_Recorder())
