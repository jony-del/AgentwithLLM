import io
import json
import ssl
import urllib.error
from email.message import Message as EmailMessage

import pytest

from agent_core.models import LLMContextTooLongError, LLMResult, LLMTransientError, Message
from agent_core.providers.claude import ClaudeProvider


def test_claude_provider_formats_tool_use_and_result_blocks() -> None:
    provider = ClaudeProvider(api_key="test-key")
    messages = [
        Message("system", "system prompt"),
        Message("user", "use the tool"),
        Message(
            "assistant",
            "Calling echo",
            metadata={
                "tool_calls": [
                    {
                        "id": "toolu_1",
                        "name": "echo",
                        "arguments": {"text": "hello"},
                    }
                ]
            },
        ),
        Message("tool", "echo: hello", name="echo", metadata={"tool_call_id": "toolu_1", "ok": True}),
    ]

    system, formatted = provider._format_messages(messages)

    assert system == "system prompt"
    assert formatted == [
        {"role": "user", "content": "use the tool"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Calling echo"},
                {"type": "tool_use", "id": "toolu_1", "name": "echo", "input": {"text": "hello"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "echo: hello"}],
        },
    ]


def test_claude_provider_marks_failed_tool_result_as_error() -> None:
    provider = ClaudeProvider(api_key="test-key")
    messages = [
        Message(
            "assistant",
            "",
            metadata={"tool_calls": [{"id": "toolu_1", "name": "read_text_file", "arguments": {"path": "x"}}]},
        ),
        Message("tool", "Tool error", metadata={"tool_call_id": "toolu_1", "ok": False}),
    ]

    _, formatted = provider._format_messages(messages)

    tool_result = formatted[1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["is_error"] is True


def test_claude_provider_converts_orphan_tool_messages_to_user_text() -> None:
    provider = ClaudeProvider(api_key="test-key")
    messages = [Message("tool", "orphan observation", metadata={"tool_call_id": "toolu_missing"})]

    _, formatted = provider._format_messages(messages)

    assert formatted == [{"role": "user", "content": [{"type": "text", "text": "orphan observation"}]}]


# --- Extended thinking -------------------------------------------------------


def _request_body(provider: ClaudeProvider, config: dict) -> dict:
    request = provider._build_request([Message("user", "hi")], [], config)
    return json.loads(request.data.decode("utf-8"))


def test_thinking_budget_enables_thinking_and_adjusts_limits() -> None:
    provider = ClaudeProvider(api_key="test-key")
    body = _request_body(provider, {"max_tokens": 512, "temperature": 0.2, "thinking_budget": 2048})

    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert body["temperature"] == 1  # forced on while thinking is enabled
    assert body["max_tokens"] > 2048  # API requires max_tokens > budget_tokens


def test_no_thinking_budget_leaves_request_unchanged() -> None:
    provider = ClaudeProvider(api_key="test-key")
    body = _request_body(provider, {"max_tokens": 512, "temperature": 0.2, "thinking_budget": None})

    assert "thinking" not in body
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 512


def test_parse_response_collects_thinking_blocks() -> None:
    provider = ClaudeProvider(api_key="test-key")
    payload = {
        "content": [
            {"type": "thinking", "thinking": "step one", "signature": "sig-abc"},
            {"type": "text", "text": "the answer"},
        ],
        "stop_reason": "end_turn",
    }

    result = provider._parse_response(payload)

    assert result.content == "the answer"
    assert result.thinking == "step one"
    assert result.thinking_blocks == [{"type": "thinking", "thinking": "step one", "signature": "sig-abc"}]


def test_assistant_content_replays_thinking_blocks_first() -> None:
    provider = ClaudeProvider(api_key="test-key")
    thinking_block = {"type": "thinking", "thinking": "because", "signature": "sig-1"}
    message = Message(
        "assistant",
        "Calling echo",
        metadata={
            "thinking_blocks": [thinking_block],
            "tool_calls": [{"id": "toolu_1", "name": "echo", "arguments": {"text": "hi"}}],
        },
    )

    content = provider._format_assistant_content(message)

    assert isinstance(content, list)
    assert content[0] == thinking_block  # replayed first, ahead of text/tool_use
    assert content[1]["type"] == "text"
    assert content[2]["type"] == "tool_use"


# --- Streaming (SSE) ---------------------------------------------------------


class _RecordingStream:
    def __init__(self) -> None:
        self.text: list[str] = []
        self.thinking: list[str] = []
        self.tool_args: list[tuple[str, str]] = []

    def on_text_delta(self, text: str) -> None:
        self.text.append(text)

    def on_thinking_delta(self, text: str) -> None:
        self.thinking.append(text)

    def on_tool_args_delta(self, tool_name: str, partial_json: str) -> None:
        self.tool_args.append((tool_name, partial_json))


def _sse_lines(*frames: dict) -> list[bytes]:
    """Render event frames as raw SSE byte-lines (data: line + blank separator)."""
    lines: list[bytes] = []
    for frame in frames:
        lines.append(f"event: {frame['type']}".encode("utf-8"))
        lines.append(("data: " + json.dumps(frame)).encode("utf-8"))
        lines.append(b"")
    return lines


def test_consume_stream_assembles_result_and_pushes_deltas() -> None:
    provider = ClaudeProvider(api_key="test-key")
    sink = _RecordingStream()
    raw = _sse_lines(
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Let me think"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig123"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hello"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": " world"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_9", "name": "echo"}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "{\"text\":"}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": " \"hi\"}"}},
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        {"type": "message_stop"},
    )

    result = provider._consume_stream(raw, sink)

    assert result.content == "Hello world"
    assert result.thinking == "Let me think"
    assert result.thinking_blocks == [{"type": "thinking", "thinking": "Let me think", "signature": "sig123"}]
    assert result.stop_reason == "tool_use"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "echo"
    assert result.tool_calls[0].id == "toolu_9"
    assert result.tool_calls[0].arguments == {"text": "hi"}
    # Deltas were pushed live, in order.
    assert sink.thinking == ["Let me think"]
    assert sink.text == ["Hello", " world"]
    assert sink.tool_args == [("echo", "{\"text\":"), ("echo", " \"hi\"}")]


def test_stream_error_event_raises_transient() -> None:
    provider = ClaudeProvider(api_key="test-key")
    raw = _sse_lines({"type": "error", "error": {"type": "overloaded_error", "message": "overloaded"}})
    with pytest.raises(LLMTransientError):
        provider._consume_stream(raw, _RecordingStream())


def test_build_request_sets_stream_flag_only_when_streaming() -> None:
    provider = ClaudeProvider(api_key="test-key")
    streamed = json.loads(provider._build_request([Message("user", "hi")], [], {}, streaming=True).data.decode("utf-8"))
    plain = json.loads(provider._build_request([Message("user", "hi")], [], {}, streaming=False).data.decode("utf-8"))
    assert streamed["stream"] is True
    assert "stream" not in plain


# --- Retry / resilience ------------------------------------------------------


def _retry_provider(**kwargs) -> tuple[ClaudeProvider, list[float]]:
    """A provider whose sleeps are recorded instead of real, and that never prints."""
    slept: list[float] = []
    provider = ClaudeProvider(api_key="test-key", sleep=slept.append, on_retry=None, **kwargs)
    return provider, slept


def _http_error(code: int, body: str = "{}", retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = EmailMessage()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(
        url="https://api.anthropic.com/v1/messages",
        code=code,
        msg="error",
        hdrs=headers,
        fp=io.BytesIO(body.encode("utf-8")),
    )


# A minimal valid Anthropic payload the parser can turn into an LLMResult.
_OK_PAYLOAD = {"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn"}


def test_succeeds_without_retry_when_first_attempt_works() -> None:
    provider, slept = _retry_provider()
    calls = {"n": 0}

    def send_once(request, timeout):
        calls["n"] += 1
        return _OK_PAYLOAD

    provider._send_once = send_once  # type: ignore[method-assign]
    result = provider.complete([], [], {})

    assert isinstance(result, LLMResult)
    assert result.content == "hi"
    assert calls["n"] == 1
    assert slept == []


def test_retries_transient_network_error_then_succeeds() -> None:
    provider, slept = _retry_provider(max_retries=2)
    calls = {"n": 0}

    def send_once(request, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError(ssl.SSLEOFError("EOF in violation of protocol"))
        return _OK_PAYLOAD

    provider._send_once = send_once  # type: ignore[method-assign]
    result = provider.complete([], [], {})

    assert result.content == "hi"
    assert calls["n"] == 3  # 1 initial + 2 retries
    assert len(slept) == 2  # one sleep per retry


def test_exhausted_retries_on_network_error_raise_transient() -> None:
    provider, slept = _retry_provider(max_retries=2)

    def send_once(request, timeout):
        raise urllib.error.URLError(ssl.SSLEOFError("boom"))

    provider._send_once = send_once  # type: ignore[method-assign]
    with pytest.raises(LLMTransientError) as exc_info:
        provider.complete([], [], {})

    assert "3 attempt" in str(exc_info.value)  # 1 + 2 retries
    assert len(slept) == 2


def test_retryable_status_is_retried_then_raises_transient() -> None:
    provider, slept = _retry_provider(max_retries=1)

    def send_once(request, timeout):
        raise _http_error(529, body="overloaded")

    provider._send_once = send_once  # type: ignore[method-assign]
    with pytest.raises(LLMTransientError):
        provider.complete([], [], {})

    assert len(slept) == 1


def test_non_retryable_status_raises_immediately() -> None:
    provider, slept = _retry_provider(max_retries=3)
    calls = {"n": 0}

    def send_once(request, timeout):
        calls["n"] += 1
        raise _http_error(401, body="invalid api key")

    provider._send_once = send_once  # type: ignore[method-assign]
    with pytest.raises(RuntimeError) as exc_info:
        provider.complete([], [], {})

    assert calls["n"] == 1  # not retried
    assert slept == []
    assert "401" in str(exc_info.value)
    assert not isinstance(exc_info.value, LLMTransientError)


def test_context_overflow_maps_to_context_error_without_retry() -> None:
    provider, slept = _retry_provider(max_retries=3)

    def send_once(request, timeout):
        raise _http_error(400, body="prompt is too long: too many tokens")

    provider._send_once = send_once  # type: ignore[method-assign]
    with pytest.raises(LLMContextTooLongError):
        provider.complete([], [], {})

    assert slept == []  # context errors are handled by compression, never retried


def test_retry_after_header_is_honored_and_capped() -> None:
    provider, _ = _retry_provider()
    assert provider._retry_after(_http_error(429, retry_after="3")) == 3.0
    # Oversized values are capped by _retry_delay, not by _retry_after itself.
    assert provider._retry_delay(0, 999.0) == 60.0
    assert provider._retry_after(_http_error(429)) is None


def test_network_error_classification() -> None:
    provider, _ = _retry_provider()
    assert provider._is_retryable_network_error(TimeoutError())
    assert provider._is_retryable_network_error(ssl.SSLError())
    assert provider._is_retryable_network_error(ConnectionResetError())
    assert provider._is_retryable_network_error(urllib.error.URLError(ssl.SSLEOFError()))
    # A non-transport reason (e.g. a value error) is not a network transient.
    assert not provider._is_retryable_network_error(urllib.error.URLError(ValueError("bad")))


def test_backoff_delay_grows_and_is_bounded() -> None:
    provider, _ = _retry_provider(initial_backoff=0.5, max_backoff=8.0, backoff_multiplier=2.0)
    # Full jitter: every delay lies within [0, cap] for its attempt.
    for attempt, cap in [(0, 0.5), (1, 1.0), (2, 2.0), (10, 8.0)]:
        for _ in range(50):
            assert 0.0 <= provider._retry_delay(attempt, None) <= cap


def test_missing_api_key_raises() -> None:
    provider = ClaudeProvider(api_key="placeholder")
    provider.api_key = None
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        provider.complete([], [], {})
