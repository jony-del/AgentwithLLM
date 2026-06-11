import json

import pytest

from agent_core.models import LLMTransientError, Message
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
    return provider._build_body([Message("user", "hi")], [], config)


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


class _FakeStreamResponse:
    """Stands in for an httpx streaming response: just an async line iterator."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def _sse_lines(*frames: dict) -> list[bytes]:
    """Render event frames as raw SSE byte-lines (data: line + blank separator)."""
    lines: list[bytes] = []
    for frame in frames:
        lines.append(f"event: {frame['type']}".encode("utf-8"))
        lines.append(("data: " + json.dumps(frame)).encode("utf-8"))
        lines.append(b"")
    return lines


async def test_consume_stream_assembles_result_and_pushes_deltas() -> None:
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

    result = await provider._consume_stream(_FakeStreamResponse(raw), sink)

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


async def test_stream_error_event_raises_transient() -> None:
    provider = ClaudeProvider(api_key="test-key")
    raw = _sse_lines({"type": "error", "error": {"type": "overloaded_error", "message": "overloaded"}})
    with pytest.raises(LLMTransientError):
        await provider._consume_stream(_FakeStreamResponse(raw), _RecordingStream())


def test_build_body_sets_stream_flag_only_when_streaming() -> None:
    provider = ClaudeProvider(api_key="test-key")
    streamed = provider._build_body([Message("user", "hi")], [], {}, streaming=True)
    plain = provider._build_body([Message("user", "hi")], [], {}, streaming=False)
    assert streamed["stream"] is True
    assert "stream" not in plain


# --- Retry policy (pure pieces) ----------------------------------------------


def test_retry_after_header_is_parsed_and_capped() -> None:
    provider = ClaudeProvider(api_key="test-key")
    assert provider._parse_retry_after("3") == 3.0
    assert provider._parse_retry_after(None) is None
    # Oversized values are capped by _retry_delay, not by parsing itself.
    assert provider._retry_delay(0, 999.0) == 60.0


def test_backoff_delay_grows_and_is_bounded() -> None:
    provider = ClaudeProvider(
        api_key="test-key", initial_backoff=0.5, max_backoff=8.0, backoff_multiplier=2.0, on_retry=None
    )
    # Full jitter: every delay lies within [0, cap] for its attempt.
    for attempt, cap in [(0, 0.5), (1, 1.0), (2, 2.0), (10, 8.0)]:
        for _ in range(50):
            assert 0.0 <= provider._retry_delay(attempt, None) <= cap


async def test_missing_api_key_raises() -> None:
    provider = ClaudeProvider(api_key="placeholder")
    provider.api_key = None
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        await provider.complete([], [], {})
