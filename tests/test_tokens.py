import json


from agent_core import tokens
from agent_core.models import LLMResult, Message, TokenUsage
from agent_core.providers.claude import ClaudeProvider
from agent_core.providers.fake import FakeProvider


# --- TokenUsage contract -----------------------------------------------------


def test_token_usage_context_tokens_sums_input_and_cache() -> None:
    usage = TokenUsage(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=20,
        cache_creation_input_tokens=5,
    )
    # context_tokens is the request footprint: input + both cache buckets (NOT output).
    assert usage.context_tokens == 125


def test_token_usage_total_tokens_adds_output_to_context() -> None:
    usage = TokenUsage(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=20,
        cache_creation_input_tokens=5,
    )
    # total_tokens = context (input + both caches) + output — the per-response footprint.
    assert usage.total_tokens == 125 + 50


def test_token_usage_defaults_are_zero() -> None:
    assert TokenUsage().context_tokens == 0
    assert TokenUsage().total_tokens == 0


# --- rough char-based estimate -----------------------------------------------


def test_rough_token_estimate_is_chars_over_four() -> None:
    assert tokens.rough_token_estimate("x" * 40) == 10
    assert tokens.rough_token_estimate("") == 0
    # Custom ratio (e.g. denser content) is honored; a 0 ratio never divides by zero.
    assert tokens.rough_token_estimate("x" * 40, bytes_per_token=2) == 20
    assert tokens.rough_token_estimate("xxxx", bytes_per_token=0) == 4


def test_rough_token_estimate_for_messages_sums_per_message() -> None:
    msgs = [Message("user", "x" * 40), Message("assistant", "y" * 8)]
    assert tokens.rough_token_estimate_for_messages(msgs) == 10 + 2
    assert tokens.rough_token_estimate_for_messages([]) == 0


def test_llm_result_usage_defaults_to_none() -> None:
    assert LLMResult(content="hi").usage is None


# --- context_window_for_model ------------------------------------------------


def test_context_window_default_for_plain_claude_model() -> None:
    assert tokens.context_window_for_model("claude-haiku-4-5-20251001") == 200_000


def test_context_window_native_1m_family() -> None:
    # The known 1M family (Opus 4.6/4.7/4.8, Sonnet 4.6, Fable/Mythos 5) reports 1M natively.
    assert tokens.context_window_for_model("claude-opus-4-8") == 1_000_000
    assert tokens.context_window_for_model("claude-sonnet-4-6") == 1_000_000
    assert tokens.context_window_for_model("claude-fable-5") == 1_000_000


def test_context_window_1m_tag_forces_1m_for_any_id() -> None:
    assert tokens.context_window_for_model("some-model[1m]") == 1_000_000


def test_context_window_default_for_200k_and_unknown_models() -> None:
    # Haiku 4.5 (genuine 200k) and unrecognised ids stay at the conservative default.
    assert tokens.context_window_for_model("claude-haiku-4-5") == 200_000
    assert tokens.context_window_for_model("some-other-model") == 200_000


def test_model_output_tokens_default_and_upper() -> None:
    assert tokens.model_output_tokens("claude-opus-4-8") == (64_000, 128_000)
    assert tokens.model_output_tokens("claude-haiku-4-5") == (32_000, 64_000)
    # Unknown ids fall back to the conservative flat default for both.
    assert tokens.model_output_tokens("some-other-model") == (8_192, 8_192)
    # max_output_tokens_for_model returns just the steady-state default.
    assert tokens.max_output_tokens_for_model("claude-opus-4-8") == 64_000


def test_is_supported_model_recognises_known_families() -> None:
    # Each known family (with realistic suffixes / tags) is accepted.
    for model in (
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-6",
        "claude-opus-4-8",
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-4-8[1m]",
    ):
        assert tokens.is_supported_model(model)


def test_is_supported_model_rejects_unknown_and_empty() -> None:
    assert not tokens.is_supported_model("gpt-4")
    assert not tokens.is_supported_model("some-other-model")
    assert not tokens.is_supported_model("")
    assert not tokens.is_supported_model(None)  # type: ignore[arg-type]


# --- effective_context_window ------------------------------------------------


def test_effective_context_window_reserves_output() -> None:
    # 200k window minus min(haiku max_output=32000, reserved=20000) = 200000 - 20000.
    assert tokens.effective_context_window("claude-haiku-4-5") == 200_000 - 20_000


def test_effective_context_window_respects_override() -> None:
    # Sonnet 4.6's native window (1M) is capped down to the override; reserve = min(32000, 20000).
    eff = tokens.effective_context_window("claude-sonnet-4-6", context_window_override=50_000)
    assert eff == 50_000 - 20_000


def test_effective_context_window_override_only_caps_down() -> None:
    # An override larger than the native window does not raise it.
    eff = tokens.effective_context_window("claude-haiku-4-5", context_window_override=999_999)
    assert eff == 200_000 - 20_000


# --- auto_compact_threshold --------------------------------------------------


def test_auto_compact_threshold_is_effective_minus_buffer() -> None:
    eff = tokens.effective_context_window("claude-haiku-4-5")
    assert tokens.auto_compact_threshold("claude-haiku-4-5") == eff - 13_000


def test_auto_compact_threshold_honors_window_override() -> None:
    eff = tokens.effective_context_window("claude-sonnet-4-6", context_window_override=40_000)
    assert (
        tokens.auto_compact_threshold("claude-sonnet-4-6", context_window_override=40_000)
        == eff - 13_000
    )


def test_auto_compact_threshold_pct_override_lowers_threshold() -> None:
    eff = tokens.effective_context_window("claude-haiku-4-5")
    base = eff - 13_000
    # 10% of the effective window is far below base for a 200k window → percent wins.
    result = tokens.auto_compact_threshold("claude-haiku-4-5", pct_override=10.0)
    assert result == min(eff // 10, base)
    assert result < base


def test_auto_compact_threshold_pct_override_cannot_exceed_base() -> None:
    # 100% of the window is above base, so the buffer-derived ceiling wins.
    base = tokens.effective_context_window("claude-haiku-4-5") - 13_000
    assert tokens.auto_compact_threshold("claude-haiku-4-5", pct_override=100.0) == base


def test_pct_override_resolver_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_AUTOCOMPACT_PCT_OVERRIDE", "25")
    assert tokens.resolve_pct_override() == 25.0
    # Explicit arg wins over env.
    assert tokens.resolve_pct_override(50.0) == 50.0


def test_pct_override_resolver_rejects_out_of_range(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_AUTOCOMPACT_PCT_OVERRIDE", "0")
    assert tokens.resolve_pct_override() is None
    monkeypatch.setenv("AGENT_AUTOCOMPACT_PCT_OVERRIDE", "150")
    assert tokens.resolve_pct_override() is None
    monkeypatch.setenv("AGENT_AUTOCOMPACT_PCT_OVERRIDE", "notanumber")
    assert tokens.resolve_pct_override() is None


# --- FakeProvider deterministic usage ----------------------------------------


async def test_fake_provider_populates_deterministic_usage() -> None:
    provider = FakeProvider()
    messages = [Message("user", "x" * 40), Message("user", "y" * 8)]
    result = await provider.complete(messages, [], {})
    assert result.usage is not None
    # char/4 of the input: (40 + 8) // 4 == 12.
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 8
    assert result.usage.context_tokens == 12


# --- ClaudeProvider usage parse: non-streaming -------------------------------


def test_claude_non_streaming_usage_parse() -> None:
    provider = ClaudeProvider(api_key="test-key")
    payload = {
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 1200,
            "output_tokens": 34,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 100,
        },
    }
    result = provider._parse_response(payload)
    assert result.usage is not None
    assert result.usage.input_tokens == 1200
    assert result.usage.output_tokens == 34
    assert result.usage.cache_read_input_tokens == 800
    assert result.usage.cache_creation_input_tokens == 100
    assert result.usage.context_tokens == 1200 + 800 + 100


def test_claude_non_streaming_usage_defaults_cache_to_zero() -> None:
    provider = ClaudeProvider(api_key="test-key")
    payload = {
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }
    result = provider._parse_response(payload)
    assert result.usage.cache_read_input_tokens == 0
    assert result.usage.cache_creation_input_tokens == 0
    assert result.usage.context_tokens == 10


# --- ClaudeProvider usage parse: streaming -----------------------------------


class _RecordingStream:
    def on_text_delta(self, text: str) -> None:
        pass

    def on_thinking_delta(self, text: str) -> None:
        pass

    def on_tool_args_delta(self, tool_name: str, partial_json: str) -> None:
        pass


class _FakeStreamResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def _sse_lines(*frames: dict) -> list[bytes]:
    lines: list[bytes] = []
    for frame in frames:
        lines.append(f"event: {frame['type']}".encode("utf-8"))
        lines.append(("data: " + json.dumps(frame)).encode("utf-8"))
        lines.append(b"")
    return lines


async def test_claude_streaming_usage_parse() -> None:
    provider = ClaudeProvider(api_key="test-key")
    raw = _sse_lines(
        {
            "type": "message_start",
            "message": {
                "usage": {
                    "input_tokens": 500,
                    "cache_read_input_tokens": 40,
                    "cache_creation_input_tokens": 10,
                    "output_tokens": 0,
                }
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 27}},
        {"type": "message_stop"},
    )

    result = await provider._consume_stream(_FakeStreamResponse(raw), _RecordingStream())

    assert result.usage is not None
    # input + cache come from message_start; output total comes from message_delta.
    assert result.usage.input_tokens == 500
    assert result.usage.cache_read_input_tokens == 40
    assert result.usage.cache_creation_input_tokens == 10
    assert result.usage.output_tokens == 27
    assert result.usage.context_tokens == 500 + 40 + 10
