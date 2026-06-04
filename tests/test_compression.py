from agent_core.compression import CompressionConfig, CompressionPipeline
from agent_core.models import Message


def test_auto_compact_triggers_when_context_is_large() -> None:
    pipeline = CompressionPipeline(CompressionConfig(max_context_chars=100, auto_threshold_ratio=0.5))
    messages = [Message("user", "x" * 80), Message("tool", "y" * 100)]
    compacted, events = pipeline.maybe_auto_compact(messages)
    assert events
    assert sum(len(message.content) for message in compacted) < 180


def test_reactive_compact_is_more_aggressive() -> None:
    pipeline = CompressionPipeline(CompressionConfig(max_message_chars=60, collapsed_keep_recent=4))
    messages = [Message("user", str(index) + "x" * 100) for index in range(10)]
    compacted, events = pipeline.reactive_compact(messages)
    assert events
    assert len(compacted) < len(messages)


def test_context_collapse_preserves_system_prompt() -> None:
    pipeline = CompressionPipeline(CompressionConfig(max_message_chars=1000, collapsed_keep_recent=4))
    messages = [Message("system", "keep these instructions")]
    messages.extend(Message("user", f"{index}: {'x' * 40}") for index in range(10))

    compacted, events = pipeline.reactive_compact(messages)

    assert events
    assert compacted[0] == Message("system", "keep these instructions")
    assert any(message.metadata.get("compressed") == "context_collapse" for message in compacted)
