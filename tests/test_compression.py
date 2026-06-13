from agent_core.compression import CompressionConfig, CompressionEvent, CompressionPipeline
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


def test_compact_reports_each_stage() -> None:
    pipeline = CompressionPipeline(CompressionConfig(max_message_chars=60, collapsed_keep_recent=4))
    messages = [Message("user", str(index) + "x" * 100) for index in range(10)]
    calls: list[tuple[int, int, str]] = []

    pipeline.compact(messages, on_stage=lambda done, total, event: calls.append((done, total, event.stage)))

    assert [(done, total) for done, total, _ in calls] == [(1, 3), (2, 3), (3, 3)]
    assert [stage for _, _, stage in calls] == ["snip", "microcompact", "context_collapse"]


def test_microcompact_preserves_pinned() -> None:
    pinned = Message("system", "PINNED " * 2000, metadata={"pinned": "claudemd"})
    pipeline = CompressionPipeline(CompressionConfig(max_message_chars=60, collapsed_keep_recent=4))
    messages = [pinned]
    messages.extend(Message("user", str(index) + "x" * 200) for index in range(10))

    compacted, _ = pipeline.compact(messages, aggressive=True)

    # The pinned block survives verbatim; ordinary long messages are still compressed.
    assert pinned in compacted
    assert any(message.metadata.get("compressed") for message in compacted)


def test_collapse_event_carries_detail() -> None:
    pipeline = CompressionPipeline(CompressionConfig(max_message_chars=1000, collapsed_keep_recent=4))
    messages = [Message("user", f"{index}: {'x' * 40}") for index in range(10)]
    events: list[CompressionEvent] = []

    pipeline.compact(messages, on_stage=lambda done, total, event: events.append(event))

    collapse = next(event for event in events if event.stage == "context_collapse")
    assert collapse.detail == "collapsed 6 msgs"  # 10 messages, keep_recent=4
