from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from agent_core.models import LLMContextTooLongError, LLMResult, Message, TokenUsage, ToolCall
from agent_core.providers.base import LLMProvider, StreamHandler


class FakeProvider(LLMProvider):
    """Deterministic provider for local demos and tests."""

    def __init__(self, fail_once_context: bool = False, stream_delay: float = 0.0) -> None:
        self.fail_once_context = fail_once_context
        # Per-chunk sleep when streaming — 0 keeps tests instant; a small value
        # (e.g. 0.02) makes the `--provider fake` demo visibly "type" its output.
        self.stream_delay = stream_delay
        self.calls = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        config: dict[str, Any],
        stream: StreamHandler | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> LLMResult:
        self.calls += 1
        if self.fail_once_context and self.calls == 1:
            raise LLMContextTooLongError("simulated context-too-long")

        result = self._compute(messages)
        # Deterministic token accounting so offline/test compaction thresholds are
        # meaningful and stable: char/4 estimate of the sent input + a fixed output.
        result.usage = self._estimate_usage(messages)
        # Stream the answer text in whitespace chunks so the live UI can render it
        # token-by-token, mirroring a real SSE stream — without needing an API key.
        if stream is not None and result.content and not result.tool_calls:
            await self._stream_text(result.content, stream, should_cancel)
        return result

    @staticmethod
    def _estimate_usage(messages: list[Message]) -> TokenUsage:
        """Char/4 input estimate plus a small fixed output, deterministic by design."""
        input_tokens = sum(len(m.content) for m in messages) // 4
        return TokenUsage(input_tokens=input_tokens, output_tokens=8)

    def _compute(self, messages: list[Message]) -> LLMResult:
        memory_response = self._maybe_memory_response(messages)
        if memory_response is not None:
            return LLMResult(content=memory_response, stop_reason="end")

        last_message = messages[-1] if messages else Message("user", "")
        last = last_message.content
        if last_message.role == "tool":
            return LLMResult(content=f"Final answer based on observation: {last}", stop_reason="end")
        if "tool:" in last:
            _, rest = last.split("tool:", 1)
            name = rest.strip().split()[0]
            return LLMResult(
                content=f"Calling {name}",
                tool_calls=[ToolCall(name=name, arguments={"text": last})],
                stop_reason="tool_use",
            )
        return LLMResult(content=f"Final answer: {last}", stop_reason="end")

    async def _stream_text(
        self,
        text: str,
        stream: StreamHandler,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        chunks = text.split(" ")
        for index, chunk in enumerate(chunks):
            if should_cancel is not None and should_cancel():
                raise asyncio.CancelledError("fake stream cancelled by user")
            stream.on_text_delta(chunk if index == 0 else " " + chunk)
            if self.stream_delay:
                await asyncio.sleep(self.stream_delay)

    @staticmethod
    def _maybe_memory_response(messages: list[Message]) -> str | None:
        """Deterministic JSON for memory extraction / insight prompts.

        Lets the whole memory pipeline (extraction + dreaming) be exercised without
        an API key. Keyed off the marker the memory subsystem embeds in its system
        prompt — see ``agent_core.memory.extraction.MEMORY_EXTRACTION_MARKER``.
        """
        # Lazy import keeps the providers layer decoupled from the memory subsystem.
        from agent_core.memory.extraction import MEMORY_EXTRACTION_MARKER

        system = next((m.content for m in messages if m.role == "system"), "")
        if MEMORY_EXTRACTION_MARKER not in system:
            return None
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")

        if "dreaming" in system.lower():
            # Insight-synthesis prompt: emit one stable, generic insight.
            return json.dumps(
                [{"content": "The user's stated preferences form a consistent profile.",
                  "kind": "insight", "importance": 0.6, "tags": ["insight"]}]
            )

        # Extraction prompt: remember the first real user line of the transcript.
        fact = FakeProvider._first_transcript_user_line(user)
        if not fact:
            return json.dumps([])
        return json.dumps(
            [{"content": fact, "kind": "preference", "importance": 0.7, "tags": ["fake"]}]
        )

    @staticmethod
    def _first_transcript_user_line(transcript: str) -> str:
        for line in transcript.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("user:"):
                return stripped[len("user:"):].strip()
        return ""
