from __future__ import annotations

import json
import os
import random
import ssl
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from agent_core.models import (
    LLMContextTooLongError,
    LLMResult,
    LLMTransientError,
    Message,
    ToolCall,
)
from agent_core.providers.base import LLMProvider, StreamHandler

# HTTP statuses worth retrying: request timeout / lock conflict, rate limiting, and
# transient upstream failures. 529 is Anthropic's "overloaded" signal. Everything
# else (401, 403, 404, 422, and 400s) is a caller/permanent error and is not retried.
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

# Never sleep longer than this for a server-supplied Retry-After, so a hostile or
# buggy header can't wedge the agent for minutes.
_MAX_RETRY_AFTER = 60.0


def _default_retry_notice(message: str) -> None:
    # ASCII only on purpose: this may print before the CLI reconfigures stderr to
    # UTF-8, and we must not reintroduce the very UnicodeEncodeError we just fixed.
    print(message, file=sys.stderr)


class ClaudeProvider(LLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com",
        *,
        max_retries: int = 2,
        initial_backoff: float = 0.5,
        max_backoff: float = 8.0,
        backoff_multiplier: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        on_retry: Callable[[str], None] | None = _default_retry_notice,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.max_retries = max(0, max_retries)
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.backoff_multiplier = backoff_multiplier
        self._sleep = sleep
        self._on_retry = on_retry
        self._rng = random.Random()

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        config: dict[str, Any],
        stream: StreamHandler | None = None,
    ) -> LLMResult:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for ClaudeProvider")

        streaming = stream is not None and config.get("stream", True)
        request = self._build_request(messages, tools, config, streaming=streaming)
        timeout = config.get("timeout", 60)
        if streaming:
            response = self._open_stream(request, timeout)
            try:
                return self._consume_stream(response, stream)
            finally:
                response.close()
        payload = self._send_with_retry(request, timeout)
        return self._parse_response(payload)

    def _build_request(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        config: dict[str, Any],
        streaming: bool = False,
    ) -> urllib.request.Request:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for ClaudeProvider")
        system, anthropic_messages = self._format_messages(messages)
        body: dict[str, Any] = {
            "model": config.get("model", "claude-sonnet-4-6"),
            "max_tokens": config.get("max_tokens", 1024),
            "temperature": config.get("temperature", 0.2),
            "messages": anthropic_messages,
        }
        self._apply_thinking(body, config.get("thinking_budget"))
        if streaming:
            body["stream"] = True
        if system:
            body["system"] = system
        if tools:
            body["tools"] = tools

        return urllib.request.Request(
            f"{self.base_url}/v1/messages",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

    @staticmethod
    def _apply_thinking(body: dict[str, Any], thinking_budget: Any) -> None:
        """Enable extended thinking in-place when a positive budget is requested.

        The Anthropic API requires ``max_tokens > budget_tokens`` and only supports
        ``temperature: 1`` while thinking is on, so we enforce both here rather than
        trusting the caller's defaults.
        """
        if not isinstance(thinking_budget, int) or isinstance(thinking_budget, bool) or thinking_budget <= 0:
            return
        body["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        body["temperature"] = 1
        body["max_tokens"] = max(body.get("max_tokens", 0), thinking_budget + 1024)

    def _send_once(self, request: urllib.request.Request, timeout: float) -> dict[str, Any]:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _send_with_retry(self, request: urllib.request.Request, timeout: float) -> dict[str, Any]:
        """Send a non-streaming request, retrying only transient faults."""
        return self._request_with_retry(lambda: self._send_once(request, timeout))

    def _open_stream(self, request: urllib.request.Request, timeout: float):
        """Open a streaming connection, retrying only connection-setup faults.

        Once the 200 event-stream is open we hand the response to ``_consume_stream``;
        a break *mid-stream* (after tokens were shown) is not retried — it surfaces
        as ``LLMTransientError`` so we never reprint half-streamed output.
        """
        return self._request_with_retry(lambda: urllib.request.urlopen(request, timeout=timeout))

    def _request_with_retry(self, op):
        """Run ``op`` with bounded exponential backoff on transient faults.

        A retryable HTTP status or a transient transport error (SSL EOF, reset,
        timeout, …) backs off and tries again; once retries are exhausted, or for
        any non-retryable error, it raises — context-overflow as
        ``LLMContextTooLongError``, transient failures as ``LLMTransientError``,
        and other API errors as ``RuntimeError``.
        """
        attempt = 0
        while True:
            try:
                return op()
            except urllib.error.HTTPError as exc:
                # HTTPError is a URLError subclass, so it must be handled first.
                error_text = exc.read().decode("utf-8", errors="replace")
                if exc.code in _RETRYABLE_STATUS and attempt < self.max_retries:
                    delay = self._retry_delay(attempt, self._retry_after(exc))
                    self._announce_retry(attempt, delay, f"HTTP {exc.code}")
                    self._sleep(delay)
                    attempt += 1
                    continue
                raise self._http_error(exc.code, error_text) from exc
            except (urllib.error.URLError, ssl.SSLError, TimeoutError, ConnectionError) as exc:
                if self._is_retryable_network_error(exc) and attempt < self.max_retries:
                    delay = self._retry_delay(attempt, None)
                    self._announce_retry(attempt, delay, self._describe_network_error(exc))
                    self._sleep(delay)
                    attempt += 1
                    continue
                raise LLMTransientError(
                    f"Network error talking to the Claude API after {attempt + 1} attempt(s): "
                    f"{self._describe_network_error(exc)}"
                ) from exc

    def _http_error(self, code: int, text: str) -> Exception:
        lowered = text.lower()
        if code == 400 and ("context" in lowered or "token" in lowered):
            return LLMContextTooLongError(text)
        if code in _RETRYABLE_STATUS:
            return LLMTransientError(f"Claude API error {code} after retries: {text}")
        return RuntimeError(f"Claude API error {code}: {text}")

    @staticmethod
    def _is_retryable_network_error(exc: BaseException) -> bool:
        if isinstance(exc, (TimeoutError, ssl.SSLError, ConnectionError)):
            return True
        if isinstance(exc, urllib.error.URLError):
            # The underlying reason is what actually failed (DNS, reset, SSL, timeout).
            # All of these are transport-level and worth one more attempt.
            return isinstance(exc.reason, (TimeoutError, ssl.SSLError, ConnectionError, OSError))
        return False

    @staticmethod
    def _describe_network_error(exc: BaseException) -> str:
        reason = getattr(exc, "reason", None) or exc
        return f"{type(exc).__name__}: {reason}"

    def _retry_delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, _MAX_RETRY_AFTER)
        # Exponential backoff with full jitter: sleep a random point in [0, cap] so
        # concurrent clients don't retry in lockstep (thundering herd).
        cap = min(self.max_backoff, self.initial_backoff * (self.backoff_multiplier ** attempt))
        return self._rng.uniform(0, cap)

    @staticmethod
    def _retry_after(exc: urllib.error.HTTPError) -> float | None:
        value = exc.headers.get("Retry-After") if exc.headers else None
        if not value:
            return None
        value = value.strip()
        if value.isdigit():
            return float(value)
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())

    def _announce_retry(self, attempt: int, delay: float, cause: str) -> None:
        if self._on_retry is None:
            return
        self._on_retry(
            f"[retry] Claude API request failed ({cause}); "
            f"retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})"
        )

    def _format_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        system_parts: list[str] = []
        anthropic_messages: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []
        pending_text_blocks: list[dict[str, Any]] = []
        expected_tool_use_ids: set[str] = set()

        def flush_tool_results() -> None:
            nonlocal pending_tool_results, pending_text_blocks, expected_tool_use_ids
            if pending_tool_results or pending_text_blocks:
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [*pending_tool_results, *pending_text_blocks],
                    }
                )
                pending_tool_results = []
                pending_text_blocks = []
                expected_tool_use_ids = set()

        for message in messages:
            if message.role == "system":
                system_parts.append(message.content)
                continue

            if message.role == "tool":
                block = self._format_tool_result_block(message, expected_tool_use_ids)
                if block["type"] == "tool_result":
                    pending_tool_results.append(block)
                else:
                    pending_text_blocks.append(block)
                continue

            flush_tool_results()
            if message.role == "assistant":
                anthropic_messages.append(
                    {
                        "role": "assistant",
                        "content": self._format_assistant_content(message),
                    }
                )
                expected_tool_use_ids = self._assistant_tool_use_ids(message)
            else:
                anthropic_messages.append({"role": "user", "content": message.content})
                expected_tool_use_ids = set()

        flush_tool_results()
        system = "\n".join(part for part in system_parts if part) or None
        return system, anthropic_messages

    @staticmethod
    def _format_assistant_content(message: Message) -> str | list[dict[str, Any]]:
        tool_calls = message.metadata.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            tool_calls = []
        blocks: list[dict[str, Any]] = []
        # Replay any preserved thinking blocks first — the API requires the prior
        # turn's thinking (with its signature) when thinking and tool use span turns.
        thinking_blocks = message.metadata.get("thinking_blocks", [])
        if isinstance(thinking_blocks, list):
            blocks.extend(block for block in thinking_blocks if isinstance(block, dict))
        if message.content:
            blocks.append({"type": "text", "text": message.content})
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_use_id = tool_call.get("id")
            name = tool_call.get("name")
            if not isinstance(tool_use_id, str) or not tool_use_id or not isinstance(name, str) or not name:
                continue
            arguments = tool_call.get("arguments") or {}
            if not isinstance(arguments, dict):
                arguments = {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": name,
                    "input": arguments,
                }
            )
        return blocks if any(block["type"] == "tool_use" for block in blocks) else message.content

    @staticmethod
    def _assistant_tool_use_ids(message: Message) -> set[str]:
        tool_calls = message.metadata.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            return set()
        ids: set[str] = set()
        for tool_call in tool_calls:
            if isinstance(tool_call, dict) and isinstance(tool_call.get("id"), str):
                ids.add(tool_call["id"])
        return ids

    @staticmethod
    def _format_tool_result_block(message: Message, expected_tool_use_ids: set[str]) -> dict[str, Any]:
        tool_use_id = message.metadata.get("tool_call_id")
        if isinstance(tool_use_id, str) and tool_use_id in expected_tool_use_ids:
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": message.content,
            }
            if message.metadata.get("ok") is False:
                block["is_error"] = True
            return block
        return {"type": "text", "text": message.content}

    def _parse_response(self, payload: dict[str, Any]) -> LLMResult:
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        thinking_blocks: list[dict[str, Any]] = []
        tool_calls: list[ToolCall] = []
        for block in payload.get("content", []):
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "thinking":
                thinking_parts.append(block.get("thinking", ""))
                thinking_blocks.append(block)
            elif block_type == "redacted_thinking":
                thinking_parts.append("[redacted thinking]")
                thinking_blocks.append(block)
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id"),
                        name=block.get("name", ""),
                        arguments=block.get("input") or {},
                    )
                )
        return LLMResult(
            content="\n".join(part for part in text_parts if part),
            tool_calls=tool_calls,
            stop_reason=payload.get("stop_reason"),
            raw=payload,
            thinking="\n".join(part for part in thinking_parts if part),
            thinking_blocks=thinking_blocks,
        )

    # --- streaming (Server-Sent Events) -----------------------------------

    def _consume_stream(self, raw_lines, stream: StreamHandler) -> LLMResult:
        """Assemble an ``LLMResult`` from the SSE event stream, pushing deltas live.

        ``raw_lines`` is any iterable of SSE lines (bytes or str) — the open HTTP
        response in production, a synthetic list in tests. Content blocks are
        accumulated by ``index`` and frozen on ``content_block_stop`` into the same
        shape ``_parse_response`` produces, so the rest of the agent is unaffected.
        """
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        thinking_blocks: list[dict[str, Any]] = []
        tool_calls: list[ToolCall] = []
        stop_reason: str | None = None
        blocks: dict[int, dict[str, Any]] = {}

        for event in self._iter_sse_events(raw_lines):
            etype = event.get("type")
            if etype == "content_block_start":
                blocks[event.get("index")] = self._start_block(event.get("content_block", {}))
            elif etype == "content_block_delta":
                self._apply_delta(blocks.get(event.get("index")), event.get("delta", {}), stream)
            elif etype == "content_block_stop":
                self._finalize_block(
                    blocks.get(event.get("index")), text_parts, thinking_parts, thinking_blocks, tool_calls
                )
            elif etype == "message_delta":
                stop_reason = event.get("delta", {}).get("stop_reason") or stop_reason
            elif etype == "error":
                err = event.get("error", {})
                raise LLMTransientError(
                    f"Claude streaming error {err.get('type', 'unknown')}: {err.get('message', '')}"
                )

        return LLMResult(
            content="\n".join(part for part in text_parts if part),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            raw={},
            thinking="\n".join(part for part in thinking_parts if part),
            thinking_blocks=thinking_blocks,
        )

    @staticmethod
    def _start_block(content_block: dict[str, Any]) -> dict[str, Any]:
        kind = content_block.get("type")
        if kind == "tool_use":
            return {"kind": "tool_use", "id": content_block.get("id"), "name": content_block.get("name", ""), "json": ""}
        if kind == "thinking":
            return {"kind": "thinking", "thinking": content_block.get("thinking", ""), "signature": content_block.get("signature", "")}
        if kind == "redacted_thinking":
            return {"kind": "redacted_thinking", "raw": content_block}
        return {"kind": "text", "text": content_block.get("text", "")}

    @staticmethod
    def _apply_delta(block: dict[str, Any] | None, delta: dict[str, Any], stream: StreamHandler) -> None:
        if block is None:
            return
        dtype = delta.get("type")
        if dtype == "text_delta":
            chunk = delta.get("text", "")
            block["text"] = block.get("text", "") + chunk
            stream.on_text_delta(chunk)
        elif dtype == "thinking_delta":
            chunk = delta.get("thinking", "")
            block["thinking"] = block.get("thinking", "") + chunk
            stream.on_thinking_delta(chunk)
        elif dtype == "signature_delta":
            block["signature"] = block.get("signature", "") + delta.get("signature", "")
        elif dtype == "input_json_delta":
            chunk = delta.get("partial_json", "")
            block["json"] = block.get("json", "") + chunk
            stream.on_tool_args_delta(block.get("name", ""), chunk)

    @staticmethod
    def _finalize_block(
        block: dict[str, Any] | None,
        text_parts: list[str],
        thinking_parts: list[str],
        thinking_blocks: list[dict[str, Any]],
        tool_calls: list[ToolCall],
    ) -> None:
        if block is None:
            return
        kind = block["kind"]
        if kind == "text":
            text_parts.append(block.get("text", ""))
        elif kind == "thinking":
            thinking_parts.append(block.get("thinking", ""))
            thinking_blocks.append(
                {"type": "thinking", "thinking": block.get("thinking", ""), "signature": block.get("signature", "")}
            )
        elif kind == "redacted_thinking":
            thinking_parts.append("[redacted thinking]")
            thinking_blocks.append(block.get("raw", {"type": "redacted_thinking"}))
        elif kind == "tool_use":
            raw_json = block.get("json", "").strip()
            try:
                arguments = json.loads(raw_json) if raw_json else {}
            except json.JSONDecodeError:
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            tool_calls.append(ToolCall(id=block.get("id"), name=block.get("name", ""), arguments=arguments))

    @staticmethod
    def _iter_sse_events(raw_lines):
        """Yield parsed JSON event objects from a raw SSE line stream.

        SSE frames are separated by blank lines; we accumulate ``data:`` payloads
        and dispatch on the blank line. ``event:`` lines and ``:`` keepalive
        comments are ignored — each Anthropic frame's JSON carries its own ``type``.
        """
        data_parts: list[str] = []
        for raw in raw_lines:
            line = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else raw
            line = line.rstrip("\n").rstrip("\r")
            if line == "":
                if data_parts:
                    payload = "".join(data_parts)
                    data_parts = []
                    try:
                        yield json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                continue
            if line.startswith(":"):
                continue  # SSE comment / ping keepalive
            if line.startswith("data:"):
                data_parts.append(line[len("data:"):].lstrip())
            # "event:" lines are intentionally ignored; we rely on the JSON "type".
        if data_parts:  # flush a trailing frame with no closing blank line
            try:
                yield json.loads("".join(data_parts))
            except json.JSONDecodeError:
                pass
