from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from agent_core.models import (
    LLMContextTooLongError,
    LLMResult,
    LLMTransientError,
    Message,
    TokenUsage,
    ToolCall,
)
from agent_core.providers.base import LLMProvider, ProviderConfig, StreamHandler

# HTTP statuses worth retrying: request timeout / lock conflict, rate limiting, and
# transient upstream failures. 529 is Anthropic's "overloaded" signal. Everything
# else (401, 403, 404, 422, and 400s) is a caller/permanent error and is not retried.
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

# Never sleep longer than this for a server-supplied Retry-After, so a hostile or
# buggy header can't wedge the agent for minutes.
_MAX_RETRY_AFTER = 60.0

# Model families that REMOVED the sampling params (``temperature``/``top_p``/``top_k``)
# and dropped manual extended thinking — on these, sending ``temperature`` or
# ``thinking:{type:"enabled", budget_tokens}`` is a 400. Thinking is adaptive-only
# (``thinking:{type:"adaptive"}``). Opus 4.7+ and the Fable/Mythos line share this
# request shape; everything older (Haiku 4.5, Sonnet, Opus <= 4.6) keeps the legacy
# shape below. Matched as substrings so suffixes / ``[1m]`` tags don't defeat it.
_ADAPTIVE_THINKING_MARKERS = ("opus-4-7", "opus-4-8", "fable-5", "mythos-5", "mythos-preview")


def _is_adaptive_thinking_model(model: str) -> bool:
    """True for models that reject sampling params and use adaptive-only thinking."""
    name = (model or "").lower()
    return any(marker in name for marker in _ADAPTIVE_THINKING_MARKERS)


# ``output_config.effort`` support (mirrors Open-ClaudeCode ``modelSupportsEffort`` /
# ``modelSupportsMaxEffort``). ``low``/``medium``/``high`` are the base levels every
# effort-capable model accepts; ``xhigh`` (Opus 4.7+/Fable) and ``max`` (Opus 4.6+/
# Sonnet 4.6/Fable) are narrower. Sending an unsupported level/model is a 400, so the
# provider drops effort it can't honor (keeps Haiku debug runs working).
_EFFORT_MARKERS = ("opus-4-5", "opus-4-6", "opus-4-7", "opus-4-8", "sonnet-4-6", "fable-5", "mythos-5", "mythos-preview")
_XHIGH_MARKERS = ("opus-4-7", "opus-4-8", "fable-5", "mythos-5", "mythos-preview")
_MAX_EFFORT_MARKERS = ("opus-4-6", "opus-4-7", "opus-4-8", "sonnet-4-6", "fable-5", "mythos-5", "mythos-preview")
_BASE_EFFORT_LEVELS = ("low", "medium", "high")


# Every effort level the provider knows, weakest → strongest. ``available_efforts``
# filters this per model so the UI offers exactly what the model will accept.
ALL_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


def _effort_for_model(model: str, level: Any) -> str | None:
    """Resolve the ``effort`` level to send for ``model``, or ``None`` to omit it.

    Returns ``None`` when no level is requested, the model doesn't support effort, or
    the requested level isn't allowed on that model (``xhigh``/``max`` gating) — so the
    caller never sends a value that would 400.
    """
    if not isinstance(level, str) or not level:
        return None
    name = (model or "").lower()
    if not any(marker in name for marker in _EFFORT_MARKERS):
        return None
    level = level.lower()
    if level in _BASE_EFFORT_LEVELS:
        return level
    if level == "xhigh" and any(marker in name for marker in _XHIGH_MARKERS):
        return level
    if level == "max" and any(marker in name for marker in _MAX_EFFORT_MARKERS):
        return level
    return None


def available_efforts(model: str) -> tuple[str, ...]:
    """Effort levels ``model`` actually accepts, weakest → strongest (``()`` for none).

    Derived from :func:`_effort_for_model` so it can never drift from what the provider
    sends: Haiku → ``()``, Sonnet 4.6 / Opus 4.6 → low/medium/high/max, Opus 4.7+ /
    Fable / Mythos → all five, Opus 4.5 → low/medium/high.
    """
    return tuple(level for level in ALL_EFFORT_LEVELS if _effort_for_model(model, level) == level)


class _StreamAccumulator:
    """Mutable assembly state for one streamed response, shared by sync + async consumers."""

    def __init__(self) -> None:
        self.text_parts: list[str] = []
        self.thinking_parts: list[str] = []
        self.thinking_blocks: list[dict[str, Any]] = []
        self.tool_calls: list[ToolCall] = []
        self.stop_reason: str | None = None
        self.blocks: dict[int, dict[str, Any]] = {}
        # Token accounting: input/cache counts arrive in ``message_start``, the
        # running output total in each ``message_delta``.
        self.usage: TokenUsage = TokenUsage()

    def result(self) -> LLMResult:
        return LLMResult(
            content="\n".join(part for part in self.text_parts if part),
            tool_calls=self.tool_calls,
            stop_reason=self.stop_reason,
            raw={},
            thinking="\n".join(part for part in self.thinking_parts if part),
            thinking_blocks=self.thinking_blocks,
            usage=self.usage,
        )


def _default_retry_notice(message: str) -> None:
    # ASCII only on purpose: this may print before the CLI reconfigures stderr to
    # UTF-8, and we must not reintroduce the very UnicodeEncodeError we just fixed.
    print(message, file=sys.stderr)


class ClaudeProvider(LLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        max_retries: int = 2,
        initial_backoff: float = 0.5,
        max_backoff: float = 8.0,
        backoff_multiplier: float = 2.0,
        on_retry: Callable[[str], None] | None = _default_retry_notice,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = (base_url or os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
        self.max_retries = max(0, max_retries)
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.backoff_multiplier = backoff_multiplier
        self._on_retry = on_retry
        self._rng = random.Random()
        # Lazily-created transport (see ``complete``). The client is bound to the
        # event loop it was created on; a new top-level ``asyncio.run`` gets a fresh
        # client so we never reuse a pool from a closed loop.
        self._client: Any = None
        self._client_loop: Any = None
        # Test seam: an httpx transport (e.g. MockTransport) injected before the first
        # call. Production leaves this None and uses httpx's default transport.
        self._transport: Any = None

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        config: ProviderConfig,
        stream: StreamHandler | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> LLMResult:
        """Send one Messages API request over a shared ``httpx.AsyncClient``.

        Each call builds, sends, retries, and (optionally) streams independently, so
        many can run concurrently over one connection pool. ``should_cancel`` is
        polled as streamed deltas arrive so a long response can be interrupted
        (Esc) promptly rather than only at the next turn boundary.
        """
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for ClaudeProvider")

        streaming = stream is not None and config.stream
        client = await self._get_client()
        body = self._build_body(messages, tools, config, streaming=streaming)
        timeout = config.timeout
        url = f"{self.base_url}/v1/messages"

        if not streaming:
            async def send_once() -> dict[str, Any]:
                response = await client.post(url, json=body, headers=self._headers(), timeout=timeout)
                if response.status_code >= 400:
                    raise httpx.HTTPStatusError("error", request=response.request, response=response)
                return response.json()

            payload = await self._request_with_retry(send_once)
            return self._parse_response(payload)

        # Streaming: retry only the connection setup (send + status). Once the body is
        # streaming we never retry — a mid-stream break surfaces as LLMTransientError so
        # we never reprint half-streamed output.
        async def open_stream():
            request = client.build_request("POST", url, json=body, headers=self._headers(), timeout=timeout)
            response = await client.send(request, stream=True)
            if response.status_code >= 400:
                await response.aread()
                await response.aclose()
                raise httpx.HTTPStatusError("error", request=request, response=response)
            return response

        response = await self._request_with_retry(open_stream)
        assert stream is not None
        try:
            return await self._consume_stream(response, stream, should_cancel)
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            raise LLMTransientError(
                f"Claude API stream interrupted: {self._describe_network_error(exc)}"
            ) from exc
        finally:
            await response.aclose()

    async def _get_client(self):
        loop = asyncio.get_running_loop()
        if self._client is None or self._client_loop is not loop:
            self._client = httpx.AsyncClient(timeout=None, transport=self._transport)
            self._client_loop = loop
        return self._client

    async def _request_with_retry(self, op):
        """Run ``op`` with bounded exponential backoff on transient faults.

        A retryable HTTP status or a transient transport error backs off and tries
        again; once retries are exhausted, or for any non-retryable error, it raises —
        context-overflow as ``LLMContextTooLongError``, transient failures as
        ``LLMTransientError``, and other API errors as ``RuntimeError``.
        """
        attempt = 0
        while True:
            try:
                return await op()
            except httpx.HTTPStatusError as exc:
                response = exc.response
                code = response.status_code
                error_text = response.text
                if code in _RETRYABLE_STATUS and attempt < self.max_retries:
                    delay = self._retry_delay(attempt, self._parse_retry_after(response.headers.get("Retry-After")))
                    self._announce_retry(attempt, delay, f"HTTP {code}")
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                raise self._http_error(code, error_text) from exc
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                if attempt < self.max_retries:
                    delay = self._retry_delay(attempt, None)
                    self._announce_retry(attempt, delay, self._describe_network_error(exc))
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                raise LLMTransientError(
                    f"Network error talking to the Claude API after {attempt + 1} attempt(s): "
                    f"{self._describe_network_error(exc)}"
                ) from exc

    def _build_body(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        config: ProviderConfig,
        streaming: bool = False,
    ) -> dict[str, Any]:
        """Build the Anthropic Messages request body (pure; shared by sync + async)."""
        system, anthropic_messages = self._format_messages(messages)
        model = config.model or "claude-opus-4-8"
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": config.max_tokens,
            "messages": anthropic_messages,
        }
        if _is_adaptive_thinking_model(model):
            # Opus 4.7+/Fable/Mythos: NO sampling params (they 400), adaptive-only thinking.
            self._apply_adaptive_thinking(body, config.thinking_budget)
        else:
            # Legacy shape (Haiku 4.5, Sonnet, Opus <= 4.6): temperature + manual thinking.
            body["temperature"] = 0.2 if config.temperature is None else config.temperature
            self._apply_thinking(body, config.thinking_budget)
        effort = _effort_for_model(model, config.effort)
        if effort is not None:
            body["output_config"] = {"effort": effort}
        if streaming:
            body["stream"] = True
        if system:
            body["system"] = system
        if tools:
            body["tools"] = tools
        return body

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for ClaudeProvider")
        return {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

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

    @staticmethod
    def _apply_adaptive_thinking(body: dict[str, Any], thinking_budget: Any) -> None:
        """Enable adaptive thinking in-place for Opus 4.7+/Fable/Mythos models.

        These models reject ``temperature`` and the manual ``budget_tokens`` shape, so
        a positive ``thinking_budget`` only toggles adaptive thinking ON — the model
        decides depth, and the numeric value is advisory (ignored here). ``display`` is
        set to ``summarized`` so the live trace can render reasoning (the API default is
        ``omitted``, i.e. empty thinking text). No sampling params are ever set; with no
        budget requested we leave ``thinking`` unset (request runs without thinking).
        """
        if not isinstance(thinking_budget, int) or isinstance(thinking_budget, bool) or thinking_budget <= 0:
            return
        body["thinking"] = {"type": "adaptive", "display": "summarized"}

    def _http_error(self, code: int, text: str) -> Exception:
        lowered = text.lower()
        if code == 400 and ("context" in lowered or "token" in lowered):
            return LLMContextTooLongError(text)
        if code in _RETRYABLE_STATUS:
            return LLMTransientError(f"Claude API error {code} after retries: {text}")
        return RuntimeError(f"Claude API error {code}: {text}")

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
    def _parse_retry_after(value: str | None) -> float | None:
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
            usage=self._parse_usage(payload.get("usage")),
        )

    @staticmethod
    def _parse_usage(usage: Any) -> TokenUsage | None:
        """Map an Anthropic ``usage`` object onto ``TokenUsage`` (cache_* default 0)."""
        if not isinstance(usage, dict):
            return None
        return TokenUsage(
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cache_read_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
            cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens") or 0),
        )

    # --- streaming (Server-Sent Events) -----------------------------------

    async def _consume_stream(
        self,
        response,
        stream: StreamHandler,
        should_cancel: Callable[[], bool] | None = None,
    ) -> LLMResult:
        """Assemble an ``LLMResult`` from an httpx SSE streaming response.

        Content blocks are accumulated by ``index`` and frozen on
        ``content_block_stop`` into the same shape ``_parse_response`` produces, so
        the rest of the agent is unaffected. A transport break here propagates to
        ``complete`` as ``LLMTransientError``; it is never retried, so half-streamed
        output is never reprinted.

        ``should_cancel`` is polled before each event so the user (Esc) can abort a
        long response mid-stream; when it fires we raise ``asyncio.CancelledError``,
        which the agent loop turns into a clean "interrupted" stop.
        """
        acc = _StreamAccumulator()
        async for event in self._iter_sse_events(response.aiter_lines()):
            if should_cancel is not None and should_cancel():
                raise asyncio.CancelledError("Claude stream cancelled by user")
            self._handle_sse_event(event, acc, stream)
        return acc.result()

    def _handle_sse_event(self, event: dict[str, Any], acc: "_StreamAccumulator", stream: StreamHandler) -> None:
        """Dispatch one parsed SSE event into the accumulator (shared sync + async)."""
        etype = event.get("type")
        if etype == "message_start":
            # Input + cache token counts are reported once, up front. Output is still
            # zero here; the running output total accrues in message_delta below.
            message = event.get("message", {})
            usage = self._parse_usage(message.get("usage")) if isinstance(message, dict) else None
            if usage is not None:
                # Preserve any output already accumulated (defensive — normally none yet).
                usage.output_tokens = acc.usage.output_tokens or usage.output_tokens
                acc.usage = usage
        elif etype == "content_block_start":
            index = event.get("index")
            if isinstance(index, int):
                acc.blocks[index] = self._start_block(event.get("content_block", {}))
        elif etype == "content_block_delta":
            index = event.get("index")
            if isinstance(index, int):
                self._apply_delta(acc.blocks.get(index), event.get("delta", {}), stream)
        elif etype == "content_block_stop":
            index = event.get("index")
            if isinstance(index, int):
                self._finalize_block(
                    acc.blocks.get(index),
                    acc.text_parts,
                    acc.thinking_parts,
                    acc.thinking_blocks,
                    acc.tool_calls,
                )
        elif etype == "message_delta":
            acc.stop_reason = event.get("delta", {}).get("stop_reason") or acc.stop_reason
            # message_delta carries the running output token total at top level.
            delta_usage = event.get("usage")
            if isinstance(delta_usage, dict) and "output_tokens" in delta_usage:
                acc.usage.output_tokens = int(delta_usage.get("output_tokens") or 0)
        elif etype == "error":
            err = event.get("error", {})
            raise LLMTransientError(
                f"Claude streaming error {err.get('type', 'unknown')}: {err.get('message', '')}"
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
    def _sse_feed(raw, data_parts: list[str]) -> dict[str, Any] | None:
        """Feed one raw SSE line into ``data_parts``; return a parsed frame on a blank line.

        ``event:`` lines and ``:`` keepalive comments are ignored — each Anthropic
        frame's JSON carries its own ``type``.
        """
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else raw
        line = line.rstrip("\n").rstrip("\r")
        if line == "":
            if data_parts:
                payload = "".join(data_parts)
                data_parts.clear()
                try:
                    return json.loads(payload)
                except json.JSONDecodeError:
                    return None
            return None
        if line.startswith(":"):
            return None  # SSE comment / ping keepalive
        if line.startswith("data:"):
            data_parts.append(line[len("data:"):].lstrip())
        # "event:" lines are intentionally ignored; we rely on the JSON "type".
        return None

    @classmethod
    async def _iter_sse_events(cls, raw_lines):
        """Yield parsed JSON event objects from an async SSE line stream (httpx)."""
        data_parts: list[str] = []
        async for raw in raw_lines:
            event = cls._sse_feed(raw, data_parts)
            if event is not None:
                yield event
        if data_parts:  # flush a trailing frame with no closing blank line
            try:
                yield json.loads("".join(data_parts))
            except json.JSONDecodeError:
                pass
