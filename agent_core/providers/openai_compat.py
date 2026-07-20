"""OpenAI-compatible chat-completions provider (decision D5).

Speaks the ``/v1/chat/completions`` protocol directly over httpx — the same
no-vendor-SDK stance as :class:`ClaudeProvider` — so it covers OpenAI itself and
every OpenAI-shaped endpoint (vLLM, llama.cpp server, LM Studio, Together, Groq,
DeepSeek, …) via ``base_url``. Its purpose is architectural before it is
functional: a second protocol family must fit behind ``LLMProvider.complete``
without touching ``providers/base.py`` or the core loop — anywhere it can't is an
Anthropic leak to fix in the abstraction, not to branch around.

ProviderConfig mapping: ``model`` / ``max_tokens`` / ``stream`` / ``timeout`` map
directly; ``temperature`` is sent only when set. ``thinking_budget`` and ``effort``
have no chat-completions equivalent and are safely ignored. ``thinking_blocks``
stays empty (provider-owned opaque data — this provider owns none).
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from collections.abc import Callable
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
from agent_core.providers.openai_errors import format_openai_error, parse_openai_error

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
_MAX_RETRY_AFTER = 30.0

# Substrings that identify a context-window overflow in an OpenAI-shaped 400 body
# (OpenAI uses code "context_length_exceeded"; compat servers vary in wording).
_CONTEXT_OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "too many tokens",
    "prompt is too long",
)


_WARNED_DEPRECATED_ENV: set[str] = set()


def _default_retry_notice(message: str) -> None:
    print(message, file=sys.stderr)  # ASCII-only; see ClaudeProvider._default_retry_notice


class OpenAICompatProvider(LLMProvider):
    """Chat-completions over httpx: streaming + non-streaming, retries, tool calls."""

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
        self.api_key = api_key or os.getenv("OPENAI_COMPAT_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = (
            base_url
            or os.getenv("OPENAI_COMPAT_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com"
        ).rstrip("/")
        if api_key is None and os.getenv("OPENAI_COMPAT_API_KEY") is None and os.getenv("OPENAI_API_KEY"):
            self._warn_deprecated_env("OPENAI_API_KEY", "OPENAI_COMPAT_API_KEY")
        if base_url is None and os.getenv("OPENAI_COMPAT_BASE_URL") is None and os.getenv("OPENAI_BASE_URL"):
            self._warn_deprecated_env("OPENAI_BASE_URL", "OPENAI_COMPAT_BASE_URL")
        self.max_retries = max(0, max_retries)
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.backoff_multiplier = backoff_multiplier
        self._on_retry = on_retry
        self._rng = random.Random()
        # Lazily-created client bound to the running loop (fresh pool per top-level
        # asyncio.run), plus the httpx MockTransport test seam — same pattern as
        # ClaudeProvider.
        self._client: Any = None
        self._client_loop: Any = None
        self._transport: Any = None

    @staticmethod
    def _warn_deprecated_env(old: str, new: str) -> None:
        if old in _WARNED_DEPRECATED_ENV:
            return
        _WARNED_DEPRECATED_ENV.add(old)
        print(
            f"[deprecated] OpenAI-compatible provider is using {old}; set {new} instead.",
            file=sys.stderr,
        )

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        config: ProviderConfig,
        stream: StreamHandler | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> LLMResult:
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_COMPAT_API_KEY is required for the OpenAI-compatible provider "
                "(set OPENAI_COMPAT_BASE_URL too for a non-OpenAI endpoint; legacy "
                "OPENAI_API_KEY/OPENAI_BASE_URL are still accepted with a deprecation warning)"
            )
        if not config.model:
            raise RuntimeError("OpenAICompatProvider needs an explicit model (--model / config.model)")

        streaming = stream is not None and config.stream
        client = await self._get_client()
        body = self._build_body(messages, tools, config, streaming=streaming)
        url = f"{self.base_url}/v1/chat/completions"

        if not streaming:
            async def send_once() -> dict[str, Any]:
                response = await client.post(url, json=body, headers=self._headers(), timeout=config.timeout)
                if response.status_code >= 400:
                    raise httpx.HTTPStatusError("error", request=response.request, response=response)
                return response.json()

            payload = await self._request_with_retry(send_once, model=config.model)
            return self._parse_response(payload)

        # Streaming: retry only connection setup; never mid-stream (no half-reprints).
        async def open_stream():
            request = client.build_request(
                "POST", url, json=body, headers=self._headers(), timeout=config.timeout
            )
            response = await client.send(request, stream=True)
            if response.status_code >= 400:
                await response.aread()
                await response.aclose()
                raise httpx.HTTPStatusError("error", request=request, response=response)
            return response

        response = await self._request_with_retry(open_stream, model=config.model)
        try:
            return await self._consume_stream(response, stream, should_cancel)
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            raise LLMTransientError(
                f"chat-completions stream interrupted: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            await response.aclose()

    # --- request shape ------------------------------------------------------------

    def _build_body(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        config: ProviderConfig,
        streaming: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "messages": self._format_messages(messages),
        }
        if config.temperature is not None:
            body["temperature"] = config.temperature
        if streaming:
            body["stream"] = True
        if tools:
            body["tools"] = [self._format_tool(schema) for schema in tools]
        return body

    @staticmethod
    def _format_tool(schema: dict[str, Any]) -> dict[str, Any]:
        """Project this project's neutral tool schema into the function-call shape."""
        return {
            "type": "function",
            "function": {
                "name": schema.get("name", ""),
                "description": schema.get("description", ""),
                "parameters": schema.get("input_schema", {"type": "object"}),
            },
        }

    @staticmethod
    def _format_messages(messages: list[Message]) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                formatted.append({"role": "system", "content": message.content})
            elif message.role == "tool":
                tool_call_id = message.metadata.get("tool_call_id")
                formatted.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(tool_call_id) if tool_call_id else (message.name or "call_0"),
                        "content": message.content,
                    }
                )
            elif message.role == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": message.content or None}
                tool_calls = message.metadata.get("tool_calls", [])
                calls: list[dict[str, Any]] = []
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if not isinstance(tool_call, dict) or not tool_call.get("name"):
                            continue
                        arguments = tool_call.get("arguments") or {}
                        calls.append(
                            {
                                "id": str(tool_call.get("id") or f"call_{len(calls)}"),
                                "type": "function",
                                "function": {
                                    "name": tool_call["name"],
                                    "arguments": json.dumps(
                                        arguments if isinstance(arguments, dict) else {},
                                        ensure_ascii=False,
                                    ),
                                },
                            }
                        )
                if calls:
                    entry["tool_calls"] = calls
                formatted.append(entry)
            else:
                formatted.append({"role": "user", "content": message.content})
        return formatted

    def _headers(self) -> dict[str, str]:
        return {
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}",
        }

    # --- response parsing -----------------------------------------------------------

    @staticmethod
    def _parse_tool_calls(raw_calls: Any) -> list[ToolCall]:
        calls: list[ToolCall] = []
        if not isinstance(raw_calls, list):
            return calls
        for raw in raw_calls:
            if not isinstance(raw, dict):
                continue
            function = raw.get("function") or {}
            name = function.get("name")
            if not name:
                continue
            arguments_raw = function.get("arguments") or "{}"
            try:
                arguments = json.loads(arguments_raw) if isinstance(arguments_raw, str) else dict(arguments_raw)
            except (ValueError, TypeError):
                arguments = {"_raw_arguments": str(arguments_raw)}
            if not isinstance(arguments, dict):
                arguments = {"_raw_arguments": str(arguments_raw)}
            calls.append(ToolCall(name=str(name), arguments=arguments, id=str(raw.get("id") or "")))
        return calls

    def _parse_response(self, payload: dict[str, Any]) -> LLMResult:
        choices = payload.get("choices") or [{}]
        message = choices[0].get("message") or {}
        usage_raw = payload.get("usage") or {}
        usage = None
        if usage_raw:
            usage = TokenUsage(
                input_tokens=int(usage_raw.get("prompt_tokens", 0) or 0),
                output_tokens=int(usage_raw.get("completion_tokens", 0) or 0),
            )
        return LLMResult(
            content=message.get("content") or "",
            tool_calls=self._parse_tool_calls(message.get("tool_calls")),
            stop_reason=choices[0].get("finish_reason"),
            raw=payload,
            usage=usage,
        )

    async def _consume_stream(
        self,
        response: Any,
        sink: StreamHandler | None,
        should_cancel: Callable[[], bool] | None,
    ) -> LLMResult:
        """Assemble one result from SSE chunks, pushing deltas to the sink live."""
        content_parts: list[str] = []
        finish_reason: str | None = None
        usage: TokenUsage | None = None
        # index → accumulating {"id", "name", "arguments"(str parts)}
        pending_calls: dict[int, dict[str, Any]] = {}

        async for line in response.aiter_lines():
            if should_cancel is not None and should_cancel():
                raise asyncio.CancelledError("chat-completions stream cancelled by user")
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            if chunk.get("error"):
                raise LLMTransientError(f"chat-completions stream error: {chunk['error']}")
            usage_raw = chunk.get("usage")
            if isinstance(usage_raw, dict) and usage_raw:
                usage = TokenUsage(
                    input_tokens=int(usage_raw.get("prompt_tokens", 0) or 0),
                    output_tokens=int(usage_raw.get("completion_tokens", 0) or 0),
                )
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if text:
                content_parts.append(text)
                if sink is not None:
                    sink.on_text_delta(text)
            for raw_call in delta.get("tool_calls") or []:
                if not isinstance(raw_call, dict):
                    continue
                index = int(raw_call.get("index", 0) or 0)
                slot = pending_calls.setdefault(index, {"id": "", "name": "", "arguments": []})
                if raw_call.get("id"):
                    slot["id"] = str(raw_call["id"])
                function = raw_call.get("function") or {}
                if function.get("name"):
                    slot["name"] = str(function["name"])
                fragment = function.get("arguments")
                if fragment:
                    slot["arguments"].append(str(fragment))
                    if sink is not None:
                        sink.on_tool_args_delta(slot["name"] or "?", str(fragment))

        tool_calls: list[ToolCall] = []
        for index in sorted(pending_calls):
            slot = pending_calls[index]
            if not slot["name"]:
                continue
            arguments_raw = "".join(slot["arguments"]) or "{}"
            try:
                arguments = json.loads(arguments_raw)
            except ValueError:
                arguments = {"_raw_arguments": arguments_raw}
            if not isinstance(arguments, dict):
                arguments = {"_raw_arguments": arguments_raw}
            tool_calls.append(ToolCall(name=slot["name"], arguments=arguments, id=slot["id"]))

        return LLMResult(
            content="".join(content_parts),
            tool_calls=tool_calls,
            stop_reason=finish_reason,
            raw={},
            usage=usage,
        )

    # --- transport plumbing (mirrors ClaudeProvider's shape, provider-local) ---------

    async def _get_client(self):
        loop = asyncio.get_running_loop()
        if self._client is None or self._client_loop is not loop:
            self._client = httpx.AsyncClient(timeout=None, transport=self._transport)
            self._client_loop = loop
        return self._client

    async def _request_with_retry(self, op, *, model: str | None = None):
        attempt = 0
        while True:
            try:
                return await op()
            except httpx.HTTPStatusError as exc:
                response = exc.response
                code = response.status_code
                text = response.text
                if code in _RETRYABLE_STATUS and attempt < self.max_retries:
                    delay = self._retry_delay(attempt, self._parse_retry_after(response.headers.get("Retry-After")))
                    self._announce_retry(attempt, delay, f"HTTP {code}")
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                raise self._http_error(code, text, model=model) from exc
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                if attempt < self.max_retries:
                    delay = self._retry_delay(attempt, None)
                    self._announce_retry(attempt, delay, f"{type(exc).__name__}: {exc}")
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                raise LLMTransientError(
                    f"Network error talking to the chat-completions API after {attempt + 1} "
                    f"attempt(s): {type(exc).__name__}: {exc}"
                ) from exc

    @staticmethod
    def _http_error(code: int, text: str, *, model: str | None = None) -> Exception:
        lowered = text.lower()
        if code == 400 and any(marker in lowered for marker in _CONTEXT_OVERFLOW_MARKERS):
            return LLMContextTooLongError(f"chat-completions context overflow: {text[:300]}")
        if code in _RETRYABLE_STATUS:
            return LLMTransientError(f"chat-completions API error {code} after retries: {text[:300]}")
        info = parse_openai_error(text)
        return RuntimeError(format_openai_error("OpenAI-compatible Chat Completions", model, code, info))

    def _retry_delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, _MAX_RETRY_AFTER)
        cap = min(self.max_backoff, self.initial_backoff * (self.backoff_multiplier ** attempt))
        return self._rng.uniform(0, cap)

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        if not value:
            return None
        value = value.strip()
        if value.isdigit():
            return float(value)
        return None

    def _announce_retry(self, attempt: int, delay: float, cause: str) -> None:
        if self._on_retry is None:
            return
        self._on_retry(
            f"[retry] chat-completions request failed ({cause}); "
            f"retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})"
        )
