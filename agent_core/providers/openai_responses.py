"""OpenAI Responses API provider.

Speaks OpenAI's ``/v1/responses`` protocol directly over httpx. This is distinct
from :class:`OpenAICompatProvider`, which intentionally remains on the older
``/v1/chat/completions`` shape for compatible endpoints.
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

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
_MAX_RETRY_AFTER = 30.0
_CONTEXT_OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "too many tokens",
    "prompt is too long",
)
_SUPPORTED_REASONING_EFFORTS = {"low", "medium", "high"}
_REASONING_MODEL_MARKERS = ("gpt-5", "o1", "o3", "o4")
_REPLAY_OUTPUT_TYPES = {"message", "function_call", "reasoning", "output_text"}


def _supports_reasoning(model: str) -> bool:
    """Return whether ``model`` accepts Responses reasoning-only request fields."""
    name = (model or "").lower()
    return any(marker in name for marker in _REASONING_MODEL_MARKERS)


def _reasoning_effort_for_model(model: str, level: Any) -> str | None:
    """Resolve a Responses reasoning effort, or ``None`` to omit unsupported values."""
    if not _supports_reasoning(model) or not isinstance(level, str):
        return None
    normalized = level.lower()
    return normalized if normalized in _SUPPORTED_REASONING_EFFORTS else None


def _default_retry_notice(message: str) -> None:
    print(message, file=sys.stderr)


def _json_clone(value: Any) -> Any:
    """Return a plain JSON-compatible clone, stringifying impossible leaves."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


class _ResponsesStreamAccumulator:
    def __init__(self) -> None:
        self.completed_response: dict[str, Any] | None = None
        self.text_parts: list[str] = []
        self.output_items: dict[str, dict[str, Any]] = {}
        self.output_order: list[str] = []
        self.function_args: dict[str, list[str]] = {}
        self.stop_reason: str | None = None
        self.usage: TokenUsage | None = None

    def result(self) -> LLMResult:
        if self.completed_response is not None:
            return OpenAIResponsesProvider._parse_response(self.completed_response)
        output = [self.output_items[key] for key in self.output_order if key in self.output_items]
        text = "".join(self.text_parts)
        calls = OpenAIResponsesProvider._parse_function_calls(output)
        return LLMResult(
            content=text,
            tool_calls=calls,
            stop_reason=self.stop_reason,
            raw={},
            usage=self.usage,
            provider_state={"output": _json_clone(output)} if output else {},
        )


class OpenAIResponsesProvider(LLMProvider):
    """OpenAI Responses API over httpx: state replay, streaming, retries, tools."""

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
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com").rstrip("/")
        self.max_retries = max(0, max_retries)
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.backoff_multiplier = backoff_multiplier
        self._on_retry = on_retry
        self._rng = random.Random()
        self._client: Any = None
        self._client_loop: Any = None
        self._transport: Any = None

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        config: ProviderConfig,
        stream: StreamHandler | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> LLMResult:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI Responses provider")
        if not config.model:
            raise RuntimeError("OpenAIResponsesProvider needs an explicit model (--model / config.model)")

        streaming = stream is not None and config.stream
        client = await self._get_client()
        body = self._build_body(messages, tools, config, streaming=streaming)
        url = f"{self.base_url}/v1/responses"

        if not streaming:

            async def send_once() -> dict[str, Any]:
                response = await client.post(url, json=body, headers=self._headers(), timeout=config.timeout)
                if response.status_code >= 400:
                    raise httpx.HTTPStatusError("error", request=response.request, response=response)
                return response.json()

            payload = await self._request_with_retry(send_once)
            return self._parse_response(payload)

        async def open_stream():
            request = client.build_request("POST", url, json=body, headers=self._headers(), timeout=config.timeout)
            response = await client.send(request, stream=True)
            if response.status_code >= 400:
                await response.aread()
                await response.aclose()
                raise httpx.HTTPStatusError("error", request=request, response=response)
            return response

        response = await self._request_with_retry(open_stream)
        try:
            return await self._consume_stream(response, stream, should_cancel)
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            raise LLMTransientError(
                f"OpenAI Responses stream interrupted: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            await response.aclose()

    # --- request shape ---------------------------------------------------------

    def _build_body(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        config: ProviderConfig,
        streaming: bool = False,
    ) -> dict[str, Any]:
        preserve_reasoning = _supports_reasoning(config.model)
        body: dict[str, Any] = {
            "model": config.model,
            "input": self._format_input(messages, preserve_reasoning=preserve_reasoning),
            "max_output_tokens": config.max_tokens,
            "store": False,
        }
        if preserve_reasoning:
            body["include"] = ["reasoning.encrypted_content"]
        effort = _reasoning_effort_for_model(config.model, config.effort)
        if effort is not None:
            body["reasoning"] = {"effort": effort}
        if streaming:
            body["stream"] = True
        if tools:
            body["tools"] = [self._format_tool(schema) for schema in tools]
        return body

    @staticmethod
    def _format_tool(schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "name": schema.get("name", ""),
            "description": schema.get("description", ""),
            "parameters": schema.get("input_schema", {"type": "object"}),
        }

    def _format_input(self, messages: list[Message], *, preserve_reasoning: bool) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                items.append({"role": "system", "content": message.content})
            elif message.role == "user":
                items.append({"role": "user", "content": message.content})
            elif message.role == "tool":
                items.append(self._format_function_call_output(message))
            elif message.role == "assistant":
                items.extend(self._format_assistant_items(message, preserve_reasoning=preserve_reasoning))
        return items

    def _format_assistant_items(self, message: Message, *, preserve_reasoning: bool) -> list[dict[str, Any]]:
        state = message.metadata.get("provider_state")
        if message.metadata.get("compressed") is None and isinstance(state, dict):
            output = state.get("output")
            if isinstance(output, list):
                replay = [
                    item
                    for item in output
                    if isinstance(item, dict) and (preserve_reasoning or item.get("type") != "reasoning")
                ]
                if replay:
                    return _json_clone(replay)

        items: list[dict[str, Any]] = []
        if message.content:
            items.append(self._message_item("assistant", message.content, output=True))
        tool_calls = message.metadata.get("tool_calls", [])
        if isinstance(tool_calls, list):
            for index, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                name = tool_call.get("name")
                if not isinstance(name, str) or not name:
                    continue
                arguments = tool_call.get("arguments") or {}
                if not isinstance(arguments, dict):
                    arguments = {}
                call_id = str(tool_call.get("id") or f"call_{index}")
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    }
                )
        return items

    @staticmethod
    def _message_item(role: str, content: str, *, output: bool = False) -> dict[str, Any]:
        block_type = "output_text" if output else "input_text"
        return {"type": "message", "role": role, "content": [{"type": block_type, "text": content}]}

    @staticmethod
    def _format_function_call_output(message: Message) -> dict[str, Any]:
        tool_call_id = message.metadata.get("tool_call_id")
        return {
            "type": "function_call_output",
            "call_id": str(tool_call_id) if tool_call_id else (message.name or "call_0"),
            "output": message.content,
        }

    def _headers(self) -> dict[str, str]:
        return {"content-type": "application/json", "authorization": f"Bearer {self.api_key}"}

    # --- response parsing ------------------------------------------------------

    @staticmethod
    def _parse_response(payload: dict[str, Any]) -> LLMResult:
        output = payload.get("output") if isinstance(payload.get("output"), list) else []
        text_parts = OpenAIResponsesProvider._parse_text_parts(output)
        tool_calls = OpenAIResponsesProvider._parse_function_calls(output)
        return LLMResult(
            content="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=OpenAIResponsesProvider._stop_reason(payload, tool_calls),
            raw=payload,
            usage=OpenAIResponsesProvider._parse_usage(payload.get("usage")),
            provider_state={"output": _json_clone(output)} if output else {},
        )

    @staticmethod
    def _parse_text_parts(output: list[Any]) -> list[str]:
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "message":
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") in {"output_text", "text"}:
                            parts.append(str(block.get("text") or ""))
            elif item_type in {"output_text", "text"}:
                parts.append(str(item.get("text") or ""))
        return parts

    @staticmethod
    def _parse_function_calls(output: list[Any]) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for index, item in enumerate(output):
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name:
                continue
            call_id = str(item.get("call_id") or item.get("id") or f"call_{index}")
            arguments = OpenAIResponsesProvider._parse_arguments(item.get("arguments") or "{}")
            calls.append(ToolCall(name=name, arguments=arguments, id=call_id))
        return calls

    @staticmethod
    def _parse_arguments(arguments_raw: Any) -> dict[str, Any]:
        try:
            arguments = json.loads(arguments_raw) if isinstance(arguments_raw, str) else dict(arguments_raw)
        except (TypeError, ValueError):
            return {"_raw_arguments": str(arguments_raw)}
        if not isinstance(arguments, dict):
            return {"_raw_arguments": str(arguments_raw)}
        return arguments

    @staticmethod
    def _parse_usage(usage_raw: Any) -> TokenUsage | None:
        if not isinstance(usage_raw, dict):
            return None
        return TokenUsage(
            input_tokens=int(usage_raw.get("input_tokens") or 0),
            output_tokens=int(usage_raw.get("output_tokens") or 0),
        )

    @staticmethod
    def _stop_reason(payload: dict[str, Any], tool_calls: list[ToolCall]) -> str | None:
        details = payload.get("incomplete_details")
        reason = details.get("reason") if isinstance(details, dict) else None
        if reason == "max_output_tokens":
            return "max_tokens"
        if tool_calls:
            return "tool_calls"
        status = payload.get("status")
        if status == "incomplete" and reason:
            return str(reason)
        return str(status) if status is not None else None

    # --- streaming -------------------------------------------------------------

    async def _consume_stream(
        self,
        response: Any,
        sink: StreamHandler | None,
        should_cancel: Callable[[], bool] | None,
    ) -> LLMResult:
        acc = _ResponsesStreamAccumulator()
        async for event in self._iter_sse_events(response.aiter_lines()):
            if should_cancel is not None and should_cancel():
                raise asyncio.CancelledError("OpenAI Responses stream cancelled by user")
            self._handle_sse_event(event, acc, sink)
        return acc.result()

    @staticmethod
    def _item_key(event: dict[str, Any], item: dict[str, Any] | None = None) -> str:
        for key in ("item_id", "output_index", "index"):
            if event.get(key) is not None:
                return str(event[key])
        if item is not None:
            for key in ("id", "call_id"):
                if item.get(key) is not None:
                    return str(item[key])
        return "0"

    def _handle_sse_event(
        self,
        event: dict[str, Any],
        acc: _ResponsesStreamAccumulator,
        sink: StreamHandler | None,
    ) -> None:
        etype = event.get("type")
        if etype == "response.created":
            return
        if etype == "response.output_text.delta":
            chunk = str(event.get("delta") or "")
            acc.text_parts.append(chunk)
            if sink is not None and chunk:
                sink.on_text_delta(chunk)
            return
        if etype in {"response.output_item.added", "response.output_item.done"}:
            item = event.get("item")
            if isinstance(item, dict):
                key = self._item_key(event, item)
                clean = _json_clone(item)
                if clean.get("type") in _REPLAY_OUTPUT_TYPES:
                    if key not in acc.output_items:
                        acc.output_order.append(key)
                    acc.output_items[key] = clean
                if clean.get("type") == "function_call" and key in acc.function_args:
                    clean["arguments"] = "".join(acc.function_args[key]) or clean.get("arguments", "")
                    acc.output_items[key] = clean
            return
        if etype in {"response.function_call_arguments.delta", "response.function_call_arguments.done"}:
            key = self._item_key(event)
            slot = acc.function_args.setdefault(key, [])
            delta = event.get("delta")
            if delta:
                slot.append(str(delta))
                if sink is not None:
                    item = acc.output_items.get(key, {})
                    sink.on_tool_args_delta(str(item.get("name") or "?"), str(delta))
            arguments = event.get("arguments")
            if isinstance(arguments, str):
                acc.function_args[key] = [arguments]
            item = acc.output_items.get(key)
            if item is not None:
                item["arguments"] = "".join(acc.function_args.get(key, [])) or item.get("arguments", "")
            return
        if etype == "response.completed":
            response_payload = event.get("response")
            if isinstance(response_payload, dict):
                acc.completed_response = response_payload
            return
        if etype in {"response.failed", "response.incomplete", "error"}:
            error = event.get("error") or event.get("response") or event
            raise LLMTransientError(f"OpenAI Responses stream error: {error}")

    @classmethod
    async def _iter_sse_events(cls, raw_lines):
        data_parts: list[str] = []
        event_type: str | None = None
        async for raw in raw_lines:
            line = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else raw
            line = line.rstrip("\n").rstrip("\r")
            if line == "":
                event = cls._parse_sse_payload(data_parts, event_type)
                data_parts = []
                event_type = None
                if event is not None:
                    yield event
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_parts.append(line[len("data:"):].lstrip())
        event = cls._parse_sse_payload(data_parts, event_type)
        if event is not None:
            yield event

    @staticmethod
    def _parse_sse_payload(data_parts: list[str], event_type: str | None) -> dict[str, Any] | None:
        if not data_parts:
            return None
        payload = "".join(data_parts)
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            return None
        if isinstance(event, dict):
            event.setdefault("type", event_type)
            return event
        return None

    # --- transport plumbing ----------------------------------------------------

    async def _get_client(self):
        loop = asyncio.get_running_loop()
        if self._client is None or self._client_loop is not loop:
            self._client = httpx.AsyncClient(timeout=None, transport=self._transport)
            self._client_loop = loop
        return self._client

    async def _request_with_retry(self, op):
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
                raise self._http_error(code, text) from exc
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                if attempt < self.max_retries:
                    delay = self._retry_delay(attempt, None)
                    self._announce_retry(attempt, delay, f"{type(exc).__name__}: {exc}")
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                raise LLMTransientError(
                    f"Network error talking to the OpenAI Responses API after {attempt + 1} "
                    f"attempt(s): {type(exc).__name__}: {exc}"
                ) from exc

    @staticmethod
    def _http_error(code: int, text: str) -> Exception:
        lowered = text.lower()
        if code == 400 and any(marker in lowered for marker in _CONTEXT_OVERFLOW_MARKERS):
            return LLMContextTooLongError(f"OpenAI Responses context overflow: {text[:300]}")
        if code in _RETRYABLE_STATUS:
            return LLMTransientError(f"OpenAI Responses API error {code} after retries: {text[:300]}")
        return RuntimeError(f"OpenAI Responses API error {code}: {text[:300]}")

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
            f"[retry] OpenAI Responses request failed ({cause}); "
            f"retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})"
        )
