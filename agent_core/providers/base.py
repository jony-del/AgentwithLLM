from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import inspect
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal, Protocol, runtime_checkable

from agent_core.models import LLMResult, Message, ToolCall
from agent_core.execution import ExecutionScope, current_execution_scope

logger = logging.getLogger(__name__)

ToolStreamBoundary = Literal["explicit", "terminal_only", "unsupported"]


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """Optional provider contract for safe incremental tool execution.

    Providers that do not expose this contract are treated as ``terminal_only``.
    That preserves third-party compatibility while preventing the core from
    interpreting a syntactically closed JSON fragment as an execution boundary.
    """

    tool_stream_boundary: ToolStreamBoundary = "terminal_only"
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class StreamedToolCall:
    """One immutable streamed tool call with provider and turn identities."""

    tool_call: ToolCall
    call_id: str
    ordinal: int
    provider_item_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Per-call completion parameters — the cross-layer contract of ``complete``.

    Provider-neutral by construction: each field is a concept every chat-completion
    protocol has some answer to. A provider that has no equivalent for a field
    ignores it (and says so in its docstring) rather than erroring; a provider MUST
    NOT require keys outside this contract — provider-specific connection settings
    (base URL, API key, retries) belong on the provider's constructor.

    ``model = ""`` means "the provider's own default model"; ``temperature = None``
    means "the provider's own default sampling" (some model families reject explicit
    sampling parameters entirely). Derived calls (summaries, hook prompts) override
    fields with :func:`dataclasses.replace` instead of mutating.
    """

    model: str = ""
    temperature: float | None = None
    max_tokens: int = 1024
    thinking_budget: int | None = None
    effort: str | None = None
    # Provider speed tier. Currently only Claude Opus 4.6 accepts ``"fast"``.
    speed: str | None = None
    stream: bool = True
    timeout: float = 60.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ProviderConfig":
        """Build from a loose mapping, dropping (and debug-logging) unknown keys.

        The tolerant entry point for config files and tests; typed call sites should
        construct the dataclass directly so typos fail loudly.
        """
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            logger.debug("ProviderConfig.from_dict dropping unknown keys: %s", unknown)
        return cls(**{key: value for key, value in data.items() if key in known})

    def to_dict(self) -> dict[str, Any]:
        """A plain-JSON projection (for logs and external-hook payload bounds)."""
        return asdict(self)


@runtime_checkable
class StreamHandler(Protocol):
    """Sink a provider calls back as tokens arrive, for live (streamed) display.

    ``AgentUI`` satisfies this structurally, so providers can stream straight to
    the UI without importing the ui module. All methods are best-effort display
    side-effects; the provider still returns a complete ``LLMResult``.
    """

    def on_text_delta(self, text: str) -> None: ...

    def on_thinking_delta(self, text: str) -> None: ...

    def on_tool_args_delta(self, tool_name: str, partial_json: str) -> None: ...


@runtime_checkable
class ToolCallStreamHandler(StreamHandler, Protocol):
    """Optional extension notified when one streamed tool call is immutable.

    This is deliberately separate from :class:`StreamHandler`: existing UI sinks and
    third-party providers only implement the three display callbacks and must keep
    working unchanged. Providers call :func:`notify_tool_call_complete`, which performs
    a structural, best-effort lookup instead of assuming the extension is present.
    """

    def on_tool_call_complete(self, tool_call: ToolCall, ordinal: int | None = None) -> None: ...


@runtime_checkable
class StreamedToolCallHandler(StreamHandler, Protocol):
    """Preferred extension carrying the complete stable event identity."""

    def on_streamed_tool_call(self, event: StreamedToolCall) -> None: ...


def provider_capabilities(provider: "LLMProvider") -> ProviderCapabilities:
    """Read optional capabilities, unwrapping gates and failing closed."""

    candidate: object = provider
    seen: set[int] = set()
    while id(candidate) not in seen:
        seen.add(id(candidate))
        raw = getattr(candidate, "capabilities", None)
        if callable(raw):
            try:
                value = raw()
            except Exception:  # pragma: no cover - defensive third-party boundary
                value = None
        else:
            value = raw
        if isinstance(value, ProviderCapabilities):
            return value
        inner = getattr(candidate, "inner", None)
        if inner is None:
            break
        candidate = inner
    return ProviderCapabilities(
        "terminal_only", "provider does not declare an explicit tool-call boundary"
    )


def notify_streamed_tool_call(
    stream: StreamHandler | None,
    event: StreamedToolCall,
) -> None:
    """Publish the rich event, adapting to the legacy callback when necessary."""

    callback = getattr(stream, "on_streamed_tool_call", None)
    if callable(callback):
        callback(event)
        return
    notify_tool_call_complete(stream, event.tool_call, event.ordinal)


def notify_tool_call_complete(
    stream: StreamHandler | None,
    tool_call: ToolCall,
    ordinal: int | None = None,
) -> None:
    """Publish a finalized call and its stable model-output ordinal.

    The optional-argument compatibility path keeps third-party display sinks working;
    the core scheduler only grants speculative admission when an ordinal is present.
    """
    callback = getattr(stream, "on_tool_call_complete", None)
    if callable(callback):
        try:
            parameters = inspect.signature(callback).parameters.values()
            accepts_ordinal = any(
                item.name == "ordinal" or item.kind is inspect.Parameter.VAR_KEYWORD
                for item in parameters
            )
        except (TypeError, ValueError):
            accepts_ordinal = False
        if accepts_ordinal:
            callback(tool_call, ordinal=ordinal)
        else:
            callback(tool_call)


class LLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        config: ProviderConfig,
        stream: StreamHandler | None = None,
        should_cancel: Callable[[], bool] | None = None,
        scope: ExecutionScope | None = None,
    ) -> LLMResult:
        """Return the next assistant response.

        When ``stream`` is given and the provider supports it, token deltas are
        pushed to the handler as they arrive; the returned ``LLMResult`` is the
        same fully-assembled result either way.

        ``should_cancel`` is the loop's cooperative-cancel probe (e.g. the user
        pressing Esc). A streaming provider should poll it as deltas arrive and
        raise ``asyncio.CancelledError`` when it fires, so a long response can be
        interrupted promptly instead of only at the next turn boundary.

        Providers backed by a blocking SDK should wrap the blocking call as an
        internal detail — ``await asyncio.to_thread(self._blocking_call, ...)`` —
        so the event loop keeps breathing; providers with a native async transport
        (see ``ClaudeProvider``) run real concurrent requests over one pool.
        """


class _TokenBucket:
    """Async token-bucket rate limiter shared across concurrent ``complete`` calls.

    ``rate_per_min`` of ``0`` disables limiting entirely. Tokens refill continuously
    at ``rate_per_min / 60`` per second up to a burst capacity of roughly one second's
    worth of requests, so a brief burst passes freely while the sustained rate is
    capped — which is exactly the pressure the higher API concurrency introduces.
    """

    def __init__(self, rate_per_min: float) -> None:
        self.rate_per_sec = max(0.0, rate_per_min) / 60.0
        self.capacity = max(1.0, self.rate_per_sec)
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, scope: ExecutionScope | None = None) -> None:
        if self.rate_per_sec <= 0:
            return
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate_per_sec)
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self.rate_per_sec
                if scope is None:
                    await asyncio.sleep(wait)
                else:
                    await scope.sleep(wait)


class ProviderGate:
    """Shared, cancel-aware concurrency limiter for provider API calls.

    Bounds how many ``complete`` calls are in flight at once (semaphore) and how
    fast they may be issued (token bucket). One gate is created at the top-level
    agent and reused by every child via :func:`gated_provider`, so the whole
    multi-agent fan-out shares a single budget.

    The asyncio primitives are created lazily on first use so they bind to the
    running event loop rather than whatever loop (if any) existed at construction.
    """

    def __init__(self, max_concurrency: int = 8, rate_limit: float = 0) -> None:
        self.max_concurrency = max(1, int(max_concurrency))
        self.rate_limit = max(0.0, float(rate_limit))
        self._semaphore: asyncio.Semaphore | None = None
        self._bucket: _TokenBucket | None = None

    def _ensure(self) -> tuple[asyncio.Semaphore, _TokenBucket]:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        if self._bucket is None:
            self._bucket = _TokenBucket(self.rate_limit)
        return self._semaphore, self._bucket

    @asynccontextmanager
    async def attempt(self, scope: ExecutionScope | None = None):
        """Acquire limits for one physical provider request, not a retry loop."""

        semaphore, bucket = self._ensure()
        if scope is None:
            await semaphore.acquire()
        else:
            scope.raise_if_cancelled()
            await scope.run_awaitable(semaphore.acquire())
        try:
            await bucket.acquire(scope)
            if scope is not None:
                scope.raise_if_cancelled()
            yield
        finally:
            semaphore.release()


@asynccontextmanager
async def provider_attempt(scope: ExecutionScope | None):
    """Bound one network attempt using the gate carried by ``scope``."""

    gate = scope.provider_gate if scope is not None else None
    if isinstance(gate, ProviderGate):
        async with gate.attempt(scope):
            yield
        return
    if scope is not None:
        scope.raise_if_cancelled()
    yield


def _supports_scope(provider: object) -> bool:
    try:
        parameters = inspect.signature(getattr(provider, "complete")).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        item.name == "scope" or item.kind is inspect.Parameter.VAR_KEYWORD
        for item in parameters
    )


class GatedProvider(LLMProvider):
    """Wrap a provider so concurrent children share one bounded API-call budget.

    For scope-aware providers (``complete`` accepts ``scope``) the gate travels
    through the scope and is acquired per physical attempt inside the provider's
    own retry loop, so the slot is released during retry backoff. Legacy
    providers are opaque and cannot be split that way: as a documented
    limitation they hold the gate slot for the entire call envelope, internal
    retries included. Either way the call is hard-bounded by the scope's
    remaining wall budget, so N concurrent children run up to
    ``max_concurrency`` attempts at a time instead of one.
    """

    def __init__(self, inner: LLMProvider, gate: ProviderGate | None = None) -> None:
        self.inner = inner
        self.gate = gate or ProviderGate()

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        config: ProviderConfig,
        stream: StreamHandler | None = None,
        should_cancel: Callable[[], bool] | None = None,
        scope: ExecutionScope | None = None,
    ) -> LLMResult:
        if should_cancel is not None and should_cancel():
            raise asyncio.CancelledError("provider call cancelled before start")
        scope = scope or current_execution_scope()
        call_scope = scope.with_provider_gate(self.gate) if scope is not None else None
        kwargs: dict[str, Any] = {}
        if should_cancel is not None:
            kwargs["should_cancel"] = should_cancel
        if _supports_scope(self.inner):
            kwargs["scope"] = call_scope
            result = self.inner.complete(messages, tools, config, stream, **kwargs)
            return await call_scope.run_awaitable(result) if call_scope is not None else await result

        # Compatibility path: a legacy provider is one opaque physical attempt, so
        # it holds the slot for the whole envelope (documented limitation); the
        # run_awaitable wrap still hard-bounds it by the scope's wall budget.
        async with self.gate.attempt(call_scope):
            result = self.inner.complete(messages, tools, config, stream, **kwargs)
            return await call_scope.run_awaitable(result) if call_scope is not None else await result


def gated_provider(
    provider: LLMProvider,
    *,
    max_concurrency: int = 8,
    rate_limit: float = 0,
) -> LLMProvider:
    """Wrap ``provider`` in a shared :class:`GatedProvider`, idempotently.

    A provider that is already gated is returned unchanged, so children spawned with
    ``provider=self.provider`` reuse the leader's single gate (and its budget). The
    ``max_concurrency`` / ``rate_limit`` knobs therefore take effect only at the
    top-level agent, which is exactly where the gate is first created.
    """
    if isinstance(provider, GatedProvider):
        return provider
    return GatedProvider(provider, ProviderGate(max_concurrency=max_concurrency, rate_limit=rate_limit))
