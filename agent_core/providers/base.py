from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from agent_core.models import LLMResult, Message


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


class LLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        config: dict[str, Any],
        stream: StreamHandler | None = None,
        should_cancel: Callable[[], bool] | None = None,
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

    async def acquire(self) -> None:
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
                await asyncio.sleep(wait)


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


class GatedProvider(LLMProvider):
    """Wrap a provider so concurrent children share one bounded API-call budget.

    ``complete`` acquires the shared semaphore and rate-limit token before issuing
    the call, so N concurrent children run up to ``max_concurrency`` at a time
    instead of one.
    """

    def __init__(self, inner: LLMProvider, gate: ProviderGate | None = None) -> None:
        self.inner = inner
        self.gate = gate or ProviderGate()

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        config: dict[str, Any],
        stream: StreamHandler | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> LLMResult:
        if should_cancel is not None and should_cancel():
            raise asyncio.CancelledError("provider call cancelled before start")
        semaphore, bucket = self.gate._ensure()
        async with semaphore:
            if should_cancel is not None and should_cancel():
                raise asyncio.CancelledError("provider call cancelled before start")
            await bucket.acquire()
            # Forward the cancel probe so a streaming provider can poll it as
            # deltas arrive (Esc interrupts mid-response, not just at turn
            # boundaries). Only pass it through when present, so minimal providers
            # whose ``complete`` omits the kwarg keep working unchanged.
            if should_cancel is not None:
                return await self.inner.complete(
                    messages, tools, config, stream, should_cancel=should_cancel
                )
            return await self.inner.complete(messages, tools, config, stream)


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
