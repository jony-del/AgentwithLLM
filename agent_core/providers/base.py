from __future__ import annotations

import threading
from abc import ABC, abstractmethod
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
    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        config: dict[str, Any],
        stream: StreamHandler | None = None,
    ) -> LLMResult:
        """Return the next assistant response.

        When ``stream`` is given and the provider supports it, token deltas are
        pushed to the handler as they arrive; the returned ``LLMResult`` is the
        same fully-assembled result either way.
        """


class SynchronizedProvider(LLMProvider):
    """Serialize access to a provider instance shared across parent/child agents."""

    def __init__(self, inner: LLMProvider, lock: threading.RLock | None = None) -> None:
        self.inner = inner
        self._lock = lock or threading.RLock()

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        config: dict[str, Any],
        stream: StreamHandler | None = None,
    ) -> LLMResult:
        with self._lock:
            return self.inner.complete(messages, tools, config, stream=stream)


def synchronized_provider(provider: LLMProvider) -> LLMProvider:
    if isinstance(provider, SynchronizedProvider):
        return provider
    return SynchronizedProvider(provider)
