from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


Role = Literal["system", "user", "assistant", "tool"]


class ToolRisk(str, Enum):
    READ = "read"
    WRITE = "write"
    DANGEROUS = "dangerous"


@dataclass(slots=True)
class Message:
    role: Role
    content: str
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
        }
        if self.name:
            data["name"] = self.name
        return data


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass(slots=True)
class ToolResult:
    name: str
    content: str
    ok: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LLMResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    # Human-readable extended-thinking text for display.
    thinking: str = ""
    # The raw thinking / redacted_thinking blocks (with their signatures), kept so
    # they can be replayed verbatim on later turns — the Anthropic API requires the
    # prior turn's thinking block when thinking and tool use span multiple turns.
    thinking_blocks: list[dict[str, Any]] = field(default_factory=list)


class LLMContextTooLongError(RuntimeError):
    """Raised when a provider rejects a request because the context is too long."""


class LLMTransientError(RuntimeError):
    """Raised when a provider fails on a transient fault (network/SSL/timeout or a
    retryable server status) that survived the provider's own retries. Callers may
    surface it and let the user try again instead of crashing the session."""


class ToolExecutionError(RuntimeError):
    """Raised when a tool cannot be executed."""

