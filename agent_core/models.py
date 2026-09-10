from __future__ import annotations

import uuid as _uuid
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
    """Frozen identity contract (schema v4):

    - ``uuid``: instance identity within one transcript chain. Stable across
      compression-derived copies; regenerated only when absent on load.
    - ``origin_id``: logical identity across compressions; points at the first
      version of the message (defaults to ``uuid``).
    - ``version``: content version, bumped by ``_evolve_message`` on every
      compression derivation.
    - ``round_id``: provider tool-round identity (defaults to ``uuid`` for
      assistant messages carrying ``tool_calls``).

    Provider serialization pairs tool calls purely via ``metadata``
    (``tool_calls`` / ``tool_call_id``); the identity fields above belong to the
    persistence/recovery layer only. ``compare=False`` keeps equality based on
    role/content/name/metadata, so two messages with the same content but
    distinct ids still compare equal.
    """

    role: Role
    content: str
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Identity for transcript persistence / resume: each message carries a stable
    # ``uuid`` and points at its predecessor via ``parent_uuid``, forming the message
    # tree the reference project stores per session. Appended last with defaults so all
    # existing positional ``Message(role, content, ...)`` construction stays valid, and
    # providers (which only read role/content/metadata/name) are unaffected.
    uuid: str = field(default_factory=lambda: _uuid.uuid4().hex, compare=False)
    parent_uuid: str | None = field(default=None, compare=False)
    origin_id: str | None = field(default=None, compare=False)
    version: int = field(default=1, compare=False)
    round_id: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.origin_id is None:
            self.origin_id = self.uuid
        self.version = max(1, int(self.version))
        if self.round_id is None and self.role == "assistant" and self.metadata.get("tool_calls"):
            self.round_id = self.uuid

    @property
    def message_id(self) -> str:
        return self.uuid

    @property
    def parent_id(self) -> str | None:
        return self.parent_uuid

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "uuid": self.uuid,
            "parent_uuid": self.parent_uuid,
            "message_id": self.uuid,
            "parent_id": self.parent_uuid,
            "origin_id": self.origin_id,
            "version": self.version,
            "round_id": self.round_id,
        }
        if self.name:
            data["name"] = self.name
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Rebuild a ``Message`` from its ``to_dict`` form (transcript load path).

        ``metadata`` round-trips verbatim — that is what preserves ``thinking_blocks``
        (Claude's thinking invariant) and ``tool_calls`` / ``tool_call_id`` (tool-result
        correlation) across a resume. ``uuid`` is regenerated only when absent so older
        records without identity still load.
        """
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            name=data.get("name"),
            metadata=dict(data.get("metadata") or {}),
            uuid=data.get("message_id") or data.get("uuid") or _uuid.uuid4().hex,
            parent_uuid=data.get("parent_id", data.get("parent_uuid")),
            origin_id=data.get("origin_id"),
            version=data.get("version", 1),
            round_id=data.get("round_id"),
        )


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
class TokenUsage:
    """Per-response token accounting reported by a provider.

    Mirrors the Anthropic ``usage`` object. ``context_tokens`` is the total token
    footprint of the *request that was sent* (the non-cached prompt plus whatever was
    read from / written to the prompt cache) — that running figure is what context
    compaction thresholds against, not the freshly generated output.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def context_tokens(self) -> int:
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens

    @property
    def total_tokens(self) -> int:
        """Full per-response footprint: the sent prompt (incl. cache) plus the output.

        Mirrors the reference ``getTokenCountFromUsage``. This is the value anchored on
        an assistant turn for the next gate estimate — once the turn is in history, its
        generated output counts toward the prompt the next request will carry.
        """
        return self.context_tokens + self.output_tokens


@dataclass(slots=True)
class LLMResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    # Human-readable extended-thinking text for display.
    thinking: str = ""
    # PROVIDER-OWNED OPAQUE DATA (E8): the raw thinking / redacted_thinking blocks
    # (with their signatures), preserved and round-tripped verbatim across turns —
    # e.g. the Anthropic API requires the prior turn's thinking block when thinking
    # and tool use span multiple turns. The core loop and other providers must never
    # interpret, edit, or depend on the contents; a provider with no equivalent
    # simply leaves it empty.
    thinking_blocks: list[dict[str, Any]] = field(default_factory=list)
    # Token accounting for this response, when the provider reports it. Appended after
    # thinking blocks so existing positional ``LLMResult(...)`` construction stays valid.
    # Compaction reads ``usage.context_tokens`` as the running prompt size.
    usage: "TokenUsage | None" = None
    # PROVIDER-OWNED OPAQUE DATA: generic JSON-serializable state a provider needs to
    # replay future turns (e.g. Responses API output items). The core persists it but
    # never interprets it. Prefer this for new provider state; ``thinking_blocks`` stays
    # for the established Anthropic thinking invariant.
    provider_state: dict[str, Any] = field(default_factory=dict)
    # Proof that the provider reached an authoritative protocol boundary.  The
    # compatibility default is true because legacy/non-streaming providers return
    # only after their terminal response has arrived.  Streaming providers must set
    # this false until their protocol-specific terminal event is observed.
    termination_proven: bool = True
    termination_event: str | None = None


class LLMContextTooLongError(RuntimeError):
    """Raised when a provider rejects a request because the context is too long."""


class LLMTransientError(RuntimeError):
    """Raised when a provider fails on a transient fault (network/SSL/timeout or a
    retryable server status) that survived the provider's own retries. Callers may
    surface it and let the user try again instead of crashing the session."""


class ToolExecutionError(RuntimeError):
    """Raised when a tool cannot be executed."""

