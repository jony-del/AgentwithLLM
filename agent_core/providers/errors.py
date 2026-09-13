"""Shared HTTP error classification for LLM providers.

Context-overflow detection must be precise. A bare substring like ``"token"`` or
``"context"`` misclassifies ``max_tokens exceeds the model limit`` (a completion
budget error) as a context overflow, triggering a needless prompt-truncation
recovery. All providers therefore classify overflow through the same markers.

HTTP 413 from a chat endpoint is treated as overflow unconditionally: a request
body the server refuses on size is, for an LLM prompt, the context-overflow
condition, and shrinking the conversation (reactive compaction) is the right
recovery regardless of how the proxy worded the body.
"""

CONTEXT_OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "too many tokens",
    "prompt is too long",
    "input length and `max_tokens` exceed context limit",
)


def is_context_overflow(status_code: int, body_text: str) -> bool:
    """True when the error means the prompt/context does not fit the model."""

    if status_code == 413:
        return True
    if status_code != 400:
        return False
    lowered = body_text.lower()
    return any(marker in lowered for marker in CONTEXT_OVERFLOW_MARKERS)
