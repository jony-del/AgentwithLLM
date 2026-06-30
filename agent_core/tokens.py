from __future__ import annotations

import math
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid an import cycle (models is import-light too, but stay defensive)
    from agent_core.models import Message

"""Pure-stdlib token math for context-window accounting and the auto-compact gate.

Ports the *logic* of the reference implementation's ``autoCompact.ts`` /
``context.ts`` (Open-ClaudeCode) so the compaction layer can decide, from a running
prompt token count, whether to fold history before the next request.

Kept dependency-free on purpose: no provider / compression imports, nothing heavy.
``import agent_core`` must stay import-light, and this module sits below both
providers and compression to avoid cycles.
"""

# Reserve this many tokens of headroom below the effective window before
# auto-compaction fires. Mirrors ``AUTOCOMPACT_BUFFER_TOKENS``.
AUTOCOMPACT_BUFFER_TOKENS = 13_000

# Tokens held back from the context window to leave room for the compaction
# summary's *output*. Mirrors ``MAX_OUTPUT_TOKENS_FOR_SUMMARY`` (reference reserves
# 20k based on p99.99 of observed summary sizes).
MAX_OUTPUT_TOKENS_FOR_SUMMARY = 20_000

# Conservative default context window for an unknown / unrecognised model.
MODEL_CONTEXT_WINDOW_DEFAULT = 200_000

# Substrings of model ids that ship a native 1M-token context window (per
# shared/models.md). Matched as substrings so suffixes / ``[1m]`` tags don't defeat it.
ONE_MILLION_WINDOW_MARKERS = (
    "opus-4-6",
    "opus-4-7",
    "opus-4-8",
    "sonnet-4-6",
    "fable-5",
    "mythos-5",
    "mythos-preview",
)

# Default per-request output ceiling we assume for an unknown model. The reference caps
# the native default to 8k for slot-reservation reasons; we keep that conservative value
# so ``effective_context_window`` reserves a realistic output slice for unrecognised ids.
MAX_OUTPUT_TOKENS_DEFAULT = 8_192

# Per-model output ceilings: ``marker -> (default, upper)`` (mirrors the reference
# ``getModelMaxOutputTokens``). ``default`` is the steady-state per-request ceiling;
# ``upper`` is the hard max the model supports — the top tier the compaction summary can
# escalate to (see compression.py). Matched as substrings; first hit wins, so order most
# specific first.
MODEL_OUTPUT_TOKENS: tuple[tuple[str, int, int], ...] = (
    ("opus-4-6", 64_000, 128_000),
    ("opus-4-7", 64_000, 128_000),
    ("opus-4-8", 64_000, 128_000),
    ("fable-5", 64_000, 128_000),
    ("mythos-5", 64_000, 128_000),
    ("mythos-preview", 64_000, 128_000),
    ("sonnet-4-6", 32_000, 128_000),
    ("haiku-4-5", 32_000, 64_000),
)

# Env override (percent of the effective window, 0 < p <= 100) that lowers the
# auto-compact threshold for easier testing. Mirrors ``CLAUDE_AUTOCOMPACT_PCT_OVERRIDE``.
AUTOCOMPACT_PCT_OVERRIDE_ENV = "AGENT_AUTOCOMPACT_PCT_OVERRIDE"

# Average characters per token for the cheap char-based fallback estimate. Mirrors the
# reference ``roughTokenCountEstimation`` default; ~4 chars/token holds for English/code.
ROUGH_BYTES_PER_TOKEN = 4


def rough_token_estimate(text: str, *, bytes_per_token: int = ROUGH_BYTES_PER_TOKEN) -> int:
    """Cheap char-based token estimate (``len // bytes_per_token``).

    The single source of truth for the ``//4`` heuristic that used to be duplicated as
    inline lambdas across the compaction layer and providers. Used only as a fallback /
    mid-turn delta estimate — real API ``usage`` is preferred whenever it is available.
    """
    if not text:
        return 0
    return len(text) // max(1, bytes_per_token)


def rough_token_estimate_for_messages(messages: Iterable["Message"]) -> int:
    """Rough token estimate over a sequence of messages (sum of per-content estimates).

    Mirrors ``roughTokenCountEstimationForMessages``. Sums ``len(content) // 4`` over the
    messages — used both as the no-usage fallback and to estimate the messages added
    since the last anchored API response.
    """
    return sum(rough_token_estimate(m.content) for m in messages)


def context_window_for_model(model: str) -> int:
    """Return the model's total context window in tokens.

    Models in :data:`ONE_MILLION_WINDOW_MARKERS` (Opus 4.6/4.7/4.8, Sonnet 4.6, Fable/
    Mythos 5) ship a native 1M window and are reported as such; an explicit ``[1m]`` tag
    forces 1M for any id; everything else falls back to the conservative 200k default
    (so auto-compaction still fires on genuine 200k models like Haiku 4.5). Configure
    ``context_window_tokens`` to override for a smaller deployment.

    NOTE: this is a deliberate divergence from the reference, which kept 1M behind a beta
    flag and defaulted every ``claude-*`` id to 200k. The project targets Opus 4.8, whose
    real window is 1M, so the known 1M family is recognised natively.
    """
    name = (model or "").lower()
    if "[1m]" in name:
        return 1_000_000
    if any(marker in name for marker in ONE_MILLION_WINDOW_MARKERS):
        return 1_000_000
    return MODEL_CONTEXT_WINDOW_DEFAULT


def model_output_tokens(model: str) -> tuple[int, int]:
    """Return ``(default, upper)`` output-token ceilings for a model.

    ``default`` is the steady-state per-request output ceiling; ``upper`` is the hard
    maximum the model supports (the top tier compaction may escalate to). Unknown ids
    get the conservative ``MAX_OUTPUT_TOKENS_DEFAULT`` for both.
    """
    name = (model or "").lower()
    for marker, default, upper in MODEL_OUTPUT_TOKENS:
        if marker in name:
            return default, upper
    return MAX_OUTPUT_TOKENS_DEFAULT, MAX_OUTPUT_TOKENS_DEFAULT


def is_supported_model(model: str) -> bool:
    """True when ``model`` belongs to a known model family.

    The known families are exactly the markers in :data:`MODEL_OUTPUT_TOKENS` (Opus
    4.6/4.7/4.8, Fable/Mythos 5, Sonnet 4.6, Haiku 4.5), matched as substrings so
    suffixes / ``[1m]`` tags don't defeat the check — same semantics as
    :func:`context_window_for_model` and :func:`model_output_tokens`.

    Used to validate a caller-supplied (LLM-driven) model id before spawning a
    sub-agent/teammate with it: an unrecognised id would silently fall back to the
    conservative 200k window here AND be sent verbatim to the provider (a likely API
    404), so it's rejected up front instead.
    """
    name = (model or "").lower()
    return any(marker in name for marker, _default, _upper in MODEL_OUTPUT_TOKENS)


def max_output_tokens_for_model(model: str) -> int:
    """Return the assumed steady-state per-request output ceiling for a model.

    Used by the window math to reserve an output slice; the reserve is itself clamped
    against ``MAX_OUTPUT_TOKENS_FOR_SUMMARY`` in :func:`effective_context_window`.
    """
    return model_output_tokens(model)[0]


def effective_context_window(
    model: str,
    *,
    context_window_override: int | None = None,
    reserved_output_tokens: int = MAX_OUTPUT_TOKENS_FOR_SUMMARY,
) -> int:
    """Window minus the tokens reserved for (summary) output.

    Mirrors ``getEffectiveContextWindowSize``: ``window - min(maxOutput, reserved)``.
    A positive ``context_window_override`` caps the window (e.g. to model a smaller
    deployment), exactly like the reference's ``CLAUDE_CODE_AUTO_COMPACT_WINDOW``.
    """
    window = context_window_for_model(model)
    if context_window_override is not None and context_window_override > 0:
        window = min(window, context_window_override)
    reserved = min(max_output_tokens_for_model(model), reserved_output_tokens)
    return window - reserved


def resolve_pct_override(explicit: float | None = None) -> float | None:
    """Resolve the auto-compact percent override: explicit arg wins, else env.

    Returns a float in ``(0, 100]`` or ``None`` when unset/invalid. Kept as a thin
    helper so ``auto_compact_threshold`` stays a pure function of its params and the
    env read is testable in isolation.
    """
    if explicit is not None:
        return explicit if 0 < explicit <= 100 else None
    raw = os.getenv(AUTOCOMPACT_PCT_OVERRIDE_ENV)
    if not raw:
        return None
    try:
        parsed = float(raw)
    except ValueError:
        return None
    return parsed if 0 < parsed <= 100 else None


def auto_compact_threshold(
    model: str,
    *,
    context_window_override: int | None = None,
    buffer_tokens: int = AUTOCOMPACT_BUFFER_TOKENS,
    reserved_output_tokens: int = MAX_OUTPUT_TOKENS_FOR_SUMMARY,
    pct_override: float | None = None,
) -> int:
    """Token count at/above which history should be auto-compacted.

    Base threshold is ``effective_context_window - buffer_tokens`` (mirrors
    ``getAutoCompactThreshold``). When a percent override is supplied (param) or set
    via ``AGENT_AUTOCOMPACT_PCT_OVERRIDE`` (env), the threshold becomes
    ``min(floor(effective * pct/100), base)`` — the percent can only *lower* the
    threshold, never raise it past the buffer-derived ceiling.
    """
    effective = effective_context_window(
        model,
        context_window_override=context_window_override,
        reserved_output_tokens=reserved_output_tokens,
    )
    base = effective - buffer_tokens

    pct = resolve_pct_override(pct_override)
    if pct is not None:
        percentage_threshold = math.floor(effective * (pct / 100))
        return min(percentage_threshold, base)
    return base
