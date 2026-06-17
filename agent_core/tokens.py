from __future__ import annotations

import math
import os

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
