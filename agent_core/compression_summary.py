"""LLM summary seam for Track A compaction.

The :class:`CompressionPipeline` must not import a provider (CLAUDE.md): it only
awaits an opaque :data:`~agent_core.compression.Summarizer` callback. This module is
the seam that builds that callback from a provider, mirroring the project's other
injected-closure patterns (e.g. ``session.subagent_factory``). It renders the
prefix to a single user turn, asks the model for a structured ``<summary>`` with no
tools, and parses the result — borrowing the prompt shape from Claude Code's
``services/compact/prompt.ts`` without porting its code.
"""

from __future__ import annotations

import re
from typing import Any

from agent_core.compression import CompressionConfig, Summarizer
from agent_core.models import Message
from agent_core.providers.base import GatedProvider, LLMProvider
from agent_core.providers.fake import FakeProvider

# No-tools preamble + the section list the summary must produce. The model replies
# with TEXT ONLY; an optional <analysis> scratchpad is stripped before the summary is
# reinserted, so only the <summary> body survives into the live context.
SUMMARY_SYSTEM = (
    "You are compacting a long agent conversation so it can continue within a smaller "
    "context window.\n"
    "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools. Do NOT ask questions. "
    "Produce only the summary described below.\n\n"
    "Optionally think first inside a single <analysis>...</analysis> block (it will be "
    "discarded). Then emit the summary inside <summary>...</summary> covering, in order:\n"
    "1. Primary request and intent — what the user is ultimately trying to achieve.\n"
    "2. Key technical concepts, technologies, and conventions in play.\n"
    "3. Files and code sections touched, with the relevant snippets/edits.\n"
    "4. Errors encountered and how they were fixed.\n"
    "5. Problem solving done and decisions made.\n"
    "6. All explicit user instructions and constraints (do not drop any).\n"
    "7. Pending tasks and the current in-progress work.\n"
    "8. The most likely next step, with a verbatim quote of the latest instruction if any.\n"
    "Be precise and information-dense; preserve identifiers, paths, and exact values. "
    "Omit pleasantries."
)


def render_prefix(prefix: list[Message], max_chars: int) -> str:
    """Render the to-be-folded prefix as one plain-text transcript for the summarizer.

    Hard-capped at ``max_chars`` (head + tail kept) so the summary call itself can't
    overflow — the prefix has already been snipped/microcompacted, but a long run can
    still exceed a comfortable single-call budget.
    """
    lines = [f"[{message.role}] {message.content}" for message in prefix]
    convo = "\n".join(lines)
    if max_chars > 0 and len(convo) > max_chars:
        half = max(200, max_chars // 2)
        convo = f"{convo[:half]}\n...[transcript truncated]...\n{convo[-half:]}"
    return convo


_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)
_ANALYSIS_RE = re.compile(r"<analysis>.*?</analysis>", re.DOTALL | re.IGNORECASE)


def extract_summary(text: str) -> str:
    """Pull the ``<summary>`` body out of the model reply; tolerate a missing tag.

    Strips any ``<analysis>`` scratchpad. If the model didn't wrap its answer in
    ``<summary>`` tags we fall back to the whole (analysis-stripped) text so a
    well-meaning but unformatted reply is still usable rather than discarded.
    """
    match = _SUMMARY_RE.search(text)
    if match:
        return match.group(1).strip()
    return _ANALYSIS_RE.sub("", text).strip()


def _unwrap(provider: LLMProvider) -> LLMProvider:
    """Peel the shared concurrency gate so we can inspect the concrete provider."""
    return provider.inner if isinstance(provider, GatedProvider) else provider


def build_summarizer(
    provider: LLMProvider,
    provider_config: dict[str, Any],
    config: CompressionConfig,
) -> Summarizer | None:
    """Build the Track A summarizer, or ``None`` to force deterministic Track B.

    Returns ``None`` when LLM summary is disabled or the provider is the deterministic
    :class:`FakeProvider` (no real key), so offline/test runs stay byte-stable on
    Track B. The returned callback issues a no-tools, non-streaming completion through
    the *gated* provider so it shares the fan-out's API budget.
    """
    if not config.use_llm_summary:
        return None
    if isinstance(_unwrap(provider), FakeProvider):
        return None

    async def summarize(prefix: list[Message]) -> str:
        convo = render_prefix(prefix, config.summary_input_max_chars)
        messages = [Message("system", SUMMARY_SYSTEM), Message("user", convo)]
        # Override the live-run knobs: a bounded summary, no streaming, no thinking,
        # and (via the empty tools list at the call site) no tool use.
        summary_config = {
            **provider_config,
            "max_tokens": config.summary_max_tokens,
            "stream": False,
            "thinking_budget": None,
        }
        result = await provider.complete(messages, [], summary_config)
        return extract_summary(result.content)

    return summarize
