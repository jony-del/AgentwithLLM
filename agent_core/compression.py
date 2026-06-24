from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from math import floor

from agent_core import tokens
from agent_core.models import Message

# Called after each compaction stage so a UI can drive a progress bar:
# (completed_stages, total_stages, event).
StageReporter = Callable[[int, int, "CompressionEvent"], None]

# Injected async callback that folds an old message prefix into a single summary
# string (Track A). It lives here — with the pipeline that consumes it — so the
# seam module (compression_summary) can depend on the pipeline without a cycle. The
# pipeline never imports a provider; it only awaits this opaque callback.
Summarizer = Callable[[list[Message]], Awaitable[str]]

# Injected sync callback that estimates the token footprint of a message list for the
# auto-compact gate. Mirrors the ``summarizer`` injection seam: the real agent passes
# ``max(last_usage, char/4)``; offline/test runs default to ``char_count // 4`` so the
# gate stays deterministic for FakeProvider.
TokenEstimator = Callable[[list[Message]], int]

# Markers stamped on a folded summary block (either track). A prior summary is itself
# foldable on the next pass, so repeated compactions re-summarize rather than pile up.
_COLLAPSE_MARKERS = frozenset({"context_collapse", "llm_summary"})

# Re-injection wrapper for a folded summary (mirrors ``getCompactUserSummaryMessage``).
# The summary is re-injected as a USER message, not a system message: the model reads
# it as continued conversation context, not as authoritative instructions.
_SUMMARY_WRAPPER_HEADER = (
    "This session is being continued from a previous conversation that ran out of "
    "context. The summary below covers the earlier portion of the conversation."
)
_SUMMARY_WRAPPER_FOOTER = (
    "Continue the conversation from where it left off without asking the user any "
    "further questions. Resume directly — do not acknowledge the summary, do not "
    "recap, do not preface. Pick up the last task as if the break never happened."
)


def build_summary_user_message(summary: str, *, marker: str, messages_collapsed: int) -> Message:
    """Wrap a folded ``summary`` body in the continuation USER message (both tracks).

    ``marker`` is the collapse marker (``"llm_summary"`` or ``"context_collapse"``) so
    the block stays re-foldable on the next pass via ``_COLLAPSE_MARKERS``. The message
    is a USER turn carrying the "This session is being continued…" framing — mirrors
    the reference ``getCompactUserSummaryMessage`` + ``suppressFollowUpQuestions`` path.
    """
    body = f"{_SUMMARY_WRAPPER_HEADER}\n\n{summary.strip()}\n\n{_SUMMARY_WRAPPER_FOOTER}"
    metadata: dict[str, object] = {
        "compressed": marker,
        "messages_collapsed": messages_collapsed,
    }
    if marker == "llm_summary":
        metadata["is_compact_summary"] = True
    return Message("user", body, metadata=metadata)


def build_post_compact_messages(
    summary_message: Message,
    recent: list[Message],
    *,
    attachments: list[Message] | None = None,
) -> list[Message]:
    """Assemble the post-compaction conversation tail: ``[summary, *recent, *attachments]``.

    Mirrors the reference ``buildPostCompactMessages`` ordering. ``attachments`` is the
    Phase 3E seam for re-injected file/tool context — accepted here so E can append
    without churning this signature. The system/pinned prefix is prepended by the
    caller (``_context_collapse``), not by this helper.
    """
    tail = [summary_message, *recent]
    if attachments:
        tail.extend(attachments)
    return tail


def group_into_rounds(messages: list[Message]) -> list[list[Message]]:
    """Group messages into API rounds, keeping a tool_call glued to its tool_results.

    An ``assistant`` message carrying ``metadata["tool_calls"]`` forms one inseparable
    group with the immediately-following ``tool`` messages whose
    ``metadata["tool_call_id"]`` matches one of that assistant's calls. Every other
    message is its own singleton group.

    Invariant: a tool_call is never separated from its tool_result, and the groups
    concatenate back to the original list in order. Pure/deterministic — Phase 3D's
    head-truncation reuses this to drop whole rounds without splitting a round.
    """
    rounds: list[list[Message]] = []
    index = 0
    total = len(messages)
    while index < total:
        message = messages[index]
        call_ids = _assistant_call_ids(message)
        if call_ids:
            group = [message]
            cursor = index + 1
            # Glue every immediately-following tool result whose id belongs to this
            # assistant's calls. Stop at the first message that isn't such a result so
            # an unrelated/interleaved message starts a fresh round.
            while cursor < total:
                following = messages[cursor]
                if following.role == "tool" and following.metadata.get("tool_call_id") in call_ids:
                    group.append(following)
                    cursor += 1
                    continue
                break
            rounds.append(group)
            index = cursor
        else:
            rounds.append([message])
            index += 1
    return rounds


def _assistant_call_ids(message: Message) -> set[str]:
    """Return the set of tool_call ids an assistant message issued (empty if none)."""
    if message.role != "assistant":
        return set()
    raw_calls = message.metadata.get("tool_calls")
    if not raw_calls:
        return set()
    ids: set[str] = set()
    for call in raw_calls:
        call_id = call.get("id") if isinstance(call, dict) else None
        if call_id is not None:
            ids.add(call_id)
    return ids


def split_on_round_boundary(
    conversation: list[Message], keep: int
) -> tuple[list[Message], list[Message]]:
    """Split ``conversation`` into ``(prefix, recent)`` snapped to a round boundary.

    ``keep`` is an approximate recent-message budget; the actual split point is moved
    *earlier* (more kept) so ``recent`` always starts at the head of a round — never on
    an orphan ``tool`` result, and never leaving an assistant-with-tool_calls as the
    last prefix element without its results. Deterministic.
    """
    if keep <= 0 or keep >= len(conversation):
        if keep >= len(conversation):
            return [], list(conversation)
        return list(conversation), []
    rounds = group_into_rounds(conversation)
    # Walk rounds from the end, accumulating until we've kept at least ``keep`` messages,
    # then cut on that round boundary.
    recent: list[Message] = []
    boundary = len(rounds)
    for boundary in range(len(rounds) - 1, -1, -1):
        recent = [m for group in rounds[boundary:] for m in group]
        if len(recent) >= keep:
            break
    prefix = [m for group in rounds[:boundary] for m in group]
    return prefix, recent


def is_preserved(message: Message) -> bool:
    """A message is preserved (never folded/dropped) if it is a non-collapse system block
    or is pinned (any role).

    This is the single source of truth for the preservation predicate, shared by
    ``_context_collapse`` (which folds the rest) and ``truncate_head_for_ptl_retry``
    (which drops the rest). The preserved set is the system/pinned front: base
    system+gitStatus, memory recall, and the pinned ``userContext`` USER message.

    Prior summary blocks (carrying a ``_COLLAPSE_MARKERS`` marker) are NOT preserved —
    they are re-foldable conversation even when carried as a USER message.
    """
    if message.metadata.get("pinned"):
        return True
    return message.role == "system" and message.metadata.get("compressed") not in _COLLAPSE_MARKERS


def is_summary_message(message: Message) -> bool:
    """True for a folded-summary block produced by ``build_summary_user_message`` (either
    track). The single source of truth for "is this the message a context_collapse fold
    just produced" — used by the transcript layer to find the new summary at a compaction
    boundary without duplicating the marker set.
    """
    return message.metadata.get("compressed") in _COLLAPSE_MARKERS


# Regex for the Anthropic "prompt is too long" 413/400 error body (case-insensitive,
# lenient about SDK/JSON wrapping): ``prompt is too long: 219763 tokens > 200000``.
# Captures (actual, limit). Mirrors the reference ``parsePromptTooLongTokenCounts``.
_PTL_TOKEN_RE = re.compile(
    r"prompt is too long[^0-9]*(\d+)\s*tokens?\s*>\s*(\d+)",
    re.IGNORECASE,
)


def parse_prompt_too_long_gap(text: str) -> int | None:
    """Parse the token overflow gap from a "prompt is too long" error body.

    Returns ``actual - limit`` when both are present and the gap is positive, else
    ``None`` (unparseable, or a non-positive/reversed gap). Lenient about casing and
    surrounding SDK/JSON wrapping. Mirrors ``getPromptTooLongTokenGap`` in the reference.
    """
    match = _PTL_TOKEN_RE.search(text)
    if match is None:
        return None
    actual = int(match.group(1))
    limit = int(match.group(2))
    gap = actual - limit
    return gap if gap > 0 else None


# Synthetic user turn prepended after a head-truncation when the first non-system
# message would otherwise be an assistant/tool turn — the Anthropic Messages API
# requires the messages array (system is a separate param) to begin with a user turn.
# Our preserved front usually begins with the pinned ``userContext`` USER message, but
# that message is optional (no git AND no CLAUDE.md), so this is the safety net. Mirrors
# the reference ``PTL_RETRY_MARKER`` + ``ensureToolResultPairing`` re-prepend.
PTL_RETRY_MARKER = "[earlier conversation truncated to fit the context window]"


def _ensure_user_first(messages: list[Message]) -> list[Message]:
    """Insert the PTL marker before the first non-system message if it isn't a user turn.

    Leaves ``messages`` untouched when the first non-system message is already a user
    turn (the common case: a pinned ``userContext`` USER message leads the conversation).
    """
    for index, message in enumerate(messages):
        if message.role == "system":
            continue
        if message.role == "user":
            return messages
        marker = Message("user", PTL_RETRY_MARKER, metadata={"ptl_retry_marker": True})
        return [*messages[:index], marker, *messages[index:]]
    return messages


def truncate_head_for_ptl_retry(
    messages: list[Message],
    *,
    token_gap: int | None = None,
    drop_fraction: float = 0.2,
    token_estimator: Callable[[list[Message]], int] | None = None,
) -> list[Message] | None:
    """Drop the oldest whole API rounds so a 413-retry's prompt fits, or ``None``.

    Splits ``messages`` into the preserved system/pinned front (never dropped) and the
    foldable ``conversation``; groups the conversation into API rounds via
    ``group_into_rounds`` (so an assistant-with-tool_calls is never separated from its
    tool results). Returns ``None`` — caller gives up — when there are ``< 2`` rounds
    (nothing safe to drop) or the computed drop count is ``< 1``.

    Drop count: if ``token_gap`` is known (> 0), accumulate per-group token estimates
    (default ``char_count // 4``) until the accumulated estimate ``>= token_gap``,
    counting that many groups; otherwise ``max(1, floor(len(rounds) * drop_fraction))``.
    Capped at ``len(rounds) - 1`` so at least one round always survives.

    The returned list is ``[*preserved, *flatten(rounds[drop:])]``. Because our preserved
    front always begins with a system message and the first non-system entry is the
    pinned ``userContext`` USER message, dropping the oldest conversation rounds can
    never produce an assistant-first slice or an orphaned tool_result. That is why we
    don't need the reference's synthetic-marker re-prepend dance.
    """
    preserved = [m for m in messages if is_preserved(m)]
    conversation = [m for m in messages if not is_preserved(m)]
    rounds = group_into_rounds(conversation)
    if len(rounds) < 2:
        return None

    estimator = token_estimator or (lambda msgs: sum(len(m.content) for m in msgs) // 4)
    if token_gap is not None and token_gap > 0:
        drop = 0
        acc = 0
        for group in rounds:
            drop += 1
            acc += estimator(group)
            if acc >= token_gap:
                break
    else:
        drop = max(1, floor(len(rounds) * drop_fraction))

    drop = min(drop, len(rounds) - 1)
    if drop < 1:
        return None
    kept_conversation = [m for group in rounds[drop:] for m in group]
    return _ensure_user_first([*preserved, *kept_conversation])


def shrink_oversize_messages(
    messages: list[Message],
    *,
    tokens_to_drop: int,
    token_estimator: Callable[[list[Message]], int] | None = None,
    min_keep_chars: int = 200,
) -> list[Message] | None:
    """Last-resort fold when whole-round head-truncation can't help (a single oversized
    round or one giant message that alone exceeds the window).

    Head/tail-truncates the largest NON-preserved messages (each with an omission marker,
    keeping ``min_keep_chars`` from each end) until at least ``tokens_to_drop`` tokens
    (estimated, default ``char_count // 4``) have been shed. Returns a NEW list, or
    ``None`` when nothing further can be shrunk (so the caller can finally give up rather
    than spin). Preserved system/pinned messages are never touched. This guarantees the
    413 retry loop always makes progress and converges.
    """
    if tokens_to_drop <= 0:
        return None
    estimator = token_estimator or (lambda msgs: sum(len(m.content) for m in msgs) // 4)
    # Largest non-preserved messages first — shrink where it frees the most.
    order = sorted(
        (i for i, m in enumerate(messages) if not is_preserved(m)),
        key=lambda i: len(messages[i].content),
        reverse=True,
    )
    floor_len = 2 * min_keep_chars + len("\n[truncated NNNN chars]")
    result = list(messages)
    dropped = 0
    changed = False
    for i in order:
        message = result[i]
        if len(message.content) <= floor_len:
            continue
        head = message.content[:min_keep_chars]
        tail = message.content[-min_keep_chars:]
        omitted = len(message.content) - 2 * min_keep_chars
        content = f"{head}\n[truncated {omitted} chars]\n{tail}"
        saved_chars = len(message.content) - len(content)
        result[i] = Message(
            message.role, content, message.name, {**message.metadata, "compressed": "ptl_shrink"}
        )
        changed = True
        dropped += max(0, saved_chars) // 4
        if dropped >= tokens_to_drop:
            break
    if not changed:
        return None
    return result


@dataclass(slots=True)
class CompressionEvent:
    stage: str
    before_chars: int
    after_chars: int
    detail: str = ""


@dataclass(slots=True)
class CompressionConfig:
    max_context_chars: int = 24000
    max_message_chars: int = 6000
    collapsed_keep_recent: int = 8
    # Track A (LLM summary) knobs. When a real summarizer is injected and
    # use_llm_summary is on, _context_collapse folds the old prefix into a model
    # summary instead of the deterministic Track B join; any failure degrades to
    # Track B. With no summarizer (FakeProvider / no key) these are inert.
    use_llm_summary: bool = True
    summary_keep_recent: int = 8
    summary_input_max_chars: int = 16000
    # Track A summary OUTPUT budget, as a 3-tier escalation ladder (mirrors the
    # reference's streamed compact: steady-state -> compaction cap -> model ceiling). The
    # call STREAMS (compression_summary.py) so the high tiers don't trip the SDK's
    # non-streaming timeout. ``compact_summary_start_tokens`` is the first attempt
    # (steady-state, ~the reference's 8k slot-cap); on a ``max_tokens`` truncation it
    # escalates to ``compact_max_output_tokens`` (the compaction cap, the reference's 20k),
    # then to the model's hard ceiling (``tokens.model_output_tokens(model)[1]`` — 128k for
    # Opus). ``compact_max_output_retries`` caps the escalations after the first attempt.
    compact_summary_start_tokens: int = 8000
    compact_max_output_tokens: int = 20000
    compact_max_output_retries: int = 2
    # Single, non-stacked timeout (seconds) wrapped around the Track A summarizer call —
    # it bounds the WHOLE escalation ladder, not each attempt (the ONLY asyncio-level
    # timeout here). Defaults to 120s so a wedged call can't hang a run yet the ladder has
    # room to escalate; set to None to disable (then only the run-level deadline bounds it).
    # On timeout the prefix degrades to the deterministic Track B fold.
    summary_timeout_seconds: float | None = 120.0
    # Token-based threshold knobs (Phase 1A). The char-based fields above still govern
    # per-message snip/microcompact budgets; the gate itself is token-based (Phase 2B).
    # ``context_window_tokens`` optionally caps the model's window; the buffer / reserved
    # values feed ``tokens.auto_compact_threshold``. ``max_consecutive_autocompact_failures``
    # is the circuit breaker that stops futile retries once context is irrecoverably over.
    context_window_tokens: int | None = None
    autocompact_buffer_tokens: int = 13000
    reserved_output_tokens_for_summary: int = 20000
    max_consecutive_autocompact_failures: int = 3
    # Resolved from AGENT_AUTOCOMPACT_PCT_OVERRIDE (percent of the effective window,
    # 0 < p <= 100) — lowers the auto-compact threshold for testing. Not a toml key.
    autocompact_pct_override: float | None = None
    # Post-compact file re-injection (Phase 3E). After a real fold the react loop re-attaches
    # the most-recently-read files (newest first) as one untrusted ``<system-reminder>`` user
    # message, so the model doesn't lose file context the summary may have dropped. Budgets
    # are TOKEN-based (parity with the reference; same char/4 estimate as the auto-compact
    # gate, so the units match): at most ``post_compact_max_files`` files, each truncated to
    # ``post_compact_max_tokens_per_file``, within a total ``post_compact_total_budget_tokens``.
    # These are larger than the old char caps (~4k tokens total) so folded file context is
    # actually restored, while staying well under the window on a 200k model.
    post_compact_max_files: int = 5
    post_compact_max_tokens_per_file: int = 5000
    post_compact_total_budget_tokens: int = 20000


@dataclass
class CompressionPipeline:
    config: CompressionConfig = field(default_factory=CompressionConfig)

    def __post_init__(self) -> None:
        # Consecutive auto-compact failures (stages ran but couldn't pull the estimate
        # below the threshold, or the stage run raised). Not a dataclass field — it is
        # mutable per-run state, reset on any success or when the gate isn't tripped.
        # After config.max_consecutive_autocompact_failures, auto_compact short-circuits.
        self._consecutive_autocompact_failures = 0

    def should_compact(
        self,
        messages: list[Message],
        *,
        model: str = "",
        token_estimator: TokenEstimator | None = None,
    ) -> bool:
        """Read-only predicate: would ``auto_compact`` run the shrink stages now?

        Mirrors the gate at the top of ``auto_compact`` (token estimate vs. threshold,
        plus the circuit breaker) without any side effects, so the loop can fire a
        ``PreCompact`` hook *before* compaction only when a fold is actually imminent —
        instead of firing it every turn. ``auto_compact`` still re-checks the gate
        itself, so this is purely advisory.
        """
        estimate = self._estimate(messages, token_estimator)
        threshold = tokens.auto_compact_threshold(
            model,
            context_window_override=self.config.context_window_tokens,
            buffer_tokens=self.config.autocompact_buffer_tokens,
            reserved_output_tokens=self.config.reserved_output_tokens_for_summary,
            pct_override=self.config.autocompact_pct_override,
        )
        if estimate < threshold:
            return False
        if self._consecutive_autocompact_failures >= self.config.max_consecutive_autocompact_failures:
            return False
        return True

    async def auto_compact(
        self,
        messages: list[Message],
        *,
        model: str = "",
        token_estimator: TokenEstimator | None = None,
        summarizer: Summarizer | None = None,
        on_stage: StageReporter | None = None,
        attachments: list[Message] | None = None,
    ) -> tuple[list[Message], list[CompressionEvent]]:
        """Proactive compaction, gated on the running token count crossing the threshold.

        The gate is token-based (parity with the reference): an injected
        ``token_estimator`` (real agent: ``max(last_usage, char/4)``; default offline:
        ``char_count // 4``) is compared against ``tokens.auto_compact_threshold(model)``.
        Below the line we return unchanged; at/above it we run the shrink stages.

        A circuit breaker tracks consecutive failures (stages ran but the result is
        still over, or the stage run raised). After
        ``config.max_consecutive_autocompact_failures`` consecutive failures the gate
        short-circuits (returns unchanged, no events) so the loop can't spin on context
        it cannot recover. The breaker never raises.
        """
        estimate = self._estimate(messages, token_estimator)
        threshold = tokens.auto_compact_threshold(
            model,
            context_window_override=self.config.context_window_tokens,
            buffer_tokens=self.config.autocompact_buffer_tokens,
            reserved_output_tokens=self.config.reserved_output_tokens_for_summary,
            pct_override=self.config.autocompact_pct_override,
        )
        if estimate < threshold:
            # Not over the line: nothing to do and the breaker resets (a later genuine
            # spike starts from a clean failure count).
            self._consecutive_autocompact_failures = 0
            return messages, []

        if self._consecutive_autocompact_failures >= self.config.max_consecutive_autocompact_failures:
            # Tripped breaker: context is irrecoverably over and compaction keeps failing.
            # Short-circuit silently rather than spin (mirrors the reference breaker).
            return messages, []

        try:
            result, events = await self._run_stages(
                messages, aggressive=False, summarizer=summarizer, on_stage=on_stage, attachments=attachments
            )
        except Exception:  # noqa: BLE001 - a stage failure counts as a breaker failure, never crashes the loop
            self._consecutive_autocompact_failures += 1
            return messages, []

        if self._estimate(result, token_estimator) >= threshold:
            # Stages ran but couldn't pull the estimate below the line: a failure.
            self._consecutive_autocompact_failures += 1
        else:
            self._consecutive_autocompact_failures = 0
        return result, events

    async def reactive_compact(
        self,
        messages: list[Message],
        *,
        model: str = "",
        token_estimator: TokenEstimator | None = None,
        summarizer: Summarizer | None = None,
        on_stage: StageReporter | None = None,
        attachments: list[Message] | None = None,
    ) -> tuple[list[Message], list[CompressionEvent]]:
        """Aggressive compaction run after a context-overflow error; always compacts.

        Accepts the same ``model``/``token_estimator`` kwargs as ``auto_compact`` so the
        loop can call either through one signature; the reactive path ignores the gate
        (it is reached only after an actual overflow) and always runs the stages.
        """
        return await self._run_stages(
            messages, aggressive=True, summarizer=summarizer, on_stage=on_stage, attachments=attachments
        )

    def _estimate(self, messages: list[Message], token_estimator: TokenEstimator | None) -> int:
        estimator = token_estimator or (lambda msgs: self._char_count(msgs) // 4)
        return estimator(messages)

    async def _run_stages(
        self,
        messages: list[Message],
        *,
        aggressive: bool,
        summarizer: Summarizer | None,
        on_stage: StageReporter | None,
        attachments: list[Message] | None = None,
    ) -> tuple[list[Message], list[CompressionEvent]]:
        total = 3
        events: list[CompressionEvent] = []
        current = messages

        def report(done: int, event: CompressionEvent) -> None:
            events.append(event)
            if on_stage is not None:
                on_stage(done, total, event)

        # Stages 1–2 are deterministic and cheap; stage 3 (prefix collapse) may await
        # the injected summarizer (Track A) and so is the only async one.
        current, event = self._snip(current, aggressive)
        report(1, event)
        current, event = self._microcompact(current, aggressive)
        report(2, event)
        current, event = await self._context_collapse(current, aggressive, summarizer, attachments)
        report(3, event)
        return current, [event for event in events if event.before_chars != event.after_chars]

    def _snip(self, messages: list[Message], aggressive: bool) -> tuple[list[Message], CompressionEvent]:
        before = self._char_count(messages)
        limit = self.config.max_message_chars // (2 if aggressive else 1)
        snipped: list[Message] = []
        count = 0
        for message in messages:
            if message.role == "tool" and len(message.content) > limit:
                half = max(100, limit // 2)
                content = f"{message.content[:half]}\n[snip]\n{message.content[-half:]}"
                snipped.append(Message(message.role, content, message.name, {**message.metadata, "compressed": "snip"}))
                count += 1
            else:
                snipped.append(message)
        detail = f"snipped {count}" if count else ""
        return snipped, CompressionEvent("snip", before, self._char_count(snipped), detail)

    def _microcompact(self, messages: list[Message], aggressive: bool) -> tuple[list[Message], CompressionEvent]:
        before = self._char_count(messages)
        budget_limit = max(40, self.config.max_context_chars // (6 if aggressive else 4))
        limit = min(self.config.max_message_chars // (3 if aggressive else 2), budget_limit)
        compacted: list[Message] = []
        count = 0
        for message in messages:
            # Pinned blocks (e.g. injected CLAUDE.md, the userContext system-reminder)
            # are kept verbatim — they are one-time context that must survive compaction.
            # _context_collapse already preserves them, but microcompact truncates by
            # length without regard to role, so it needs its own guard.
            if message.metadata.get("pinned"):
                compacted.append(message)
                continue
            if len(message.content) > limit:
                content = f"{message.content[:limit]} [microcompact: omitted {len(message.content) - limit} chars]"
                compacted.append(Message(message.role, content, message.name, {**message.metadata, "compressed": "microcompact"}))
                count += 1
            else:
                compacted.append(message)
        detail = f"microcompacted {count}" if count else ""
        return compacted, CompressionEvent("microcompact", before, self._char_count(compacted), detail)

    async def _context_collapse(
        self,
        messages: list[Message],
        aggressive: bool,
        summarizer: Summarizer | None,
        attachments: list[Message] | None = None,
    ) -> tuple[list[Message], CompressionEvent]:
        before = self._char_count(messages)
        use_llm = summarizer is not None and self.config.use_llm_summary
        # Track A keeps its own (configurable) recent window; Track B keeps the legacy
        # collapsed_keep_recent so a no-LLM history collapses byte-for-byte as before.
        base_keep = self.config.summary_keep_recent if use_llm else self.config.collapsed_keep_recent
        keep = max(4, base_keep // (2 if aggressive else 1))
        # Preserved messages are never folded: a non-collapse system block (base prompt,
        # memory recall) OR anything pinned (any role — protects the new userContext
        # USER system-reminder). Everything else — incl. a prior summary block now
        # carrying a USER role — is foldable conversation, so repeated compactions
        # re-summarize rather than stacking stale summaries.
        preserved = [m for m in messages if self._is_preserved(m)]
        conversation_messages = [m for m in messages if not self._is_preserved(m)]
        # All preserved messages sit at the front by construction (base system, recall,
        # userContext are injected before any conversation), so reassembly stays
        # [*preserved, summary, *recent] with original ordering intact.
        assert messages[: len(preserved)] == preserved, "preserved messages must form a prefix"
        if len(conversation_messages) <= keep + 1:
            return messages, CompressionEvent("context_collapse", before, before)
        # Snap the prefix/recent split to a round boundary so a tool_call is never
        # separated from its tool_result and ``recent`` never starts with an orphan tool.
        prefix, recent = split_on_round_boundary(conversation_messages, keep)
        if not prefix:
            # Boundary snapping kept everything recent (e.g. one giant round) — nothing
            # to fold; leave the history unchanged rather than emit an empty summary.
            return messages, CompressionEvent("context_collapse", before, before)
        summary_message, note = await self._collapse_prefix(prefix, summarizer if use_llm else None)
        # Attachments (re-injected file context) are appended ONLY here, on a real fold —
        # the early-return no-fold paths above leave history untouched and inject nothing.
        # They live in the conversation TAIL (after ``recent``) and are NOT pinned, so the
        # preserved-front invariant holds and a later compaction can re-fold them.
        collapsed = [
            *preserved,
            *build_post_compact_messages(summary_message, recent, attachments=attachments),
        ]
        detail = f"collapsed {len(prefix)} msgs ({note})"
        return collapsed, CompressionEvent("context_collapse", before, self._char_count(collapsed), detail)

    @staticmethod
    def _is_preserved(message: Message) -> bool:
        """Delegate to the module-level ``is_preserved`` — the single source of truth for
        the preservation predicate, shared with ``truncate_head_for_ptl_retry``."""
        return is_preserved(message)

    async def _collapse_prefix(
        self, prefix: list[Message], summarizer: Summarizer | None
    ) -> tuple[Message, str]:
        """Fold the old prefix into one USER summary message, returning it plus a note.

        Track A (``summarizer`` present): ask the model for a structured summary, wrapped
        in a single ``asyncio.wait_for`` when ``summary_timeout_seconds`` is set — the ONLY
        asyncio-level timeout on this path. Any failure — exception, timeout, or an
        empty/blank result — degrades to the deterministic Track B fold so compaction
        always succeeds and the run never breaks. The note (``llm_summary`` /
        ``summary_fallback: ...`` / ``track_b``) flows into the compression log event.
        """
        if summarizer is None:
            return self._collapse_prefix_naive(prefix), "track_b"
        try:
            timeout = self.config.summary_timeout_seconds
            if timeout is not None:
                summary = await asyncio.wait_for(summarizer(prefix), timeout)
            else:
                summary = await summarizer(prefix)
            if not summary or not summary.strip():
                raise ValueError("empty summary")
        except Exception as exc:  # noqa: BLE001 - any summarizer failure must degrade, not crash
            return self._collapse_prefix_naive(prefix), f"summary_fallback: {type(exc).__name__}"
        block = build_summary_user_message(
            summary, marker="llm_summary", messages_collapsed=len(prefix)
        )
        return block, "llm_summary"

    @staticmethod
    def _collapse_prefix_naive(prefix: list[Message]) -> Message:
        """Deterministic Track B fold: join the old prefix into one USER summary message.

        Pure string manipulation, no model call — the offline/no-key default and the
        fallback when an LLM summary (Track A) is unavailable or fails. Byte-stable for a
        given input (the continuation wrapper is fixed text)."""
        summary = " | ".join(f"{m.role}: {m.content[:160]}" for m in prefix)
        return build_summary_user_message(
            summary, marker="context_collapse", messages_collapsed=len(prefix)
        )

    @staticmethod
    def _char_count(messages: list[Message]) -> int:
        return sum(len(message.content) for message in messages)
