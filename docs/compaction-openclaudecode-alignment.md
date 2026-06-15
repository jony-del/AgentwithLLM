# Compaction & context-assembly alignment with Open-ClaudeCode

> Supersedes the "future work" roadmap in `p0-3-compaction-landed.md` §6 for the
> items now implemented. This documents the alignment pass that brought auto/reactive
> compaction **and** the system/system-context/user-context assembly to parity with
> the reference `Open-ClaudeCode`. Locate code by symbol name; line numbers drift.

## Why

The previous compaction was char-count gated, injected its summary as a *system*
message, used a condensed summary prompt, and recovered from context-overflow by
re-running aggressive stages. Prompt assembly injected CLAUDE.md and git as separate
pinned *system* messages. The reference does all of this differently. This pass closes
those gaps. Delivered as six phases (all landed; full suite green at **349 passed**).

## What changed

### 1. Token-based auto-compact gate (`agent_core/tokens.py`, `compression.py`)
- New pure-stdlib `tokens.py`: `context_window_for_model` (200k default; `[1m]` tag →
  1M), `effective_context_window = window − min(max_output, 20k reserved)`,
  `auto_compact_threshold = effective − 13k buffer` with `AGENT_AUTOCOMPACT_PCT_OVERRIDE`.
- `auto_compact` now compares an **injected `token_estimator`** against the model
  threshold instead of `chars < cap*ratio`. The agent's estimator is
  `max(last_response_usage.context_tokens, chars//4)`; offline/Fake defaults to
  `chars//4`, keeping tests deterministic. Char fields now only bound per-message
  snip/microcompact budgets.
- **Token usage** is surfaced end-to-end: `LLMResult.usage: TokenUsage | None`
  (`input/output/cache_read/cache_creation`, `.context_tokens` = input+cache).
  `ClaudeProvider` parses it from non-streaming `usage` and streaming
  `message_start`/`message_delta`; `FakeProvider` stamps a deterministic chars/4.
- **Circuit breaker**: after `max_consecutive_autocompact_failures` (3) consecutive
  failures (stages ran but estimate still over, or raised), `auto_compact`
  short-circuits; resets on success / when under the line. Mirrors
  `MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES`.

### 2. Context assembly restructure (`react.py`, `context.py`, `providers/claude.py`)
Run-start message order is now:
```
system( base prompt + "\n\ngitStatus: <git block>" )
(memory recall system, if any)
userContext  <system-reminder> USER message (pinned)  # claudeMd + currentDate
user( task )
```
- `append_system_context(system_text, {"gitStatus": …})` folds git into the single
  system block (`key: value`), preserving the untrusted-data `<git_status>` framing.
- `prepend_user_context({"claudeMd": …, "currentDate": …})` emits one pinned
  `<system-reminder>` USER message (exact reference wrapper), preserving CLAUDE.md's
  OVERRIDE preamble. CLAUDE.md is no longer a standalone system message.
- The provider already collapses system parts into one `system` string and emits the
  leading `<system-reminder>` as the first user message (the API accepts the two
  consecutive user messages, as the reference relies on).

### 3. Summary prompt + USER-message re-injection (`compression_summary.py`, `compression.py`)
- `SUMMARY_SYSTEM` now ports the full reference prompt: `NO_TOOLS_PREAMBLE` + the
  9-section `BASE_COMPACT_PROMPT` (incl. **All user messages** and **Optional Next
  Step** with a verbatim quote) + `NO_TOOLS_TRAILER`. The transcript is sent as the
  untrusted user turn inside `<transcript>` delimiters with a preamble.
- Summary output budget raised: `compact_max_output_tokens` (default 8192) replaces the
  hard-coded 2048.
- A **single, non-stacked** `asyncio.wait_for(summarizer, summary_timeout_seconds)`
  wraps the Track A call (only when set); timeout degrades to Track B.
- The fold is re-injected as a **USER** message (`build_summary_user_message`) with the
  reference's `"This session is being continued…"` header + `"Continue… without asking
  further questions"` footer; metadata `compressed=llm_summary|context_collapse` keeps
  it re-foldable. Track B stays deterministic/byte-stable (fixed wrapper text).

### 4. Round-boundary grouping (`compression.py`)
- `group_into_rounds` glues an `assistant`-with-`tool_calls` to its matching `tool`
  results; `split_on_round_boundary` snaps the prefix/recent cut to a round edge so a
  tool call is never separated from its result. Used by both the fold and the reactive
  head-truncation.
- Preservation generalized: `is_preserved` keeps non-collapse **system** blocks **and
  any pinned message (any role)** — protecting the pinned userContext USER message.
  Prior summaries (collapse-marked, now USER role) re-fold rather than stack.

### 5. Reactive 413 recovery (`compression.py`, `react.py`)
- `parse_prompt_too_long_gap` reads `actual > limit` from the error body.
- `truncate_head_for_ptl_retry` drops the oldest whole rounds (gap-driven, else 20%
  fallback; keeps ≥1 round; never drops the preserved front).
- The loop now: summarize → up to `MAX_PTL_RETRIES` (5) retries that peel oldest rounds
  → give up if `<2` rounds remain. Bounded — no 413→compact→413 loop; CancelledError
  handling and usage tracking preserved on every retry.

### 6. Post-compact file re-injection (`session.py`, `react.py`, `compression.py`)
- `SessionContext.read_file_state` + `record_read` track recently-read files (recency
  order, ~20 cap). The react loop records `read_text_file` results (no tool/mixin
  change). `_build_read_attachments` builds **one** untrusted `<system-reminder>` USER
  message of the most-recent ≤`post_compact_max_files` (5) files (per-file + total char
  budgets), tagged `post_compact_attachment` (**not** pinned).
- Threaded as `attachments` through `auto_compact`/`reactive_compact` →
  `build_post_compact_messages`, injected **only on a real fold** (tail position).
- Deferred-tool delta announcements intentionally skipped: this framework re-sends all
  tool schemas every turn, so tools are never lost after compaction.

## Verification
- `pytest` — **358 passed** (349 alignment + 9 hardening). Focused: `tests/test_tokens.py`, `test_compression.py`,
  `test_context.py`, `test_react.py`, `test_session.py`, `test_postcompact.py`,
  `test_claude_provider.py`.
- Manual acceptance (offline `--provider fake`, forced fold): the produced messages are
  `system(+gitStatus)` → pinned `<system-reminder>` userContext → summary **USER**
  message with the continuation wrapper → recent rounds intact on round boundaries →
  `post_compact_attachment` USER message at the tail.

## Post-alignment hardening (critical-review follow-up)

A critical review against the reference source surfaced robustness/parity gaps that
this pass closed (full suite **358 passed**):

- **Single oversized round/message converges** (`shrink_oversize_messages`): when whole-
  round head-truncation can't help (one round, or one message, alone exceeds the window),
  the 413 loop falls back to head/tail-truncating the largest **non-preserved** messages
  (omission marker, `min_keep_chars` each end) until the gap is shed. Guarantees the retry
  loop always makes progress instead of re-raising / spinning. Logged as `ptl_shrink`.
- **`PTL_RETRY_MARKER`**: after a head-truncation, if the first non-system message would be
  an assistant/tool turn (possible when the pinned `userContext` is absent — no git AND no
  CLAUDE.md), a synthetic user turn is prepended so the Anthropic messages array still
  begins with a user message. Mirrors the reference marker.
- **Bounded summary timeout by default**: `summary_timeout_seconds` now defaults to `60`
  (was `None`/unbounded) so a wedged Track A call can't hang a run; set to `None` to rely
  only on the run-level deadline.
- **Post-compact re-injection budgets are token-based** (`post_compact_max_tokens_per_file`
  = 5000, `post_compact_total_budget_tokens` = 20000), matching the gate's unit and
  restoring meaningfully more file context than the old ~4k-char total. **CLAUDE.md is
  excluded** from re-injection (already pinned as `userContext`).
- **Dead config removed**: `auto_threshold_ratio`, `summary_max_tokens` (both unused since
  the token gate / `compact_max_output_tokens` landed). `max_context_chars` is retained —
  it still bounds the microcompact budget. `compact_max_output_tokens` stays 8192 (safe
  across models; raise per-config for high-output models).

## Deliberately out of scope
- `cache_edits` / time-based microcompaction (needs server-side prompt-cache editing
  the provider abstraction does not expose).
- `sessionMemoryCompact` (this project's `memory/` is cross-conversation fact
  extraction — different semantics).
- Map-reduce chunked summaries for very long prefixes (still head/tail-capped).
