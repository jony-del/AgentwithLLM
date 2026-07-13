# CLAUDE.md

Engineering guidance for working in this repository. This document describes what
the project is, the principles that govern changes, and the invariants that must
hold. It is a living contract: when code and this file disagree, fix one of them
in the same change — never leave them drifting.

## Project

Polaris (package `agent_core`) is an async Python ReAct agent framework
(Python >= 3.11) evolving into an industrial-grade coding/automation agent:
provider-neutral at its core, safe by construction in its tool execution,
observable and replayable in its runs, and testable at every layer.

Design authority lives HERE, in this project's own goals. External projects
(including Claude Code and its source-recovered copies) are inspiration and a
source of battle-tested patterns — never a spec to mirror, never a ceiling, and
never a justification by themselves. "The reference does X" is not a reason;
"X solves our problem because…" is.

`revision guide.md` (repo root) holds the current audit, the gap analysis, and
the staged roadmap (R0–R3). Consult it before large structural changes.

## Engineering Principles

1. **Provider-neutral core.** `agent_core.models` and `providers/base.py` are the
   contracts. No provider SDK types may leak into them. `ClaudeProvider` speaks
   the Anthropic Messages protocol directly over httpx. External protocol
   choices are explicit, never inferred from model names or base URLs:
   `claude` → Anthropic Messages (`/v1/messages`), `openai` → OpenAI Responses
   (`/v1/responses`), and `openai-compat` → OpenAI-compatible Chat Completions
   (`/v1/chat/completions`) for DeepSeek/Qwen/GLM/Moonshot/vLLM/LM Studio/Groq
   style endpoints. Never infer or switch the protocol from a model name or base
   URL. `fake` remains the deterministic offline provider. Any new provider must
   fit behind `LLMProvider.complete` unchanged. Provider-specific payloads (e.g.
   `LLMResult.thinking_blocks` and `provider_state`) are provider-owned opaque
   data: preserved and round-tripped, never interpreted by the core loop.

2. **Clear capability boundaries.** A capability (web, MCP, memory, skills,
   sandbox, terminal UI) is enabled via config, initialized at startup with an
   actionable error when its dependencies are missing, and reachable from tools
   only through explicit seams (`SessionAwareMixin`, `SandboxAwareMixin`).
   Startup initialization means *validate and construct*, with bounded,
   idempotent side effects: heavyweight work (pulling a container image, booting
   a VM) must be explicit, logged, and done once per process — not repeated per
   agent/sub-agent construction, and never hidden in an import. Library
   embedding is a supported deployment: `asyncio.run` appears only at CLI entry
   points, `NullUI` stays silent, and constructor injection
   (`ReActAgent(provider=…, tools=…, hooks=…)`) is a stable surface.

3. **Fail-closed security posture.** The direction of travel for every
   permission/sandbox decision is: when in doubt, ask or deny — and when a
   safety mechanism itself fails, the gated action does not silently proceed.
   Concretely:
   - *Trust is assigned by origin, not content.* Repo-controlled inputs
     (CLAUDE.md, git output, file contents, tool output, web pages) are data,
     not instructions; framework-injected framing is trusted only because it
     comes from the framework's own injection path (`context.py`,
     `PromptValidationHook`). Repo instructions inform style and architecture;
     they must never be treated as authorization to weaken permission rules,
     hooks, or sandbox settings.
   - *Repo config must not widen privileges (enforced).* An in-repo `agent.toml`
     may tighten policy (deny/ask rules and every ordinary table apply); the
     privilege-widening subset (allow rules, external hooks, sandbox exclusions/
     relaxations, MCP servers, `[web].allowed_domains`) is gated by trust-on-first-use in
     `agent_core/trust.py`: inert until the user approves it once (recorded per
     project in `~/.polaris/trusted.json`, re-prompted on change), dropped with an
     audit line on unattended/CI runs. A config passed explicitly via `--config`
     is user-chosen and not filtered. Never extend what repo config can widen.
   - *Unattended modes require a sandbox (enforced).* Constructing an agent in an
     unattended permission mode (`auto`/`dontask`/`bypass`) with no working sandbox
     raises `SandboxRequiredError`; interactive runs get a one-off confirmation,
     and `[sandbox] allow_unattended_unsandboxed` / `AGENT_SANDBOX_ALLOW_UNATTENDED`
     is the explicit, logged opt-out.
   - *Unattended web egress is allowlist-only (enforced).* The `[web]` domain
     policy (`tools/web.py`, decision D10): `blocked_domains` always refuse; in
     unattended modes a domain not in `allowed_domains` is refused fail-closed
     (checked on every redirect hop, and against the search backend's host for
     `web_search`). Attended runs stay open unless blocked.
   - *Degrade-to-allow is only for observation.* Observational hooks and
     context injection may fail open (a run is never sunk by telemetry).
     Control-path mechanisms must support fail-closed behavior: external hooks
     carry `fail_mode = "open"|"closed"` (command/http), and a child's effective
     permissions never exceed its parent's (`_child_permission_mode`). New
     control-path features default to fail-closed.
   - *Programmatic approval of asks (`PermissionRequest` hook).* Fires only when
     the pipeline reached an *ask* (interactive, or its headless collapse) — never
     for hard denies, so a hook cannot launder a deny rule. Folding is fail-closed
     (any deny wins); a crashed hook yields no opinion (the normal ask path
     resumes, never a silent allow); external adapters on this event default to
     `fail_mode = "closed"`. This is the headless deployment's ask outlet.
   - Every tool sets a correct `ToolRisk` (`READ`/`WRITE`/`DANGEROUS`); the
     permission pipeline (`permissions.py`: deny → sensitive-path safety net →
     ask → sandbox auto-allow → bypass → allow → per-mode matrix) depends on it.

4. **Observable, replayable, debuggable.** Every decision that affects a run
   (permission verdicts, hook outcomes, compaction events, tool results, spawn
   lineage) is written to `runs/<run_id>.jsonl`; conversations persist as
   resumable transcripts (`transcript.py`). Silent degradation is a bug: any
   `except Exception` recovery path must emit an event or log line saying what
   was skipped and why. New record types carry a schema version so tooling can
   evolve.

5. **Progressive evolution.** Prefer the smallest reversible change that can be
   verified; land risky work behind config flags with conservative defaults;
   measure before optimizing. Do not design around a small fixed step cap — a
   coding agent needs many tool turns; bound runs with explicit guards instead
   (cooperative cancel, optional `max_steps`, shared wall-clock deadline).

6. **Docs move with code.** A change that adds/removes a module, config table,
   CLI flag, or invariant updates this file (and `agent.toml.example`) in the
   same commit. Never reference files that do not exist in the repo.

## Environment

Primary dev environment is **Windows + PowerShell** (`$env:NAME="..."`); CI must
also cover Linux. Use OS-appropriate paths. Linting/type-checking is being
introduced incrementally (start: ruff E/F on changed code); do not add sweeping
format-only diffs.

## Commands

```powershell
# Deterministic offline run (no API key). `polaris` is the installed CLI;
# `python -m agent_core` is equivalent.
polaris run "Say hello without tools" --provider fake
polaris chat --provider fake          # interactive chat, one event loop per session

# Real API
$env:ANTHROPIC_API_KEY="your-key"
polaris run "Use the echo tool" --provider claude --model claude-haiku-4-5-20251001

# OpenAI Responses API
$env:OPENAI_API_KEY="your-key"
polaris run "Use the echo tool" --provider openai --model gpt-4.1-mini

# OpenAI-compatible chat-completions endpoint (vLLM / Groq / DeepSeek / Qwen / ...)
$env:OPENAI_COMPAT_API_KEY="your-key"
$env:OPENAI_COMPAT_BASE_URL="https://compat.example.com"
polaris run "Use the echo tool" --provider openai-compat --model your-endpoint-model

# Live-display flags (CLI-only): --quiet, --no-stream, --thinking-budget N
# Other subcommands: sessions / dream / memory / mcp / health
polaris replay <run_id>                  # re-render a runs/*.jsonl as a timeline (no API)
# --config PATH reads settings from another toml (explicit path = user-chosen,
# no TOFU filtering) — e.g. the shipped agent.strict.toml / agent.lax.toml profiles
polaris run "task" --provider fake --config agent.strict.toml

# Install: core is the minimal library surface (httpx+pyyaml); capabilities are
# extras — [web] (fetch/search), [mcp], [terminal] (interactive console), [all].
pip install -e .[all,dev]                # recommended: everything + test/lint tooling

# Tests
pytest                                   # full suite must stay green
pytest tests/test_react.py::test_react_executes_demo_tool
```

`pyproject.toml` sets `pythonpath = ["."]`, so tests import `agent_core` without
installation. `asyncio_mode = "auto"`; an un-awaited coroutine fails the suite.

## Code Map

Core loop & contracts:
- `agent_core/react.py` — `ReActAgent.run()`, the central async loop; lifecycle
  hook boundaries; deadline/cancel guards; sub-agent & teammate factories.
- `agent_core/models.py` — cross-layer contracts: `Message`, `ToolCall`,
  `ToolResult`, `LLMResult`, `TokenUsage`. Change deliberately; update tests.
- `agent_core/session.py` — per-run state shared with session-aware tools
  (todos, read-file state, spawn factories, depth limits).
- `agent_core/interrupt.py` — cooperative Esc-cancel watcher (no-op non-TTY).
- `agent_core/compression.py` / `compression_summary.py` — token-gated proactive
  + reactive (413) compaction; LLM-summary Track A degrading to deterministic
  Track B; bounded PTL retry.
- `agent_core/tokens.py` / `model_catalog.py` — context-window math and known
  model families.
- `agent_core/context.py` — run-start context assembly: CLAUDE.md discovery
  (stops at project root), one-shot git snapshot (wrapped as untrusted data),
  pinned user-context message.
- `agent_core/transcript.py` — resumable session transcripts (uuid/parent_uuid
  chains, compact boundaries, fork); `agent_core/storage.py` — per-run JSONL
  event log (`JSONLRunLogger` writer + `read_events` reader; `polaris replay`
  renders a log as a timeline).

Providers:
- `agent_core/providers/base.py` — `LLMProvider` interface, `ProviderConfig`
  (the frozen per-call parameter contract of `complete`; derived calls override
  via `dataclasses.replace`), `GatedProvider` (shared semaphore + token-bucket
  across the multi-agent fan-out).
- `agent_core/providers/claude.py` — Anthropic Messages API (`/v1/messages`) over
  httpx: streaming, retries with jitter, model-aware request shape (adaptive
  thinking vs legacy), effort levels. `openai_capabilities.py` — provider-local
  GPT/Responses capability profiles (anchored model-family matching, conservative
  defaults for unknown models). `openai_responses.py` — OpenAI official Responses
  API (`/v1/responses`): manual item replay (`store=false`), model-gated encrypted
  reasoning include/replay, displayable reasoning summaries, flat function tools,
  streaming typed SSE, provider-state preservation, structured OpenAI errors.
  `openai_compat.py` — OpenAI-compatible Chat Completions (`/v1/chat/completions`;
  `OPENAI_COMPAT_API_KEY`/`OPENAI_COMPAT_BASE_URL`, with deprecated fallback to
  `OPENAI_API_KEY`/`OPENAI_BASE_URL`) for vLLM, LM Studio, Groq, DeepSeek, Qwen,
  GLM, Moonshot, etc. `openai_errors.py` — shared OpenAI error envelope parsing
  and actionable unsupported-parameter diagnostics. `fake.py` — deterministic
  offline provider for tests/demos.

Tools:
- `agent_core/tools/base.py` — `Tool`, `ToolRisk`, `WorkspacePathMixin`
  (workspace confinement), concurrency specs; `catalog.py` — `@builtin_tool`
  self-registration; `registry.py` / `executor.py` — registry and wave-partitioned
  parallel execution; `adapters.py` — adapter base.
- `builtin.py` (fs/search/command/tests; `search_text` probes for ripgrep and
  uses it when present, falling back to the pure-Python scan — identical output
  either way), `editing.py`, `web.py` (SSRF-guarded + `[web]` domain policy),
  `planning.py`, `subagent.py`, `team.py`, `skill.py`.

Permissions & sandbox:
- `agent_core/permissions.py` — `PermissionPolicy` decision pipeline + modes
  (default/acceptedits/plan/auto/dontask/bypass).
- `agent_core/permission_rules.py` — argument-aware `ToolName(content)` rules:
  shell decomposition (allow needs ALL sub-commands, deny needs ANY),
  anti-evasion (env-var/wrapper stripping with hijack-var guard), path globs,
  `domain:` matching. Parse failures drop the rule, never raise.
- `agent_core/sandbox/` — OS enforcement layer: `SandboxManager` selects a
  backend by tier (`native` bwrap/seatbelt, `container` podman/docker/nerdctl,
  `vm` Kata/Hyper-V/Lima) with downgrade-on-unavailable; `fail_if_unavailable`
  turns "can't sandbox" into a hard startup error. `prepare()` is idempotent and
  managers are process-shared via `get_shared_manager` (children reuse their
  parent's via `ReActAgent(sandbox=…)`), so heavyweight readying runs once per
  process. Command tools reach it via `SandboxAwareMixin`.

Hooks:
- `agent_core/hooks.py` — tool hooks (sync, in-executor) + async lifecycle
  events: the decision events (`UserPromptSubmit`, `PreCompact`, `PostCompact`,
  blockable `Stop`), fire-and-forget `PostSampling`, and the observational C5
  events (`SessionStart`/`SessionEnd`, `SubagentStart`/`SubagentStop`,
  `PostToolUseFailure`) which are awaited but fail-open and never alter control
  flow, plus the control-path `PermissionRequest` (programmatic ask approval,
  fail-closed — see the security principles). `SessionEnd` is host-driven: the
  CLI calls `agent.fire_session_end()` at run/chat exit; library embedders call
  it at their own session boundary. Also
  `MaxOutputPostHook` (oversized output → on-disk spill + `<tool_output_ref>`
  pointer).
- `agent_core/builtin_hooks.py` — in-process hooks (stop-on-open-todos,
  observers, `PromptValidationHook` input firewall: provenance-based,
  neutralize-not-reject).
- `agent_core/hook_adapters.py` — config-driven external hooks
  (`command`/`http`/`prompt`/`agent`), single non-stacked timeout, kill+await,
  bounded JSON projection across the boundary. Currently observation-oriented
  and fail-open; control-path use requires the fail-closed option (roadmap R1).

Multi-agent:
- `agent_core/agents/multi.py` — parallel fan-out; `agents/team.py` —
  `TeamStore`/`FileLock` shared state.

Skills / chat / UI / CLI:
- `agent_core/skills/` — markdown (`bundled/` + user/project dirs) and
  `@programmatic_skill` Python skills merged into one per-run registry;
  malformed skills degrade to fewer skills, never a crash.
- `agent_core/chat_commands.py` — interactive `/commands` (CLI-only).
- `agent_core/ui.py` — `NullUI` (default, silent) / `ConsoleUI`;
  `agent_core/terminal/` — prompt_toolkit interactive app, completion,
  keybindings, model picker.
- `agent_core/cli.py` — argparse entry (`polaris`).

Cross-cutting:
- `agent_core/config.py` — resolution: defaults → `agent.toml` → env → CLI.
- `agent_core/memory/` — optional cross-conversation memory (recall/extract/
  dream). `agent_core/mcp/` — MCP client/config/tool adapter (stdio +
  streamable-http; MCP tools default to `dangerous` risk).
- `agent_core/tool_use_summary.py` — optional UI-only progress labels.

## Agent Loop

`async ReActAgent.run()` — hook ordering is part of the loop contract:

0. `[UserPromptSubmit]` after the task is in place, before the first model call
   (may block the run, rewrite the task, or inject grounding).
1. `[PreCompact]` → token-gated auto-compaction → `[PostCompact]` (a PreCompact
   block skips only the proactive fold; the reactive 413 path always compacts).
2. Provider call, then `[PostSampling]` (fire-and-forget).
3. No tool calls → `[Stop]`: a hook may block the stop (bounded by
   `max_stop_blocks`); otherwise return the final answer.
4. Tool calls run through `ToolExecutor` (permission pipeline + tool hooks,
   wave-partitioned concurrency), observations append as `tool` messages, loop.

A run's `deadline` is shared with all spawned children so the whole fan-out is
bounded by one budget; crossing the soft deadline injects a one-time wrap-up
nudge. Stop reasons are always explicit.

## Async Execution Model

Async-only: every public execution API is `async def` — no sync twins.
`asyncio.run` only at CLI entry points. Blocking work is an internal detail:
ordinary tools implement `_invoke()` (thread-offloaded by `Tool.run`, bounded by
executor `max_workers`); stores/loggers expose async methods over `_xxx_sync`
internals; blocking-SDK providers wrap calls in `asyncio.to_thread` inside their
own `complete`. Never call blocking IO directly from a coroutine.

## Invariants

Contracts & providers:
- `Message`/`ToolCall`/`ToolResult`/`LLMResult` are cross-layer contracts —
  change them deliberately, update affected tests in the same change.
- Preserve `LLMResult.thinking_blocks` (with signatures) across turns; treat the
  contents as opaque provider data.

Tools & permissions:
- Every tool sets a correct `ToolRisk`; permission behavior depends on it.
- A tool implements EITHER `_invoke()` OR an async `run()` override — never both.
- Workspace-scoped tools MUST use `WorkspacePathMixin`; no absolute-path or
  `..` escapes.
- Tools self-register via `@builtin_tool`; adding a built-in must not require
  editing `react.py`.
- Session-aware tools use `SessionAwareMixin` and are rebound by
  `ReActAgent.__init__`; never import `ReActAgent` from a tool.
- Sub-agents MUST NOT receive `dispatch_agent` or `skill` (recursion guard),
  and a child's effective permissions must never be broader than its parent's:
  children run the preset-mapped mode from `_child_permission_mode`
  (`read_only`→default, `full`→acceptedits), never an inherited unattended mode.
  Do not add spawn paths that escalate.

Degradation & security:
- Optional/advanced capabilities degrade with an actionable install/config
  error, never an import-time crash; malformed skills/config degrade to
  fewer features, never a failed construction.
- Every degradation is observable: log or JSONL event, no bare
  `except Exception: pass` without a trace.
- Fail-open is acceptable only on observation paths; anything gating an action
  must be able to fail closed (and new gates default to it).
- Untrusted text (user task framing, git output, web content, tool output)
  crosses into the prompt only wrapped/neutralized by the framework's own
  injection paths; never weaken those wrappers.

UI, memory, state:
- `NullUI` stays silent and non-interactive (tests, library use); streaming UI
  hooks are finalizers when deltas were already printed — do not duplicate
  streamed content.
- Memory is off by built-in default (this repo's `agent.toml` may enable it);
  recall at run start, extraction only after natural termination, and it must
  never fail an otherwise completed run.
- `runs/`, `memory/`, and user/project skill dirs (`.polaris/skills/`) are
  runtime state, gitignored; only `agent_core/skills/bundled/*.md` is in-repo.
- Built-in chat `/commands` live in `chat_commands.py` (CLI-only); `/model`
  mutates `config.model` only, never rebuilds the agent.

## Configuration

Scalar precedence: built-in defaults → `agent.toml` → env → CLI flags. The
provider scalar is an explicit protocol selector: `claude` uses `/v1/messages`,
`openai` uses `/v1/responses`, `openai-compat` uses `/v1/chat/completions`, and
`fake` is offline. Do not infer or change provider from model ID or base URL;
legacy `--provider openai + OPENAI_BASE_URL=<compat endpoint>` users must migrate
to `--provider openai-compat`. The OpenAI Responses capability matrix is
provider-local and fail-safe: unknown models do not receive reasoning-only fields,
unsupported effort values are omitted, and encrypted reasoning stays in opaque
`provider_state` while only displayable reasoning summaries reach the UI.
Tables resolved separately in `config.py`: `[memory]`, `[limits]`, `[session]`,
`[context]`, `[compression]`, `[concurrency]`, `[mcp]`, `[output]`,
`[tool_use_summary]`, `[skills]`, `[hooks]` (+ `[hooks.builtin]`,
`[hooks.prompt_validation]`, `[[hooks.external]]`), `[permissions]`, `[web]`,
`[sandbox]` (+ filesystem/network/container/vm). Live-display knobs are
CLI-only. See `agent.toml.example` for the full supported shape; keep it in
sync with code. Two ready-made posture profiles ship at the repo root —
`agent.strict.toml` (web denied, mandatory sandbox, everything asks) and
`agent.lax.toml` (trusted-dev: pre-approved read-only commands, sandbox
auto-allow) — usable via `--config <file>` or by copying over `agent.toml`.

Security note: an in-repo `agent.toml` is repo-controlled input. Its
privilege-widening keys (allow rules, external hooks, sandbox exclusions/
relaxations, MCP servers, `[web].allowed_domains`) are filtered through the trust-on-first-use policy in
`agent_core/trust.py` — inert on an untrusted clone until approved. Because this
repo's own `agent.toml` declares MCP servers, working here interactively prompts
once to trust them (headless runs drop them with a warning). Never extend the set
of keys repo config can widen.

## Testing & Acceptance

- The full suite stays green (`pytest`, currently ~780 tests); run the focused
  file(s) for the area touched plus the full suite for shared behavior.
- Security-affecting changes ship with paired tests: the blocked case, the
  allowed case, and the degradation path (what happens when the mechanism
  itself fails).
- Provider changes: test streaming AND non-streaming paths.
- Tool changes: test permissions, workspace confinement, and failure output.
- Config changes: test precedence and unknown-key tolerance.
- Memory changes: test disabled and enabled modes.
- UI changes: keep non-interactive runs byte-stable.
- Hook/loop-ordering changes: assert event order, and manually inspect a real
  `runs/*.jsonl` + transcript once — unit assertions alone don't count as
  acceptance for message-injection changes.
- New subprocess calls: single non-stacked timeout + kill-and-await; new
  external calls: bounded input projection, never the full history.

## When Changing Code

- Prefer existing seams over new abstractions; when a new abstraction is
  warranted, optimize for the long-term shape: clear contracts, safe execution,
  observability, testability.
- Keep changes reversible and flag-gated when they alter runtime behavior;
  conservative defaults.
- Update this file, `agent.toml.example`, and `revision guide.md` (if the
  roadmap is affected) in the same change.
- Do not copy external projects' code, naming, or prompt text verbatim; adapt
  ideas into this project's own vocabulary and justify them by this project's
  goals.
