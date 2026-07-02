# CLAUDE.md

This file gives Claude Code the high-value context needed to work in this
repository and evolve it toward a capable coding/automation agent.

## Project

A Python ReAct agent framework (Python >= 3.11) growing toward an advanced
coding agent in the spirit of Claude Code and Codex: it should reason over
tasks, use tools safely, modify code, preserve useful context, and stay
debuggable.

Dependency philosophy:

- `import agent_core` MUST succeed without any optional/heavy dependency at
  import time.
- There are currently no extras — runtime, MCP reference-server, and test deps
  are all core, so `pip install -e .` yields a fully working agent.
- Capabilities that are *enabled* load eagerly: import and initialize their deps
  at startup, NOT lazily at first use. (See the eager-loading invariant below;
  this supersedes any older "behind extras / lazy import" guidance.)
- Provider SDK choices MUST NOT leak into the core abstractions.

Prefer existing project patterns over new abstractions, but optimize for the
long-term shape of a serious agent: clear contracts, safe tool execution,
observability, testability.

## Environment

Primary dev environment is **Windows + PowerShell**. Command blocks below are
PowerShell (`$env:NAME="..."`). Use OS-appropriate paths; there is no
lint/format tooling configured (don't reach for ruff/black/mypy).

## Commands

```powershell
# Run a single deterministic task without an API key.
# `polaris` is the preferred installed CLI; `python -m agent_core` also works
# while the internal package is still named agent_core.
polaris run "Say hello without tools" --provider fake
python -m agent_core run "Say hello without tools" --provider fake

# Interactive chat loop (one event loop for the whole session)
polaris chat --provider fake

# Use the real Claude API
$env:ANTHROPIC_API_KEY="your-key"
polaris run "Use the echo tool" --provider claude --model claude-haiku-4-5-20251001

# Live display flags (CLI-only)
polaris run "Think, then answer" --provider claude --thinking-budget 1024
polaris run "Say hello" --provider fake --no-stream
polaris run "Say hello" --provider fake --quiet

# Tests
pip install -e .
pytest
pytest tests/test_react.py
pytest tests/test_react.py::test_react_executes_demo_tool
```

`pyproject.toml` exposes the `polaris` console script and sets
`pythonpath = ["."]`, so tests import `agent_core` without installation.

## Code Map

Core loop & contracts:
- `agent_core/react.py`: central `ReActAgent.run()` loop (async).
- `agent_core/models.py`: cross-layer dataclasses `Message`, `ToolCall`,
  `ToolResult`, `LLMResult`.
- `agent_core/session.py`: per-run state shared with session-aware tools.
- `agent_core/interrupt.py`: `KeyInterrupt` — background Esc watcher exposing a
  cooperative cancellation flag the loop polls at safe points (no-op on
  non-TTY/CI).
- `agent_core/compression.py`: proactive and reactive context compaction, aligned
  with Open-ClaudeCode (token-gated; folds the prefix into a USER summary message;
  round-boundary grouping; 413 head-truncation + circuit breaker). See
  `docs/compaction-openclaudecode-alignment.md`.
- `agent_core/compression_summary.py`: Track A summarizer seam (full 9-section
  summary prompt; bounded, no-tools call; single non-stacked timeout).
- `agent_core/tokens.py`: pure-stdlib model context-window + auto-compact threshold math.
- `agent_core/context.py`: run-start assembly — base system prompt + `systemContext`
  (git folded in as `key: value`) + a pinned `<system-reminder>` `userContext` USER
  message (CLAUDE.md + date), mirroring the reference's `appendSystemContext`/
  `prependUserContext`.

Providers:
- `agent_core/providers/`: `FakeProvider` (deterministic, for tests/demos) and
  `ClaudeProvider` (maps messages/tools to the Anthropic Messages API, streams,
  preserves extended-thinking blocks). `base.py` is the shared interface.

Tools:
- `agent_core/tools/base.py`: `Tool` base, `ToolRisk`, mixins
  (`WorkspacePathMixin`, `SessionAwareMixin`).
- `agent_core/tools/catalog.py`: `@builtin_tool` self-registration +
  `discover()`/`default_tools()`.
- `agent_core/tools/registry.py`, `executor.py`, `adapters.py`: registry,
  `ToolExecutor` (wave partitioning, `max_workers`), and adapter base.
- `agent_core/tools/builtin.py`, `editing.py`, `web.py`, `planning.py`: built-in,
  file-editing, web, and planning tools.
- `agent_core/tools/subagent.py`, `team.py`: sub-agent dispatch and team tools.
- `agent_core/tools/skill.py`: the model-facing `skill` tool (invoke a loaded skill
  by name; inline returns the rendered prompt, fork runs a sub-agent).

Skills / slash commands:
- `agent_core/skills/`: the skill subsystem. `models.py` (`Skill`, `SkillContext`),
  `frontmatter.py` (PyYAML-backed frontmatter parser; owns fence detection + key
  normalisation, degrades to no-metadata on bad YAML), `loader.py`
  (`discover_skill_dirs`/`load_skills`, dedupe by realpath, precedence
  bundled→user→project→extra), `registry.py` (`SkillRegistry`), `dispatch.py`
  (`parse_slash_command`/`looks_like_command`/`render_skill_prompt`/`build_skill_prompt`/
  `fork_preset`), `programmatic.py` + `builtin_programmatic.py` (`@programmatic_skill`
  self-registered Python skills whose prompt is computed at call time, mirroring the
  reference's `getPromptForCommand`), `config.py` (`SkillsConfig`), `bundled/*.md`
  (shipped markdown skills loaded like any other).

Multi-agent:
- `agent_core/agents/multi.py`: `MultiAgentCoordinator`/`SubAgent` parallel
  fan-out over one task.
- `agent_core/agents/team.py`: `TeamStore`/`FileLock` shared-state coordination.

Cross-cutting:
- `agent_core/cli.py`: argparse CLI (real entry for `polaris` / `__main__`).
- `agent_core/chat_commands.py`: built-in interactive-chat `/commands` (`dispatch` →
  `ChatTurn`); also routes `/skill` invocations. CLI-only, imports no `cli`.
- `agent_core/permissions.py`: `PermissionPolicy` — the argument-aware decision
  pipeline (deny rules → sensitive-path safety net → ask rules → sandbox auto-allow →
  `bypass` mode → allow rules → the legacy per-mode `ToolRisk` matrix). No rules + no
  sandbox ⇒ identical to the old coarse gate. Modes: default/acceptedits/plan/auto/
  dontask/bypass.
- `agent_core/permission_rules.py`: cross-platform *policy layer* — `ToolName(content)`
  allow/deny/ask rules, shell exact/prefix/wildcard matching, compound-command
  decomposition (allow needs ALL sub-commands, deny needs ANY) + anti-evasion
  (`SAFE_ENV_VARS`/`BINARY_HIJACK_VARS`/wrapper stripping), path-glob + `domain:` matching.
  Parse failures degrade (drop the rule), never raise.
- `agent_core/sandbox/`: OS *enforcement layer* — `SandboxManager` wraps
  `run_command`/`run_tests` under a **pluggable backend chosen by isolation tier**
  (`config.backend`): `NativeBackend` (bwrap/sandbox-exec, no-op on Windows),
  `ContainerBackend` (podman/docker/nerdctl `run`), `VmBackend` (Kata/Hyper-V/Lima with
  snapshot-rollback). `auto` prefers `container → native → noop` (never auto-selects vm);
  an explicit tier degrades to the next weaker available one; `fail_if_unavailable` turns
  "can't sandbox" into a hard startup error. All backends share the launcher-prefix
  `wrap()` seam; container/vm add a lifecycle (`prepare` at construct, `reset` per run for
  vm, `teardown` at close). Command tools reach the manager via `SandboxAwareMixin`
  (rebound by `ReActAgent.__init__` like `SessionAwareMixin`). See
  `docs/sandbox-openclaudecode-alignment.md`.
- `agent_core/hooks.py`: hook surface + `[hooks]` config dataclasses
  (`HooksConfig`/`BuiltinHooksConfig`/`ExternalHookSpec`). Sync pre/post *tool* hooks run
  inside the executor; async *lifecycle* hooks (`UserPromptSubmit`, `PostSampling`,
  `PreCompact`, `PostCompact`, blockable `Stop`) fire at `react.py` loop
  boundaries via `HookPipeline.run_*`. A `HookOutcome.block` is event-specific (abort
  prompt / skip compaction / keep running). Stop-blocking is bounded by
  `config.max_stop_blocks`. The lifecycle pipeline has **two sources**, assembled by
  `ReActAgent._build_hook_pipeline` from `config.hooks` (default content, no longer empty):
  - `agent_core/builtin_hooks.py`: built-in *programmatic* hooks (hold live objects —
    `StopCompletionHook` reads `session.todos`; observers write `runs/*.jsonl`), toggled by
    `[hooks.builtin]` (observation/control all on by default). Plus `PromptValidationHook`, the
    `UserPromptSubmit` input firewall (`[hooks.prompt_validation]`, **on by default**):
    provenance-based — empty/oversize/control-char prompts are blocked, and reserved framing
    tags inside the *user task* are *neutralized, not rejected* (escaped + wrapped in an
    `<untrusted_user_input>` envelope + guard preamble + audited). It rewrites the task via
    `HookOutcome.transformed_prompt`, which the loop persists/sends in place of the raw prompt.
  - `agent_core/hook_adapters.py`: config-driven *external* hooks from `[[hooks.external]]`
    (`command`/`http`/`prompt`/`agent`), each adapter implements the lifecycle Protocols and
    folds in the same pipeline. Every external call has a single non-stacked timeout +
    kill-await and **degrades to allow + log on any failure — never sink a run**; the context
    crosses the boundary as a bounded JSON projection (`project_hook_input`), never the full
    history. A still-injectable override `hooks=` arg to `ReActAgent` bypasses assembly.
- `agent_core/config.py`: config resolution (defaults → `agent.toml` → env → CLI).
- `agent_core/memory/`: optional cross-conversation memory.
- `agent_core/mcp/`: MCP client, config, tool adapter.
- `agent_core/ui.py`: live terminal UI event sink (`NullUI`/`ConsoleUI`).
- `agent_core/storage.py`: JSONL run logging (inspect a run under `runs/`).

## Agent Loop

`async ReActAgent.run()` is the central loop. Lifecycle hooks fire at fixed
boundaries (shown as `[Hook]`), so hook ordering is part of the loop contract:

0. `[UserPromptSubmit]` once the task is in place, before the first model call —
   may abort the run or inject grounding context.
1. `[PreCompact]` (only when the gate would fold) → auto-compact history if the
   running token estimate crosses the model's threshold (token-gated, not
   char-gated; reactive compaction additionally recovers from a 413) →
   `[PostCompact]` when a fold actually happened. `PreCompact.block` skips the
   proactive fold; it is ignored on the forced reactive path.
2. Call the configured `LLMProvider`, then `[PostSampling]` (fire-and-forget) once
   the assistant turn is recorded.
3. If the result has no tool calls, fire `[Stop]`: a hook may *block* the stop and
   force the loop to keep running (bounded by `max_stop_blocks`); otherwise return
   the final answer.
4. Execute tool calls through `ToolExecutor` (which runs the sync pre/post *tool*
   hooks around each call).
5. Append tool observations as `tool` messages and continue.

Do NOT design around a small fixed step cap — a capable coding agent often needs
many tool turns. Use explicit safety guards instead: cooperative cancellation
(`KeyInterrupt`/Esc, checked at turn boundaries and around tool calls), optional
`max_steps`, and a wall-clock deadline. A run's `deadline` is shared with
sub-agents/teammates so the whole fan-out is bounded by one budget (see
`react.py`); crossing the soft deadline injects a message so the model can land a
final answer before the hard stop. Stop reasons are explicit.

## Async Execution Model

Async-only: every public execution API is `async def` (`run`, `complete`,
`execute_many`, `extract`, `dream`) — no sync twins, no `a`-prefixed variants.
`asyncio.run` appears ONLY at CLI entry points, never in library code (the CLI
does exactly one per command; `chat` keeps one loop for the session).

Blocking work is an internal detail, never public API:

- Ordinary blocking tools implement `_invoke()`; the base `Tool.run()` offloads
  it via `asyncio.to_thread`, throttled by the executor's `max_workers`.
- Stores/logger (`MemoryStore`, `TeamStore`, `JSONLRunLogger`) expose async
  methods whose disk IO runs in `_xxx_sync` internals on worker threads.
- Concurrency budgets: `GatedProvider` (semaphore + token bucket) bounds API
  calls across the fan-out; `ToolExecutor` wave-partitioning serializes tools
  with conflicting declared resources.
- Providers backed by a blocking SDK MUST wrap the call in `asyncio.to_thread`
  inside their own `complete`. Never call blocking IO directly from a coroutine.

## Important Invariants

- Treat `Message`, `ToolCall`, `ToolResult`, `LLMResult` as cross-layer
  contracts. Change them deliberately and update affected tests.
- Preserve `LLMResult.thinking_blocks` and their signatures — Claude's API needs
  prior thinking blocks when extended thinking and tool use span turns.
- Enabled capabilities (MCP, web) are imported normally and eagerly. Do NOT use
  lazy imports / first-use loading for project capabilities; initialize their
  deps at startup. The same applies to any future heavyweight capability
  (browser control, LSP, vector stores, remote runtimes) once included.
- Every tool MUST set the correct `ToolRisk` (`READ`/`WRITE`/`DANGEROUS`) —
  permission behavior depends on it.
- A tool implements EITHER `_invoke()` OR an async `run()` override — never both.
  The executor awaits an overridden `run` on the loop instead of thread-offload.
- Workspace-scoped tools MUST use `WorkspacePathMixin`; NEVER allow absolute-path
  or `..` escapes.
- Tools self-register with `@builtin_tool`; adding a built-in MUST NOT require
  editing `react.py`.
- Session-aware tools use `SessionAwareMixin` and are rebound by
  `ReActAgent.__init__`. Do NOT import `ReActAgent` from tools — use the session
  seam.
- Sub-agents MUST NOT receive the `dispatch_agent` or `skill` tool, or recursion
  escapes the intended limit.
- Skills load eagerly at agent startup (per the eager-loading invariant) into a
  per-run `SkillRegistry` on the session; a malformed skill file MUST degrade to
  fewer/zero skills, never crash construction. The model-facing `skill` tool is
  dropped from the registry when no skill is model-invocable. Skills come from two
  sources merged into one registry: markdown files (`bundled/` + user/project dirs)
  and `@programmatic_skill` self-registered Python skills; a programmatic skill's
  `prompt_fn` MUST degrade to text on error, never raise into the run.
- Built-in chat `/commands` live in `chat_commands.py` (CLI-only; `dispatch` returns
  a `ChatTurn`). They wire to real subsystems (compaction/tokens/sessions/mcp/memory)
  and do NOT exist for the reference's TUI/account/cloud commands. `/model` only
  mutates `config.model` (read per-turn), never rebuilds the agent.
- `NullUI` is the default and MUST stay silent/non-interactive for tests and
  library use; `ConsoleUI` is wired only for interactive CLI runs. Streaming UI
  hooks are finalizers when deltas were already printed — do NOT duplicate
  streamed content.
- Memory is off by default. Recall at run start; extraction only after natural
  termination, and it MUST NOT fail an otherwise completed run.
- Advanced capabilities MUST degrade with an actionable install/config error
  rather than an import-time crash.
- `runs/` and `memory/` are runtime state and are gitignored. User/project skills
  live under `.polaris/skills/`; only `agent_core/skills/bundled/*.md` is in-repo.

## Configuration

Scalar precedence: built-in defaults → `agent.toml` → environment → CLI flags.
Memory, output truncation, skills (`[skills]`), and MCP tables resolve separately;
live-display knobs (`--quiet`, `--no-stream`, `--thinking-budget`) are CLI-only. See
`agent.toml.example` for the supported shape.

## When Changing Code

- Run focused tests for the area touched; run the full suite for shared behavior.
- Provider changes: test streaming and non-streaming paths.
- Tool changes: test permissions, workspace confinement, and failure output.
- Config changes: test precedence and unknown-key behavior.
- Memory changes: test both disabled and enabled modes.
- UI changes: keep non-interactive runs stable.
