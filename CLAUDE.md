# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A Python ReAct agent framework (Python >= 3.11) growing toward Claude-Code-level capability. **The goal is a feature-rich, high-performance agent; adding a third-party dependency is fine when a capability needs it**. The pattern, though, is to keep the *core* lean and importable without optional extras, and put heavier capabilities behind **opt-in extras with lazy imports** (so `import agent_core` never requires them):

- `mcp` — the official MCP SDK (`pip install -e ".[mcp]"`); imported lazily only when `[mcp.servers.*]` is configured.
- `mcp-servers` — the SDK plus pip reference servers (git/fetch/time).
- `web` — `httpx`/`beautifulsoup4`/`markdownify`/`ddgs` for the `web_fetch`/`web_search` tools; imported lazily inside those tools.

The Claude provider still uses stdlib `urllib` (not the `anthropic` SDK). When you add a capability that needs a library, prefer a new extra + lazy import over bloating the core import path.

## Commands

```powershell
# Run a single task (fake provider needs no API key; deterministic)
python -m agent_core run "Say hello without tools" --provider fake

# Interactive chat loop
python -m agent_core chat --provider fake

# Use the real Claude API
$env:ANTHROPIC_API_KEY="your-key"
python -m agent_core run "Use the echo tool" --provider claude --model claude-sonnet-4-6

# Live display flags (CLI-only; auto-on in a TTY): stream tokens, show thinking
python -m agent_core run "Think, then answer" --provider claude --thinking-budget 1024
python -m agent_core run "Say hello" --provider fake --no-stream   # render per-turn, not token-by-token
python -m agent_core run "Say hello" --provider fake --quiet       # suppress the live trace entirely

# Tests (pytest is the only dev dependency)
pip install -e ".[dev]"
pytest                              # all tests
pytest tests/test_react.py          # one file
pytest tests/test_react.py::test_react_executes_demo_tool   # one test
```

`pyproject.toml` sets `pythonpath = ["."]`, so tests import `agent_core` without installation. There is no lint/format tooling configured.

## Architecture

`ReActAgent.run()` ([agent_core/react.py](agent_core/react.py)) is the central loop. Each step:

1. `CompressionPipeline.maybe_auto_compact()` shrinks history if it exceeds a char threshold (proactive).
2. The `LLMProvider` is called. If it raises `LLMContextTooLongError`, `reactive_compact()` runs (aggressive) and the call is retried once.
3. If the result has no `tool_calls`, that content is the final answer and the loop returns.
4. Otherwise each `ToolCall` goes through the `ToolExecutor`, and the `ToolResult` is appended as a `tool`-role `Message` (the observation) for the next step.

The loop is a `while True` whose **primary** exit is the model returning no tool calls — a task can take as many tool turns as it needs (like Claude Code, there is no small fixed step cap). Three safety-net guards, checked at the top of each turn (the cancel check is also re-checked between tool calls), only catch runaway/stuck/unwanted loops: a cooperative `should_cancel` signal, an optional `max_steps` hard ceiling (default `None` = no cap), and `max_wall_seconds` (default 300s wall-clock deadline). Hitting any returns via `_stopped(...)`. This complements the provider's own per-request `timeout`, which bounds a single LLM call.

Throughout the loop the agent emits display events to an `AgentUI` (see the UI layer below): `on_turn_start()` before each LLM call, streamed token deltas during it, then `on_thinking`/`on_reasoning`/`on_final` (and tool events from the executor). With the default `NullUI` these are no-ops, so `run()` is unchanged for tests/library use.

`should_cancel: Callable[[], bool]` is how Esc-to-interrupt is wired (cooperative cancellation — the keypress sets a flag, the loop polls it and unwinds; it never kills a thread). The CLI supplies it via `KeyInterrupt` ([agent_core/interrupt.py](agent_core/interrupt.py)), a context manager that watches the terminal for Esc on a daemon thread (`msvcrt` on Windows, `termios`/`select` on POSIX) and is a no-op when stdin is not a TTY (pipes, tests).

Everything flows through the dataclasses in [agent_core/models.py](agent_core/models.py): `Message`, `ToolCall`, `ToolResult`, `LLMResult`. These are the contracts between layers — change them deliberately. `LLMResult` also carries `thinking` (human-readable extended-thinking text for display) and `thinking_blocks` (the raw thinking/redacted blocks **with their signatures**, stashed on the assistant `Message.metadata` and replayed by `ClaudeProvider` — the API requires the prior turn's thinking block when thinking and tool use span turns).

### Layers and their seams

- **Providers** ([agent_core/providers/](agent_core/providers/)) implement `LLMProvider.complete(messages, tools, config, stream=None) -> LLMResult`. The optional `stream` is a `StreamHandler` (defined in [base.py](agent_core/providers/base.py); `AgentUI` satisfies it structurally) the provider pushes token deltas to as they arrive — `complete()` still returns one fully-assembled `LLMResult` either way, so the loop is agnostic to streaming. `FakeProvider` is deterministic and key-free for tests/demos (it triggers a tool when the user message contains `tool:`, can simulate a context-overflow once via `fail_once_context`, and chunks its answer to `stream` so `--provider fake` demos streaming key-free). `ClaudeProvider` translates `Message`s into the Anthropic Messages API and maps HTTP 400s mentioning "context"/"token" to `LLMContextTooLongError`. It optionally enables **extended thinking** (`thinking_budget` in the provider config forces `temperature=1` and bumps `max_tokens > budget`) and, when given a `stream`, opens an **SSE** stream and parses `text`/`thinking`/`tool_use` content-block deltas by hand (`_consume_stream`/`_iter_sse_events`, stdlib only). Connection setup is retried with backoff (shared `_request_with_retry`); a mid-stream break raises `LLMTransientError` rather than reprinting half-shown output.
- **Tools** ([agent_core/tools/](agent_core/tools/)) subclass `Tool` with a `name`, `description`, `input_schema`, and a `risk` (`ToolRisk.READ`/`WRITE`/`DANGEROUS`). `ToolRegistry` holds them; `ToolExecutor.execute()` orchestrates pre-hooks → permission decision → optional dry-run → `tool.run()` → post-hooks, logging at each stage. `risk` is what the permission layer keys off of, so set it correctly on new tools. File/command access is confined to the workspace by `WorkspacePathMixin` ([base.py](agent_core/tools/base.py)), whose `resolve_workspace_path` rejects `..`/absolute escapes. The default registry ([`ReActAgent.default_registry`](agent_core/react.py)) ships these built-ins:
  - **Navigation/read** (READ): `echo`; `read_text_file` (optional `offset`/`limit` paging); `list_dir`; `search_text` (content grep); `glob` (file-name match, newest-first — [tools/editing.py](agent_core/tools/editing.py)); `git_diff`.
  - **Edit** (WRITE): `write_text_file`; `edit_file` (exact-string replace, unique-unless-`replace_all`); `multi_edit` (a sequence of exact edits to one file, atomic — all-or-nothing); `apply_patch` (context-anchored unified diff across files, atomic) — the last two in [tools/editing.py](agent_core/tools/editing.py), reusing `edit_file`'s shared `_apply_exact_edit` helper.
  - **Exec** (DANGEROUS — denied in `auto`/`dontask`, confirmed in `default`): `run_command`; `run_tests`.
  - **Planning/orchestration**: `update_todos` (READ; a Claude-Code-style TodoWrite — [tools/planning.py](agent_core/tools/planning.py)) and `dispatch_agent` (WRITE; spawn a sub-agent — [tools/subagent.py](agent_core/tools/subagent.py)). Both are **session-aware** (see the SessionContext seam below), not workspace-scoped.
  - **Web** (READ; needs the `web` extra, imported lazily): `web_fetch` (URL → markdown, with an SSRF guard that re-checks every redirect hop) and `web_search` (keyless `ddgs` by default; Brave/Tavily when `BRAVE_API_KEY`/`TAVILY_API_KEY` is set) — [tools/web.py](agent_core/tools/web.py).

  The core file/exec built-ins live in [tools/builtin.py](agent_core/tools/builtin.py). Subprocess tools run with the workspace as cwd, capture combined stdout/stderr + exit code, and treat a non-zero exit as `ok=False` (the output is still returned). No LSP — that remains a stub (see extension points).

  **MCP** ([agent_core/mcp/](agent_core/mcp/)) is real (LCP/LSP are not). `[mcp.servers.<name>]` tables in `agent.toml` (resolved by `resolve_mcp_config`) each connect one MCP server over **stdio** (subprocess) or **streamable-http** (the 2025 single-endpoint transport); `MCPClientManager` ([mcp/client.py](agent_core/mcp/client.py)) bridges the **synchronous** agent to the **async** `mcp` SDK by owning one asyncio event loop on a background thread — a single long-lived `_serve` coroutine opens every server inside one `AsyncExitStack`, then awaits a stop `Event` so the stack unwinds in the same task (the anyio cancel-scope rule), while sync `MCPTool.run()` submits `session.call_tool` via `run_coroutine_threadsafe(...).result()`. `MCPAdapter` ([mcp/adapter.py](agent_core/mcp/adapter.py)) wraps each remote tool as a `Tool` named `"<server>__<tool>"` (collision-proof) with the server's configured `risk` (default `DANGEROUS`, overridable per server to `read`/`write`). The CLI wires this in `build_agent` (only when a server is configured) and tears it down in `run`/`chat`; `python -m agent_core mcp list` shows discovered tools.

  Tools **self-register**: each class is tagged `@builtin_tool` ([tools/catalog.py](agent_core/tools/catalog.py)), which appends it to a module-level list at import time. `catalog.discover()` lazily imports every submodule of the `tools` package (via `pkgutil`, at `default_tools()` call time — no import-time cycle) so all decorators have fired, and `default_tools(workspace=None, session=None)` builds the instances (passing `workspace` to `WorkspacePathMixin` subclasses, `session` to `SessionAwareMixin` subclasses). `ReActAgent.default_registry()` just registers `default_tools()`. **Adding a tool = drop a `@builtin_tool`-decorated class into a file in the package; `react.py` and the catalog need no edits.**

  **SessionContext seam** ([agent_core/session.py](agent_core/session.py)): most tools need only a workspace path, but the planning and sub-agent tools need *per-run shared state*. `SessionContext` holds the mutable `TodoStore`, a `subagent_factory` (a closure over the agent — so `dispatch_agent` can spawn a child `ReActAgent` **without importing the agent class**, avoiding an import cycle), a `ui_notify` callback (so a tool can surface a change to the live UI without holding a UI reference — `update_todos` uses it to drive `AgentUI.on_todos`), and `depth`/`max_depth` (sub-agent recursion ceiling). A tool opts in by subclassing `SessionAwareMixin` (which sets `needs_session = True` and a `bind_session`). `ReActAgent.__init__` builds one `SessionContext` and **rebinds every session-aware tool in the registry to it** — this matters because the CLI builds the registry *before* the agent exists (and registers MCP tools into it), so binding-after-construction is how the live session reaches those tools. `_spawn_subagent` builds the child with a narrowed tool set (`read_only` = READ tools; `full` = +WRITE) that **never includes `dispatch_agent`** (so sub-agents can't recurse), runs it under a `NullUI`, and returns only its final answer.
- **Permissions** ([agent_core/permissions.py](agent_core/permissions.py)) map a `PermissionMode` × `ToolRisk` to a `PermissionDecision` (allow / deny / dry-run / ask-user). Modes: `default`, `acceptedits`, `plan` (writes become dry-runs), `auto`, `dontask`. An "ask" is resolved through an injected `prompter` (the live `ConsoleUI.confirm_tool`, wired only when the UI is interactive) returning `once`/`always`/`deny`; `always` adds the tool to a session-allow set so it isn't re-asked. With no prompter (non-interactive: pipes, tests, `NullUI`) an "ask" becomes a denial.
- **Hooks** ([agent_core/hooks.py](agent_core/hooks.py)) are pre/post interceptors around tool execution. A pre-hook can rewrite a `ToolCall` or block it; a post-hook can rewrite the `ToolResult`. The agent installs `MaxOutputPostHook` by default: when a single tool's output exceeds a **line** budget (`max_lines`, default 100) — or, for pathological single-line output, a **char** budget (`max_chars`) — it keeps only head+tail with a `[... omitted N lines ...]` notice, and spills the full, untouched output to `runs/outputs/` (path appended to the result and recorded in its metadata). So only the truncated version enters context while the complete result stays retrievable.
- **Compression** ([agent_core/compression.py](agent_core/compression.py)) runs three sequential stages — `snip` (truncate long tool outputs), `microcompact` (cap any long message), `context_collapse` (summarize old messages, keep recent N). `aggressive=True` halves the budgets. Char counts are the proxy for token budget throughout.
- **Memory** ([agent_core/memory/](agent_core/memory/)) is cross-conversation state, **off by default** (gated on `config.memory.enabled`; when off, `run()` is unchanged and nothing is written). `ReActAgent` wires three pieces from `MemoryConfig`: at the **start** of `run()`, `MemoryRetriever.recall()` scores stored memories by `relevance × importance × recency` (lexical only — no embeddings; `text.py` has the shared `tokenize`/`lexical_relevance`) and injects the top-`k` as a pinned `system` message tagged `metadata={"memory":"recall"}`; on **natural termination only**, `MemoryExtractor.extract()` asks the LLM (best-effort, wrapped so it never fails a finished run) to distil durable memories as JSON. `MemoryStore` ([store.py](agent_core/memory/store.py)) persists `MemoryRecord`s to `memory/memory.jsonl` with an atomic rewrite (unlike the append-only run log, memories are mutable). `Dreamer.dream()` ([dreaming.py](agent_core/memory/dreaming.py)) is the offline consolidation pass — decay/forget → merge near-duplicates → LLM insight synthesis — and supports `commit=False` for a side-effect-free dry run (it snapshots records via `dataclasses.replace` first). `FakeProvider` returns deterministic JSON when it sees `MEMORY_EXTRACTION_MARKER`, so the whole pipeline is exercisable key-free. CLI: `--memory`/`--no-memory`, `dream [--dry-run]`, `memory list|add|forget`.
- **Storage** ([agent_core/storage.py](agent_core/storage.py)): `JSONLRunLogger` appends one JSON record per event to `runs/<timestamp>-<uuid>.jsonl`. The loop logs `user`, `compression`, `llm`, `tool_pre`, `permission`, `tool_result`, `final`, and (when memory is on) `memory_recall`/`memory_extract` events — this file is the primary trace for debugging a run. `runs/` and `memory/` are gitignored.
- **UI / display** ([agent_core/ui.py](agent_core/ui.py)) is the live terminal trace, decoupled from the loop via an `AgentUI` event sink. `NullUI` (the default) is silent and `confirm_tool` denies, so non-interactive runs are byte-for-byte unchanged; `ConsoleUI` renders a Claude-Code-style trace and the interactive `y`/`a`/`n` permission panel, and is wired by the CLI **only when both stdin and stdout are TTYs** and `--quiet` isn't set (same gating spirit as `KeyInterrupt`). It supports two coexisting modes: **streaming** (the provider pushes `on_text_delta`/`on_thinking_delta`/`on_tool_args_delta` and `on_thinking`/`on_reasoning`/`on_final` act as *finalizers* — close the streamed block, never reprint) and **per-turn** (no deltas this turn → those hooks print the full section). `on_turn_start()` resets the per-turn streamed-state flags. `is_live` lets the CLI skip its own final-answer print (the UI already showed it) and lets `ReActAgent` decide whether to wire the interactive prompter.

### Config precedence

`resolve_config()` ([agent_core/config.py](agent_core/config.py)) merges the scalar settings, lowest to highest priority: built-in defaults → `agent.toml` (optional, via `tomllib`; only top-level scalars) → `AGENT_MODEL`/`AGENT_PERMISSION`/`AGENT_PROVIDER` env vars → CLI flags. `None` values never override. Memory is resolved separately by `resolve_memory_config()`: the `[memory]` toml **table** supplies all tunables, and only `enabled` is then overridable by `AGENT_MEMORY` → `--memory`/`--no-memory`. The `[output]` table (tool-output truncation limits) is resolved the same way by `resolve_output_config()`, and the `[mcp]` table (servers, via the `[mcp.servers.<name>]` sub-tables) by `resolve_mcp_config()`. All use the shared `overlay_dataclass()`/`coerce_to_type()` helpers in [config.py](agent_core/config.py) to build a dataclass from a toml table (unknown keys ignored, values coerced to field types; `overlay_dataclass` passes list/dict fields like a server's `args`/`env`/`headers` through untouched). See [agent.toml.example](agent.toml.example).

The live-display knobs are deliberately **CLI-only** (not read from `agent.toml` or env): `--quiet`, `--no-stream`, and `--thinking-budget N` map straight onto `ReActConfig` in `build_agent` ([cli.py](agent_core/cli.py)) without going through `resolve_config`. They only take effect with the live `ConsoleUI`.

### Placeholders / extension points

`LCPAdapter` ([agent_core/tools/adapters.py](agent_core/tools/adapters.py)) is an intentionally minimal stub (`list_tools()` returns `[]`), marking where local/custom-protocol tool integration is meant to grow. (`MCPAdapter` is no longer a stub — it now lives in [agent_core/mcp/](agent_core/mcp/); see the Tools seam above.) `MultiAgentCoordinator` ([agent_core/agents/multi.py](agent_core/agents/multi.py)) is also no longer a stub — it runs several `SubAgent`s over one task concurrently (`ThreadPoolExecutor`, one child's failure isolated as an `[error] …` string). The single-child path used by `dispatch_agent` goes through `SessionContext.subagent_factory` directly; the coordinator is the fan-out-to-many path.
