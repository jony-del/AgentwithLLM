# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A minimal, dependency-free Python ReAct agent framework (Python >= 3.11). The standard library is the only runtime dependency — the Claude provider uses `urllib`, not the `anthropic` SDK. Keep it that way unless explicitly asked to add a dependency.

## Commands

```powershell
# Run a single task (fake provider needs no API key; deterministic)
python -m agent_core run "Say hello without tools" --provider fake

# Interactive chat loop
python -m agent_core chat --provider fake

# Use the real Claude API
$env:ANTHROPIC_API_KEY="your-key"
python -m agent_core run "Use the echo tool" --provider claude --model claude-sonnet-4-6

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

`should_cancel: Callable[[], bool]` is how Esc-to-interrupt is wired (cooperative cancellation — the keypress sets a flag, the loop polls it and unwinds; it never kills a thread). The CLI supplies it via `KeyInterrupt` ([agent_core/interrupt.py](agent_core/interrupt.py)), a context manager that watches the terminal for Esc on a daemon thread (`msvcrt` on Windows, `termios`/`select` on POSIX) and is a no-op when stdin is not a TTY (pipes, tests).

Everything flows through the dataclasses in [agent_core/models.py](agent_core/models.py): `Message`, `ToolCall`, `ToolResult`, `LLMResult`. These are the contracts between layers — change them deliberately.

### Layers and their seams

- **Providers** ([agent_core/providers/](agent_core/providers/)) implement `LLMProvider.complete(messages, tools, config) -> LLMResult`. `FakeProvider` is deterministic and key-free for tests/demos (it triggers a tool when the user message contains `tool:`, and can simulate a context-overflow once via `fail_once_context`). `ClaudeProvider` translates `Message`s into the Anthropic Messages API and maps HTTP 400s mentioning "context"/"token" to `LLMContextTooLongError`.
- **Tools** ([agent_core/tools/](agent_core/tools/)) subclass `Tool` with a `name`, `description`, `input_schema`, and a `risk` (`ToolRisk.READ`/`WRITE`/`DANGEROUS`). `ToolRegistry` holds them; `ToolExecutor.execute()` orchestrates pre-hooks → permission decision → optional dry-run → `tool.run()` → post-hooks, logging at each stage. `risk` is what the permission layer keys off of, so set it correctly on new tools.
- **Permissions** ([agent_core/permissions.py](agent_core/permissions.py)) map a `PermissionMode` × `ToolRisk` to a `PermissionDecision` (allow / deny / dry-run / ask-user). Modes: `default`, `acceptedits`, `plan` (writes become dry-runs), `auto`, `dontask`. In non-interactive mode an "ask" becomes a denial.
- **Hooks** ([agent_core/hooks.py](agent_core/hooks.py)) are pre/post interceptors around tool execution. A pre-hook can rewrite a `ToolCall` or block it; a post-hook can rewrite the `ToolResult`. The agent installs `MaxOutputPostHook` by default: when a single tool's output exceeds a **line** budget (`max_lines`, default 100) — or, for pathological single-line output, a **char** budget (`max_chars`) — it keeps only head+tail with a `[... omitted N lines ...]` notice, and spills the full, untouched output to `runs/outputs/` (path appended to the result and recorded in its metadata). So only the truncated version enters context while the complete result stays retrievable.
- **Compression** ([agent_core/compression.py](agent_core/compression.py)) runs three sequential stages — `snip` (truncate long tool outputs), `microcompact` (cap any long message), `context_collapse` (summarize old messages, keep recent N). `aggressive=True` halves the budgets. Char counts are the proxy for token budget throughout.
- **Memory** ([agent_core/memory/](agent_core/memory/)) is cross-conversation state, **off by default** (gated on `config.memory.enabled`; when off, `run()` is unchanged and nothing is written). `ReActAgent` wires three pieces from `MemoryConfig`: at the **start** of `run()`, `MemoryRetriever.recall()` scores stored memories by `relevance × importance × recency` (lexical only — no embeddings; `text.py` has the shared `tokenize`/`lexical_relevance`) and injects the top-`k` as a pinned `system` message tagged `metadata={"memory":"recall"}`; on **natural termination only**, `MemoryExtractor.extract()` asks the LLM (best-effort, wrapped so it never fails a finished run) to distil durable memories as JSON. `MemoryStore` ([store.py](agent_core/memory/store.py)) persists `MemoryRecord`s to `memory/memory.jsonl` with an atomic rewrite (unlike the append-only run log, memories are mutable). `Dreamer.dream()` ([dreaming.py](agent_core/memory/dreaming.py)) is the offline consolidation pass — decay/forget → merge near-duplicates → LLM insight synthesis — and supports `commit=False` for a side-effect-free dry run (it snapshots records via `dataclasses.replace` first). `FakeProvider` returns deterministic JSON when it sees `MEMORY_EXTRACTION_MARKER`, so the whole pipeline is exercisable key-free. CLI: `--memory`/`--no-memory`, `dream [--dry-run]`, `memory list|add|forget`.
- **Storage** ([agent_core/storage.py](agent_core/storage.py)): `JSONLRunLogger` appends one JSON record per event to `runs/<timestamp>-<uuid>.jsonl`. The loop logs `user`, `compression`, `llm`, `tool_pre`, `permission`, `tool_result`, `final`, and (when memory is on) `memory_recall`/`memory_extract` events — this file is the primary trace for debugging a run. `runs/` and `memory/` are gitignored.

### Config precedence

`resolve_config()` ([agent_core/config.py](agent_core/config.py)) merges the scalar settings, lowest to highest priority: built-in defaults → `agent.toml` (optional, via `tomllib`; only top-level scalars) → `AGENT_MODEL`/`AGENT_PERMISSION`/`AGENT_PROVIDER` env vars → CLI flags. `None` values never override. Memory is resolved separately by `resolve_memory_config()`: the `[memory]` toml **table** supplies all tunables, and only `enabled` is then overridable by `AGENT_MEMORY` → `--memory`/`--no-memory`. The `[output]` table (tool-output truncation limits) is resolved the same way by `resolve_output_config()`. Both use the shared `overlay_dataclass()`/`coerce_to_type()` helpers in [config.py](agent_core/config.py) to build a dataclass from a toml table (unknown keys ignored, values coerced to field types). See [agent.toml.example](agent.toml.example).

### Placeholders / extension points

`MCPAdapter` and `LCPAdapter` ([agent_core/tools/adapters.py](agent_core/tools/adapters.py)) and `MultiAgentCoordinator` ([agent_core/agents/multi.py](agent_core/agents/multi.py)) are intentionally minimal stubs (`list_tools()` returns `[]`). They mark where tool-protocol integration and multi-agent coordination are meant to grow.
