# CLAUDE.md

This file gives Claude Code the high-value context needed to work in this
repository without rereading the whole codebase.

## Project

This is a Python ReAct agent framework (Python >= 3.11) growing toward
Claude-Code-level capability.

The goal is a feature-rich, high-performance agent. Adding third-party
dependencies is acceptable when a capability needs them, but keep the core lean:

- `import agent_core` must not require optional heavy dependencies.
- Put optional capabilities behind extras and lazy imports.
- Current extras include `mcp`, `mcp-servers`, and `web`.
- The Claude provider intentionally uses stdlib `urllib`, not the `anthropic`
  SDK.

Prefer local project patterns over new abstractions. Keep changes scoped and add
tests proportional to the behavior touched.

## Commands

```powershell
# Run a single deterministic task without an API key
python -m agent_core run "Say hello without tools" --provider fake

# Interactive chat loop
python -m agent_core chat --provider fake

# Use the real Claude API
$env:ANTHROPIC_API_KEY="your-key"
python -m agent_core run "Use the echo tool" --provider claude --model claude-haiku-4-5-20251001

# Live display flags
python -m agent_core run "Think, then answer" --provider claude --thinking-budget 1024
python -m agent_core run "Say hello" --provider fake --no-stream
python -m agent_core run "Say hello" --provider fake --quiet

# Tests
pip install -e ".[dev]"
pytest
pytest tests/test_react.py
pytest tests/test_react.py::test_react_executes_demo_tool
```

`pyproject.toml` sets `pythonpath = ["."]`, so tests can import `agent_core`
without installation. There is no lint/format tooling configured.

## Code Map

- `agent_core/react.py`: central `ReActAgent.run()` loop.
- `agent_core/models.py`: dataclass contracts between layers:
  `Message`, `ToolCall`, `ToolResult`, `LLMResult`.
- `agent_core/providers/`: LLM providers. `FakeProvider` is deterministic for
  tests and demos. `ClaudeProvider` maps messages/tools to the Anthropic
  Messages API, supports streaming, and preserves extended-thinking blocks.
- `agent_core/tools/`: tool base classes, registry, executor, built-ins, editing
  tools, web tools, planning, and subagent dispatch.
- `agent_core/permissions.py`: permission modes and risk decisions.
- `agent_core/hooks.py`: pre/post tool execution hooks.
- `agent_core/compression.py`: proactive and reactive context compaction.
- `agent_core/config.py`: config resolution from defaults, `agent.toml`, env,
  and CLI flags.
- `agent_core/memory/`: optional cross-conversation memory.
- `agent_core/mcp/`: MCP client, config, and tool adapter.
- `agent_core/ui.py`: live terminal UI event sink.
- `agent_core/storage.py`: JSONL run logging.
- `agent_core/session.py`: per-run state shared with session-aware tools.

## Core Flow

`ReActAgent.run()` is the central loop:

1. Auto-compact history if needed.
2. Call the configured `LLMProvider`.
3. If the result has no tool calls, return the final answer.
4. Execute tool calls through `ToolExecutor`.
5. Append tool observations as `tool` messages and continue.

There is no small fixed step cap by default. The loop relies on safety guards:
cooperative cancellation, optional `max_steps`, and `max_wall_seconds`.

The provider's `complete()` method may stream deltas to a UI, but it must still
return one fully assembled `LLMResult`.

## Important Invariants

- Treat `Message`, `ToolCall`, `ToolResult`, and `LLMResult` as cross-layer
  contracts. Change them deliberately and update affected tests.
- Preserve `LLMResult.thinking_blocks` and their signatures. Claude's API needs
  prior thinking blocks when extended thinking and tool use span turns.
- Keep optional dependencies out of the core import path. Use extras plus lazy
  imports for capabilities such as MCP and web access.
- Every tool must set the correct `ToolRisk`: `READ`, `WRITE`, or `DANGEROUS`.
  Permission behavior depends on this.
- Workspace-scoped tools should use `WorkspacePathMixin`; do not allow absolute
  path or `..` escapes.
- Tools self-register with `@builtin_tool`. Adding a built-in tool should not
  require editing `react.py`.
- Session-aware tools use `SessionAwareMixin` and are rebound by
  `ReActAgent.__init__`. Avoid importing `ReActAgent` from tools; use the
  session seam instead.
- Subagents must not receive the `dispatch_agent` tool, otherwise recursion can
  escape the intended limit.
- `NullUI` is the default and must remain silent/non-interactive for tests and
  library use. `ConsoleUI` is wired only for interactive CLI runs.
- Streaming UI hooks are finalizers when deltas were already printed; do not
  duplicate streamed content.
- Memory is off by default. Recall happens at run start; extraction happens only
  after natural termination and must never fail an otherwise completed run.
- `runs/` and `memory/` are runtime state and are gitignored.

## Configuration Notes

Scalar config precedence is:

1. built-in defaults
2. `agent.toml`
3. environment variables
4. CLI flags

Memory, output truncation, and MCP tables are resolved separately. Live display
knobs such as `--quiet`, `--no-stream`, and `--thinking-budget` are CLI-only.

See `agent.toml.example` for supported configuration shape.

## When Changing Code

- Run focused tests for the area touched; run the full suite for shared behavior.
- For provider changes, test streaming and non-streaming paths where applicable.
- For tool changes, test permissions, workspace confinement, and failure output.
- For config changes, test precedence and unknown-key behavior.
- For memory changes, test both disabled and enabled modes.
- For UI changes, keep non-interactive runs stable.
