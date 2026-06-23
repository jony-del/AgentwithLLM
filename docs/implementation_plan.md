# Terminal UI Alignment Plan

This document outlines the plan to overhaul the project's terminal output, transitioning from a basic "log-style" print out to a beautiful, interactive "message stream component-style" interface that visually aligns with Anthropic's Claude Code.

## User Review Required

Please review the open questions below regarding `prompt_toolkit` key bindings and TTY fallbacks, and confirm if you are happy with the phased approach.

## Open Questions

> [!WARNING]
> 1. **Windows Console Compatibility:** `prompt_toolkit` and `rich` generally handle Windows well, but certain keybindings like `Ctrl+O` might conflict with OS shortcuts or terminal emulators in some environments. Do you have a preferred fallback key if `Ctrl+O` behaves unexpectedly?
> 2. **Fallback for Non-TTY environments:** The current `ConsoleUI` gracefully falls back to `NullUI` when not in an interactive terminal. We will ensure the new `rich` renderer disables colors/complex UI if the output is piped, but do we still want to keep `NullUI` as the base implementation for non-TTY? (Assume yes unless specified).

## Proposed Changes

We will execute this in a phased approach, migrating the existing `ConsoleUI` to a new `agent_core/terminal` package.

### Phase 1: Foundational Rendering (Rich & Prompt Toolkit)
Introduce the new dependencies and set up the new `terminal` package structure. `ConsoleUI` will delegate its rendering to a `TerminalRenderer`.

#### [MODIFY] [pyproject.toml](file:///E:/ZNGZ/Code_copy/learning_proj/AgentwithLLM/pyproject.toml)
- Add `rich` and `prompt_toolkit` to the `dependencies` array.

#### [NEW] [agent_core/terminal/__init__.py](file:///E:/ZNGZ/Code_copy/learning_proj/AgentwithLLM/agent_core/terminal/__init__.py)
- Package initialization.

#### [NEW] [agent_core/terminal/theme.py](file:///E:/ZNGZ/Code_copy/learning_proj/AgentwithLLM/agent_core/terminal/theme.py)
- Define standard colors, layout symbols (e.g., `╭─`, `│`, `╰─`, `●`, `⎿`), and spacing rules using `rich.theme.Theme`.

#### [NEW] [agent_core/terminal/app.py](file:///E:/ZNGZ/Code_copy/learning_proj/AgentwithLLM/agent_core/terminal/app.py)
- Implement `TerminalRenderer`, an abstraction over `rich.console.Console`.

#### [MODIFY] [agent_core/ui.py](file:///E:/ZNGZ/Code_copy/learning_proj/AgentwithLLM/agent_core/ui.py)
- Refactor `ConsoleUI` to delegate formatting and printing to `TerminalRenderer` rather than using standard `print` and manual ANSI escape codes. 
- Retain the streaming logic, but upgrade the styling using `rich.text.Text`.

---

### Phase 2: Claude Code Style Tool Cards
Implement customized display logic for different tool executions instead of a generic "Tool ok" response.

#### [NEW] [agent_core/terminal/components.py](file:///E:/ZNGZ/Code_copy/learning_proj/AgentwithLLM/agent_core/terminal/components.py)
- Define `ToolCard`, `MessageBlock`, and `DiffBlock` utilizing `rich.panel.Panel`, `rich.syntax.Syntax`, and `rich.tree.Tree`.

#### [MODIFY] [agent_core/tools/base.py](file:///E:/ZNGZ/Code_copy/learning_proj/AgentwithLLM/agent_core/tools/base.py) or related
- Introduce a `ToolDisplayProvider` protocol or mixin for tools to dictate how their arguments and results should be rendered.

#### [MODIFY] [agent_core/ui.py](file:///E:/ZNGZ/Code_copy/learning_proj/AgentwithLLM/agent_core/ui.py)
- Update `on_tool_call` and `on_tool_result` to use the new `ToolCard` components based on the specific tool being executed. Implement specialized beautiful diffs for `write`/`edit` tools.

---

### Phase 3: Read/Search Folding
Implement the folding mechanism to prevent the terminal from being flooded by repetitive reads/searches.

#### [NEW] [agent_core/terminal/transcript.py](file:///E:/ZNGZ/Code_copy/learning_proj/AgentwithLLM/agent_core/terminal/transcript.py)
- Introduce a terminal-specific message model that manages UI state (like `is_collapsed`).

#### [MODIFY] [agent_core/terminal/app.py](file:///E:/ZNGZ/Code_copy/learning_proj/AgentwithLLM/agent_core/terminal/app.py)
- Track consecutive `read`/`search`/`list` tool calls.
- Render them compactly (e.g., `● Searched 4 patterns, read 7 files`).
- Introduce the `--verbose` flag rendering logic. *(Note: The dynamic `Ctrl+O` toggle will be fully wired in Phase 4 when `prompt_toolkit` handles the event loop).*

---

### Phase 4: Interactive Input with Prompt Toolkit
Replace the built-in `input()` in the chat CLI with a rich, multi-line `prompt_toolkit` application.

#### [NEW] [agent_core/terminal/keybindings.py](file:///E:/ZNGZ/Code_copy/learning_proj/AgentwithLLM/agent_core/terminal/keybindings.py)
- Define `prompt_toolkit` key bindings for `Enter` (Send), `Shift+Enter` (Newline), `Esc` (Interrupt), and `Ctrl+O` (Toggle Details).

#### [MODIFY] [agent_core/cli.py](file:///E:/ZNGZ/Code_copy/learning_proj/AgentwithLLM/agent_core/cli.py)
- Refactor the `_async_input` function in `chat_command` to use a `prompt_toolkit.PromptSession` running asynchronously.
- Draw the bottom input box: `╭─ Message ─╮ ... ╰──╯`.

---

### Phase 5: Interactive Permissions Block
Enhance the current simple `[y/a/n]` prompt into a visually distinct block.

#### [MODIFY] [agent_core/ui.py](file:///E:/ZNGZ/Code_copy/learning_proj/AgentwithLLM/agent_core/ui.py)
- Refactor `confirm_tool` to render a `rich.panel.Panel` (e.g., `╭─ Permission Required ─╮`).
- We will leverage `prompt_toolkit` to handle the keypress for `y`/`a`/`n` smoothly without requiring `Enter` if possible, otherwise keep standard input but visually framed.

---

### Phase 6: Final Run Summary (Recap)
Present a neat summary at the end of a run.

#### [MODIFY] [agent_core/react.py](file:///E:/ZNGZ/Code_copy/learning_proj/AgentwithLLM/agent_core/react.py)
- At the end of `run()`, collect statistics from the executed tools.

#### [MODIFY] [agent_core/ui.py](file:///E:/ZNGZ/Code_copy/learning_proj/AgentwithLLM/agent_core/ui.py)
- Implement `on_run_completed(stats)` to print the recap using `rich.table.Table` or `rich.columns.Columns` (e.g., `✓ Churned for 2m 18s \n Read 9 files...`).

## Verification Plan

### Automated Tests
- Run `pytest` to ensure core event routing and hooks in `ReActAgent` are not broken by `ConsoleUI` changes.
- Ensure tests relying on `NullUI` (which is the default) remain unaffected and silent.

### Manual Verification
- Run `polaris chat --provider fake` and verify the terminal UI rendering, interactive `prompt_toolkit` bottom bar, and formatting.
- Execute commands requiring tools to verify `ToolCard` styling, diffs, and the collapsible read/search logic.
- Trigger a dangerous tool to verify the Permission Box format.
