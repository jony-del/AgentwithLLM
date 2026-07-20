# Tool platform

Polaris tools share a session lifecycle, permission policy, sandbox scope, process
supervisor and JSONL audit stream. Core tools are advertised at startup; less common
Notebook, LSP, Worktree, Scheduler, Config and MCP-resource tools are activated through
`tool_search` only when configured and permitted.

## Shell and tasks

Use `bash` for Bash syntax and `powershell` for PowerShell syntax. Both accept `command`,
`timeout`, `description`, `run_in_background` and `dangerously_disable_sandbox`.
Foreground commands automatically become background tasks after the configured threshold;
pressing Ctrl+B moves the same running process into the background without restarting it.
`task_output` reads bounded output (including completed-task history after a Polaris restart)
and `task_stop` terminates the process tree. Session end stops every unfinished task.

Windows agent runs require Git for Windows Bash. Polaris rejects the WindowsApps/WSL shim;
set `POLARIS_BASH_PATH` or `[tools.shell.bash].executable` when Git is installed outside its
standard locations. PowerShell prefers `pwsh`, then Windows PowerShell 5.1.

## Deferred capabilities

- `notebook_edit` edits cells by stable ID after `read_text_file` records a notebook
  fingerprint. External changes make the edit fail closed.
- `lsp` lazily starts the configured server for a file extension and supports definitions,
  references, symbols, hover, implementations, call hierarchy and diagnostics.
- `enter_worktree` and `exit_worktree` switch only the current session. Removal verifies
  dirty files and new commits and never merges automatically.
- `cron_create`, `cron_list` and `cron_delete` store jobs in SQLite WAL. The daemon only
  routes prompts to live-agent queues; it never creates an Agent or calls a model.
- `config` reads effective settings. Allowlisted user-setting writes are atomic and always
  require interactive approval.

See `agent.toml.example` for every `[tools]` setting.

## Scheduler service

The supported installers register the scheduler as a least-privilege current-user service
(Windows Task Scheduler, systemd user service, or a macOS LaunchAgent). Installation fails
with diagnostics when the platform user-service manager is unavailable. Manage it directly
when developing from source:

```console
polaris scheduler-service install
polaris scheduler-service status
polaris scheduler-service uninstall
polaris scheduler-service uninstall --purge-data
```

The service computes due times and writes delivery records only. A live main session or
teammate publishes a TTL heartbeat and drains its own queue after a complete `agent.run()`;
the daemon never starts a headless agent. Overlapping occurrences coalesce. A recurring job
missed while offline is delivered at most once on resume and then rescheduled from the
current time. A missed one-shot is reported in headless mode and requires an interactive
Run-now/Discard decision before it can execute.

Service removal is receipt-checked. Ordinary uninstall preserves scheduler history; only
`--purge-data` removes the SQLite database.

## LSP configuration

Servers are opt-in and repo-defined server commands are TOFU-gated. Each server starts lazily
for its mapped extension in a per-call scope with a read-only workspace, denied network, and
a private writable temp directory:

```toml
[tools.lsp]
autodetect = false
max_restarts = 3

[[tools.lsp.servers]]
name = "pyright"
command = "pyright-langserver"
args = ["--stdio"]
extensions = { ".py" = "python" }
timeout = 15
```
