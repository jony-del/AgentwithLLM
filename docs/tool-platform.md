# Tool platform

Polaris tools share a session lifecycle, permission policy, sandbox scope, process
supervisor and JSONL audit stream. Core tools are advertised at startup; less common
Notebook, LSP, Worktree, Scheduler, Config and MCP-resource tools are activated through
`tool_search` only when configured and permitted. Configured MCP business tools remain connected but deferred;
`capability_search` can find them and `capability_activate` exposes an exact returned tool
at the next model-call boundary.

## Streaming execution safety

Tool risk and execution timing are separate contracts. Every tool defaults to
`final_only`: it is not authorized or run until the provider's authoritative response
has been reconciled. Audited pure reads and `echo`
declare `speculative_safe`; workspace-native file editors declare `transactional` and run
against a private, path-scoped sparse overlay. Only paths named by filesystem resource
locks are materialized, and commit rejects undeclared changes, symlinks, changed file
types, subtree membership races, and fingerprint conflicts. The real workspace is
unchanged during sampling. A valid turn commits the overlay with atomic replacement;
stream failure, cancellation, schema failure, protocol mismatch, or any transactional
failure rolls the whole overlay back.

Provider capability negotiation is fail-closed. `explicit` providers may emit a
`StreamedToolCall` with a stable call ID, contiguous zero-based tool ordinal, and provider
item identity. `terminal_only` providers still stream display text but never start a tool
from partial JSON; unknown third-party providers get this compatibility default.
`unsupported` providers disable the optimization. Claude proves `message_stop` plus a
valid stop reason, OpenAI Responses proves one `response.completed`, and their completed
tool calls must exactly match the authoritative response. OpenAI-compatible endpoints use
`terminal_only`; Fake emits deterministic explicit events for offline regression. A
silent EOF or incomplete/error terminal event invalidates the round and cannot commit a
workspace transaction or start a final-only tool.

The scheduler builds resource-conflict edges by stable model-output ordinal. Independent
ready nodes can bypass a blocked node, while write/read and write/write conflicts preserve
model order. A consumer lock with `requires_success` propagates a structured
`DependencyFailed` result. Exclusive calls are global barriers. Background shell tasks
retain their resource lease after the tool returns; a conflicting same-turn call receives
`DependencyStillRunning` until the process exits.

Arguments are validated against `input_schema` before execution and again after hook or
permission rewrites. Invalid provider JSON is never converted to executable `{}`. Direct
same-turn `$tool_result` references are rejected with
`ResultDependencyRequiresNextTurn`; the model must consume the observation in the next
inference.

Each tool round is recorded in a durable execution journal and persisted to the transcript
as one checksummed record containing the assistant calls, ordered results, and execution
manifest. Journal writes use a single-writer queue: discovery telemetry does not block the
SSE reader, while authorization, external-effect intent, recovery history, and commit gates
wait for durable acknowledgement. Recovery payloads are secret-redacted before commit;
cross-process file ownership (with a per-process start token in every record) replaces PID
guessing, and checksummed transcript recovery is idempotent. A staged result is never sent
through PostToolUse, the UI, or the run log. After reconciliation it is observed exactly
once as either finalized or rolled back; external PreToolUse, permission classifiers, and
interactive prompts are also deferred until the response is authoritative. Legacy
per-message transcript records remain readable. Trusted local
`[tools.execution_policies."qualified_name"]` entries can provide safety and static or
argument-derived resource templates; malformed policies fail closed. MCP annotations by
themselves never upgrade execution timing.

Cancellation follows the declared tool policy. Async-native tools are directly cancelled
only when `safely_cancellable = true`. Other early tools must finish within
`execution_timeout`; an unkillable worker-thread call becomes an explicit indeterminate
cleanup state and its overlay is never committed. The JSONL timing event reports provider
degradation, argument completion, tool start, model termination, commit time, and one
critical-path overlap estimate (never a sum across parallel tools). Run
`python benchmarks/streaming_tools.py --runs 9` for median/p95 end-to-end comparisons of
read, transactional, multi-call, hook-deferred, and provider-capability fixtures.

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

- `capability_search` searches model-invocable skills, connected/deferred MCP tools,
  installed plugins, and configured trusted marketplaces. `capability_activate` accepts
  only a stable catalog id plus its snapshot digest. Local activation is the default;
  remote installation requires `autonomous-trusted`, an explicitly trusted marketplace,
  and a pinned SHA-256 or immutable Git commit. Plugin swaps occur atomically at a turn
  boundary. Autonomous hooks are deny-by-default and executable plugin components are
  sandbox- and network-policy constrained.
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
