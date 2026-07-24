# Long-term memory

Polaris long-term memory is disabled by its built-in defaults. When enabled, the
main agent stores project-private memory outside the checkout:

```text
~/.polaris/projects/<canonical-project-id>/memory/private/
  MEMORY.md
  topic-name-<id>.md
```

The project id is derived from Git's common directory, so normal worktrees share
private memory. Team memory lives in the current checkout at
`.polaris/memory/team/`; it can be reviewed and synchronized with Git, but Polaris
does not commit or push it.

Each topic Markdown file is authoritative. Its YAML frontmatter records schema
version, stable id, name, description, type (`user`, `feedback`, `project`, or
`reference`), timestamps, confidence, tags, sources, and whether it was explicitly
saved. `MEMORY.md` is a compact generated index, capped at 200 lines and 25 KB.

## Safety model

- A checked-out `agent.toml` cannot redirect private memory to an external path.
  Trusted overrides are `AGENT_MEMORY_DIR`, gitignored `agent.local.toml`, or the
  `polaris memory --memory-dir` option.
- Repository and topic paths are contained and reject null bytes, `..`, UNC paths,
  drive roots, prefix collisions, and symbolic-link escapes.
- Writes are protected by a cross-process lock and use fsync plus atomic replace.
- Secrets and credentials are rejected before persistence; diagnostics contain
  rule names, never the matching value.
- Recalled content is untrusted historical data. It cannot grant permission or
  override current instructions. Claims about current files, functions,
  configuration, and runtime state must be verified.
- Forgetting moves private memory to `.trash`; automatic consolidation archives
  rather than hard-deleting protected memory.

## Commands

```text
polaris memory list [--scope private|team]
polaris memory show <id>
polaris memory search "<query>"
polaris memory add "<content>" --name NAME --description TEXT --type project
polaris memory edit <id> --text "<replacement>"
polaris memory forget <id>
polaris memory validate [--repair]
polaris memory migrate [path/to/memory.jsonl]
```

The model has corresponding constrained tools: `memory_search`, `memory_write`,
and `memory_forget`. Team automatic extraction is off by default.

## JSONL migration

When an old `memory.jsonl` is found, Polaris performs a staged, checksum-marked,
idempotent import. Every valid row becomes `legacy-<id>.md`; content, timestamps,
importance, access count, tags, kind, and run source are retained in frontmatter.
The source JSONL is never changed or deleted. Corrupt lines are isolated and
reported without preventing the old store from remaining readable.

Use `polaris memory migrate` to run the import explicitly and
`polaris memory validate --repair` to validate topics and rebuild the index.

## Privacy

Private memory is local user data and is never added to the repository. Team
memory is intentionally repository-visible and should be reviewed like source
code. Polaris provides no private cloud synchronization service and never
automatically commits or pushes memory.
