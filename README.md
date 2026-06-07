# Agent with LLM

A Python ReAct agent framework with:

- Claude API provider
- ReAct loop
- Tool registry and executor
- MCP/LCP adapter interfaces
- pre/post tool hooks
- permission modes
- multi-layer context compaction
- cross-conversation memory (recall, extraction, dreaming)
- JSONL run logging
- subagent and multi-agent coordination interfaces

## Quick start

```powershell
python -m agent_core run "Say hello without tools" --provider fake
python -m agent_core chat --provider fake
```

To use Claude:

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your-key
AGENT_MODEL=claude-sonnet-4-6
AGENT_PROVIDER=claude
```

Then run:

```powershell
python -m agent_core run "Use the echo tool"
python -m agent_core chat
```

## Cross-conversation memory

By default the agent is stateless: each run starts fresh. Opt into memory and it will
**recall** relevant facts from past runs, **extract** durable memories when a run
finishes, and **consolidate** them offline ("dreaming"). It is dependency-free —
relevance is lexical (token overlap × importance × recency), not embeddings.

```powershell
# Enable per-invocation with --memory (works with the key-free fake provider too):
python -m agent_core run "I prefer dark mode and use Python 3.11" --provider fake --memory
python -m agent_core run "remind me about my preferences" --provider fake --memory

# Inspect / curate what's stored:
python -m agent_core memory list
python -m agent_core memory add "The user deploys on Fridays"
python -m agent_core memory forget <id>

# Dreaming: decay & forget weak memories, merge duplicates, synthesise insights.
python -m agent_core dream --dry-run     # preview, writes nothing
python -m agent_core dream               # apply the consolidation
```

Memories live in `memory/memory.jsonl` (gitignored). Each run also logs
`memory_recall` / `memory_extract` events into its `runs/*.jsonl` trace.

### How it works

| Mechanism | Module | What it does |
| --- | --- | --- |
| Recall | `memory/retrieval.py` | Scores stored memories by relevance × importance × recency and injects the top-`k` as a pinned system block before the task. Recall reinforces a memory (raises its access count). |
| Extraction | `memory/extraction.py` | After a completed run, asks the LLM to distil durable facts/preferences as JSON, de-duplicated against what's already stored. |
| Dreaming | `memory/dreaming.py` | Offline pass: a forgetting curve drops weak, unused memories; near-duplicates are merged; the LLM synthesises higher-level `insight` memories. |

### Configuration

Enable memory and tune it via `agent.toml` (copy [`agent.toml.example`](agent.toml.example)),
the `AGENT_MEMORY` env var, or the `--memory` / `--no-memory` flags. Precedence is
defaults → `agent.toml` → env → CLI. Numeric tunables (recall weights, decay,
forget/merge thresholds) live only in the `[memory]` table:

```toml
[memory]
enabled = true
recall_k = 5
auto_extract = true
forget_threshold = 0.15
merge_threshold = 0.6
```
