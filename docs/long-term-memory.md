# Long-term memory

Polaris long-term memory is disabled by its built-in defaults. When enabled, the
main agent stores project-private memory outside the checkout:

```text
~/.polaris/projects/<canonical-project-id>/memory/private/
  MEMORY.md
  topic-name-<id>.md
```

The topic Markdown files are the only authoritative data. `MEMORY.md` is a
generated human summary capped at 200 lines and 25 KB; that cap never limits topic
enumeration or retrieval. Team memory lives at `.polaris/memory/team/` and may be
reviewed and synchronized through Git, but Polaris does not commit or push it.

## Derived local index

Search data is disposable and rebuildable:

```text
~/.polaris/indexes/<memory-root-hash>/<index-fingerprint>/memory.sqlite3
~/.polaris/indexes/<memory-root-hash>/<index-fingerprint>/dense/active.json
~/.polaris/indexes/<memory-root-hash>/<index-fingerprint>/dense/<generation>.usearch
```

The fingerprint covers the schema, chunker, lexical normalizer, chunk sizes, and
embedding model. A model or chunker upgrade therefore cannot mix incompatible
vectors. SQLite uses WAL, a busy timeout, and the same cross-process file-lock
boundary as memory mutations. Full rebuilds write a new database before atomically
switching it into place.

Markdown is split on headings, paragraphs, and fenced code blocks, then packed to
384 approximate tokens with 64-token overlap. The index contains document
metadata, chunks, exact atoms, FTS5 data, embeddings, diagnostics, and a resumable
embedding queue. Manual Markdown additions, edits, archives, and deletions are
detected by SHA-256 during the next sync.

## Retrieval

The default local CPU pipeline is:

1. Deterministic exact atoms and `id`, `tag`, `type`, and `source` filters.
2. SQLite FTS5 BM25 with `name=4`, `tags=3`, `description=2`, and `content=1`.
3. normalized BGE-M3 dense embeddings. Collections below 10,000 eligible
   vectors use bounded exact scans; larger collections use memory-mapped USearch
   HNSW recall followed by exact FP32 cosine rescoring.
4. weighted reciprocal-rank fusion (`exact=2`, `BM25=1`, `dense=1`).
5. BGE reranking of the first 24 chunks, then document aggregation.

Unicode is NFKC-normalized and case-folded. Paths normalize to `/`; CJK produces
bi/tri-grams; code identifiers keep whole and split forms. The raw query is never
passed to FTS syntax. Generated tokens are individually quoted and the SQL remains
parameterized.

At most three chunks per topic enter reranking. A final topic contains no more
than two merged, non-overlapping passages. Automatic recall injects only those
complete passages—not `MEMORY.md` or the entire topic. No qualifying hit means no
memory block. The byte budget skips passages that do not fit; it never slices a
UTF-8 sequence or a passage.

Exact/filter hits are hard-prioritized. Relevance determines normal ordering;
confidence, verification/explicit status, update time, and stable chunk id are
tie-breaks only. Stored importance and access recency do not alter relevance.

## Models and degradation

The `[memory]` Python extra contains NumPy, tokenizers, `onnxruntime==1.23.2`,
and `usearch==2.26.0`. Core imports do not load them. Model bundles are INT8 CPU
ONNX artifacts with pinned upstream commits, per-file SHA-256 values, licenses,
quantization metadata, and golden vectors. Remote code is forbidden.

```text
polaris memory models status
polaris memory models install
polaris memory models install --model-bundle path/to/models.zip
polaris memory index status
polaris memory index rebuild
```

The standard installer downloads the four commit-pinned model/tokenizer artifacts
(about 1.15 GB total), verifies each SHA-256, builds a temporary bundle, and
atomically activates it. Completed download cache files are removed after
activation; interrupted downloads resume on the next run. An offline bundle may
be used, or `--skip-memory-models` may explicitly opt into lexical-only operation.
Missing models otherwise fail installation with a recovery command.

Dense indexing is resumable in the background. Until coverage reaches 100%, exact
and BM25 are immediately available and traces report `lexical_degraded`. A missing
reranker falls back to RRF. Missing or damaged ANN data falls back to a
memory-bounded exact dense scan and schedules a rebuild; damaged SQLite falls back
to a full Unicode lexical scan. Memory failures never make an otherwise executable
agent run fail.

## Configuration

```toml
[memory]
enabled = true
recall_k = 5
content_budget_bytes = 65536

[memory.retrieval]
mode = "hybrid"
exact_k = 32
bm25_k = 64
dense_k = 64
rrf_k = 60
rerank_k = 24
chunk_tokens = 384
chunk_overlap_tokens = 64
min_rerank_score = 0.5
dense_fallback_min = 0.45
timeout_seconds = 10
model_threads = 4
dense_strategy = "auto"
ann_min_vectors = 10000
ann_candidate_multiplier = 4
ann_expansion_search = 256
```

`dense_strategy=exact` is the rollback/compatibility mode. `auto` uses exact
search for small or selective filtered sets and ANN for larger sets. ANN files
are immutable, versioned sidecars next to `memory.sqlite3`; SQLite FP32
embeddings remain the rebuild and exact-rescore source of truth.

Removed keys (`semantic_selection`, `memory_model`, `w_relevance`,
`w_importance`, `w_recency`, and `recency_decay_per_hour`) produce explicit
warnings and are ignored; their old semantics are not mapped.

## Commands and tools

```text
polaris memory list [--scope private|team]
polaris memory show <id>
polaris memory search "<query>" [--id ID] [--tag TAG] [--type TYPE]
                      [--source SOURCE] [--explain] [--full-content]
polaris memory add "<content>" --name NAME --description TEXT --type project
polaris memory edit <id> --text "<replacement>"
polaris memory forget <id>
polaris memory validate [--repair]
polaris memory migrate [path/to/memory.jsonl]
```

`memory_search` accepts the same filters plus `include_content=false` and
`explain=false`. Passage content is the default; full topic content requires an
explicit request.

`memory_recall` events contain per-stage counts and timing, coverage, model/index
fingerprints, degradation reasons, and final ids. They never record the query,
passages, full documents, or embeddings.

## Dense benchmark

After an index has complete embedding coverage, compare exhaustive search with
ANN plus exact rescoring on the same stored BGE-M3 vectors:

```text
python benchmarks/memory_ann.py path/to/memory.sqlite3 --max-vectors 100000
```

The report includes build time, exact and ANN p50/p95 latency, candidate count,
and mean recall at K. Release validation targets recall@64 of at least 0.98 and
measures 10K, 100K, and 1M-vector indexes on the same host.

## Safety and privacy

- A checked-out `agent.toml` cannot redirect private memory to an external path.
- Paths reject null bytes, `..`, UNC roots, drive roots, and symlink escapes.
- Writes are locked, fsynced, and atomically replaced.
- Secrets are rejected before persistence; diagnostics do not echo values.
- Malformed, conflicting, duplicate-id, and oversized topics are skipped with
  visible index diagnostics.
- Recalled text is explicitly marked untrusted and possibly stale.
- Forgetting moves memory to recoverable trash. Uninstall removes only
  installer-owned model bundles and derived indexes; authoritative Markdown is
  preserved unless the user explicitly purges all Polaris data.
