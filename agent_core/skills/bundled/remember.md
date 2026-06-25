---
name: remember
description: Review what was learned this session and propose promoting durable facts into CLAUDE.md or cross-conversation memory.
when-to-use: When the user wants to capture a preference, convention, or project fact so it persists into future sessions.
argument-hint: optional note about what to remember
context: inline
---
Capture durable knowledge from this session so it survives into future ones.

1. Review the conversation for facts worth keeping: user preferences, project
   conventions, constraints, or decisions that are NOT already obvious from the code,
   git history, or the existing CLAUDE.md.
2. For each candidate, decide where it belongs:
   - **CLAUDE.md** (project-level instructions injected every run) — conventions and
     constraints that apply to anyone working in this repo.
   - **cross-conversation memory** — user-specific or longer-lived facts (only if the
     memory subsystem is enabled).
3. Do NOT record things the repo already encodes, or that only mattered to this one
   conversation. If something seems obvious, ask what was non-obvious about it and keep
   that instead.
4. Propose the concrete additions (the exact text and target file) and apply them once
   the user agrees.

What to remember (if specified):

$ARGUMENTS
