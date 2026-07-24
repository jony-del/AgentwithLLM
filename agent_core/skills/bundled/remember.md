---
name: remember
description: Review what was learned and propose scoped, deduplicated durable-memory changes.
when-to-use: When the user wants a preference, convention, or project fact to persist into future sessions.
argument-hint: optional note about what to remember
context: inline
---
Capture durable knowledge from this session so it survives into future ones.

1. Review the conversation for stable preferences, project conventions, constraints,
   feedback, references, or decisions that are not already obvious from code or Git.
2. Compare each candidate with CLAUDE.md, project-private memory, Git-reviewed team
   memory, and any named-agent memory visible in this session. Identify duplicates,
   contradictions, stale claims, and narrower versions first.
3. Choose the smallest correct scope:
   - **CLAUDE.md** for instructions and conventions that apply to everyone in the repo.
   - **project-private memory** for personal or local project knowledge that must not
     be committed.
   - **team memory** for reviewed, non-secret knowledge that should follow Git.
   - **named-agent memory** for knowledge useful only to one stable user/project/local
     agent identity.
4. Recommend promotion, deduplication, conflict resolution, verification, or archival
   when that is better than creating another topic. Never silently overwrite a conflict.
5. Do not record secrets, transient task chatter, or facts the repository already
   encodes. Claims about current files, functions, configuration, and runtime state
   must be verified before saving.
6. Propose the exact operation, scope, topic id/name, and text. Apply it only once the
   user agrees, using memory tools rather than editing generated MEMORY.md directly.

What to remember (if specified):

$ARGUMENTS
