---
name: init
description: Analyze the codebase and create or update a CLAUDE.md with the high-value context an agent needs to work here.
when-to-use: When the user wants to bootstrap or refresh the project's CLAUDE.md documentation.
argument-hint: optional emphasis (e.g. "focus on the build system")
context: inline
---
Create or update `CLAUDE.md` at the repository root with the context an agent needs to
be productive in this codebase.

1. Explore the repo: read the build/config files, the entry points, the directory
   layout, and any existing docs. If a `CLAUDE.md` already exists, read it and improve
   it rather than overwriting useful content.
2. Capture the high-value, non-obvious things:
   - what the project is and how to build/test/run it (exact commands),
   - the code map (where the important pieces live),
   - conventions, invariants, and gotchas a newcomer would trip on.
3. Be concise and concrete — prefer the specific command or file path over prose. Do not
   pad it with generic advice that isn't specific to this repo.
4. Write the result to `CLAUDE.md` and summarise what you added or changed.

Emphasis (if specified):

$ARGUMENTS
