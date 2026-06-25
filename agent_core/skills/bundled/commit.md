---
name: commit
description: Stage the current changes and write a clear, conventional git commit message.
when-to-use: When the user asks to commit the working changes or wants a commit message drafted.
argument-hint: optional short hint about the change
context: inline
---
Create a git commit for the current working changes.

1. Run `git status` and `git diff` (and `git diff --staged`) to see exactly what changed.
2. Group the changes into a single coherent commit. If the changes are unrelated, say so
   and propose splitting them rather than committing everything at once.
3. Write a commit message: a concise imperative subject line (<= 72 chars), a blank line,
   then a short body explaining the *why* when it isn't obvious from the diff.
4. Stage the relevant files and create the commit. Do not push unless explicitly asked.

If the user gave extra context, incorporate it into the message:

$ARGUMENTS
