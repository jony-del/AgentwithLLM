---
name: review
description: Review the current code changes for correctness, clarity, and risk, then summarize findings.
when-to-use: When the user wants the pending diff or a specific area reviewed before committing or merging.
argument-hint: optional path or area to focus the review on
context: fork
allowed-tools: [read_text_file, search_text, list_dir, glob, git_diff]
---
Perform a focused code review of the current changes and report back.

1. Inspect the pending diff (`git diff` and `git diff --staged`) and read the surrounding
   code for the files involved so you understand the change in context.
2. Look for: correctness bugs, missing error handling, unsafe edge cases, broken
   invariants, and anything that contradicts the project's conventions.
3. Note clarity/simplification opportunities only when they are clearly worthwhile —
   avoid nitpicking style.
4. Return a concise report grouped by severity (blocking / should-fix / optional), each
   with the file:line and a one-line rationale. If the change looks good, say so plainly.

This review is read-only: do not modify files. Focus area (if given):

$ARGUMENTS
