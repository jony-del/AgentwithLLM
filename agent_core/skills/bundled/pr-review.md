---
name: pr-review
aliases: [review-pr]
description: Review the current branch / pull-request changes for correctness, clarity, and risk, then report findings by severity.
when-to-use: When the user wants the branch diff or a PR reviewed before merging.
argument-hint: optional path or area to focus the review on
context: fork
allowed-tools: [read_text_file, search_text, list_dir, glob, git_diff]
---
Review the current branch's changes and report back. This is read-only: do not modify
files.

1. Inspect the diff against the base branch (`git diff` / `git diff --staged`, and the
   per-file changes) and read the surrounding code so you understand each change in
   context.
2. Look for: correctness bugs, missing error handling, unsafe edge cases, broken
   invariants, security issues, and anything contradicting the project's conventions.
3. Note clarity/simplification opportunities only when clearly worthwhile — avoid
   nitpicking style.
4. Return a concise report grouped by severity (blocking / should-fix / optional), each
   with the `file:line` and a one-line rationale. If the change looks good, say so
   plainly.

Focus area (if given):

$ARGUMENTS
