---
name: security-review
description: Complete a security review of the pending changes on the current branch and report exploitable issues.
when-to-use: When the user wants the branch changes checked for security vulnerabilities before merging.
argument-hint: optional area to focus on
context: fork
allowed-tools: [read_text_file, search_text, list_dir, glob, git_diff]
---
Perform a focused security review of the pending changes on the current branch. This is
read-only: do not modify files.

1. Inspect the diff (`git diff` / `git diff --staged`) and read the affected code paths,
   including how untrusted input reaches them.
2. Hunt specifically for exploitable issues introduced or exposed by the change:
   - injection (command/SQL/path/template), unsafe deserialization,
   - missing authn/authz or trust-boundary checks,
   - secrets in code/logs, unsafe file or network access,
   - SSRF, path traversal, and unsafe handling of user-controlled data.
3. For each finding, give the `file:line`, the concrete attack scenario, the severity,
   and a suggested fix. Distinguish real exploitable issues from theoretical concerns.
4. If you find nothing exploitable, say so plainly rather than inventing low-value notes.

Focus area (if given):

$ARGUMENTS
