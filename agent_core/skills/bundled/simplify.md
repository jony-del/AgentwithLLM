---
name: simplify
description: Review the changed code for reuse, simplification, efficiency, and altitude cleanups, then apply the fixes. Quality only — it does not hunt for bugs.
when-to-use: When the user wants the pending diff tidied up for clarity and reuse without a full correctness review.
argument-hint: optional path or area to focus on
context: fork
allowed-tools: [read_text_file, search_text, list_dir, glob, edit_file, multi_edit, apply_patch]
---
Review the current changes for quality and apply worthwhile cleanups. This is a
quality pass, NOT a bug hunt — do not change behaviour.

1. Inspect the pending diff (`git diff` / `git diff --staged`) and read the surrounding
   code so you understand the change in context.
2. Look for, and fix, only clearly worthwhile improvements:
   - duplicated logic that should reuse an existing helper,
   - needless complexity that a simpler construct expresses more directly,
   - obvious inefficiency (redundant work, avoidable passes),
   - wrong altitude (a detail inlined where a named helper reads better, or vice versa).
3. Match the surrounding code's style, naming, and comment density. Skip subjective
   nitpicks and anything that would alter behaviour.
4. Apply the edits, then briefly summarise what you changed and why.

If nothing is worth changing, say so plainly. Focus area (if given):

$ARGUMENTS
