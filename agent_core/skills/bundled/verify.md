---
name: verify
description: Verify a code change actually works by running the app or relevant commands and observing the behaviour.
when-to-use: When the user wants to confirm a fix or feature works, validate local changes, or check a change before committing.
argument-hint: optional description of what to verify
context: inline
---
Verify that the recent change does what it is supposed to do — don't just re-read the
code, actually exercise it.

1. Work out what behaviour the change is meant to produce and the cheapest way to
   observe it: run the test suite, run the app/CLI, or run the specific command the
   change affects.
2. Run it. Capture the real output. If a quick targeted test or one-off command proves
   the behaviour, prefer that over a full run.
3. Compare what you observed against what was expected. State plainly whether it works.
4. If it fails or is inconclusive, report the exact output and what it implies — do not
   claim success without evidence.

Report faithfully: if you could not run something, say so rather than guessing. What to
verify (if specified):

$ARGUMENTS
