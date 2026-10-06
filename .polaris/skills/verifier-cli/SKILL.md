---
name: verifier-cli
description: Verify Polaris public CLI behavior with isolated offline functional and error probes.
context: inline
---

Polaris is a Python 3.11+ CLI (`python -m agent_core`, installed command `polaris`).
Core dependencies from pyproject.toml must already be installed in the selected interpreter.
No API key, server, browser, network or extra package installation is needed for these probes.
The fake provider validates CLI/runtime behavior; it does not establish real model quality.

Run these through `verifier_probe` in the independent `run_verifier` context, from the
clean copied source root. Use explicit argv, expected_exit_code=0 and
expected_output="CLI_PROBE_VERIFIED". The helper performs assertions before printing
that marker and records the actual application exit codes and output.

Functional probe:

```json
{"argv":["python","benchmarks/verifier_cli_smoke.py","functional"],"kind":"functional","criterion":"A plain fake-provider CLI task returns the expected answer, exits 0 and writes a run log","expected_exit_code":0,"expected_output":"CLI_PROBE_VERIFIED","timeout":60}
```

Adversarial probe:

```json
{"argv":["python","benchmarks/verifier_cli_smoke.py","adversarial"],"kind":"adversarial","criterion":"Unknown arguments exit 2; required verification without a sandbox exits 2 with unverified and a persisted PARTIAL verdict","expected_exit_code":0,"expected_output":"CLI_PROBE_VERIFIED","timeout":120}
```

Each helper creates a temporary working directory, explicit minimal configuration and
private user-state root. It disables memory, hooks, capability discovery and nested automatic
verification for the child application and removes those directories when finished. The
adversarial case explicitly requests required verification to check failure behavior.

For a specific code change, add probes that exercise its actual public behavior and a
relevant failure/boundary input. These generic smoke checks alone cannot certify every feature.
Inspect actual probe IDs and expected versus actual results; return PASS/FAIL/PARTIAL.
Missing interpreter dependencies or sandbox capabilities are PARTIAL. Do not use host
execution as a substitute for the outer verifier's required sandbox.

The snapshot excludes .git, .venv, node_modules, root .polaris/runs/tmp/memory and secret
files. Dependencies must exist in the verifier runtime; project .venv is not copied. This
guide is passed separately as untrusted context. After changing a guide, call run_verifier
explicitly to create fresh evidence. Existing browser MCP tools, if connected, remain subject
to the parent's tool permission policy; browser checks are unnecessary for this CLI profile.
