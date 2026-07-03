"""Shared test setup.

Two process-wide arrangements:

- The D3 rule (unattended permission modes require a working sandbox) is opted out:
  many tests construct agents with ``permission="auto"`` to exercise loop behavior,
  and no sandbox backend exists on CI runners. The enforcement itself is tested
  explicitly in ``test_sandbox_required.py``, which removes this env var.
- The D2 TOFU trust store is redirected to a temp file so tests never read or write
  the developer's real ``~/.polaris/trusted.json``. ``test_trust.py`` overrides it
  per-test with tmp_path fixtures.
"""

import os
import tempfile

os.environ.setdefault("AGENT_SANDBOX_ALLOW_UNATTENDED", "1")
os.environ.setdefault(
    "AGENT_TRUST_STORE", os.path.join(tempfile.mkdtemp(prefix="polaris-test-trust-"), "trusted.json")
)
