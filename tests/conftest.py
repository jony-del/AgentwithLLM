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

import pytest

os.environ.setdefault("AGENT_SANDBOX_ALLOW_UNATTENDED", "1")
os.environ.setdefault(
    "AGENT_TRUST_STORE", os.path.join(tempfile.mkdtemp(prefix="polaris-test-trust-"), "trusted.json")
)


@pytest.fixture(autouse=True)
def _fresh_shared_sandbox_managers():
    """Isolate the process-level SandboxManager cache (§5.6) between tests.

    Without this, a test that constructs an agent while backend probing is
    monkeypatched would leak its manager to any later test using an identical
    sandbox config.
    """
    from agent_core.sandbox import reset_shared_managers

    reset_shared_managers()
    yield
    reset_shared_managers()
