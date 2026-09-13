"""Property/fuzz tests for audit-confirmed contracts (hypothesis).

Shares the tests/ sys.path shim with tests/nightly/ so helpers living directly under
tests/ stay importable, and pins two sampling profiles: ``ci`` (small, deterministic)
and ``nightly`` (large, unbounded deadline). Select with ``HYPOTHESIS_PROFILE``.
"""

import os
import sys
from pathlib import Path

from hypothesis import settings

_TESTS_DIR = Path(__file__).resolve().parent.parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

settings.register_profile("ci", max_examples=50, derandomize=True)
settings.register_profile("nightly", max_examples=1000, deadline=None)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "ci"))
