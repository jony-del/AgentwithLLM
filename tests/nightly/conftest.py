"""Make the shared crash harness importable from this subdirectory."""

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent.parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
