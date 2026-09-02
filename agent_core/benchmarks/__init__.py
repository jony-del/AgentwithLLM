"""Benchmark integrations for Polaris.

The benchmark packages are intentionally isolated from the normal interactive
agent path.  Importing :mod:`agent_core` therefore never imports Docker,
HuggingFace datasets, or the SWE-bench package; those dependencies are loaded
only when a ``polaris swebench`` command is used.
"""

from __future__ import annotations

__all__ = ["swebench"]
