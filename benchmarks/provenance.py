"""Shared provenance record for offline benchmark reports.

Every performance comparison must keep the raw before/after JSON, the source
SHA, a configuration fingerprint, the random seed (or an explicit deterministic
note) and the environment fingerprint, so results stay comparable and failures
keep their denominators. No network access; the git call is best-effort.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any


def source_sha(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True,
            text=True, timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def worktree_dirty(repo_root: Path) -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True,
            text=True, timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(completed.stdout.strip())


def config_sha256(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def provenance(repo_root: Path, config: dict[str, Any], *, seed: int | None) -> dict[str, Any]:
    return {
        "source_sha": source_sha(repo_root),
        "worktree_dirty": worktree_dirty(repo_root),
        "config_sha256": config_sha256(config),
        "random_seed": seed,
        "corpus_generation": "deterministic" if seed is None else "seeded",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
    }
