"""SWE-bench Lite runner.

The public API is deliberately small so an embedding application can use the
same runner as the CLI without importing optional dependencies until it needs
them.
"""

from __future__ import annotations

from .models import (
    FailureKind,
    InstanceState,
    SWEbenchInstance,
    SWEbenchRunConfig,
    SWEbenchSelection,
)
from .dataset import SWEbenchDataset, load_swebench_dataset
from .patch import export_patch, prediction_record, upsert_prediction
from .prompt import build_swebench_prompt
from .runner import SWEbenchRunner
from .selection import load_selection, make_selection

__all__ = [
    "FailureKind",
    "InstanceState",
    "SWEbenchInstance",
    "SWEbenchRunConfig",
    "SWEbenchSelection",
    "SWEbenchDataset",
    "load_swebench_dataset",
    "SWEbenchRunner",
    "build_swebench_prompt",
    "export_patch",
    "prediction_record",
    "upsert_prediction",
    "load_selection",
    "make_selection",
]
