from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Any


@dataclass(frozen=True, slots=True)
class CodeIntelConfig:
    enabled: bool = True
    strict_versions: bool = False
    # Background construction/reconciliation in agent sessions; explicit
    # `polaris code build` works regardless.
    maintain: bool = True
    # OS-level change events (watchdog); periodic reconciliation is the fallback.
    watch: bool = True
    reconcile_seconds: float = 60.0
    query_seconds: float = 5.0
    max_files: int = 2000
    max_bytes: int = 64 * 1024 * 1024
    max_file_bytes: int = 2 * 1024 * 1024
    max_results: int = 100
    max_candidates: int = 10000
    max_context_tokens: int = 8000
    max_depth: int = 2
    max_edges: int = 2000
    max_shards: int = 4
    cache_entries: int = 128
    max_index_bytes: int = 8 * 1024 * 1024 * 1024

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> CodeIntelConfig:
        defaults = cls()
        values: dict[str, Any] = {}
        for field in fields(cls):
            if field.name not in (raw or {}):
                continue
            value = (raw or {})[field.name]
            default = getattr(defaults, field.name)
            if isinstance(default, bool):
                if not isinstance(value, bool):
                    raise ValueError(f"codeintel.{field.name} must be boolean")
            else:
                if isinstance(value, bool):
                    raise ValueError(f"codeintel.{field.name} must be numeric")
                value = type(default)(value)
                if value <= 0 or not math.isfinite(value):
                    raise ValueError(f"codeintel.{field.name} must be positive")
            values[field.name] = value
        return cls(**values)
