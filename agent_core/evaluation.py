"""Bounded offline task-outcome aggregation; never execute log contents."""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
from typing import Any


def evaluate_logs(paths: list[Path]) -> dict[str, Any]:
    if len(paths) > 1000:
        raise ValueError("evaluation run budget exceeded")
    tasks: dict[str, tuple[float, dict[str, Any]]] = {}
    missing = 0
    for path in paths:
        total = 0
        found = False
        with path.open("rb") as handle:
            while line := handle.readline(1024 * 1024 + 1):
                total += len(line)
                if len(line) > 1024 * 1024 or total > 64 * 1024 * 1024:
                    raise ValueError("evaluation log budget exceeded")
                record = json.loads(line)
                if record.get("event") != "task_metrics":
                    continue
                if record.get("status") not in {"completed", "unverified", "blocked", "failed", "cancelled"}:
                    raise ValueError("evaluation requires terminal task metrics")
                key = str(record.get("task_id") or path.resolve())
                stamp = float(record.get("ts", 0))
                numeric = ("duration_seconds", "input_tokens", "output_tokens", "verification_attempts",
                           "verification_failures", "tool_failures", "review_attempts", "checkpoint_count")
                if not math.isfinite(stamp) or any(not isinstance(record.get(field, 0), (int, float)) or
                    not math.isfinite(record.get(field, 0)) or record.get(field, 0) < 0 for field in numeric):
                    raise ValueError("invalid task metrics")
                if key not in tasks or stamp >= tasks[key][0]:
                    tasks[key] = (stamp, record)
                found = True
        missing += not found
    records = [record for _, record in tasks.values()]
    completed = [record for record in records if record["status"] == "completed"]
    total_tasks = len(records)
    return {"schema_version": 1, "runs_read": len(paths), "runs_without_metrics": missing,
            "tasks": total_tasks, "completed": len(completed),
            "completion_rate": len(completed) / total_tasks if total_tasks else None,
            "statuses": {status: sum(r["status"] == status for r in records) for status in
                         ("completed", "unverified", "blocked", "failed", "cancelled")},
            "reviewed_completions": sum(bool(r.get("review_passed")) for r in completed),
            "completed_with_checks": sum(r.get("verification_attempts", 0) > 0 for r in completed),
            "median_duration_seconds": statistics.median(r.get("duration_seconds", 0) for r in records) if records else None,
            "totals": {field: sum(r.get(field, 0) for r in records) for field in
                       ("input_tokens", "output_tokens", "verification_attempts", "verification_failures", "tool_failures", "review_attempts")},
            "limitations": ["Uses runtime outcomes; does not prove semantic task success",
                            "Resumed tasks counted once using latest terminal metrics", "No pre-change quality baseline implied"]}
