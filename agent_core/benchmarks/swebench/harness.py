"""Subprocess adapter for the official SWE-bench evaluation Harness."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class HarnessResult:
    command: list[str]
    returncode: int
    duration: float
    report_dir: Path
    stdout: str = ""
    stderr: str = ""
    status_by_instance: dict[str, str] = field(default_factory=dict)
    failure_by_instance: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    @property
    def completed(self) -> bool:
        return self.returncode == 0

    @property
    def resolved_ids(self) -> tuple[str, ...]:
        return tuple(key for key, value in self.status_by_instance.items() if value == "resolved")


class OfficialHarnessAdapter:
    """Invoke the installed official Harness without importing its private modules."""

    module = "swebench.harness.run_evaluation"

    def __init__(self, *, python: str | None = None) -> None:
        self.python = python or sys.executable

    def available(self) -> bool:
        try:
            result = subprocess.run(
                [self.python, "-m", self.module, "--help"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0 or "run evaluation" in (result.stdout + result.stderr).casefold()

    def build_command(
        self,
        *,
        dataset: str,
        split: str,
        predictions_path: str | Path,
        run_id: str,
        report_dir: str | Path,
        instance_ids: Iterable[str] = (),
        max_workers: int = 1,
        timeout: int = 1800,
    ) -> list[str]:
        dataset_arg = str(dataset)
        dataset_path = Path(dataset_arg).expanduser()
        if dataset_path.exists():
            dataset_arg = str(dataset_path.resolve())
        command = [
            self.python,
            "-m",
            self.module,
            "--dataset_name",
            dataset_arg,
            "--split",
            split,
            "--predictions_path",
            str(Path(predictions_path).resolve()),
            "--run_id",
            run_id,
            "--max_workers",
            str(max(1, int(max_workers))),
            "--timeout",
            str(max(1, int(timeout))),
            "--report_dir",
            str(Path(report_dir).resolve()),
        ]
        ids = [str(item) for item in instance_ids if str(item).strip()]
        if ids:
            command.extend(["--instance_ids", *ids])
        return command

    def evaluate(
        self,
        *,
        dataset: str,
        split: str,
        predictions_path: str | Path,
        run_id: str,
        report_dir: str | Path,
        instance_ids: Iterable[str] = (),
        max_workers: int = 1,
        timeout: int = 1800,
        cwd: str | Path | None = None,
    ) -> HarnessResult:
        target = Path(report_dir).resolve()
        target.mkdir(parents=True, exist_ok=True)
        ids = [str(item) for item in instance_ids if str(item).strip()]
        command = self.build_command(
            dataset=dataset,
            split=split,
            predictions_path=predictions_path,
            run_id=run_id,
            report_dir=target,
            instance_ids=ids,
            max_workers=max_workers,
            timeout=timeout,
        )
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=str(Path(cwd).resolve()) if cwd else None,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(1, int(timeout)) * max(1, len(ids) or 1) + 120,
            )
            harness_error = None
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                harness_error = f"Harness exited with code {completed.returncode}"
                if detail:
                    harness_error += f": {detail[-2000:]}"
            result = HarnessResult(
                command,
                completed.returncode,
                time.monotonic() - started,
                target,
                completed.stdout or "",
                completed.stderr or "",
                error=harness_error,
            )
        except FileNotFoundError as exc:
            result = HarnessResult(command, 127, time.monotonic() - started, target, error=str(exc))
        except subprocess.TimeoutExpired as exc:
            result = HarnessResult(command, 124, time.monotonic() - started, target, error=f"Harness timed out: {exc}")
        except OSError as exc:
            result = HarnessResult(command, 126, time.monotonic() - started, target, error=str(exc))
        (target / "command.json").write_text(json.dumps(command, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (target / "stdout.txt").write_text(result.stdout, encoding="utf-8", errors="replace")
        (target / "stderr.txt").write_text(result.stderr + (f"\n{result.error}" if result.error else ""), encoding="utf-8", errors="replace")
        result.status_by_instance, result.failure_by_instance = parse_harness_reports(target)
        return result


def parse_harness_reports(report_dir: str | Path) -> tuple[dict[str, str], dict[str, str]]:
    """Read common official report layouts into ``instance_id -> status`` maps."""
    statuses: dict[str, str] = {}
    failures: dict[str, str] = {}
    root = Path(report_dir)
    if not root.exists():
        return statuses, failures
    for path in sorted(root.rglob("*.json")):
        if path.name in {"command.json"}:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        _collect_report_value(value, statuses, failures)
    # Some Harness versions print a compact table but do not emit per-instance
    # reports.  Parse the stable ``resolved: id``/``unresolved: id`` forms as a
    # last-resort diagnostic, without treating unknown output as a grade.
    for path in (root / "stdout.txt", root / "stderr.txt"):
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for status, instance_id in re.findall(r"(?im)^\s*(resolved|unresolved|error)\s*[:\-]\s*(\S+)", text):
            statuses.setdefault(instance_id, "failed" if status == "error" else status)
    return statuses, failures


def _collect_report_value(value: Any, statuses: dict[str, str], failures: dict[str, str], *, inherited_id: str | None = None) -> None:
    if isinstance(value, list):
        for item in value:
            _collect_report_value(item, statuses, failures, inherited_id=inherited_id)
        return
    if not isinstance(value, dict):
        return
    explicit_id = str(value.get("instance_id") or value.get("instanceId") or "").strip()
    candidate_status = _status_from_mapping(value)
    # An inherited key is an instance ID only when the current object actually
    # carries a status.  This prevents wrapper objects such as
    # ``{"results": {"instance-id": {"resolved": true}}}`` from turning
    # ``results`` into the ID and losing the nested key.
    instance_id = explicit_id or (inherited_id if candidate_status is not None else "")
    if instance_id:
        status = candidate_status
        if status is not None:
            statuses[instance_id] = status
            reason = value.get("failure_reason") or value.get("error") or value.get("reason")
            if reason:
                failures[instance_id] = str(reason)[:1000]
    for key in ("resolved_ids", "resolved", "resolved_instances"):
        ids = value.get(key)
        if isinstance(ids, (list, tuple, set)):
            for item in ids:
                if isinstance(item, str):
                    statuses[item] = "resolved"
    for key in ("unresolved_ids", "unresolved", "unresolved_instances"):
        ids = value.get(key)
        if isinstance(ids, (list, tuple, set)):
            for item in ids:
                if isinstance(item, str):
                    statuses[item] = "unresolved"
    for key in ("error_ids", "errors", "error_instances", "incomplete_ids", "infra_failure_ids", "ambiguous_failure_ids", "empty_patch_ids"):
        ids = value.get(key)
        if isinstance(ids, dict):
            for item, reason in ids.items():
                statuses[str(item)] = "failed"
                failures[str(item)] = str(reason)[:1000]
        elif isinstance(ids, (list, tuple, set)):
            for item in ids:
                if isinstance(item, str):
                    statuses[item] = "failed"
                    failures.setdefault(item, key.replace("_ids", "").replace("_", " "))
    reasons = value.get("failure_reasons")
    if isinstance(reasons, dict):
        for item, reason in reasons.items():
            failures[str(item)] = str(reason)[:1000]
    aggregate_keys = {
        "resolved_ids",
        "resolved",
        "resolved_instances",
        "unresolved_ids",
        "unresolved",
        "unresolved_instances",
        "error_ids",
        "errors",
        "error_instances",
        "incomplete_ids",
        "infra_failure_ids",
        "ambiguous_failure_ids",
        "empty_patch_ids",
        "failure_reasons",
    }
    for key, child in value.items():
        if key in aggregate_keys or not isinstance(child, (dict, list)):
            continue
        # Older Harness releases emit one JSON object per instance as
        # ``{instance_id: {resolved: true, ...}}`` rather than a top-level
        # aggregate report.  Treat a mapping key as the inherited instance ID;
        # child objects without a status simply contribute no result.
        context_id = explicit_id or (inherited_id if candidate_status is not None else "")
        child_id = context_id or str(key)
        _collect_report_value(child, statuses, failures, inherited_id=child_id)


def _status_from_mapping(value: dict[str, Any]) -> str | None:
    raw = value.get("status") or value.get("result") or value.get("resolution")
    if isinstance(raw, bool):
        return "resolved" if raw else "unresolved"
    if raw is not None:
        text = str(raw).casefold()
        # Check negative forms first: ``unresolved`` contains the substring
        # ``resolved`` and would otherwise be misclassified as a pass.
        if any(token in text for token in ("unresolved", "not_resolved", "fail", "error", "incomplete")):
            return "failed" if "error" in text or "incomplete" in text else "unresolved"
        if any(token in text for token in ("resolved", "pass", "success")):
            return "resolved"
    for key in ("resolved", "is_resolved", "passed", "success"):
        if key in value and isinstance(value[key], bool):
            return "resolved" if value[key] else "unresolved"
    return None
