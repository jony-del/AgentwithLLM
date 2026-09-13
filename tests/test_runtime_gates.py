"""Self-tests for the runtime-gate harness itself (audit phase 5).

The nightly gate is only trustworthy if every failure mode actually fails. Each
test here drives ``benchmarks/runtime_gates.py`` as a real subprocess with one
deliberately broken input and asserts: nonzero exit, a clear reason on
stdout/stderr, and a preserved ``--json-out`` report. The happy path (a passing
gate) is exercised implicitly by the nightly workflow and explicitly by the
calibration runs that produced benchmarks/thresholds.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GATE = _REPO_ROOT / "benchmarks" / "runtime_gates.py"
sys.path.insert(0, str(_REPO_ROOT / "benchmarks"))

from runtime_gates import SCENARIO_NAMES, _platform_key  # noqa: E402

pytestmark = pytest.mark.integration

# The cheapest scenario, so each failure-mode probe stays a few seconds.
_FAST = "cancellation_unwind"


def _thresholds(tmp_path: Path, *, value: float | str = 3600.0, omit: tuple[str, ...] = ()):
    """A per-OS v2 threshold file with every metric pass-generous by default."""

    platform_thresholds = {}
    for name in SCENARIO_NAMES:
        if name in omit:
            continue
        platform_thresholds[f"{name}_p95_seconds"] = value
    path = tmp_path / "thresholds.json"
    path.write_text(
        json.dumps({"version": 2, "platforms": {_platform_key(): platform_thresholds}}),
        encoding="utf-8",
    )
    return path


def _run_gate(thresholds: Path, report: Path, *extra: str, env_extra: dict | None = None):
    command = [
        sys.executable,
        str(_GATE),
        "--only",
        _FAST,
        "--runs",
        "20",
        "--check",
        str(thresholds),
        "--json-out",
        str(report),
        *extra,
    ]
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(command, capture_output=True, text=True, timeout=300, env=env)


def test_artificially_tight_threshold_fails_the_gate(tmp_path: Path) -> None:
    thresholds = _thresholds(tmp_path, value=1e-9)
    report = tmp_path / "gates.json"
    completed = _run_gate(thresholds, report)
    assert completed.returncode != 0, completed.stdout
    assert f"FAIL {_FAST}" in completed.stdout
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert payload["scenarios"][_FAST]["status"] == "pass"
    assert len(payload["scenarios"][_FAST]["samples"]) == 20


def test_missing_threshold_fails_the_gate(tmp_path: Path) -> None:
    thresholds = _thresholds(tmp_path, omit=(SCENARIO_NAMES[0],))
    report = tmp_path / "gates.json"
    completed = _run_gate(thresholds, report)
    assert completed.returncode != 0, completed.stdout
    message = completed.stdout + completed.stderr
    assert f"{SCENARIO_NAMES[0]}_p95_seconds" in message
    assert "missing" in message.lower()


def test_invalid_threshold_value_fails_the_gate(tmp_path: Path) -> None:
    thresholds = _thresholds(tmp_path, value=-1)
    completed = _run_gate(thresholds, tmp_path / "gates.json")
    assert completed.returncode != 0, completed.stdout
    assert "invalid" in (completed.stdout + completed.stderr).lower()


def test_insufficient_samples_fail_the_gate(tmp_path: Path) -> None:
    thresholds = _thresholds(tmp_path)
    report = tmp_path / "gates.json"
    command = [
        sys.executable,
        str(_GATE),
        "--only",
        _FAST,
        "--runs",
        "5",
        "--check",
        str(thresholds),
        "--json-out",
        str(report),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
    assert completed.returncode != 0, completed.stdout
    assert "below the required 20" in completed.stdout


def test_scenario_timeout_fails_the_gate_and_preserves_report(tmp_path: Path) -> None:
    thresholds = _thresholds(tmp_path)
    report = tmp_path / "gates.json"
    completed = _run_gate(
        thresholds,
        report,
        "--scenario-timeout",
        "2",
        env_extra={"RUNTIME_GATES_SELFTEST": "hang"},
    )
    assert completed.returncode != 0, completed.stdout
    assert "terminated" in completed.stdout
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["failures"], "the timeout must be recorded as a failure"
    assert payload["scenarios"][_FAST]["status"] == "failed"
    assert "timeout" in payload["scenarios"][_FAST]["reason"].lower()


def test_default_selection_runs_every_scenario_without_only(tmp_path: Path) -> None:
    """No ``--only`` means all five scenarios — the parent must not crash on the
    default (this exact path once died on ``list(None)``)."""

    thresholds = _thresholds(tmp_path)
    report = tmp_path / "gates.json"
    command = [
        sys.executable,
        str(_GATE),
        "--scenario-timeout",
        "2",
        "--check",
        str(thresholds),
        "--json-out",
        str(report),
    ]
    env = {**os.environ, "RUNTIME_GATES_SELFTEST": "hang"}
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=300, env=env
    )
    assert completed.returncode != 0, completed.stdout
    assert "TypeError" not in completed.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert set(payload["scenarios"]) == set(SCENARIO_NAMES)


def test_semantic_assertion_failure_fails_the_gate_and_preserves_report(
    tmp_path: Path,
) -> None:
    thresholds = _thresholds(tmp_path)
    report = tmp_path / "gates.json"
    completed = _run_gate(
        thresholds,
        report,
        env_extra={"RUNTIME_GATES_SELFTEST": "fail-semantics"},
    )
    assert completed.returncode != 0, completed.stdout
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["failures"], "the semantic failure must be recorded"
    assert payload["scenarios"][_FAST]["status"] == "failed"
    assert "injected semantic failure" in payload["scenarios"][_FAST]["reason"]
