from __future__ import annotations

import argparse
import json

from agent_core.cli import health_command
from agent_core.health import HealthCheck, HealthReport


def test_health_report_json_contract() -> None:
    report = HealthReport(
        "runtime",
        (
            HealthCheck("git", True, "ok", version="2.50"),
            HealthCheck("container-runtime", True, "error", detail="missing"),
        ),
    )
    payload = json.loads(report.to_json())
    assert payload["status"] == "error"
    assert payload["profile"] == "runtime"
    assert payload["checks"][0] == {
        "detail": "",
        "name": "git",
        "required": True,
        "status": "ok",
        "version": "2.50",
    }


def test_optional_health_failure_is_degraded_not_error() -> None:
    report = HealthReport(
        "runtime", (HealthCheck("memory", False, "error", detail="unavailable"),)
    )
    assert report.status == "degraded"


def test_cli_health_json_includes_real_tool_catalog(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent_core.health.collect_dependency_checks", lambda profile: [])
    args = argparse.Namespace(
        config=None,
        model=None,
        permission=None,
        provider="fake",
        effort=None,
        memory=False,
        profile="runtime",
        json=True,
    )
    assert health_command(args) == 0
    payload = json.loads(capsys.readouterr().out)
    registry = next(item for item in payload["checks"] if item["name"] == "tool-registry")
    assert registry["status"] == "ok"
    assert int(registry["detail"].split()[0]) > 0
