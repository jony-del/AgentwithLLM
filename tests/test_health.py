from __future__ import annotations

import argparse
import json

from agent_core.cli import health_command
from agent_core.health import HealthCheck, HealthReport, collect_dependency_checks


def _health_args(*, config: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        config=config,
        model=None,
        permission=None,
        provider="fake",
        effort=None,
        memory=False,
        profile="runtime",
        json=True,
    )


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
    monkeypatch.setattr(
        "agent_core.health.collect_dependency_checks",
        lambda profile, **overrides: [],
    )
    assert health_command(_health_args()) == 0
    payload = json.loads(capsys.readouterr().out)
    registry = next(item for item in payload["checks"] if item["name"] == "tool-registry")
    assert registry["status"] == "ok"
    assert int(registry["detail"].split()[0]) > 0


def test_dependency_health_forwards_configured_shell_executables(
    monkeypatch, tmp_path,
) -> None:
    seen: dict[str, str | None] = {}

    def resolve_bash(configured: str | None = None) -> str:
        seen["bash"] = configured
        return configured or "bash"

    def resolve_powershell(configured: str | None = None) -> str:
        seen["powershell"] = configured
        return configured or "powershell"

    monkeypatch.setattr("agent_core.health._RUNTIME_DISTRIBUTIONS", ())
    monkeypatch.setattr("agent_core.health._HOST_COMMANDS", ())
    monkeypatch.setattr("agent_core.health._command_version", lambda executable: "version")
    monkeypatch.setattr("agent_core.health._usable_container_runtime", lambda: None)
    monkeypatch.setattr(
        "agent_core.process_supervisor.resolve_bash_executable", resolve_bash
    )
    monkeypatch.setattr(
        "agent_core.process_supervisor.resolve_powershell_executable",
        resolve_powershell,
    )
    monkeypatch.setattr(
        "agent_core.scheduler_service.default_receipt_path",
        lambda: tmp_path / "missing-receipt.json",
    )

    checks = collect_dependency_checks(
        bash_executable="D:/Git/bin/bash.exe",
        powershell_executable="D:/PowerShell/pwsh.exe",
    )

    assert seen == {
        "bash": "D:/Git/bin/bash.exe",
        "powershell": "D:/PowerShell/pwsh.exe",
    }
    bash = next(check for check in checks if check.name in {"bash", "git-bash"})
    assert bash.status == "ok"
    assert bash.detail == "D:/Git/bin/bash.exe"

    from agent_core.process_supervisor import ShellUnavailableError

    def missing_bash(configured: str | None = None) -> str:
        raise ShellUnavailableError(f"configured Bash does not exist: {configured}")

    monkeypatch.setattr(
        "agent_core.process_supervisor.resolve_bash_executable", missing_bash
    )
    checks = collect_dependency_checks(bash_executable="D:/Missing/bash.exe")
    bash = next(check for check in checks if check.name in {"bash", "git-bash"})
    assert bash.status == "error"
    assert bash.detail == "configured Bash does not exist: D:/Missing/bash.exe"


def test_cli_health_uses_bash_environment_when_unconfigured(
    monkeypatch, tmp_path, capsys,
) -> None:
    seen: dict[str, str | None] = {}

    def collect(profile: str, **overrides: str | None) -> list[HealthCheck]:
        seen.update(overrides)
        return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POLARIS_BASH_PATH", "D:/Git/bin/bash.exe")
    monkeypatch.setattr("agent_core.health.collect_dependency_checks", collect)

    assert health_command(_health_args()) == 0
    capsys.readouterr()
    assert seen["bash_executable"] == "D:/Git/bin/bash.exe"


def test_cli_health_prefers_configured_bash_over_environment(
    monkeypatch, tmp_path, capsys,
) -> None:
    config = tmp_path / "health.toml"
    config.write_text(
        '[tools.shell.bash]\nexecutable = "D:/Configured/Git/bash.exe"\n',
        encoding="utf-8",
    )
    seen: dict[str, str | None] = {}

    def collect(profile: str, **overrides: str | None) -> list[HealthCheck]:
        seen.update(overrides)
        return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POLARIS_BASH_PATH", "D:/Environment/Git/bash.exe")
    monkeypatch.setattr("agent_core.health.collect_dependency_checks", collect)

    assert health_command(_health_args(config=str(config))) == 0
    capsys.readouterr()
    assert seen["bash_executable"] == "D:/Configured/Git/bash.exe"
