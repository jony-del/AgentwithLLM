"""Machine-readable health checks for runtime and development installations."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from agent_core.sandbox.config import SandboxContainerConfig

_RUNTIME_DISTRIBUTIONS = (
    "agent-with-llm",
    "httpx",
    "pyyaml",
    "markdownify",
    "beautifulsoup4",
    "ddgs",
    "mcp",
    "mcp-server-git",
    "mcp-server-fetch",
    "mcp-server-time",
    "rich",
    "prompt-toolkit",
)
_DEV_DISTRIBUTIONS = ("pytest", "pytest-asyncio", "ruff", "mypy")
_HOST_COMMANDS = ("git", "rg", "node", "npm", "npx")
_CONTAINER_RUNTIMES = ("podman", "docker", "nerdctl")


@dataclass(frozen=True, slots=True)
class HealthCheck:
    name: str
    required: bool
    status: str
    version: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True, slots=True)
class HealthReport:
    profile: str
    checks: tuple[HealthCheck, ...]

    @property
    def status(self) -> str:
        required_failed = any(check.required and not check.ok for check in self.checks)
        if required_failed:
            return "error"
        return "degraded" if any(not check.ok for check in self.checks) else "ok"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "profile": self.profile,
            "checks": [asdict(check) for check in self.checks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


def collect_dependency_checks(
    profile: str = "runtime",
    *,
    bash_executable: str | None = None,
    powershell_executable: str | None = None,
) -> list[HealthCheck]:
    if profile not in {"runtime", "dev"}:
        raise ValueError(f"unknown health profile: {profile}")
    checks = [
        HealthCheck(
            name="python",
            required=True,
            status="ok" if sys.version_info >= (3, 11) else "error",
            version=platform_python_version(),
            detail="Python 3.11 or newer is required",
        )
    ]
    distributions: Iterable[str] = _RUNTIME_DISTRIBUTIONS
    if profile == "dev":
        distributions = (*_RUNTIME_DISTRIBUTIONS, *_DEV_DISTRIBUTIONS)
    checks.extend(_distribution_check(name) for name in distributions)
    checks.extend(_command_check(name) for name in _HOST_COMMANDS)
    from agent_core.process_supervisor import (
        ShellUnavailableError,
        resolve_bash_executable,
        resolve_powershell_executable,
    )

    for name, required, resolver, configured in (
        (
            "git-bash" if sys.platform == "win32" else "bash",
            True,
            resolve_bash_executable,
            bash_executable,
        ),
        ("powershell", False, resolve_powershell_executable, powershell_executable),
    ):
        try:
            executable = resolver(configured)
        except ShellUnavailableError as exc:
            checks.append(HealthCheck(name, required, "error" if required else "missing", detail=str(exc)))
        else:
            checks.append(HealthCheck(name, required, "ok", version=_command_version(executable), detail=executable))
    from agent_core.scheduler_service import SERVICE_ID, default_receipt_path

    scheduler_receipt = default_receipt_path()
    try:
        receipt = json.loads(scheduler_receipt.read_text(encoding="utf-8"))
        service_executable = Path(str(receipt.get("executable", "")))
        scheduler_ok = receipt.get("service_id") == SERVICE_ID and service_executable.is_file()
    except (OSError, ValueError):
        scheduler_ok = False
    checks.append(
        HealthCheck(
            "scheduler-service", False, "ok" if scheduler_ok else "missing",
            detail=str(scheduler_receipt) if scheduler_ok else "current-user service is not installed",
        )
    )
    runtime = _usable_container_runtime()
    checks.append(
        HealthCheck(
            name="container-runtime",
            required=True,
            status="ok" if runtime else "error",
            version=_command_version(runtime) if runtime else "",
            detail=runtime or "no usable podman/docker/nerdctl runtime",
        )
    )
    image = SandboxContainerConfig().image
    image_ok = bool(runtime and _probe([runtime, "image", "inspect", image]))
    checks.append(
        HealthCheck(
            name="sandbox-image",
            required=True,
            status="ok" if image_ok else "error",
            detail=image if image_ok else f"{image} is not present",
        )
    )
    return checks


def render_human(report: HealthReport) -> str:
    lines = ["Polaris Health Check", "=" * 40]
    for check in report.checks:
        marker = "OK" if check.ok else "FAIL"
        description = check.version or check.detail
        if check.version and check.detail:
            description = f"{check.version} ({check.detail})"
        lines.append(f"[{marker:4}] {check.name}: {description}")
    lines.extend(["=" * 40, f"Overall status: {report.status}"])
    return "\n".join(lines)


def platform_python_version() -> str:
    return ".".join(str(item) for item in sys.version_info[:3])


def _distribution_check(name: str) -> HealthCheck:
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return HealthCheck(name, True, "error", detail="Python distribution is not installed")
    return HealthCheck(name, True, "ok", version=version)


def _command_check(name: str) -> HealthCheck:
    version = _command_version(name)
    if not version:
        return HealthCheck(name, True, "error", detail="command is missing or unusable")
    if name == "node" and _node_major(version) < 24:
        return HealthCheck(name, True, "error", version=version, detail="Node.js 24 LTS or newer is required")
    return HealthCheck(name, True, "ok", version=version)


def _command_version(name: str | None) -> str:
    if not name:
        return ""
    executable = shutil.which(name)
    if executable is None:
        return ""
    try:
        proc = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode:
        return ""
    output = (proc.stdout or proc.stderr or "").strip().splitlines()
    return output[0] if output else name


def _node_major(version: str) -> int:
    for token in version.replace("v", " ").split():
        head = token.split(".", 1)[0]
        if head.isdigit():
            return int(head)
    return 0


def _usable_container_runtime() -> str | None:
    for runtime in _CONTAINER_RUNTIMES:
        if shutil.which(runtime) and _probe([runtime, "info"]):
            return runtime
    return None


def _probe(argv: list[str]) -> bool:
    try:
        proc = subprocess.run(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0

