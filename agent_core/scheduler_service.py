"""Cross-platform user-service management for the scheduler routing daemon."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import tempfile
import time
from typing import Any

from agent_core.scheduler import SchedulerStore


SERVICE_ID = "polaris-scheduler"


def _uid() -> int:
    getter = getattr(os, "getuid", None)
    if getter is None:
        raise RuntimeError("this user-service manager requires a POSIX user id")
    return int(getter())


def default_receipt_path() -> Path:
    return Path.home() / ".polaris" / "scheduler-service.json"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _run(argv: list[str]) -> None:
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=30, check=False)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"service command failed ({completed.returncode}): {detail[:4000]}")


def install_user_service(
    *, executable: str | Path = sys.executable, database: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    executable = Path(executable).resolve()
    database = Path(database).expanduser().resolve()
    receipt_path = Path(receipt_path) if receipt_path is not None else default_receipt_path()
    if receipt_path.exists():
        uninstall_user_service(
            expected_executable=executable, receipt_path=receipt_path, purge_data=False
        )
    command = [str(executable), "-m", "agent_core.scheduler_service", "run", "--database", str(database)]
    platform = sys.platform
    resources: list[str] = []
    if platform == "win32":
        task_name = "Polaris Scheduler"
        task_command = subprocess.list2cmdline(command)
        if len(task_command) > 261:
            raise RuntimeError(
                "scheduler command exceeds the Windows Task Scheduler /TR limit; "
                "install Polaris in a shorter path"
            )
        _run([
            "schtasks", "/Create", "/TN", task_name, "/SC", "ONLOGON", "/RL", "LIMITED",
            "/TR", task_command,
            "/F",
        ])
        _run(["schtasks", "/Run", "/TN", task_name])
        resources.append(task_name)
    elif platform.startswith("linux"):
        if not os.environ.get("XDG_RUNTIME_DIR"):
            raise RuntimeError("systemd user services require XDG_RUNTIME_DIR and a user service manager")
        unit = Path.home() / ".config" / "systemd" / "user" / f"{SERVICE_ID}.service"
        unit.parent.mkdir(parents=True, exist_ok=True)
        _atomic_bytes(unit, (
            "[Unit]\nDescription=Polaris scheduler router\n[Service]\nType=simple\n"
            + "ExecStart=" + " ".join(command) + "\nRestart=on-failure\nNoNewPrivileges=true\n"
            + "[Install]\nWantedBy=default.target\n"
        ).encode("utf-8"))
        _run(["systemctl", "--user", "daemon-reload"])
        _run(["systemctl", "--user", "enable", "--now", unit.name])
        resources.append(str(unit))
    elif platform == "darwin":
        plist = Path.home() / "Library" / "LaunchAgents" / f"com.openai.{SERVICE_ID}.plist"
        plist.parent.mkdir(parents=True, exist_ok=True)
        _atomic_bytes(plist, plistlib.dumps({
            "Label": f"com.openai.{SERVICE_ID}", "ProgramArguments": command,
            "RunAtLoad": True, "KeepAlive": True, "ProcessType": "Background",
        }))
        _run(["launchctl", "bootstrap", f"gui/{_uid()}", str(plist)])
        resources.append(str(plist))
    else:
        raise RuntimeError(f"unsupported user-service manager on {platform}")
    receipt = {
        "version": 1, "service_id": SERVICE_ID, "platform": platform,
        "executable": str(executable), "database": str(database), "resources": resources,
        "installed_at": time.time(),
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def uninstall_user_service(
    *, expected_executable: str | Path = sys.executable,
    receipt_path: str | Path | None = None, purge_data: bool = False,
) -> None:
    receipt_path = Path(receipt_path) if receipt_path is not None else default_receipt_path()
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"scheduler install receipt is missing or invalid: {exc}") from exc
    if receipt.get("service_id") != SERVICE_ID or Path(receipt.get("executable", "")).resolve() != Path(expected_executable).resolve():
        raise RuntimeError("scheduler receipt does not match this installation; refusing to remove external resources")
    platform = receipt.get("platform")
    resources = [str(item) for item in receipt.get("resources", [])]
    if platform == "win32":
        if resources:
            _run(["schtasks", "/Delete", "/TN", resources[0], "/F"])
    elif platform.startswith("linux"):
        for item in resources:
            unit = Path(item)
            _run(["systemctl", "--user", "disable", "--now", unit.name])
            unit.unlink(missing_ok=True)
        _run(["systemctl", "--user", "daemon-reload"])
    elif platform == "darwin":
        for item in resources:
            plist = Path(item)
            _run(["launchctl", "bootout", f"gui/{_uid()}", str(plist)])
            plist.unlink(missing_ok=True)
    else:
        raise RuntimeError(f"receipt names unsupported platform: {platform}")
    if purge_data:
        Path(receipt["database"]).unlink(missing_ok=True)
    receipt_path.unlink(missing_ok=True)


async def run_daemon(database: str | Path, *, interval: float = 30.0) -> None:
    store = SchedulerStore(database)
    while True:
        await asyncio.to_thread(store.route_due)
        await asyncio.sleep(max(1.0, min(interval, 60.0)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--database", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        asyncio.run(run_daemon(args.database))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
