"""Bounded, opt-in startup of an existing Windows WSL2 Podman Machine."""

from __future__ import annotations

import json
import os
from pathlib import PureWindowsPath
import re
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urlsplit

from agent_core.sandbox.config import SandboxContainerConfig

_START_TIMEOUT = 120.0
_PROBE_TIMEOUT = 10.0
_POLL_INTERVAL = 1.0


class PodmanUnavailable(RuntimeError):
    """The requested Podman connection could not be made ready safely."""


def _summary(value: bytes | str | None) -> str:
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    return " ".join("".join(c for c in text if c.isprintable() or c.isspace()).split())[:1200]


def _failure(proc: subprocess.CompletedProcess[bytes]) -> str:
    return f"exit {proc.returncode}: {_summary(proc.stderr or proc.stdout) or 'no diagnostic output'}"


class _Commands:
    def __init__(self, runtime: str) -> None:
        self.runtime = runtime
        self.deadline = time.monotonic() + _START_TIMEOUT

    def remaining(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise PodmanUnavailable(f"Podman startup/readiness timed out after {_START_TIMEOUT:g}s")
        return remaining

    def run(self, *args: str, start: bool = False) -> subprocess.CompletedProcess[bytes]:
        timeout = self.remaining()
        if not start:
            timeout = min(timeout, _PROBE_TIMEOUT)
        try:
            return subprocess.run(
                [self.runtime, *args], capture_output=True, check=False, timeout=timeout,
                stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            detail = _summary(exc.stderr or exc.stdout)
            raise PodmanUnavailable(
                f"Podman {' '.join(args)} timed out after {timeout:g}s"
                + (f": {detail}" if detail else "")
            ) from exc
        except OSError as exc:
            raise PodmanUnavailable(f"Podman {' '.join(args)} could not execute: {_summary(str(exc))}") from exc

    def records(self, *args: str) -> list[dict[str, Any]]:
        proc = self.run(*args)
        if proc.returncode:
            raise PodmanUnavailable(f"Podman {' '.join(args)} failed ({_failure(proc)})")
        try:
            value = json.loads(proc.stdout)
        except (ValueError, UnicodeError) as exc:
            raise PodmanUnavailable(f"Podman {' '.join(args)} returned invalid JSON") from exc
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise PodmanUnavailable(f"Podman {' '.join(args)} returned an invalid record list")
        return value

    def machine(self, name: str) -> dict[str, Any]:
        machines = self.records("machine", "list", "--format", "json")
        matches = [item for item in machines if item.get("Name") == name]
        if len(matches) != 1:
            raise PodmanUnavailable(
                f"Podman Machine {name!r} is not initialized; initialize it explicitly "
                f"with 'podman machine init {name}'"
            )
        machine = matches[0]
        if machine.get("VMType") != "wsl":
            raise PodmanUnavailable(f"Podman Machine {name!r} is not a WSL2 machine")
        if not all(isinstance(machine.get(key), bool) for key in ("Running", "Starting")):
            raise PodmanUnavailable(f"Podman Machine {name!r} returned invalid running/starting state")
        return machine


def _check_connection(commands: _Commands, machine: dict[str, Any], name: str) -> None:
    if os.getenv("CONTAINER_HOST"):
        raise PodmanUnavailable(
            "CONTAINER_HOST overrides the Podman connection; automatic machine startup is unavailable"
        )
    connections = commands.records("system", "connection", "list", "--format", "json")
    requested = os.getenv("CONTAINER_CONNECTION")
    selected = [
        item for item in connections
        if (item.get("Name") == requested if requested else item.get("Default") is True)
    ]
    if len(selected) == 1:
        connection = selected[0]
        try:
            uri = urlsplit(str(connection.get("URI", "")))
            matches = (
                connection.get("Name") in {name, f"{name}-root"}
                and connection.get("IsMachine") is True
                and uri.scheme == "ssh"
                and uri.hostname in {"localhost", "127.0.0.1", "::1"}
                and uri.port == int(machine["Port"])
                and bool(machine.get("IdentityPath"))
                and PureWindowsPath(str(connection.get("Identity", "")))
                == PureWindowsPath(str(machine["IdentityPath"]))
            )
        except (KeyError, ValueError, TypeError):
            matches = False
        if matches:
            return
    raise PodmanUnavailable(
        f"active Podman connection does not match Machine {name!r}; "
        "check 'podman system connection list' and sandbox.container.podman_machine_name"
    )


def ensure_podman_ready(runtime: str, config: SandboxContainerConfig) -> None:
    """Probe first; start at most once, without initializing or changing connections.

    A shared deadline includes discovery, startup and polling. A competing process
    may win the start race; only confirmed running/starting state permits waiting
    after a failed start. Readiness always requires a successful ``podman info``.
    """
    commands = _Commands(runtime)
    info = commands.run("info")
    if info.returncode == 0:
        return
    name = config.podman_machine_name
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name):
        raise PodmanUnavailable("sandbox.container.podman_machine_name is invalid")
    try:
        machine = commands.machine(name)
        _check_connection(commands, machine, name)
    except PodmanUnavailable as exc:
        raise PodmanUnavailable(f"{exc}; podman info: {_failure(info)}") from exc

    active = machine["Running"] or machine["Starting"]
    if not config.auto_start_machine:
        state = "starting" if machine["Starting"] else "running" if machine["Running"] else "stopped"
        hint = f"run 'podman machine start {name}'" if not active else "check the Podman connection"
        raise PodmanUnavailable(
            f"Podman Machine {name!r} is {state}; {hint}; podman info: {_failure(info)}"
        )

    action = "Waiting for" if active else "Starting"
    print(f"[sandbox] {action} Podman Machine {name!r} (up to {_START_TIMEOUT:g}s)...", file=sys.stderr)
    if not active:
        started = commands.run("machine", "start", name, start=True)
        if started.returncode:
            # Do not infer a successful race from localized stderr text.
            try:
                machine = commands.machine(name)
            except PodmanUnavailable as exc:
                raise PodmanUnavailable(
                    f"Podman Machine {name!r} failed to start ({_failure(started)}); {exc}"
                ) from exc
            if not (machine["Running"] or machine["Starting"]):
                raise PodmanUnavailable(f"Podman Machine {name!r} failed to start ({_failure(started)})")
    while True:
        try:
            info = commands.run("info")
            if info.returncode == 0:
                return
            time.sleep(min(_POLL_INTERVAL, commands.remaining()))
        except PodmanUnavailable as exc:
            raise PodmanUnavailable(f"{exc}; last podman info: {_failure(info)}") from exc
