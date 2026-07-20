"""Safely remove an installer-owned Polaris command and its private dependencies.

This module deliberately uses only the Python standard library.  It is imported by
the installed CLI, but the release bootstraps can also execute this file directly
with an existing uv-managed Python when the ``polaris`` command is damaged.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Sequence

PACKAGE_NAME = "agent-with-llm"
COMMAND_NAME = "polaris"
STATE_SCHEMA = 2

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_FAILED = 10
EXIT_UNSUPPORTED = 30


class UninstallError(RuntimeError):
    """A safe, user-actionable uninstall failure."""


class OwnershipError(UninstallError):
    """The selected Polaris installation is not proven to be installer-owned."""


class UsageError(UninstallError):
    """Confirmation or command-line input is invalid."""


CommandRunner = Callable[..., subprocess.CompletedProcess[Any]]


@dataclass(slots=True)
class UninstallPlan:
    """A fully resolved plan that can survive deletion of the calling environment."""

    kind: str
    package: str
    executable: str
    environment: str
    source: str
    uv: str
    tool_root: str
    bin_dir: str
    bootstrap_python: str
    state_path: str
    data_path: str
    runtime_root: str
    node_path: str = ""
    node_bin: str = ""
    node_links: tuple[str, ...] = ()
    purge_data: bool = False
    legacy: bool = False
    already_absent: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> UninstallPlan:
        known = {field.name for field in fields(cls)}
        payload = {key: item for key, item in value.items() if key in known}
        payload["node_links"] = tuple(payload.get("node_links", ()))
        return cls(**payload)


def default_state_path() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "Polaris" / "install-state.json"
    root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return root / "polaris" / "install-state.json"


def default_data_path() -> Path:
    return Path.home() / ".polaris"


def managed_runtime_root() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "Polaris" / "runtime"
    root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return root / "polaris" / "runtime"


def _resolved(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _path_key(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.normpath(str(_resolved(path))))


def same_path(left: str | os.PathLike[str], right: str | os.PathLike[str]) -> bool:
    return _path_key(left) == _path_key(right)


def path_within(path: str | os.PathLike[str], root: str | os.PathLike[str]) -> bool:
    child = _resolved(path)
    parent = _resolved(root)
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _command_text(argv: Sequence[str]) -> str:
    return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return value.decode("utf-8", errors="replace")


def _run_process(
    argv: Sequence[str],
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(list(argv), capture_output=True, check=False, env=env)


def _read_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise UninstallError(f"could not read install state {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") not in {1, STATE_SCHEMA}:
        raise UninstallError(f"unsupported or damaged install state: {path}")
    if not isinstance(value.get("components", {}), dict):
        raise UninstallError(f"damaged component receipts in install state: {path}")
    return value


def _atomic_write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _external_guidance(current_executable: Path | None = None) -> str:
    python = sys.executable
    command_path = current_executable
    if command_path is None:
        located = shutil.which(COMMAND_NAME)
        command_path = _resolved(located) if located else None
    if command_path:
        if command_path.name.lower().startswith("python"):
            python = str(command_path)
        elif command_path.parent.name.lower() == "scripts":
            candidate = command_path.parent.parent / "python.exe"
            if candidate.is_file():
                python = str(candidate)
        elif command_path.parent.name == "bin":
            candidate = command_path.parent / "python"
            if candidate.is_file():
                python = str(candidate)
    command = _command_text([python, "-m", "pip", "uninstall", PACKAGE_NAME])
    return (
        "This Polaris installation is not proven to be owned by the Polaris installer; "
        "it was left unchanged. Uninstall it with the environment that owns it:\n"
        f"  {command}"
    )


class Uninstaller:
    """Resolve and apply deletion plans using exact, state-backed ownership."""

    def __init__(
        self,
        *,
        state_path: Path | None = None,
        data_path: Path | None = None,
        runtime_root: Path | None = None,
        current_executable: Path | None = None,
        runner: CommandRunner = _run_process,
    ) -> None:
        self.state_path = _resolved(state_path or default_state_path())
        self.data_path = _resolved(data_path or default_data_path())
        self.runtime_root = _resolved(runtime_root or managed_runtime_root())
        self.current_executable = (
            _resolved(current_executable) if current_executable is not None else None
        )
        self.runner = runner

    def build_plan(self, *, purge_data: bool = False) -> UninstallPlan:
        state = _read_state(self.state_path)
        if not state:
            if self.current_executable is None and shutil.which(COMMAND_NAME) is None:
                return self._empty_plan(purge_data=purge_data)
            raise OwnershipError(_external_guidance(self.current_executable))

        schema = int(state.get("schema", 0))
        components = state.get("components", {})
        if schema == 1:
            receipt = self._legacy_receipt(components)
            legacy = True
        else:
            receipt = components.get("polaris")
            legacy = False

        if not isinstance(receipt, dict):
            if self.current_executable is None and shutil.which(COMMAND_NAME) is None:
                return self._empty_plan(purge_data=purge_data)
            raise OwnershipError(_external_guidance(self.current_executable))
        if not receipt.get("installed_by_polaris"):
            recorded = str(receipt.get("executable", ""))
            command = self.current_executable or (_resolved(recorded) if recorded else None)
            raise OwnershipError(_external_guidance(command))

        kind = str(receipt.get("install_kind", ""))
        if legacy and not kind:
            kind = str(receipt.get("_legacy_kind", ""))
        if kind == "uv-tool":
            plan = self._uv_plan(receipt, purge_data=purge_data, legacy=legacy)
        elif kind == "dev-venv":
            plan = self._dev_plan(receipt, purge_data=purge_data, legacy=legacy)
        else:
            raise OwnershipError(
                f"Install receipt has unsupported kind {kind!r}; no files were changed.\n"
                + _external_guidance(self.current_executable)
            )

        self._attach_node(plan, components.get("node"))
        return plan

    def _empty_plan(self, *, purge_data: bool) -> UninstallPlan:
        return UninstallPlan(
            kind="none",
            package=PACKAGE_NAME,
            executable="",
            environment="",
            source="",
            uv="",
            tool_root="",
            bin_dir="",
            bootstrap_python="",
            state_path=str(self.state_path),
            data_path=str(self.data_path),
            runtime_root=str(self.runtime_root),
            purge_data=purge_data,
            already_absent=True,
        )

    def _legacy_receipt(self, components: dict[str, Any]) -> dict[str, Any] | None:
        dev = components.get("polaris-dev")
        if isinstance(dev, dict) and dev.get("installed_by_polaris"):
            source = _resolved(str(dev.get("source", "")))
            environment = source / ".venv"
            executable = environment / (
                "Scripts/polaris.exe" if os.name == "nt" else "bin/polaris"
            )
            if not (environment / "pyvenv.cfg").is_file() or not executable.exists():
                raise OwnershipError(
                    "Legacy development receipt cannot be matched to source/.venv; "
                    "no files were changed."
                )
            if self.current_executable and not same_path(self.current_executable, executable):
                raise OwnershipError(_external_guidance(self.current_executable))
            return {
                **dev,
                "_legacy_kind": "dev-venv",
                "package": PACKAGE_NAME,
                "source": str(source),
                "environment": str(environment),
                "executable": str(executable),
                "bootstrap_python": self._find_bootstrap_python(shutil.which("uv") or ""),
            }

        tool = components.get("polaris")
        if not isinstance(tool, dict) or not tool.get("installed_by_polaris"):
            return None
        return self._discover_legacy_uv_tool(tool)

    def _discover_legacy_uv_tool(self, receipt: dict[str, Any]) -> dict[str, Any]:
        uv = shutil.which("uv")
        if not uv:
            raise OwnershipError(
                "Legacy receipt needs uv to prove the isolated tool ownership; "
                "uv was not found and no files were changed."
            )
        listed = self.runner([uv, "tool", "list", "--show-paths"])
        output = _decode(listed.stdout) + "\n" + _decode(listed.stderr)
        if listed.returncode or not re.search(
            rf"(?m)^{re.escape(PACKAGE_NAME)}(?:\s|$)", output
        ):
            raise OwnershipError(_external_guidance(self.current_executable))
        tool_root = self._uv_directory(uv, ["tool", "dir"])
        bin_dir = self._uv_directory(uv, ["tool", "dir", "--bin"])
        executable = bin_dir / ("polaris.exe" if os.name == "nt" else "polaris")
        if not executable.exists() or str(executable) not in output:
            raise OwnershipError(_external_guidance(self.current_executable))
        if self.current_executable and not same_path(self.current_executable, executable):
            raise OwnershipError(_external_guidance(self.current_executable))
        return {
            **receipt,
            "_legacy_kind": "uv-tool",
            "package": PACKAGE_NAME,
            "executable": str(executable),
            "environment": str(tool_root / PACKAGE_NAME),
            "uv": uv,
            "tool_root": str(tool_root),
            "bin_dir": str(bin_dir),
            "bootstrap_python": self._find_bootstrap_python(uv),
        }

    def _uv_directory(self, uv: str, arguments: list[str]) -> Path:
        result = self.runner([uv, *arguments])
        value = _decode(result.stdout).strip().splitlines()
        if result.returncode or not value:
            detail = _decode(result.stderr).strip()
            raise UninstallError(
                f"could not locate uv tool directories with {_command_text([uv, *arguments])}"
                + (f": {detail}" if detail else "")
            )
        return _resolved(value[-1])

    def _find_bootstrap_python(self, uv: str) -> str:
        if not uv:
            return ""
        env = dict(os.environ)
        env["UV_PYTHON_DOWNLOADS"] = "never"
        result = self.runner([uv, "python", "find", "3.12"], env=env)
        lines = _decode(result.stdout).strip().splitlines()
        if result.returncode or not lines:
            return ""
        candidate = _resolved(lines[-1])
        return str(candidate) if candidate.is_file() else ""

    def _uv_plan(
        self, receipt: dict[str, Any], *, purge_data: bool, legacy: bool
    ) -> UninstallPlan:
        package = str(receipt.get("package", ""))
        if package != PACKAGE_NAME:
            raise OwnershipError(f"Unexpected package in install receipt: {package!r}")
        required = {
            name: str(receipt.get(name, ""))
            for name in (
                "executable",
                "environment",
                "uv",
                "tool_root",
                "bin_dir",
                "bootstrap_python",
            )
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise OwnershipError(
                "Install receipt is missing safe uninstall metadata: " + ", ".join(missing)
            )
        raw_environment = Path(required["environment"]).expanduser()
        if raw_environment.is_symlink():
            raise OwnershipError("Refusing to use a symlinked uv tool environment")
        environment = _resolved(raw_environment)
        tool_root = _resolved(required["tool_root"])
        executable = _resolved(required["executable"])
        bin_dir = _resolved(required["bin_dir"])
        uv_path = _resolved(required["uv"])
        if environment.parent != tool_root or environment.name != PACKAGE_NAME:
            raise OwnershipError("Tool environment is outside the recorded uv tool root")
        if executable.parent != bin_dir or executable.stem.lower() != COMMAND_NAME:
            raise OwnershipError("Polaris executable is outside the recorded uv bin directory")
        if not uv_path.is_file():
            raise UninstallError(
                f"The recorded uv executable is unavailable: {uv_path}. Restore uv, then rerun "
                "the recovery uninstall command."
            )
        if self.current_executable and not same_path(self.current_executable, executable):
            raise OwnershipError(_external_guidance(self.current_executable))
        bootstrap_python = _resolved(required["bootstrap_python"])
        if path_within(bootstrap_python, environment):
            raise OwnershipError("Uninstall worker Python is inside the environment being removed")
        if not bootstrap_python.is_file():
            replacement = self._find_bootstrap_python(required["uv"])
            if not replacement:
                raise UninstallError(
                    "The external uv-managed Python needed for self-uninstall is unavailable. "
                    "Use install.ps1 -Uninstall or install.sh --uninstall after restoring uv Python 3.12."
                )
            bootstrap_python = _resolved(replacement)
        return UninstallPlan(
            kind="uv-tool",
            package=package,
            executable=str(executable),
            environment=str(environment),
            source=str(receipt.get("source", "")),
            uv=str(uv_path),
            tool_root=str(tool_root),
            bin_dir=str(bin_dir),
            bootstrap_python=str(bootstrap_python),
            state_path=str(self.state_path),
            data_path=str(self.data_path),
            runtime_root=str(self.runtime_root),
            purge_data=purge_data,
            legacy=legacy,
            already_absent=not executable.exists() and not environment.exists(),
        )

    def _dev_plan(
        self, receipt: dict[str, Any], *, purge_data: bool, legacy: bool
    ) -> UninstallPlan:
        source = _resolved(str(receipt.get("source", "")))
        raw_environment = Path(str(receipt.get("environment", ""))).expanduser()
        if raw_environment.is_symlink():
            raise OwnershipError("Refusing to remove a symlinked development environment")
        environment = _resolved(raw_environment)
        executable = _resolved(str(receipt.get("executable", "")))
        expected_executable = environment / (
            "Scripts/polaris.exe" if os.name == "nt" else "bin/polaris"
        )
        if environment != source / ".venv" or executable != expected_executable:
            raise OwnershipError("Development receipt does not point to source/.venv")
        sentinel = environment / ".polaris-install.json"
        if environment.exists() and not legacy:
            try:
                marker = json.loads(sentinel.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise OwnershipError(
                    f"Development environment ownership marker is missing or damaged: {exc}"
                ) from exc
            if marker.get("package") != PACKAGE_NAME or not same_path(
                str(marker.get("source", "")), source
            ):
                raise OwnershipError("Development environment ownership marker does not match")
        if legacy and environment.exists() and not (environment / "pyvenv.cfg").is_file():
            raise OwnershipError("Legacy development environment has no pyvenv.cfg")
        if self.current_executable and not same_path(self.current_executable, executable):
            raise OwnershipError(_external_guidance(self.current_executable))
        bootstrap = str(receipt.get("bootstrap_python", ""))
        if not bootstrap or not _resolved(bootstrap).is_file():
            bootstrap = self._find_bootstrap_python(str(receipt.get("uv", shutil.which("uv") or "")))
        if not bootstrap or path_within(bootstrap, environment):
            raise UninstallError(
                "An external uv-managed Python is required to remove the active development .venv"
            )
        return UninstallPlan(
            kind="dev-venv",
            package=PACKAGE_NAME,
            executable=str(executable),
            environment=str(environment),
            source=str(source),
            uv=str(receipt.get("uv", "")),
            tool_root="",
            bin_dir="",
            bootstrap_python=str(_resolved(bootstrap)),
            state_path=str(self.state_path),
            data_path=str(self.data_path),
            runtime_root=str(self.runtime_root),
            purge_data=purge_data,
            legacy=legacy,
            already_absent=not environment.exists(),
        )

    def _attach_node(self, plan: UninstallPlan, receipt: Any) -> None:
        if not isinstance(receipt, dict) or not receipt.get("installed_by_polaris"):
            return
        if receipt.get("install_kind") != "managed-runtime":
            # Schema-1 Node receipts did not contain a path and cannot safely own it.
            return
        raw_destination = Path(str(receipt.get("path", ""))).expanduser()
        if raw_destination.is_symlink():
            raise OwnershipError("Refusing to follow a symlinked managed Node runtime")
        destination = _resolved(raw_destination)
        node_bin = _resolved(str(receipt.get("bin", "")))
        if destination.parent != self.runtime_root or not path_within(node_bin, destination):
            raise OwnershipError("Managed Node receipt points outside the Polaris runtime root")
        links = tuple(
            str(Path(item).expanduser().absolute()) for item in receipt.get("links", [])
        )
        plan.node_path = str(destination)
        plan.node_bin = str(node_bin)
        plan.node_links = links


def render_plan(plan: UninstallPlan) -> str:
    lines = ["\nUninstall plan", "--------------"]
    if plan.kind == "none" or plan.already_absent:
        lines.append("[ABSENT  ] Polaris program files are already absent")
    elif plan.kind == "uv-tool":
        lines.append(f"[REMOVE  ] uv tool: {plan.package} ({plan.environment})")
        lines.append(f"[REMOVE  ] command: {plan.executable}")
    else:
        lines.append(f"[REMOVE  ] installer-owned development environment: {plan.environment}")
    if plan.node_path:
        lines.append(f"[REMOVE  ] Polaris private Node runtime: {plan.node_path}")
    if plan.purge_data:
        lines.append(f"[REMOVE  ] user data: {plan.data_path}")
        lines.append(f"[REMOVE  ] install state: {plan.state_path}")
    else:
        lines.append(f"[PRESERVE] user data: {plan.data_path}")
        lines.append("[UPDATE  ] install state (Polaris/private-runtime receipts only)")
    lines.extend(
        (
            "[PRESERVE] project source and project-local .polaris/runs/memory",
            "[PRESERVE] WSL, container runtimes/images, Git, ripgrep, system Node, uv and uv Python",
        )
    )
    return "\n".join(lines)


def _confirm(*, yes: bool, dry_run: bool, non_interactive: bool) -> bool:
    if dry_run or yes:
        return True
    if non_interactive or not sys.stdin.isatty():
        raise UsageError("non-interactive uninstall requires --yes (or use --dry-run)")
    try:
        answer = input("Remove the listed Polaris files? [y/N] ").strip().lower()
    except EOFError as exc:
        raise UsageError("confirmation input ended; rerun with --yes or --dry-run") from exc
    return answer in {"y", "yes"}


def _validate_worker_plan(plan: UninstallPlan) -> UninstallPlan:
    current = Uninstaller(
        state_path=Path(plan.state_path),
        data_path=Path(plan.data_path),
        runtime_root=Path(plan.runtime_root),
    ).build_plan(purge_data=plan.purge_data)
    keys = (
        "kind",
        "package",
        "executable",
        "environment",
        "source",
        "uv",
        "tool_root",
        "bin_dir",
        "state_path",
        "data_path",
        "runtime_root",
        "node_path",
        "node_bin",
        "node_links",
        "purge_data",
    )
    if any(getattr(current, key) != getattr(plan, key) for key in keys):
        raise OwnershipError("Install state changed after uninstall was scheduled; no files changed")
    return current


def _remove_component_receipts(state_path: Path, names: set[str]) -> None:
    state = _read_state(state_path)
    if not state:
        return
    if state.get("schema") == 1:
        for item in state.get("components", {}).values():
            if isinstance(item, dict) and item.get("installed_by_polaris"):
                item["legacy_unverified"] = True
    state["schema"] = STATE_SCHEMA
    components = state.setdefault("components", {})
    for name in names:
        components.pop(name, None)
    # A migrated legacy dev receipt uses this old key.
    if "polaris" in names:
        components.pop("polaris-dev", None)
    completed = state.setdefault("completed", {})
    prefixes = {
        "polaris": ("project:",),
        "node": ("runtime:node",),
    }
    for name in names:
        for key in list(completed):
            if any(key.startswith(prefix) for prefix in prefixes.get(name, ())):
                completed.pop(key, None)
    _atomic_write_state(state_path, state)


def _remove_windows_user_path(directory: Path) -> None:
    import winreg

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
        try:
            current, value_type = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            return
        entries = [item for item in str(current).split(";") if item]
        kept = [item for item in entries if not same_path(item.strip('"'), directory)]
        if kept != entries:
            winreg.SetValueEx(key, "Path", 0, value_type, ";".join(kept))
    try:
        ctypes.windll.user32.SendMessageTimeoutW(
            0xFFFF, 0x001A, 0, "Environment", 0, 5000, None
        )
    except (AttributeError, OSError):
        pass


def _remove_private_node(plan: UninstallPlan) -> None:
    if not plan.node_path:
        return
    raw_destination = Path(plan.node_path).expanduser()
    if raw_destination.is_symlink():
        raise OwnershipError("Refusing to follow a symlinked managed Node runtime")
    destination = _resolved(raw_destination)
    runtime_root = _resolved(plan.runtime_root)
    node_bin = _resolved(plan.node_bin)
    if destination.parent != runtime_root or not path_within(node_bin, destination):
        raise OwnershipError("Refusing to remove Node outside the Polaris runtime root")
    if os.name == "nt":
        _remove_windows_user_path(node_bin)
    else:
        for raw_link in plan.node_links:
            link = Path(raw_link)
            if not link.is_symlink():
                continue
            try:
                target = link.resolve(strict=False)
            except OSError:
                continue
            if path_within(target, destination):
                link.unlink()
    if destination.exists():
        if destination.is_symlink():
            raise OwnershipError("Refusing to follow a symlinked managed Node runtime")
        shutil.rmtree(destination)
    try:
        runtime_root.rmdir()
    except OSError:
        pass


def _remove_user_data(path: Path) -> None:
    expected = _resolved(default_data_path())
    if not same_path(path, expected) or same_path(path, Path.home()):
        raise OwnershipError("Refusing to purge an unexpected user-data path")
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def apply_plan(plan: UninstallPlan) -> None:
    plan = _validate_worker_plan(plan)
    state_path = Path(plan.state_path)
    if plan.kind == "uv-tool" and not plan.already_absent:
        command = [plan.uv, "tool", "uninstall", plan.package]
        result = _run_process(command)
        if result.returncode and (
            Path(plan.executable).exists() or Path(plan.environment).exists()
        ):
            detail = (_decode(result.stderr) or _decode(result.stdout)).strip()
            raise UninstallError(
                f"command failed ({result.returncode}): {_command_text(command)}"
                + (f"\n{detail}" if detail else "")
            )
    elif plan.kind == "dev-venv" and not plan.already_absent:
        raw_environment = Path(plan.environment).expanduser()
        if raw_environment.is_symlink():
            raise OwnershipError("Development environment became a symlink")
        environment = _resolved(raw_environment)
        source = _resolved(plan.source)
        if environment != source / ".venv" or environment.is_symlink():
            raise OwnershipError("Development environment failed its final boundary check")
        shutil.rmtree(environment)
    _remove_component_receipts(state_path, {"polaris"})

    if plan.node_path:
        _remove_private_node(plan)
        _remove_component_receipts(state_path, {"node"})

    if plan.purge_data:
        _remove_user_data(Path(plan.data_path))
        try:
            state_path.unlink()
        except FileNotFoundError:
            pass


def _stage_worker(plan: UninstallPlan) -> Path:
    python = _resolved(plan.bootstrap_python)
    if not python.is_file() or (
        plan.environment and path_within(python, plan.environment)
    ):
        raise UninstallError("A safe external Python could not be selected for self-uninstall")
    directory = Path(tempfile.mkdtemp(prefix="polaris-uninstall-"))
    try:
        if os.name != "nt":
            directory.chmod(0o700)
        worker = directory / "uninstall-worker.py"
        plan_path = directory / "plan.json"
        log_path = directory / "uninstall.log"
        shutil.copy2(Path(__file__), worker)
        plan_path.write_text(
            json.dumps(asdict(plan), indent=2) + "\n", encoding="utf-8"
        )
        if os.name != "nt":
            worker.chmod(0o600)
            plan_path.chmod(0o600)
        command = [
            str(python),
            str(worker),
            "--worker",
            str(plan_path),
            "--parent-pid",
            str(os.getpid()),
            "--log",
            str(log_path),
        ]
        kwargs: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "close_fds": True,
        }
        if os.name == "nt":
            kwargs["creationflags"] = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen(command, **kwargs)
    except OSError as exc:
        shutil.rmtree(directory, ignore_errors=True)
        raise UninstallError(f"could not start the self-uninstall worker: {exc}") from exc
    return log_path


def _parent_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _worker_main(plan_path: Path, parent_pid: int, log_path: Path) -> int:
    worker_path = Path(__file__)
    deadline = time.monotonic() + 300
    while _parent_running(parent_pid):
        if time.monotonic() >= deadline:
            log_path.write_text(
                "[error] timed out waiting for the Polaris process to exit\n",
                encoding="utf-8",
            )
            return EXIT_FAILED
        time.sleep(0.1)
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        apply_plan(UninstallPlan.from_dict(payload))
    except (OSError, ValueError, TypeError, UninstallError) as exc:
        log_path.write_text(f"[error] {exc}\n", encoding="utf-8")
        return EXIT_FAILED
    else:
        log_path.write_text("[uninstalled] Polaris removal completed successfully.\n", encoding="utf-8")
        return EXIT_OK
    finally:
        for path in (plan_path, worker_path):
            try:
                path.unlink()
            except OSError:
                pass


def run_uninstall(
    *,
    purge_data: bool = False,
    dry_run: bool = False,
    yes: bool = False,
    non_interactive: bool = False,
    detach: bool = False,
    current_executable: Path | None = None,
    state_path: Path | None = None,
) -> int:
    uninstaller = Uninstaller(
        state_path=state_path,
        current_executable=current_executable,
    )
    plan = uninstaller.build_plan(purge_data=purge_data)
    print(render_plan(plan))
    if dry_run:
        print("\n[dry-run] No files or system settings were changed.")
        return EXIT_OK
    if not _confirm(yes=yes, dry_run=dry_run, non_interactive=non_interactive):
        print("[cancelled] No files or system settings were changed.")
        return EXIT_OK
    if plan.kind == "none" and not purge_data:
        print("[uninstalled] Polaris is already absent.")
        return EXIT_OK
    _remove_scheduler_service(plan)
    if detach and plan.kind != "none":
        try:
            log_path = _stage_worker(plan)
        except OSError as exc:
            raise UninstallError(f"could not stage the self-uninstall worker: {exc}") from exc
        print(
            "[scheduled] Polaris will be removed after this command exits.\n"
            f"Completion log: {log_path}"
        )
        return EXIT_OK
    try:
        apply_plan(plan)
    except OSError as exc:
        raise UninstallError(f"could not remove an installer-owned path: {exc}") from exc
    print("[uninstalled] Polaris removal completed successfully.")
    return EXIT_OK


def _remove_scheduler_service(plan: UninstallPlan) -> None:
    """Remove only a scheduler service proven by both installer and service receipts."""
    state_path = Path(plan.state_path)
    try:
        state = _read_state(state_path)
        component = state.get("components", {}).get("scheduler", {})
    except UninstallError:
        return
    if not isinstance(component, dict) or not component.get("installed_by_polaris"):
        return
    receipt_path = Path(str(component.get("receipt", ""))).expanduser()
    if not receipt_path.is_file() or component.get("service_id") != "polaris-scheduler":
        return
    try:
        service_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise UninstallError(f"scheduler service receipt is invalid; refusing unsafe cleanup: {exc}") from exc
    recorded_executable = Path(str(service_receipt.get("executable", ""))).expanduser()
    if (
        service_receipt.get("service_id") != "polaris-scheduler"
        or not recorded_executable.is_absolute()
        or _resolved(str(component.get("service_executable", ""))) != recorded_executable.resolve()
    ):
        raise UninstallError("scheduler service receipt does not match the installer receipt")
    from agent_core.scheduler_service import uninstall_user_service

    try:
        uninstall_user_service(
            expected_executable=recorded_executable, receipt_path=receipt_path,
            purge_data=plan.purge_data,
        )
    except RuntimeError as exc:
        raise UninstallError(f"could not safely remove scheduler user service: {exc}") from exc
    _remove_component_receipts(state_path, {"scheduler"})


def cli_executable() -> Path | None:
    candidate = Path(sys.argv[0])
    if candidate.stem.lower() == COMMAND_NAME and candidate.exists():
        return _resolved(candidate)
    located = shutil.which(COMMAND_NAME)
    return _resolved(located) if located else None


def uninstall_from_cli(args: argparse.Namespace) -> int:
    try:
        return run_uninstall(
            purge_data=bool(args.purge_data),
            dry_run=bool(args.dry_run),
            yes=bool(args.yes),
            detach=True,
            current_executable=cli_executable(),
        )
    except UsageError as exc:
        print(f"[usage] {exc}", file=sys.stderr)
        return EXIT_USAGE
    except OwnershipError as exc:
        print(f"[not installer-owned] {exc}", file=sys.stderr)
        return EXIT_FAILED
    except UninstallError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return EXIT_FAILED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--purge-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--state-path", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--parent-pid", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--log", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker is not None:
        if args.parent_pid <= 0 or args.log is None:
            print("[usage] worker mode requires --parent-pid and --log", file=sys.stderr)
            return EXIT_USAGE
        return _worker_main(args.worker, args.parent_pid, args.log)
    try:
        return run_uninstall(
            purge_data=args.purge_data,
            dry_run=args.dry_run,
            yes=args.yes,
            non_interactive=args.non_interactive,
            state_path=args.state_path,
        )
    except UsageError as exc:
        print(f"[usage] {exc}", file=sys.stderr)
        return EXIT_USAGE
    except OwnershipError as exc:
        print(f"[not installer-owned] {exc}", file=sys.stderr)
        return EXIT_FAILED
    except UninstallError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
