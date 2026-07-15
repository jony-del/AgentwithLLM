"""Install Polaris and the non-Python tools required by its full runtime profile.

The public entrypoints are ``install.ps1`` and ``install.sh``.  Keeping the actual
orchestration in Python gives every supported platform the same detection, state,
dry-run, and verification semantics while the two shell files only bootstrap uv and
obtain a release bundle.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import json
import locale
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

_MANIFEST = json.loads(Path(__file__).with_name("manifest.json").read_text(encoding="utf-8"))
PYTHON_SERIES = str(_MANIFEST["python_series"])
UV_VERSION = str(_MANIFEST["uv_version"])
NODE_VERSION = str(_MANIFEST["node_version"])
NODE_MAJOR = int(NODE_VERSION.removeprefix("v").split(".", 1)[0])
MIN_NODE_MAJOR = int(_MANIFEST["minimum_node_major"])
DEFAULT_IMAGE = str(_MANIFEST["sandbox_image"])
RUNTIME_COMMANDS = ("git", "rg", "node", "npm", "npx")
CONTAINER_RUNTIMES = ("podman", "docker", "nerdctl")

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_FAILED = 10
EXIT_RESTART_REQUIRED = 20
EXIT_UNSUPPORTED = 30
STATE_SCHEMA = 2
PACKAGE_NAME = "agent-with-llm"

WINDOWS_UAC_CANCELLED = 1223
WINDOWS_UAC_CANCELLED_HRESULT = 0x800704C7
WINDOWS_SUCCESS_OR_REBOOT_CODES = frozenset({0, 1641, 3010})

GITHUB_API = "https://api.github.com"
NODE_DIST = "https://nodejs.org/dist"


class InstallError(RuntimeError):
    """A user-actionable installation failure."""


class RestartRequired(InstallError):
    """The host must restart before the same installer command is run again."""


class UnsupportedHost(InstallError):
    """The current OS, architecture, or package manager is unsupported."""


@dataclass(slots=True)
class Options:
    source: Path
    dev: bool = False
    upgrade: bool = False
    check: bool = False
    dry_run: bool = False
    skip_sandbox: bool = False
    non_interactive: bool = False


@dataclass(slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


class StateStore:
    """Small resumable state file; it records ownership but never credentials."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_state_path()
        self.data: dict[str, Any] = {
            "schema": STATE_SCHEMA,
            "completed": {},
            "components": {},
        }
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            return
        if not isinstance(loaded, dict) or loaded.get("schema") not in {1, STATE_SCHEMA}:
            return
        completed = loaded.get("completed", {})
        components = loaded.get("components", {})
        if isinstance(completed, dict):
            self.data["completed"] = completed
        if isinstance(components, dict):
            self.data["components"] = components
        if loaded.get("schema") == 1:
            # Old receipts asserted ownership without recording an exact target. Keep
            # them for diagnostics, but never use them for upgrades or deletion until
            # the installer/uninstaller can independently prove the target.
            for item in self.data["components"].values():
                if isinstance(item, dict) and item.get("installed_by_polaris"):
                    item["legacy_unverified"] = True

    def completed(self, step: str) -> bool:
        return bool(self.data.get("completed", {}).get(step))

    def component_owned(self, name: str) -> bool:
        item = self.data.get("components", {}).get(name, {})
        return bool(
            item.get("installed_by_polaris") and not item.get("legacy_unverified")
        )

    def component(self, name: str) -> dict[str, Any]:
        item = self.data.get("components", {}).get(name, {})
        return dict(item) if isinstance(item, dict) else {}

    def mark(
        self,
        step: str,
        *,
        component: str | None = None,
        source: str = "",
        installed_by_polaris: bool = True,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.data.setdefault("completed", {})[step] = int(time.time())
        if component:
            receipt = {
                "installed_by_polaris": installed_by_polaris,
                "source": source,
                "updated_at": int(time.time()),
            }
            if details:
                receipt.update(details)
            components = self.data.setdefault("components", {})
            if component == "polaris":
                components.pop("polaris-dev", None)
            components[component] = receipt
        self.save()

    def save(self) -> None:
        self.data["schema"] = STATE_SCHEMA
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(self.path)


class Runner:
    def __init__(self, *, dry_run: bool = False, non_interactive: bool = False) -> None:
        self.dry_run = dry_run
        self.non_interactive = non_interactive

    def which(self, command: str) -> str | None:
        return shutil.which(command)

    def run(
        self,
        argv: Sequence[str | os.PathLike[str]],
        *,
        check: bool = False,
        capture: bool = False,
        mutates: bool = False,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        display_argv: Sequence[str | os.PathLike[str]] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        rendered = [str(item) for item in argv]
        displayed = [str(item) for item in display_argv] if display_argv else rendered
        print("+", subprocess.list2cmdline(displayed))
        if self.dry_run and mutates:
            return subprocess.CompletedProcess(rendered, 0, "", "")
        try:
            proc = subprocess.run(
                rendered,
                cwd=str(cwd) if cwd else None,
                env=env,
                capture_output=capture,
                check=False,
            )
        except OSError as exc:
            if check:
                raise InstallError(f"could not run {rendered[0]}: {exc}") from exc
            return subprocess.CompletedProcess(rendered, 127, "", str(exc))
        if capture:
            proc = subprocess.CompletedProcess(
                proc.args,
                proc.returncode,
                decode_process_output(proc.stdout),
                decode_process_output(proc.stderr),
            )
        if check and proc.returncode:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise InstallError(
                f"command failed ({proc.returncode}): {subprocess.list2cmdline(rendered)}"
                + (f"\n{detail}" if detail else "")
            )
        return proc


class Installer:
    def __init__(
        self,
        options: Options,
        *,
        runner: Runner | None = None,
        state: StateStore | None = None,
        system: str | None = None,
        machine: str | None = None,
    ) -> None:
        self.options = options
        self.runner = runner or Runner(
            dry_run=options.dry_run, non_interactive=options.non_interactive
        )
        self.state = state or StateStore()
        self.system = (system or platform.system()).lower()
        self.machine = normalize_machine(machine or platform.machine())
        self.package_family = self._detect_package_family()

    def install(self) -> list[Check]:
        self._validate_host()
        before = self.checks(include_project=self.options.check)
        self._print_checks("Preflight", before)
        if self.options.check:
            return before

        self._install_host_commands()
        self._install_node()
        if not self.options.skip_sandbox:
            self._ensure_container_runtime()
        self._install_project()
        if self.options.dry_run:
            return before

        after = self.checks(include_project=True)
        after.append(self._verify_installed_health())
        self._print_checks("Verification", after)
        return after

    def checks(self, *, include_project: bool = True) -> list[Check]:
        results: list[Check] = []
        for command in RUNTIME_COMMANDS:
            version = command_version(self.runner, command)
            ok = version is not None
            if command == "node" and ok:
                ok = parse_node_major(version or "") >= MIN_NODE_MAJOR
            results.append(Check(command, ok, version or "not found"))
        if not self.options.skip_sandbox:
            runtime = self._select_usable_runtime()
            detail = runtime or "no usable podman/docker/nerdctl runtime"
            results.append(Check("container-runtime", runtime is not None, detail))
            if runtime:
                present = self._image_present(runtime)
                results.append(
                    Check("sandbox-image", present, DEFAULT_IMAGE if present else "image missing")
                )
        if include_project:
            command = self._project_command()
            results.append(Check("polaris", command is not None, command or "not installed"))
        return results

    def _validate_host(self) -> None:
        if self.system == "windows":
            if self.machine != "x86_64":
                raise UnsupportedHost("Windows ARM is not supported by the first installer release")
            if os.name == "nt":
                fields = platform.version().split(".")
                build = int(fields[2]) if len(fields) >= 3 and fields[2].isdigit() else 0
                if build and build < 19043:
                    raise UnsupportedHost("Podman WSL2 requires Windows build 19043 or newer")
            return
        if self.system in {"darwin", "linux"} and self.machine in {"x86_64", "arm64"}:
            return
        raise UnsupportedHost(f"unsupported host: {self.system}/{self.machine}")

    def _detect_package_family(self) -> str:
        if self.system == "windows":
            return "winget"
        if self.system == "darwin":
            return "brew"
        if self.system == "linux":
            if shutil.which("apt-get"):
                return "apt"
            if shutil.which("dnf"):
                return "dnf"
        return "unsupported"

    def _install_host_commands(self) -> None:
        missing = [name for name in ("git", "rg") if self.runner.which(name) is None]
        upgrades = [
            name
            for name in ("git", "rg")
            if name not in missing and self.options.upgrade and self.state.component_owned(name)
        ]
        if not missing and not upgrades:
            return
        newly_owned: set[str] = set()
        if missing:
            newly_owned = self._install_packages(missing, upgrade=False)
        if upgrades:
            self._install_packages(upgrades, upgrade=True)
        if not self.options.dry_run:
            refresh_windows_path()
        if self.options.dry_run:
            return
        for name in missing:
            self.state.mark(
                f"host:{name}",
                component=name if name in newly_owned else None,
                source=self.package_family,
            )
        for name in upgrades:
            self.state.mark(f"host:{name}")

    def _install_packages(self, names: list[str], *, upgrade: bool = False) -> set[str]:
        package_map = {
            "winget": {"git": "Git.Git", "rg": "BurntSushi.ripgrep.MSVC", "podman": "RedHat.Podman"},
            "brew": {"git": "git", "rg": "ripgrep", "podman": "podman"},
            "apt": {"git": "git", "rg": "ripgrep", "podman": "podman"},
            "dnf": {"git": "git", "rg": "ripgrep", "podman": "podman"},
        }
        if self.package_family not in package_map:
            raise UnsupportedHost("Linux requires apt-get or dnf")
        packages = [package_map[self.package_family][name] for name in names]
        if self.package_family == "winget":
            if self.runner.which("winget") is None:
                raise InstallError(
                    "WinGet is required. Install Microsoft App Installer, then rerun this command."
                )
            verb = "upgrade" if upgrade else "install"
            preexisting = {
                name: self._winget_package_registered(package)
                for name, package in zip(names, packages, strict=True)
            }
            for name, package in zip(names, packages, strict=True):
                self._install_winget_package(name, package, verb=verb)
            return {
                name
                for name in names
                if not upgrade and not preexisting.get(name, False)
            }
        if self.package_family == "brew":
            self._ensure_homebrew()
            verb = "upgrade" if upgrade else "install"
            for package in packages:
                result = self.runner.run(["brew", verb, package], capture=True, mutates=True)
                if result.returncode and "already installed" not in (result.stderr or ""):
                    raise InstallError(result.stderr.strip() or f"brew {verb} {package} failed")
            return set() if upgrade else set(names)

        prefix = self._sudo_prefix()
        if self.package_family == "apt":
            self.runner.run([*prefix, "apt-get", "update"], check=True, mutates=True)
            args = [*prefix, "apt-get", "install", "-y"]
            if upgrade:
                args.append("--only-upgrade")
            self.runner.run([*args, *packages], check=True, mutates=True)
        else:
            verb = "upgrade" if upgrade else "install"
            self.runner.run(
                [*prefix, "dnf", verb, "-y", *packages], check=True, mutates=True
            )
        return set() if upgrade else set(names)

    def _winget_package_registered(self, package: str) -> bool:
        """Probe an exact package id before mutation without parsing localized text."""

        result = self.runner.run(
            [
                "winget",
                "list",
                "--id",
                package,
                "--exact",
                "--accept-source-agreements",
                "--disable-interactivity",
            ],
            capture=True,
        )
        return result.returncode == 0

    def _install_winget_package(self, command: str, package: str, *, verb: str) -> None:
        """Install one WinGet package and judge success by the usable command.

        ``winget install`` changes to its upgrade flow when its database already has
        a package record.  A fully up-to-date package then returns a non-zero code,
        even though the requested command may already be usable.  Conversely, stale
        portable-package records can say "installed" while the executable and shim
        are gone.  Neither case can be handled reliably by parsing localized output,
        so probe the postcondition and use one forced reinstall only when necessary.
        """

        base = [
            "winget",
            verb,
            "--id",
            package,
            "--exact",
            "--accept-package-agreements",
            "--accept-source-agreements",
        ]
        result = self.runner.run(base, mutates=True)
        if self.options.dry_run:
            return

        refresh_windows_path()
        if self.runner.which(command) is not None:
            if result.returncode:
                print(
                    f"WinGet returned {format_windows_exit_code(result.returncode)} for "
                    f"{package}, but {command} is available; continuing."
                )
            return

        # The WinGet registration exists but its portable payload or command shim may
        # be missing.  --force keeps the command in the install flow and recreates the
        # current version instead of switching to a no-op upgrade.
        print(f"WinGet did not expose {command}; repairing {package} with a forced reinstall...")
        repair_command = ["winget", "install", *base[2:], "--force"]
        repair = self.runner.run(repair_command, mutates=True)
        refresh_windows_path()
        if self.runner.which(command) is not None:
            return

        codes = (
            f"initial={format_windows_exit_code(result.returncode)}, "
            f"repair={format_windows_exit_code(repair.returncode)}"
        )
        raise InstallError(
            f"WinGet could not make {command} available after reinstalling {package} "
            f"({codes}). The WinGet package record may be damaged. Run "
            f"'winget uninstall --id {package} --exact', then rerun this installer."
        )

    def _ensure_homebrew(self) -> None:
        if self.runner.which("brew"):
            return
        if self.options.non_interactive:
            raise InstallError("Homebrew is missing; install it before using --non-interactive")
        script = (
            '/bin/bash -c "$(curl -fsSL '
            'https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        )
        self.runner.run(["/bin/bash", "-c", script], check=True, mutates=True)
        candidates = (Path("/opt/homebrew/bin/brew"), Path("/usr/local/bin/brew"))
        found = next((candidate for candidate in candidates if candidate.exists()), None)
        if found:
            os.environ["PATH"] = f"{found.parent}{os.pathsep}{os.environ.get('PATH', '')}"
        if not self.options.dry_run and self.runner.which("brew") is None:
            raise InstallError("Homebrew was installed but is not available on PATH; reopen the shell")

    def _sudo_prefix(self) -> list[str]:
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            return []
        if self.runner.which("sudo") is None:
            raise InstallError("system package installation requires root or sudo")
        return ["sudo", "-n"] if self.options.non_interactive else ["sudo"]

    def _install_node(self) -> None:
        current = command_version(self.runner, "node")
        current_major = parse_node_major(current or "")
        owned = self.state.component_owned("node")
        if current_major >= MIN_NODE_MAJOR and not (self.options.upgrade and owned):
            return
        if self.options.dry_run:
            print(f"Would install the latest Node.js {NODE_MAJOR}.x LTS official binary")
            return
        version = NODE_VERSION
        archive_name = node_archive_name(version, self.system, self.machine)
        base = f"{NODE_DIST}/{version}"
        with tempfile.TemporaryDirectory(prefix="polaris-node-") as raw_tmp:
            tmp = Path(raw_tmp)
            archive = tmp / archive_name
            checksums = tmp / "SHASUMS256.txt"
            download_file(f"{base}/{archive_name}", archive)
            download_file(f"{base}/SHASUMS256.txt", checksums)
            expected = checksum_for(archive_name, checksums.read_text(encoding="utf-8"))
            verify_sha256(archive, expected)
            runtime_root = managed_runtime_root()
            runtime_root.mkdir(parents=True, exist_ok=True)
            destination = runtime_root / archive_name.removesuffix(".zip").removesuffix(".tar.gz").removesuffix(".tar.xz")
            unpacked_root = tmp / "unpacked"
            unpacked_root.mkdir()
            extract_archive(archive, unpacked_root)
            unpacked = unpacked_root / destination.name
            if not unpacked.is_dir():
                raise InstallError(f"Node.js archive did not contain {destination.name}")
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(unpacked), destination)
        node_bin = destination if self.system == "windows" else destination / "bin"
        links = expose_node(node_bin, self.system)
        self.state.mark(
            "runtime:node",
            component="node",
            source="nodejs.org",
            details={
                "install_kind": "managed-runtime",
                "path": str(destination.resolve()),
                "bin": str(node_bin.resolve()),
                "links": [str(link.absolute()) for link in links],
                "path_entry": str(node_bin.resolve()) if self.system == "windows" else "",
            },
        )

    def _ensure_container_runtime(self) -> str:
        # Reuse any engine that is already *usable* before trying to repair a dormant
        # Podman client. This avoids enabling WSL or starting a VM when Docker/nerdctl
        # is already serving the host.
        runtime = self._select_usable_runtime()
        if (
            runtime is None
            and self.runner.which("podman")
            and self.system in {"windows", "darwin"}
        ):
            if self.system == "windows":
                self._ensure_windows_wsl()
            self._ensure_podman_machine()
            runtime = self._select_usable_runtime()
        if runtime is None:
            podman_owned = False
            if self.system == "windows":
                self._ensure_windows_wsl()
            if self.system == "darwin":
                self._install_macos_podman()
                podman_owned = True
            else:
                podman_owned = "podman" in self._install_packages(["podman"])
            if not self.options.dry_run:
                refresh_windows_path()
            if not self.options.dry_run:
                self.state.mark(
                    "host:podman",
                    component="podman" if podman_owned else None,
                    source=self.package_family,
                )
            if not self.options.dry_run and self.system in {"windows", "darwin"}:
                self._ensure_podman_machine()
            runtime = (
                "podman"
                if self.options.dry_run
                else self._select_usable_runtime()
            )
        if runtime is None:
            raise InstallError("a container runtime was installed but did not become usable")
        if not self._image_present(runtime):
            self.runner.run([runtime, "pull", DEFAULT_IMAGE], check=True, mutates=True)
        if not self.options.dry_run and not self._image_present(runtime):
            raise InstallError(f"{runtime} could not prepare sandbox image {DEFAULT_IMAGE}")
        if not self.options.dry_run:
            self.state.mark("sandbox:image", component="sandbox-image", source=runtime)
        return runtime

    def _select_usable_runtime(self) -> str | None:
        for runtime in CONTAINER_RUNTIMES:
            if self.runner.which(runtime) is None:
                continue
            if self.runner.run([runtime, "info"], capture=True).returncode == 0:
                return runtime
        return None

    def _ensure_podman_machine(self) -> None:
        if self.options.dry_run:
            print("Would initialize and start the Podman machine if needed")
            return
        listed = self.runner.run(["podman", "machine", "list", "--format", "json"], capture=True)
        machines: list[Any] = []
        if listed.returncode == 0:
            try:
                decoded = json.loads(listed.stdout or "[]")
                machines = decoded if isinstance(decoded, list) else []
            except ValueError:
                machines = []
        if not machines:
            self.runner.run(["podman", "machine", "init"], check=True, mutates=True)
        info = self.runner.run(["podman", "info"], capture=True)
        if info.returncode:
            start = self.runner.run(
                ["podman", "machine", "start"], capture=True, mutates=True
            )
            if start.returncode and "already running" not in (start.stderr or "").lower():
                raise InstallError(start.stderr.strip() or "could not start the Podman machine")

    def _ensure_windows_wsl(self) -> None:
        status = self.runner.run(["wsl", "--status"], capture=True)
        if status.returncode == 0:
            return
        if self.options.non_interactive:
            detail = (status.stderr or status.stdout or "").strip()
            message = (
                "WSL2 is unavailable; run wsl --install --no-distribution as administrator"
            )
            raise InstallError(message + (f"\n{detail}" if detail else ""))

        # ``wsl --update`` fails when WSL has not been installed yet.  Use
        # ``--version`` to distinguish a present-but-unhealthy installation from an
        # absent one, then perform exactly the applicable elevated action.
        version = self.runner.run(["wsl", "--version"], capture=True)
        if version.returncode == 0:
            arguments = ["--update"]
            result = self._run_windows_elevated("wsl.exe", arguments)
            if self.options.dry_run:
                return
            self._raise_if_uac_cancelled("wsl.exe", arguments, result)
            if self.runner.run(["wsl", "--status"], capture=True).returncode == 0:
                self.state.mark("windows:wsl-updated", component="wsl2", source="windows")
                return
            if windows_process_succeeded(result.returncode):
                self.state.mark("windows:wsl-updated", component="wsl2", source="windows")
                raise RestartRequired(
                    "WSL2 was updated but is not available yet; restart Windows and run "
                    "the same command again"
                )
            raise InstallError(
                "WSL2 is installed but its elevated update failed.\n\n"
                + format_windows_command_diagnostic("wsl.exe", arguments, result)
            )

        attempts: list[tuple[list[str], subprocess.CompletedProcess[str]]] = []
        install_commands = (
            ["--install", "--no-distribution"],
            ["--install", "--no-distribution", "--inbox"],
        )
        for arguments in install_commands:
            result = self._run_windows_elevated("wsl.exe", arguments)
            if self.options.dry_run:
                return
            self._raise_if_uac_cancelled("wsl.exe", arguments, result)
            attempts.append((arguments, result))

            # The postcondition wins over the process code. Some Windows servicing
            # paths return a failure-looking code even though WSL became available.
            if self.runner.run(["wsl", "--status"], capture=True).returncode == 0:
                source = "windows-inbox" if "--inbox" in arguments else "windows"
                self.state.mark("windows:wsl-enabled", component="wsl2", source=source)
                return
            if windows_process_succeeded(result.returncode):
                source = "windows-inbox" if "--inbox" in arguments else "windows"
                self.state.mark("windows:wsl-enabled", component="wsl2", source=source)
                raise RestartRequired(
                    "Windows accepted the WSL2 installation "
                    f"({format_windows_exit_code(result.returncode)}), but WSL is not "
                    "available yet; restart Windows and run the same command again"
                )

        diagnostics = "\n\n".join(
            format_windows_command_diagnostic("wsl.exe", arguments, result)
            for arguments, result in attempts
        )
        raise InstallError(
            "WSL2 installation failed through both the standard and Windows inbox paths.\n\n"
            + diagnostics
        )

    def _raise_if_uac_cancelled(
        self,
        executable: str,
        arguments: list[str],
        result: subprocess.CompletedProcess[str],
    ) -> None:
        if not is_windows_uac_cancelled(result.returncode):
            return
        raise InstallError(
            "Administrator approval was canceled (Windows error 1223 / 0x000004C7). "
            "No further WSL system changes were attempted. Rerun the installer and "
            "approve the UAC prompt, or use -SkipSandbox if no container sandbox is needed.\n\n"
            + format_windows_command_diagnostic(executable, arguments, result)
        )

    def _run_windows_elevated(
        self, executable: str, arguments: list[str]
    ) -> subprocess.CompletedProcess[str]:
        """Run one command through UAC while preserving its real output.

        ``Start-Process -Verb RunAs`` cannot directly redirect standard handles. Run a
        small elevated PowerShell wrapper instead; that wrapper starts the target with
        separate output files and returns its exit code to the non-elevated parent.
        """

        target = [executable, *arguments]
        with tempfile.TemporaryDirectory(prefix="polaris-elevated-") as raw_tmp:
            capture_dir = Path(raw_tmp)
            stdout_path = capture_dir / "stdout.bin"
            stderr_path = capture_dir / "stderr.bin"
            quoted_arguments = ",".join(
                powershell_string_literal(item) for item in arguments
            )
            argument_clause = (
                f" -ArgumentList @({quoted_arguments})" if quoted_arguments else ""
            )
            exception_code_script = (
                "$exception=$_.Exception;$code=1;$cursor=$exception;"
                "while ($null -ne $cursor) {"
                "if ($cursor -is [System.ComponentModel.Win32Exception] "
                "-and $cursor.NativeErrorCode -ne 0) {$code=$cursor.NativeErrorCode};"
                "if (($cursor.HResult -band 0xFFFF) -eq 1223) {$code=1223;break};"
                "$cursor=$cursor.InnerException};"
            )
            inner_script = (
                "$ErrorActionPreference='Stop';"
                "try {"
                f"$p=Start-Process -FilePath {powershell_string_literal(executable)}"
                f"{argument_clause}"
                f" -RedirectStandardOutput {powershell_string_literal(str(stdout_path))}"
                f" -RedirectStandardError {powershell_string_literal(str(stderr_path))}"
                " -WindowStyle Hidden -Wait -PassThru;exit $p.ExitCode"
                "} catch {"
                f"{exception_code_script}"
                f"[System.IO.File]::WriteAllText({powershell_string_literal(str(stderr_path))},"
                "$exception.Message + [Environment]::NewLine,[System.Text.Encoding]::Unicode);"
                "exit $code}"
            )
            encoded_script = base64.b64encode(inner_script.encode("utf-16le")).decode("ascii")
            target_description = subprocess.list2cmdline(target)
            outer_script = (
                "$ErrorActionPreference='Stop';"
                f"$targetDescription={powershell_string_literal(target_description)};"
                "try {"
                "$p=Start-Process -FilePath 'powershell.exe' "
                f"-ArgumentList @('-NoProfile','-NonInteractive','-EncodedCommand','{encoded_script}') "
                "-Verb RunAs -WindowStyle Hidden -Wait -PassThru;exit $p.ExitCode"
                "} catch {"
                f"{exception_code_script}"
                "[Console]::Error.WriteLine($exception.Message);exit $code}"
            )
            launcher = self.runner.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    outer_script,
                ],
                capture=True,
                mutates=True,
                display_argv=["[administrator]", *target],
            )

            capture_errors: list[str] = []
            stdout = read_captured_process_output(stdout_path, "stdout", capture_errors)
            stderr = read_captured_process_output(stderr_path, "stderr", capture_errors)
            stdout = combine_process_output(stdout, decode_process_output(launcher.stdout))
            stderr = combine_process_output(stderr, decode_process_output(launcher.stderr))
            if capture_errors:
                stderr = combine_process_output(stderr, "\n".join(capture_errors))
            return subprocess.CompletedProcess(
                target,
                normalize_uac_cancelled_code(launcher.returncode),
                stdout,
                stderr,
            )

    def _install_macos_podman(self) -> None:
        if self.options.dry_run:
            print("Would download and verify the official signed Podman macOS installer")
            return
        release = download_json(f"{GITHUB_API}/repos/containers/podman/releases/latest")
        architecture = "arm64" if self.machine == "arm64" else "amd64"
        assets = release.get("assets", []) if isinstance(release, dict) else []
        asset = next(
            (
                item
                for item in assets
                if str(item.get("name", "")).endswith(".pkg")
                and architecture in str(item.get("name", "")).lower()
            ),
            None,
        )
        if not asset:
            raise InstallError(f"no official Podman macOS {architecture} package was found")
        with tempfile.TemporaryDirectory(prefix="polaris-podman-") as raw_tmp:
            package = Path(raw_tmp) / str(asset["name"])
            download_file(str(asset["browser_download_url"]), package)
            self.runner.run(["pkgutil", "--check-signature", package], check=True)
            self.runner.run(
                [*self._sudo_prefix(), "installer", "-pkg", package, "-target", "/"],
                check=True,
                mutates=True,
            )

    def _image_present(self, runtime: str) -> bool:
        return self.runner.run([runtime, "image", "inspect", DEFAULT_IMAGE], capture=True).returncode == 0

    def _install_project(self) -> None:
        source = self.options.source.resolve()
        if not (source / "pyproject.toml").is_file():
            raise InstallError(f"project source is missing pyproject.toml: {source}")
        spec = f"{source}[all,dev]" if self.options.dev else f"{source}[all]"
        uv = self.runner.which("uv") or "uv"
        if self.options.dev:
            venv = source / ".venv"
            self.runner.run(
                [uv, "venv", "--python", PYTHON_SERIES, venv],
                check=True,
                mutates=True,
            )
            python = venv_python(venv, self.system)
            self.runner.run(
                [uv, "pip", "install", "--python", python, "-e", spec],
                check=True,
                mutates=True,
            )
            if not self.options.dry_run:
                executable = python.parent / (
                    "polaris.exe" if self.system == "windows" else "polaris"
                )
                if not venv.is_dir() or not executable.exists():
                    raise InstallError(
                        "uv completed the development install but its Polaris command is missing"
                    )
                marker = {
                    "schema": 1,
                    "package": PACKAGE_NAME,
                    "source": str(source),
                }
                (venv / ".polaris-install.json").write_text(
                    json.dumps(marker, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                self.state.mark(
                    "project:dev",
                    component="polaris",
                    source=str(source),
                    details={
                        "install_kind": "dev-venv",
                        "package": PACKAGE_NAME,
                        "executable": str(executable.resolve()),
                        "environment": str(venv.resolve()),
                        "uv": str(Path(uv).resolve()),
                        "bootstrap_python": str(Path(sys.executable).resolve()),
                    },
                )
            return
        existing = self._project_command()
        owned = self.state.component_owned("polaris")
        if existing and (not self.options.upgrade or not owned):
            suffix = " (not installer-owned; left unchanged)" if self.options.upgrade else ""
            print(f"Reusing existing Polaris command: {existing}{suffix}")
            if not self.options.dry_run:
                self.state.mark(
                    "project:external",
                    component="polaris",
                    source=str(source),
                    installed_by_polaris=False,
                    details={
                        "install_kind": "external",
                        "package": PACKAGE_NAME,
                        "executable": str(Path(existing).resolve()),
                        "environment": str(Path(sys.prefix).resolve()),
                    },
                )
            return
        args = [uv, "tool", "install", "--python", PYTHON_SERIES]
        if self.options.upgrade and owned:
            args.append("--force")
        args.append(spec)
        self.runner.run(args, check=True, mutates=True)
        if not self.options.dry_run:
            tool_root = self._uv_tool_directory(uv, ["tool", "dir"])
            bin_dir = self._uv_tool_directory(uv, ["tool", "dir", "--bin"])
            executable = bin_dir / (
                "polaris.exe" if self.system == "windows" else "polaris"
            )
            environment = tool_root / PACKAGE_NAME
            if not executable.exists() or not environment.is_dir():
                raise InstallError(
                    "uv installed Polaris but its isolated environment or command is missing"
                )
            self.state.mark(
                "project:tool",
                component="polaris",
                source=str(source),
                details={
                    "install_kind": "uv-tool",
                    "package": PACKAGE_NAME,
                    "executable": str(executable.resolve()),
                    "environment": str(environment.resolve()),
                    "uv": str(Path(uv).resolve()),
                    "tool_root": str(tool_root),
                    "bin_dir": str(bin_dir),
                    "bootstrap_python": str(Path(sys.executable).resolve()),
                },
            )

    def _uv_tool_directory(self, uv: str, arguments: list[str]) -> Path:
        result = self.runner.run([uv, *arguments], capture=True)
        lines = (result.stdout or "").strip().splitlines()
        if result.returncode or not lines:
            detail = (result.stderr or "").strip()
            raise InstallError(
                f"could not locate uv tool directories with "
                f"{subprocess.list2cmdline([uv, *arguments])}"
                + (f": {detail}" if detail else "")
            )
        return Path(lines[-1]).expanduser().resolve()

    def _project_command(self) -> str | None:
        if self.options.dev:
            candidate = venv_python(self.options.source / ".venv", self.system).parent / (
                "polaris.exe" if self.system == "windows" else "polaris"
            )
            return str(candidate) if candidate.exists() else None
        receipt = self.state.component("polaris")
        recorded = Path(str(receipt.get("executable", "")))
        if self.state.component_owned("polaris") and recorded.is_file():
            return str(recorded.resolve())
        direct = self.runner.which("polaris")
        if direct:
            return direct
        uv = self.runner.which("uv")
        if not uv:
            return None
        located = self.runner.run([uv, "tool", "dir", "--bin"], capture=True)
        if located.returncode:
            return None
        name = "polaris.exe" if self.system == "windows" else "polaris"
        candidate = Path((located.stdout or "").strip()) / name
        return str(candidate) if candidate.exists() else None

    def _verify_installed_health(self) -> Check:
        command = self._project_command()
        if command is None:
            return Check("polaris-health", False, "Polaris command is unavailable")
        profile = "dev" if self.options.dev else "runtime"
        proc = self.runner.run(
            [command, "health", "--provider", "fake", "--profile", profile, "--json"],
            capture=True,
            cwd=Path.home(),
        )
        try:
            payload = json.loads(proc.stdout or "{}")
        except ValueError:
            payload = {}
        status = str(payload.get("status", "error"))
        # ``--skip-sandbox`` is an explicit degraded installation. The command still
        # runs, but its expected container failures do not turn the opted-out install
        # itself into a failure.
        if self.options.skip_sandbox and status == "error":
            checks = payload.get("checks", [])
            failures = {
                str(item.get("name"))
                for item in checks
                if isinstance(item, dict) and item.get("status") != "ok"
            }
            if failures and failures <= {"container-runtime", "sandbox-image"}:
                return Check("polaris-health", True, "degraded: sandbox explicitly skipped")
        ok = proc.returncode == 0 and status in {"ok", "degraded"}
        return Check("polaris-health", ok, status if payload else "invalid health JSON")

    @staticmethod
    def _print_checks(title: str, checks: Iterable[Check]) -> None:
        print(f"\n{title}")
        print("-" * len(title))
        for item in checks:
            marker = "OK" if item.ok else "MISSING"
            print(f"[{marker:7}] {item.name}: {item.detail}")


def default_state_path() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "Polaris" / "install-state.json"
    root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return root / "polaris" / "install-state.json"


def managed_runtime_root() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "Polaris" / "runtime"
    root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return root / "polaris" / "runtime"


def normalize_machine(value: str) -> str:
    lowered = value.lower()
    if lowered in {"amd64", "x86_64", "x64"}:
        return "x86_64"
    if lowered in {"arm64", "aarch64"}:
        return "arm64"
    return lowered


def decode_process_output(data: bytes | str | None) -> str:
    """Decode captured command output without trusting one Windows code page.

    Most tools emit UTF-8 or the active ANSI/OEM code page, while ``wsl.exe`` emits
    UTF-16LE even when stdout/stderr are pipes and may omit a byte-order mark.  Python's
    default ``text=True`` decoding therefore crashes on localized WSL output.  Detect
    UTF-16 by its NUL-byte layout, then fall back through common text encodings with a
    replacement-only final path so diagnostics can never kill the installer.
    """

    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if not data:
        return ""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")

    even_nuls = data[0::2].count(0)
    odd_nuls = data[1::2].count(0)
    pairs = max(1, len(data) // 2)
    if max(even_nuls, odd_nuls) >= max(2, pairs // 5):
        encoding = "utf-16le" if odd_nuls >= even_nuls else "utf-16be"
        return data.decode(encoding, errors="replace")

    encodings = ["utf-8", locale.getpreferredencoding(False)]
    if os.name == "nt":
        try:
            encodings.append(f"cp{ctypes.windll.kernel32.GetOEMCP()}")
        except (AttributeError, OSError):
            pass
    tried: set[str] = set()
    for encoding in encodings:
        normalized = encoding.lower()
        if normalized in tried:
            continue
        tried.add(normalized)
        try:
            return data.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode(locale.getpreferredencoding(False), errors="replace")


def read_captured_process_output(
    path: Path, stream_name: str, errors: list[str]
) -> str:
    """Read an elevated stream without allowing diagnostics to mask the result."""

    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return ""
    except OSError as exc:
        errors.append(f"could not read elevated {stream_name} capture: {exc}")
        return ""
    return decode_process_output(data)


def combine_process_output(primary: str, secondary: str) -> str:
    if not primary:
        return secondary
    if not secondary:
        return primary
    separator = "" if primary.endswith(("\r", "\n")) else "\n"
    return primary + separator + secondary


def powershell_string_literal(value: str) -> str:
    """Quote an arbitrary value as one PowerShell single-quoted string."""

    return "'" + value.replace("'", "''") + "'"


def command_version(runner: Runner, command: str) -> str | None:
    executable = runner.which(command)
    if executable is None:
        return None
    proc = runner.run([executable, "--version"], capture=True)
    if proc.returncode:
        return None
    return ((proc.stdout or proc.stderr or "").strip().splitlines() or [command])[0]


def parse_node_major(version: str) -> int:
    for token in version.replace("v", " ").split():
        head = token.split(".", 1)[0]
        if head.isdigit():
            return int(head)
    return 0


def format_windows_exit_code(returncode: int) -> str:
    """Render signed or unsigned Win32/HRESULT-style process codes consistently."""

    return f"{returncode} (0x{returncode & 0xFFFFFFFF:08X})"


def normalize_uac_cancelled_code(returncode: int) -> int:
    unsigned = returncode & 0xFFFFFFFF
    if unsigned in {WINDOWS_UAC_CANCELLED, WINDOWS_UAC_CANCELLED_HRESULT}:
        return WINDOWS_UAC_CANCELLED
    return returncode


def is_windows_uac_cancelled(returncode: int) -> bool:
    return normalize_uac_cancelled_code(returncode) == WINDOWS_UAC_CANCELLED


def windows_process_succeeded(returncode: int) -> bool:
    return (returncode & 0xFFFFFFFF) in WINDOWS_SUCCESS_OR_REBOOT_CODES


def format_windows_command_diagnostic(
    executable: str,
    arguments: Sequence[str],
    result: subprocess.CompletedProcess[str],
) -> str:
    """Format one Windows command with both decoded streams for actionable errors."""

    command = subprocess.list2cmdline([executable, *arguments])
    lines = [
        f"Command: {command}",
        f"Exit code: {format_windows_exit_code(result.returncode)}",
    ]
    stdout = decode_process_output(result.stdout).strip()
    stderr = decode_process_output(result.stderr).strip()
    if stdout:
        lines.extend(("stdout:", stdout))
    if stderr:
        lines.extend(("stderr:", stderr))
    if not stdout and not stderr:
        lines.append("Output: (no output captured)")
    return "\n".join(lines)


def node_archive_name(version: str, system: str, machine: str) -> str:
    arch = "x64" if machine == "x86_64" else "arm64"
    if system == "windows":
        return f"node-{version}-win-{arch}.zip"
    if system == "darwin":
        return f"node-{version}-darwin-{arch}.tar.gz"
    if system == "linux":
        return f"node-{version}-linux-{arch}.tar.xz"
    raise UnsupportedHost(f"Node.js binaries are unsupported on {system}/{machine}")


def download_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "polaris-installer"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except (OSError, ValueError) as exc:
        raise InstallError(f"could not read {url}: {exc}") from exc


def download_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "polaris-installer"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            destination.write_bytes(response.read())
    except OSError as exc:
        raise InstallError(f"could not download {url}: {exc}") from exc


def checksum_for(filename: str, content: str) -> str:
    for line in content.splitlines():
        fields = line.strip().split()
        if len(fields) >= 2 and fields[-1].lstrip("*") == filename:
            return fields[0]
    raise InstallError(f"official checksum is missing for {filename}")


def verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest.lower() != expected.lower():
        raise InstallError(f"SHA-256 mismatch for {path.name}")


def extract_archive(archive: Path, destination: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(destination)
        return
    with tarfile.open(archive, "r:*") as bundle:
        bundle.extractall(destination, filter="data")


def expose_node(node_bin: Path, system: str) -> list[Path]:
    if system == "windows":
        prepend_windows_user_path(node_bin)
        return []
    bin_dir = Path.home() / ".local" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    links: list[Path] = []
    for name in ("node", "npm", "npx"):
        source = node_bin / name
        target = bin_dir / name
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(source)
        links.append(target)
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    return links


def prepend_windows_user_path(directory: Path) -> None:
    import winreg

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
        try:
            current, _ = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            current = ""
        entries = [item for item in str(current).split(";") if item]
        if str(directory).lower() not in {entry.lower() for entry in entries}:
            value = ";".join([str(directory), *entries])
            winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, value)
    os.environ["PATH"] = f"{directory};{os.environ.get('PATH', '')}"
    try:
        ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, "Environment", 0, 5000, None)
    except (AttributeError, OSError):
        pass


def refresh_windows_path() -> None:
    if os.name != "nt":
        return
    import winreg

    registry_values: list[str] = []
    locations = (
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
        (winreg.HKEY_CURRENT_USER, "Environment"),
    )
    for hive, path in locations:
        try:
            with winreg.OpenKey(hive, path) as key:
                value, _ = winreg.QueryValueEx(key, "Path")
                registry_values.append(str(value))
        except OSError:
            continue
    # Keep process-only entries first.  Activated Conda environments, uv bootstrap
    # paths, and terminal-specific tools do not necessarily exist in the registry;
    # replacing PATH with only machine/user values can break the installer halfway.
    values = [os.environ.get("PATH", ""), *registry_values]
    merged = merge_path_values(*values)
    if merged:
        os.environ["PATH"] = merged


def merge_path_values(*values: str) -> str:
    """Merge PATH strings without dropping session entries or adding duplicates."""

    entries: list[str] = []
    seen: set[str] = set()
    for value in values:
        for raw_entry in value.split(os.pathsep):
            entry = os.path.expandvars(raw_entry.strip().strip('"'))
            if not entry:
                continue
            key = os.path.normcase(os.path.normpath(entry))
            if key in seen:
                continue
            seen.add(key)
            entries.append(entry)
    return os.pathsep.join(entries)


def venv_python(venv: Path, system: str) -> Path:
    return venv / ("Scripts/python.exe" if system == "windows" else "bin/python")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--dev", action="store_true", help="install editable source + dev tools")
    parser.add_argument("--upgrade", action="store_true", help="upgrade installer-owned components")
    parser.add_argument("--check", action="store_true", help="detect dependencies without installing")
    parser.add_argument("--dry-run", action="store_true", help="print changes without executing them")
    parser.add_argument("--skip-sandbox", action="store_true", help="explicitly omit a container runtime")
    parser.add_argument("--non-interactive", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = Options(
        source=args.source,
        dev=args.dev,
        upgrade=args.upgrade,
        check=args.check,
        dry_run=args.dry_run,
        skip_sandbox=args.skip_sandbox,
        non_interactive=args.non_interactive,
    )
    try:
        checks = Installer(options).install()
    except RestartRequired as exc:
        print(f"[restart required] {exc}", file=sys.stderr)
        return EXIT_RESTART_REQUIRED
    except UnsupportedHost as exc:
        print(f"[unsupported] {exc}", file=sys.stderr)
        return EXIT_UNSUPPORTED
    except InstallError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return EXIT_FAILED
    if options.dry_run:
        return EXIT_OK
    required_failures = [item for item in checks if item.required and not item.ok]
    return EXIT_FAILED if required_failures else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
