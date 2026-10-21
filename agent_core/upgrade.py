"""In-CLI upgrade and rollback for installer-owned Polaris releases.

``polaris upgrade`` proves installer ownership from the install-state receipts,
downloads the target GitHub release bundle with SHA-256 verification, and stages
a detached, stdlib-only worker that performs the actual ``uv tool install`` only
after this process exits. On Windows the running CLI locks its uv tool
environment, so no upgrade mutation may happen in this process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

import httpx

from agent_core.semver import Version

PACKAGE_NAME = "agent-with-llm"
COMMAND_NAME = "polaris"
RELEASES_URL = "https://github.com/jony-del/AgentwithLLM/releases"
PYTHON_SERIES = "3.12"

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_FAILED = 10

_TAG_PATTERN = re.compile(r"v\d+\.\d+\.\d+([-.][0-9A-Za-z.-]+)?")

Fetcher = Callable[[str], bytes]
Runner = Callable[..., subprocess.CompletedProcess[Any]]
Spawner = Callable[..., Any]

_OFFLINE_GUIDANCE = (
    "Check the network connection, or upgrade offline instead: download "
    "polaris-installer.zip (Windows) or polaris-installer.tar.gz (macOS/Linux) from "
    f"{RELEASES_URL}, extract it, and run its install script with the upgrade flag "
    "(install.ps1 -Upgrade / install.sh --upgrade)."
)

_EXTERNAL_GUIDANCE = (
    "Upgrade it with the tool that installed it:\n"
    "  - source checkout: git pull (rerun the dev installer if dependencies changed)\n"
    "  - pip/Conda install: use that environment's package manager\n"
    "  - release install: rerun install.ps1 -Upgrade / install.sh --upgrade"
)


class UpgradeError(RuntimeError):
    """A user-actionable upgrade failure."""


class OwnershipError(UpgradeError):
    """The active Polaris install is not proven to be an installer-owned uv tool."""


class UsageError(UpgradeError):
    """Command-line input is invalid."""


@dataclass(slots=True)
class StagedRelease:
    """A downloaded, verified and extracted release bundle kept for the worker."""

    tag: str
    version: str
    temp_root: Path
    source: Path
    archive: str


def default_state_path() -> Path:
    # Mirrors installer/install.py; the installer package is not shipped in the wheel.
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "Polaris" / "install-state.json"
    root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return root / "polaris" / "install-state.json"


def polaris_home() -> Path:
    configured = os.environ.get("POLARIS_HOME")
    return Path(configured).expanduser() if configured else Path.home() / ".polaris"


def current_version() -> str | None:
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return None


def normalize_tag(raw: str) -> str:
    value = raw.strip()
    if value and not value.startswith(("v", "V")):
        value = "v" + value
    value = "v" + value[1:]
    if not _TAG_PATTERN.fullmatch(value):
        raise UsageError(f"invalid release tag {raw!r}; use vX.Y.Z (for example v0.1.1)")
    return value


def compare_versions(left: str, right: str) -> int:
    """Semver compare: -1/0/1, with prereleases ordering before their release."""

    parsed_left, parsed_right = Version.parse(left), Version.parse(right)
    if parsed_left == parsed_right:
        return 0
    return -1 if parsed_left < parsed_right else 1


def release_asset_url(tag: str | None, asset: str) -> str:
    if tag is None:
        return f"{RELEASES_URL}/latest/download/{asset}"
    return f"{RELEASES_URL}/download/{tag}/{asset}"


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return value.decode("utf-8", errors="replace")


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.expanduser().resolve(strict=False).relative_to(
            root.expanduser().resolve(strict=False)
        )
    except ValueError:
        return False
    return True


def _fetch_bytes(url: str) -> bytes:
    try:
        with httpx.Client(
            timeout=httpx.Timeout(120.0, connect=15.0),
            follow_redirects=True,
            headers={"User-Agent": "polaris-upgrade"},
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.content
    except httpx.HTTPStatusError as exc:
        raise UpgradeError(
            f"download of {url} failed with HTTP {exc.response.status_code}\n"
            + _OFFLINE_GUIDANCE
        ) from exc
    except httpx.HTTPError as exc:
        raise UpgradeError(f"could not download {url}: {exc}\n" + _OFFLINE_GUIDANCE) from exc


def _run_process(argv: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(argv, **kwargs)


def _spawn_process(argv: Any, **kwargs: Any) -> Any:
    return subprocess.Popen(argv, **kwargs)


def read_owned_install(state_path: Path) -> dict[str, Any]:
    """Return the polaris component receipt, or refuse with upgrade guidance."""

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise OwnershipError(
            "no Polaris installer state was found; this command was not installed by the "
            "Polaris installer, so 'polaris upgrade' cannot manage it.\n" + _EXTERNAL_GUIDANCE
        ) from None
    except (OSError, ValueError) as exc:
        raise UpgradeError(f"could not read the install state {state_path}: {exc}") from exc
    if not isinstance(state, dict) or state.get("schema") not in {1, 2}:
        raise UpgradeError(f"unsupported or damaged install state: {state_path}")
    if state.get("schema") == 1:
        raise OwnershipError(
            "the install receipts predate verified ownership records, so 'polaris upgrade' "
            "cannot prove this installation is installer-owned.\n" + _EXTERNAL_GUIDANCE
        )
    components = state.get("components", {})
    receipt = components.get("polaris") if isinstance(components, dict) else None
    if not isinstance(receipt, dict) or not receipt.get("installed_by_polaris"):
        raise OwnershipError(
            "this Polaris installation is not proven to be owned by the Polaris installer; "
            "it was left unchanged.\n" + _EXTERNAL_GUIDANCE
        )
    kind = str(receipt.get("install_kind", ""))
    if kind != "uv-tool":
        raise OwnershipError(
            f"this Polaris installation has install kind {kind or 'unknown'!r}; "
            "'polaris upgrade' only manages installer-owned uv-tool installs.\n"
            + _EXTERNAL_GUIDANCE
        )
    return receipt


def _validate_metadata(data: Any, requested_tag: str | None, url: str) -> dict[str, Any]:
    invalid = UpgradeError(f"invalid release metadata at {url}\n" + _OFFLINE_GUIDANCE)
    if not isinstance(data, dict):
        raise invalid
    if data.get("schema") != 1 or data.get("package") != PACKAGE_NAME:
        raise invalid
    release_version, tag = data.get("version"), data.get("tag")
    if (
        not isinstance(release_version, str)
        or not isinstance(tag, str)
        or tag != f"v{release_version}"
        or not _TAG_PATTERN.fullmatch(tag)
    ):
        raise invalid
    if requested_tag is not None and tag != requested_tag:
        raise UpgradeError(
            f"the release feed returned {tag}, not the requested {requested_tag}\n"
            + _OFFLINE_GUIDANCE
        )
    return data


def fetch_release_metadata(fetch: Fetcher, tag: str | None) -> dict[str, Any]:
    url = release_asset_url(tag, "release.json")
    raw = fetch(url)
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise UpgradeError(
            f"release metadata at {url} is not valid JSON\n" + _OFFLINE_GUIDANCE
        ) from exc
    return _validate_metadata(data, tag, url)


def _archive_name(os_name: str) -> str:
    return "polaris-installer.zip" if os_name == "nt" else "polaris-installer.tar.gz"


def _checksum_for(filename: str, content: str) -> str:
    matches = []
    for line in content.splitlines():
        fields = line.strip().split()
        if (
            len(fields) >= 2
            and re.fullmatch(r"[0-9a-fA-F]{64}", fields[0])
            and fields[-1].lstrip("*") == filename
        ):
            matches.append(fields[0].lower())
    if len(matches) != 1:
        raise UpgradeError(f"SHA256SUMS must contain exactly one {filename} entry")
    return matches[0]


def _verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest().lower() != expected.lower():
        raise UpgradeError(f"SHA-256 mismatch for {path.name}; the download was discarded")


def _extract_archive(archive: Path, destination: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(destination)
        return
    with tarfile.open(archive, "r:*") as bundle:
        bundle.extractall(destination, filter="data")


def stage_release(fetch: Fetcher, metadata: dict[str, Any], *, os_name: str) -> StagedRelease:
    """Download, checksum-verify and extract the release bundle for the worker."""

    tag = str(metadata["tag"])
    release_version = str(metadata["version"])
    archive_name = _archive_name(os_name)
    temp_root = Path(tempfile.mkdtemp(prefix="polaris-upgrade-"))
    try:
        base = release_asset_url(tag, "")
        archive_path = temp_root / archive_name
        archive_path.write_bytes(fetch(f"{base}{archive_name}"))
        sums = _decode(fetch(f"{base}SHA256SUMS"))
        _verify_sha256(archive_path, _checksum_for(archive_name, sums))
        source = temp_root / "source"
        source.mkdir()
        _extract_archive(archive_path, source)
        for required in (
            source / "installer" / "install.py",
            source / "installer" / "manifest.json",
        ):
            if not required.is_file():
                raise UpgradeError(
                    f"release bundle is missing {required.relative_to(source).as_posix()}"
                )
        try:
            bundled = json.loads((source / "release.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise UpgradeError(f"release bundle has no valid release.json: {exc}") from exc
        if not isinstance(bundled, dict) or bundled.get("tag") != tag:
            raise UpgradeError(
                f"release bundle tag does not match the requested {tag}; discarded"
            )
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    return StagedRelease(tag, release_version, temp_root, source, archive_name)


def find_worker_python(receipt: dict[str, Any], *, runner: Runner) -> str:
    """Locate a Python 3.12 interpreter outside the uv tool environment."""

    raw_environment = str(receipt.get("environment", ""))
    environment = Path(raw_environment).expanduser() if raw_environment else None

    def usable(candidate: Path) -> bool:
        return candidate.is_file() and (
            environment is None or not _path_within(candidate, environment)
        )

    recorded = str(receipt.get("bootstrap_python", ""))
    if recorded:
        candidate = Path(recorded).expanduser()
        if usable(candidate):
            return str(candidate)
    uv = str(receipt.get("uv", ""))
    uv_command = uv if uv and Path(uv).is_file() else shutil.which("uv")
    if not uv_command:
        raise UpgradeError(
            "uv is required to find a worker Python outside the Polaris tool environment, "
            "but uv was not found. Restore uv, or upgrade with the release install script "
            "(install.ps1 -Upgrade / install.sh --upgrade)."
        )
    env = dict(os.environ)
    env["UV_PYTHON_DOWNLOADS"] = "never"
    result = runner([uv_command, "python", "find", "--no-project", PYTHON_SERIES], env=env)
    lines = _decode(getattr(result, "stdout", "")).strip().splitlines()
    if getattr(result, "returncode", 1) == 0 and lines:
        candidate = Path(lines[-1]).expanduser()
        if usable(candidate):
            return str(candidate)
    raise UpgradeError(
        "uv could not locate a Python 3.12 interpreter outside the Polaris tool "
        "environment. Run 'uv python install 3.12', or upgrade with the release install "
        "script (install.ps1 -Upgrade / install.sh --upgrade)."
    )


def capture_health_snapshot(
    receipt: dict[str, Any], home: Path, *, runner: Runner
) -> tuple[Path | None, str | None]:
    """Best-effort pre-upgrade health record; never blocks the upgrade."""

    executable = str(receipt.get("executable", ""))
    if not executable or not Path(executable).is_file():
        executable = shutil.which(COMMAND_NAME) or ""
    if not executable:
        return None, "the Polaris command was not found; skipped the pre-upgrade health snapshot"
    command = [
        executable, "health", "--provider", "fake", "--profile", "runtime", "--json",
    ]
    try:
        proc = runner(command, capture_output=True, timeout=120)
    except Exception as exc:
        return None, f"the pre-upgrade health check could not run ({exc}); continuing"
    stdout = _decode(getattr(proc, "stdout", ""))
    try:
        report: Any = json.loads(stdout)
    except ValueError:
        report = None
    record: dict[str, Any] = {
        "schema": 1,
        "recorded_at": int(time.time()),
        "executable": executable,
        "exit_code": getattr(proc, "returncode", None),
        "report": report if isinstance(report, dict) else None,
    }
    if not isinstance(report, dict):
        record["stdout"] = stdout
    path = home / "upgrade-last-health.json"
    try:
        home.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    except OSError as exc:
        return None, f"could not store the pre-upgrade health snapshot ({exc}); continuing"
    return path, None


_WORKER_SCRIPT = '''\
"""Detached Polaris upgrade worker written by 'polaris upgrade' (stdlib-only)."""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

STILL_ACTIVE = 259
WAIT_TIMEOUT_SECONDS = 600.0


def parent_running(pid):
    # On Windows os.kill(pid, 0) would terminate the process, so probe the
    # exit code through a limited-information handle instead.
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--temp-root", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("ab") as log:
            def note(message):
                log.write((message + "\\n").encode("utf-8", "replace"))
                log.flush()

            note(f"[worker] waiting for polaris process {args.parent_pid} to exit")
            deadline = time.monotonic() + WAIT_TIMEOUT_SECONDS
            while parent_running(args.parent_pid):
                if time.monotonic() >= deadline:
                    note("[error] timed out waiting for the polaris process to exit")
                    return 10
                time.sleep(0.2)
            source = Path(args.source)
            command = [
                sys.executable,
                str(source / "installer" / "install.py"),
                "--source",
                str(source),
                "--upgrade",
                "--non-interactive",
                "--only-project",
            ]
            note("+ " + subprocess.list2cmdline(command))
            try:
                proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            except OSError as exc:
                note(f"[error] could not run the release installer: {exc}")
                return 10
            if proc.returncode:
                note(f"[error] the release installer exited with {proc.returncode}")
                note("retry with install.ps1 -Upgrade / install.sh --upgrade")
                return int(proc.returncode)
            note("[upgraded] done; open a new terminal and run 'polaris --version'")
            return 0
    finally:
        shutil.rmtree(args.temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
'''


def stage_worker(
    staged: StagedRelease,
    worker_python: str,
    log_path: Path,
    *,
    os_name: str,
    spawner: Spawner,
) -> None:
    """Write the detached worker and start it without inheriting this console."""

    worker = staged.temp_root / "upgrade_worker.py"
    worker.write_text(_WORKER_SCRIPT, encoding="utf-8")
    if os_name != "nt":
        staged.temp_root.chmod(0o700)
        worker.chmod(0o600)
    command = [
        worker_python,
        str(worker),
        "--parent-pid",
        str(os.getpid()),
        "--source",
        str(staged.source),
        "--temp-root",
        str(staged.temp_root),
        "--log",
        str(log_path),
    ]
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os_name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "DETACHED_PROCESS", 0
        )
    else:
        kwargs["start_new_session"] = True
    spawner(command, **kwargs)


def run_upgrade(
    *,
    to: str | None = None,
    check: bool = False,
    force: bool = False,
    state_path: Path | None = None,
    home: Path | None = None,
    fetch: Fetcher | None = None,
    runner: Runner | None = None,
    spawner: Spawner | None = None,
    version_probe: Callable[[], str | None] | None = None,
    os_name: str | None = None,
) -> int:
    if check and to:
        raise UsageError("--check cannot be combined with --to")
    if check and force:
        raise UsageError("--check cannot be combined with --force")
    os_name = os_name or os.name
    fetch = fetch or _fetch_bytes
    runner = runner or _run_process
    spawner = spawner or _spawn_process
    home = home or polaris_home()

    receipt = read_owned_install(state_path or default_state_path())
    current = (version_probe or current_version)()
    if current is None:
        raise UpgradeError(
            "the installed Polaris version could not be determined; this looks like a "
            "source checkout.\n" + _EXTERNAL_GUIDANCE
        )
    requested_tag = normalize_tag(to) if to else None
    metadata = fetch_release_metadata(fetch, requested_tag)
    target_tag = str(metadata["tag"])
    target_version = str(metadata["version"])

    if check:
        print(f"Installed version: {current}")
        print(f"Latest release:    {target_version} ({target_tag})")
        comparison = compare_versions(current, target_version)
        if comparison == 0:
            print("Polaris is up to date.")
        elif comparison > 0:
            print("The installed version is newer than the latest published release.")
        else:
            print(f"A newer release is available; run 'polaris upgrade' to install {target_tag}.")
        return EXIT_OK

    comparison = compare_versions(current, target_version)
    if comparison == 0 and not force:
        print(f"polaris {current} is already up to date (use --force to reinstall).")
        return EXIT_OK

    worker_python = find_worker_python(receipt, runner=runner)
    staged = stage_release(fetch, metadata, os_name=os_name)
    try:
        snapshot, warning = capture_health_snapshot(receipt, home, runner=runner)
        if warning:
            print(f"[warning] {warning}", file=sys.stderr)
        log_path = home / "upgrade-last.log"
        try:
            home.mkdir(parents=True, exist_ok=True)
            stage_worker(staged, worker_python, log_path, os_name=os_name, spawner=spawner)
        except OSError as exc:
            raise UpgradeError(f"could not start the detached upgrade worker: {exc}") from exc
    except BaseException:
        shutil.rmtree(staged.temp_root, ignore_errors=True)
        raise

    relation = "reinstall" if comparison == 0 else "rollback" if comparison > 0 else "upgrade"
    print(f"[scheduled] Polaris {relation} to {target_tag} runs when this process exits.")
    print(f"Target release: {target_tag} (installed: {current})")
    if snapshot is not None:
        print(f"Pre-upgrade health snapshot: {snapshot}")
    print(f"Completion log: {log_path}")
    print("Open a new terminal after it finishes and run 'polaris --version' to confirm.")
    return EXIT_OK


def upgrade_from_cli(args: argparse.Namespace, **overrides: Any) -> int:
    try:
        return run_upgrade(
            to=getattr(args, "to", None),
            check=bool(getattr(args, "check", False)),
            force=bool(getattr(args, "force", False)),
            **overrides,
        )
    except UsageError as exc:
        print(f"[usage] {exc}", file=sys.stderr)
        return EXIT_USAGE
    except OwnershipError as exc:
        print(f"[not installer-owned] {exc}", file=sys.stderr)
        return EXIT_FAILED
    except UpgradeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return EXIT_FAILED
