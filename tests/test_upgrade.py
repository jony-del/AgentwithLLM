"""Hermetic tests for 'polaris upgrade' and the installer's --only-project mode.

No real network, no real uv invocations and no detached processes: the release
feed is served from in-memory bundles, process execution goes through fake
runners/spawners, and all state lives under tmp_path.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

import agent_core.upgrade as upgrade
import installer.install as install_module
from installer.install import Installer, Options, StateStore


def _release_json(version: str, *, tag: str | None = None) -> dict:
    return {
        "schema": 1,
        "package": "agent-with-llm",
        "version": version,
        "tag": tag or f"v{version}",
        "wheel": {
            "filename": f"agent_with_llm-{version}-py3-none-any.whl",
            "sha256": "0" * 64,
        },
        "constraints": {"filename": "polaris-constraints.txt", "sha256": "0" * 64},
    }


def _bundle_bytes(version: str, *, windows: bool, tag: str | None = None) -> bytes:
    members = {
        "installer/install.py": "# release installer\n",
        "installer/manifest.json": "{}\n",
        "agent_core/uninstall.py": "# uninstall helper\n",
        "release.json": json.dumps(_release_json(version, tag=tag)),
    }
    buffer = io.BytesIO()
    if windows:
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, payload in members.items():
                archive.writestr(name, payload)
    else:
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for name, payload in members.items():
                data = payload.encode()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class _FakeFeed:
    """Serve release.json, the installer archive and SHA256SUMS from memory."""

    def __init__(
        self,
        versions: list[str],
        *,
        latest: str | None = None,
        windows: bool = False,
        bad_sums: bool = False,
        bundle_tag: str | None = None,
    ) -> None:
        self.tags = {f"v{item}": item for item in versions}
        self.latest = f"v{latest or versions[-1]}"
        self.windows = windows
        self.bad_sums = bad_sums
        self.bundle_tag = bundle_tag
        self.urls: list[str] = []
        self._archives: dict[str, bytes] = {}

    def _archive(self, tag: str) -> bytes:
        if tag not in self._archives:
            self._archives[tag] = _bundle_bytes(
                self.tags[tag], windows=self.windows, tag=self.bundle_tag
            )
        return self._archives[tag]

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        tag = self.latest if "/latest/download/" in url else url.split("/download/")[1].split("/")[0]
        if tag not in self.tags:
            raise upgrade.UpgradeError(f"download of {url} failed with HTTP 404")
        if url.endswith("/release.json"):
            return json.dumps(_release_json(self.tags[tag])).encode()
        name = "polaris-installer.zip" if self.windows else "polaris-installer.tar.gz"
        if url.endswith("/SHA256SUMS"):
            digest = "0" * 64 if self.bad_sums else hashlib.sha256(self._archive(tag)).hexdigest()
            return f"{digest}  {name}\n".encode()
        if url.endswith(name):
            return self._archive(tag)
        raise AssertionError(f"unexpected url: {url}")


class _FakeRunner:
    """Answer only the worker-Python probe and the health snapshot."""

    def __init__(self, worker_python: Path, *, health_status: str = "ok") -> None:
        self.worker_python = worker_python
        self.health_status = health_status
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs) -> subprocess.CompletedProcess[str]:
        args = [str(item) for item in argv]
        self.calls.append(args)
        if args[1:4] == ["python", "find", "--no-project"]:
            return subprocess.CompletedProcess(args, 0, str(self.worker_python), "")
        if args[1:2] == ["health"]:
            return subprocess.CompletedProcess(
                args, 0, json.dumps({"status": self.health_status}), ""
            )
        raise AssertionError(f"unexpected command: {args}")


class _FailingHealthRunner(_FakeRunner):
    def __call__(self, argv, **kwargs) -> subprocess.CompletedProcess[str]:
        args = [str(item) for item in argv]
        if args[1:2] == ["health"]:
            raise subprocess.TimeoutExpired(args[0], 120)
        return super().__call__(argv, **kwargs)


class _Spawner:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, argv, **kwargs) -> object:
        self.calls.append(([str(item) for item in argv], dict(kwargs)))
        return object()


def _make_owned_state(
    root: Path,
    *,
    install_kind: str = "uv-tool",
    owned: bool = True,
    schema: int = 2,
) -> dict[str, Path]:
    environment = root / "uv-tools" / "agent-with-llm"
    bin_dir = root / "uv-bin"
    environment.mkdir(parents=True)
    bin_dir.mkdir()
    executable = bin_dir / ("polaris.exe" if os.name == "nt" else "polaris")
    executable.write_text("launcher", encoding="utf-8")
    uv = root / ("uv.exe" if os.name == "nt" else "uv")
    uv.write_text("uv", encoding="utf-8")
    worker_python = root / ("python.exe" if os.name == "nt" else "python")
    worker_python.write_text("python", encoding="utf-8")
    state_path = root / "state" / "install-state.json"
    state_path.parent.mkdir()
    state = {
        "schema": schema,
        "completed": {},
        "components": {
            "polaris": {
                "installed_by_polaris": owned,
                "install_kind": install_kind,
                "package": "agent-with-llm",
                "executable": str(executable),
                "environment": str(environment),
                "uv": str(uv),
                "bootstrap_python": str(worker_python),
                "release_version": "0.1.0",
                "release_tag": "v0.1.0",
            }
        },
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return {
        "environment": environment,
        "bin_dir": bin_dir,
        "executable": executable,
        "uv": uv,
        "worker_python": worker_python,
        "state": state_path,
    }


def _spawned_temp_root(spawner: _Spawner) -> Path:
    argv, _ = spawner.calls[0]
    return Path(argv[argv.index("--temp-root") + 1])


def test_compare_versions_orders_releases_and_prereleases() -> None:
    assert upgrade.compare_versions("1.2.3", "1.2.3") == 0
    assert upgrade.compare_versions("v1.2.3", "1.2.3") == 0
    assert upgrade.compare_versions("1.2.3", "1.2.10") == -1
    assert upgrade.compare_versions("1.10.0", "1.9.9") == 1
    assert upgrade.compare_versions("1.2.3-rc.1", "1.2.3") == -1
    assert upgrade.compare_versions("1.2.3", "1.2.3-rc.1") == 1
    assert upgrade.compare_versions("1.2.3-rc.1", "1.2.3-rc.2") == -1
    with pytest.raises(ValueError):
        upgrade.compare_versions("not-a-version", "1.2.3")


def test_normalize_tag_accepts_bare_versions_and_rejects_garbage() -> None:
    assert upgrade.normalize_tag("1.2.3") == "v1.2.3"
    assert upgrade.normalize_tag("v0.1.1") == "v0.1.1"
    assert upgrade.normalize_tag(" v2.0.0-rc.1 ") == "v2.0.0-rc.1"
    for raw in ("", "latest", "1.2", "v1.2.x"):
        with pytest.raises(upgrade.UsageError):
            upgrade.normalize_tag(raw)


def test_check_reports_newer_release(tmp_path, capsys) -> None:
    paths = _make_owned_state(tmp_path)
    feed = _FakeFeed(["0.1.0", "0.2.0"], latest="0.2.0")

    code = upgrade.upgrade_from_cli(
        Namespace(to=None, check=True, force=False),
        state_path=paths["state"],
        fetch=feed,
        version_probe=lambda: "0.1.0",
    )

    assert code == upgrade.EXIT_OK
    out = capsys.readouterr().out
    assert "Installed version: 0.1.0" in out
    assert "Latest release:    0.2.0 (v0.2.0)" in out
    assert "polaris upgrade" in out
    assert feed.urls == [upgrade.release_asset_url(None, "release.json")]


def test_check_reports_up_to_date(tmp_path, capsys) -> None:
    paths = _make_owned_state(tmp_path)
    feed = _FakeFeed(["0.2.0"], latest="0.2.0")

    code = upgrade.upgrade_from_cli(
        Namespace(to=None, check=True, force=False),
        state_path=paths["state"],
        fetch=feed,
        version_probe=lambda: "0.2.0",
    )

    assert code == upgrade.EXIT_OK
    assert "up to date" in capsys.readouterr().out


def test_check_network_failure_is_actionable(tmp_path, capsys) -> None:
    paths = _make_owned_state(tmp_path)

    def offline(url: str) -> bytes:
        raise upgrade.UpgradeError(
            f"could not download {url}: connection failed\n" + upgrade._OFFLINE_GUIDANCE
        )

    code = upgrade.upgrade_from_cli(
        Namespace(to=None, check=True, force=False),
        state_path=paths["state"],
        fetch=offline,
        version_probe=lambda: "0.1.0",
    )

    assert code == upgrade.EXIT_FAILED
    err = capsys.readouterr().err
    assert "[error]" in err
    assert "install.sh --upgrade" in err


def test_upgrade_without_state_refuses_with_guidance(tmp_path, capsys) -> None:
    code = upgrade.upgrade_from_cli(
        Namespace(to=None, check=False, force=False),
        state_path=tmp_path / "missing" / "install-state.json",
        fetch=_FakeFeed(["0.2.0"]),
        version_probe=lambda: "0.1.0",
    )

    assert code == upgrade.EXIT_FAILED
    err = capsys.readouterr().err
    assert "[not installer-owned]" in err
    assert "git pull" in err


def test_upgrade_refuses_non_owned_and_dev_receipts(tmp_path, capsys) -> None:
    external = _make_owned_state(tmp_path / "external", owned=False, install_kind="external")
    code = upgrade.upgrade_from_cli(
        Namespace(to=None, check=False, force=False),
        state_path=external["state"],
        fetch=_FakeFeed(["0.2.0"]),
        version_probe=lambda: "0.1.0",
    )
    assert code == upgrade.EXIT_FAILED
    assert "[not installer-owned]" in capsys.readouterr().err

    dev = _make_owned_state(tmp_path / "dev", install_kind="dev-venv")
    code = upgrade.upgrade_from_cli(
        Namespace(to=None, check=False, force=False),
        state_path=dev["state"],
        fetch=_FakeFeed(["0.2.0"]),
        version_probe=lambda: "0.1.0",
    )
    assert code == upgrade.EXIT_FAILED
    assert "uv-tool" in capsys.readouterr().err


def test_upgrade_refuses_legacy_schema_1_receipts(tmp_path, capsys) -> None:
    paths = _make_owned_state(tmp_path, schema=1)

    code = upgrade.upgrade_from_cli(
        Namespace(to=None, check=False, force=False),
        state_path=paths["state"],
        fetch=_FakeFeed(["0.2.0"]),
        version_probe=lambda: "0.1.0",
    )

    assert code == upgrade.EXIT_FAILED
    assert "[not installer-owned]" in capsys.readouterr().err


def test_source_checkout_without_package_version_gets_git_guidance(tmp_path, capsys) -> None:
    paths = _make_owned_state(tmp_path)

    code = upgrade.upgrade_from_cli(
        Namespace(to=None, check=False, force=False),
        state_path=paths["state"],
        fetch=_FakeFeed(["0.2.0"]),
        version_probe=lambda: None,
    )

    assert code == upgrade.EXIT_FAILED
    assert "git pull" in capsys.readouterr().err


def test_already_up_to_date_exits_without_downloading(tmp_path, capsys) -> None:
    paths = _make_owned_state(tmp_path)
    feed = _FakeFeed(["0.2.0"], latest="0.2.0")
    spawner = _Spawner()

    code = upgrade.upgrade_from_cli(
        Namespace(to=None, check=False, force=False),
        state_path=paths["state"],
        home=tmp_path / "home",
        fetch=feed,
        runner=_FakeRunner(paths["worker_python"]),
        spawner=spawner,
        version_probe=lambda: "0.2.0",
        os_name="nt",
    )

    assert code == upgrade.EXIT_OK
    assert "already up to date" in capsys.readouterr().out
    assert spawner.calls == []
    assert feed.urls == [upgrade.release_asset_url(None, "release.json")]


@pytest.mark.parametrize("os_name", ["nt", "posix"])
def test_upgrade_schedules_detached_worker_without_running_uv_itself(
    tmp_path, capsys, os_name
) -> None:
    paths = _make_owned_state(tmp_path)
    home = tmp_path / "polaris-home"
    feed = _FakeFeed(["0.1.0", "0.2.0"], latest="0.2.0", windows=os_name == "nt")
    runner = _FakeRunner(paths["worker_python"])
    spawner = _Spawner()

    code = upgrade.upgrade_from_cli(
        Namespace(to=None, check=False, force=False),
        state_path=paths["state"],
        home=home,
        fetch=feed,
        runner=runner,
        spawner=spawner,
        version_probe=lambda: "0.1.0",
        os_name=os_name,
    )

    assert code == upgrade.EXIT_OK
    temp_root = _spawned_temp_root(spawner)
    try:
        assert len(spawner.calls) == 1
        argv, kwargs = spawner.calls[0]
        # The worker runs on the recorded external Python, never the tool env.
        assert argv[0] == str(paths["worker_python"])
        assert not upgrade._path_within(Path(argv[0]), paths["environment"])
        worker = Path(argv[1])
        assert worker.parent == temp_root
        script = worker.read_text(encoding="utf-8")
        assert "--only-project" in script
        assert "--upgrade" in script
        assert "install.py" in script
        assert argv[argv.index("--parent-pid") + 1] == str(os.getpid())
        source = Path(argv[argv.index("--source") + 1])
        assert (source / "installer" / "install.py").is_file()
        assert json.loads((source / "release.json").read_text(encoding="utf-8"))["tag"] == "v0.2.0"
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["close_fds"] is True
        if os_name == "nt":
            expected = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
                subprocess, "DETACHED_PROCESS", 0
            )
            assert kwargs["creationflags"] == expected
            assert "start_new_session" not in kwargs
        else:
            assert kwargs["start_new_session"] is True
            assert "creationflags" not in kwargs
        # The CLI never performed the uv tool mutation itself.
        assert [call for call in runner.calls if call[1:3] == ["tool", "install"]] == []
        # The pre-upgrade health snapshot was recorded best-effort.
        snapshot = json.loads((home / "upgrade-last-health.json").read_text(encoding="utf-8"))
        assert snapshot["report"] == {"status": "ok"}
        out = capsys.readouterr().out
        assert "[scheduled]" in out
        assert "v0.2.0" in out
        assert "upgrade-last.log" in out
        assert "polaris --version" in out
        # State receipts are only rewritten by the worker's installer run.
        stored = json.loads(paths["state"].read_text(encoding="utf-8"))
        assert stored["components"]["polaris"]["release_version"] == "0.1.0"
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_force_reinstalls_the_same_version(tmp_path, capsys) -> None:
    paths = _make_owned_state(tmp_path)
    spawner = _Spawner()

    code = upgrade.upgrade_from_cli(
        Namespace(to=None, check=False, force=True),
        state_path=paths["state"],
        home=tmp_path / "home",
        fetch=_FakeFeed(["0.2.0"], latest="0.2.0", windows=True),
        runner=_FakeRunner(paths["worker_python"]),
        spawner=spawner,
        version_probe=lambda: "0.2.0",
        os_name="nt",
    )

    assert code == upgrade.EXIT_OK
    assert "reinstall" in capsys.readouterr().out
    temp_root = _spawned_temp_root(spawner)
    shutil.rmtree(temp_root, ignore_errors=True)


def test_rollback_to_older_tag_uses_pinned_release_urls(tmp_path, capsys) -> None:
    paths = _make_owned_state(tmp_path)
    feed = _FakeFeed(["0.1.0", "0.2.0"], latest="0.2.0", windows=True)
    spawner = _Spawner()

    code = upgrade.upgrade_from_cli(
        Namespace(to="v0.1.0", check=False, force=False),
        state_path=paths["state"],
        home=tmp_path / "home",
        fetch=feed,
        runner=_FakeRunner(paths["worker_python"]),
        spawner=spawner,
        version_probe=lambda: "0.2.0",
        os_name="nt",
    )

    assert code == upgrade.EXIT_OK
    temp_root = _spawned_temp_root(spawner)
    try:
        assert feed.urls
        assert all("/download/v0.1.0/" in url for url in feed.urls)
        assert "rollback" in capsys.readouterr().out
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_unknown_target_tag_reports_the_feed_mismatch(tmp_path, capsys) -> None:
    paths = _make_owned_state(tmp_path)

    code = upgrade.upgrade_from_cli(
        Namespace(to="v9.9.9", check=False, force=False),
        state_path=paths["state"],
        fetch=_FakeFeed(["0.2.0"], windows=True),
        version_probe=lambda: "0.2.0",
    )

    assert code == upgrade.EXIT_FAILED
    assert "HTTP 404" in capsys.readouterr().err


def test_usage_errors_for_invalid_flag_combinations(tmp_path, capsys) -> None:
    paths = _make_owned_state(tmp_path)
    for args in (
        Namespace(to="v0.1.0", check=True, force=False),
        Namespace(to=None, check=True, force=True),
        Namespace(to="garbage", check=False, force=False),
    ):
        code = upgrade.upgrade_from_cli(
            args,
            state_path=paths["state"],
            fetch=_FakeFeed(["0.2.0"]),
            version_probe=lambda: "0.1.0",
        )
        assert code == upgrade.EXIT_USAGE
        assert "[usage]" in capsys.readouterr().err


def test_sha256_mismatch_discards_the_bundle(tmp_path, capsys, monkeypatch) -> None:
    paths = _make_owned_state(tmp_path)
    created: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def tracking_mkdtemp(prefix="", dir=None, **kwargs):
        path = real_mkdtemp(prefix=prefix, dir=tmp_path)
        created.append(Path(path))
        return path

    monkeypatch.setattr(tempfile, "mkdtemp", tracking_mkdtemp)

    code = upgrade.upgrade_from_cli(
        Namespace(to=None, check=False, force=False),
        state_path=paths["state"],
        home=tmp_path / "home",
        fetch=_FakeFeed(["0.2.0"], bad_sums=True, windows=True),
        runner=_FakeRunner(paths["worker_python"]),
        spawner=_Spawner(),
        version_probe=lambda: "0.1.0",
        os_name="nt",
    )

    assert code == upgrade.EXIT_FAILED
    assert "SHA-256 mismatch" in capsys.readouterr().err
    assert created and all(not path.exists() for path in created)


def test_bundle_tag_mismatch_is_rejected(tmp_path, capsys, monkeypatch) -> None:
    paths = _make_owned_state(tmp_path)
    created: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def tracking_mkdtemp(prefix="", dir=None, **kwargs):
        path = real_mkdtemp(prefix=prefix, dir=tmp_path)
        created.append(Path(path))
        return path

    monkeypatch.setattr(tempfile, "mkdtemp", tracking_mkdtemp)
    feed = _FakeFeed(["0.2.0"], windows=True, bundle_tag="v0.9.9")

    code = upgrade.upgrade_from_cli(
        Namespace(to=None, check=False, force=False),
        state_path=paths["state"],
        home=tmp_path / "home",
        fetch=feed,
        runner=_FakeRunner(paths["worker_python"]),
        spawner=_Spawner(),
        version_probe=lambda: "0.1.0",
        os_name="nt",
    )

    assert code == upgrade.EXIT_FAILED
    assert "does not match" in capsys.readouterr().err
    assert created and all(not path.exists() for path in created)


def test_missing_uv_and_worker_python_is_actionable(tmp_path, capsys, monkeypatch) -> None:
    paths = _make_owned_state(tmp_path)
    state = json.loads(paths["state"].read_text(encoding="utf-8"))
    state["components"]["polaris"]["bootstrap_python"] = ""
    state["components"]["polaris"]["uv"] = str(paths["uv"]) + ".missing"
    paths["state"].write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(upgrade.shutil, "which", lambda command: None)

    code = upgrade.upgrade_from_cli(
        Namespace(to=None, check=False, force=False),
        state_path=paths["state"],
        fetch=_FakeFeed(["0.2.0"]),
        runner=_FakeRunner(paths["worker_python"]),
        spawner=_Spawner(),
        version_probe=lambda: "0.1.0",
    )

    assert code == upgrade.EXIT_FAILED
    err = capsys.readouterr().err
    assert "uv is required" in err
    assert "install.sh --upgrade" in err


def test_worker_python_inside_the_tool_environment_is_rejected(tmp_path) -> None:
    paths = _make_owned_state(tmp_path)
    inside = paths["environment"] / ("python.exe" if os.name == "nt" else "python")
    inside.write_text("python", encoding="utf-8")
    state = json.loads(paths["state"].read_text(encoding="utf-8"))
    state["components"]["polaris"]["bootstrap_python"] = str(inside)
    paths["state"].write_text(json.dumps(state), encoding="utf-8")
    # uv exists but also resolves to a Python inside the environment being replaced.
    runner = _FakeRunner(inside)

    with pytest.raises(upgrade.UpgradeError, match="outside the Polaris tool"):
        upgrade.find_worker_python(
            json.loads(paths["state"].read_text(encoding="utf-8"))["components"]["polaris"],
            runner=runner,
        )


def test_health_snapshot_failure_only_warns(tmp_path, capsys) -> None:
    paths = _make_owned_state(tmp_path)
    spawner = _Spawner()

    code = upgrade.upgrade_from_cli(
        Namespace(to=None, check=False, force=False),
        state_path=paths["state"],
        home=tmp_path / "home",
        fetch=_FakeFeed(["0.2.0"], windows=True),
        runner=_FailingHealthRunner(paths["worker_python"]),
        spawner=spawner,
        version_probe=lambda: "0.1.0",
        os_name="nt",
    )

    assert code == upgrade.EXIT_OK
    assert len(spawner.calls) == 1
    assert "[warning]" in capsys.readouterr().err
    shutil.rmtree(_spawned_temp_root(spawner), ignore_errors=True)


def test_cli_parser_dispatches_upgrade_flags(monkeypatch) -> None:
    from agent_core.cli import main

    received: list[tuple] = []
    monkeypatch.setattr(
        upgrade,
        "upgrade_from_cli",
        lambda args: received.append((args.to, args.check, args.force)) or 0,
    )

    assert main(["upgrade", "--to", "v1.2.3", "--force"]) == 0
    assert main(["upgrade", "--check"]) == 0
    assert received == [("v1.2.3", False, True), (None, True, False)]


# --- installer --only-project -------------------------------------------------


class _ToolRunner(install_module.Runner):
    """Inert uv/Polaris command stand-in for the installer's project-only path."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.calls: list[list[str]] = []
        self.uv = root / "uv"
        self.uv.write_text("uv", encoding="utf-8")
        self.tool_root = root / "uv-tools"
        self.bin_dir = root / "uv-bin"
        self.bin_dir.mkdir(parents=True)
        self.executable = self.bin_dir / "polaris"
        self.executable.write_text("launcher", encoding="utf-8")

    def which(self, command: str) -> str | None:
        return str(self.uv) if command == "uv" else None

    def run(self, argv, **kwargs) -> subprocess.CompletedProcess[str]:
        args = [str(item) for item in argv]
        self.calls.append(args)
        if args[0] == str(self.uv):
            return self._run_uv(args)
        if args[0] == str(self.executable) and args[1:2] == ["health"]:
            return subprocess.CompletedProcess(args, 0, json.dumps({"status": "ok"}), "")
        return subprocess.CompletedProcess(args, 0, "", "")

    def _run_uv(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[1:3] == ["tool", "install"]:
            (self.tool_root / "agent-with-llm").mkdir(parents=True, exist_ok=True)
            self.executable.write_text("launcher", encoding="utf-8")
        elif args[1:] == ["tool", "dir"]:
            return subprocess.CompletedProcess(args, 0, str(self.tool_root), "")
        elif args[1:] == ["tool", "dir", "--bin"]:
            return subprocess.CompletedProcess(args, 0, str(self.bin_dir), "")
        return subprocess.CompletedProcess(args, 0, "", "")


def _write_release_source(source: Path, version: str) -> None:
    source.mkdir(parents=True, exist_ok=True)
    wheel = source / f"agent_with_llm-{version}-py3-none-any.whl"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr(
            f"agent_with_llm-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: agent-with-llm\nVersion: {version}\n",
        )
    wheel.write_bytes(buffer.getvalue())
    constraints = source / "polaris-constraints.txt"
    constraints.write_text("httpx==0.27.0\n", encoding="utf-8")
    release = {
        "schema": 1,
        "package": "agent-with-llm",
        "version": version,
        "tag": f"v{version}",
        "wheel": {
            "filename": wheel.name,
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        },
        "constraints": {
            "filename": constraints.name,
            "sha256": hashlib.sha256(constraints.read_bytes()).hexdigest(),
        },
    }
    (source / "release.json").write_text(json.dumps(release), encoding="utf-8")


def _owned_uv_state(state_path: Path, runner: _ToolRunner) -> StateStore:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "schema": 2,
                "completed": {"project:tool": 1},
                "components": {
                    "polaris": {
                        "installed_by_polaris": True,
                        "install_kind": "uv-tool",
                        "package": "agent-with-llm",
                        "executable": str(runner.executable),
                        "environment": str(runner.tool_root / "agent-with-llm"),
                        "uv": str(runner.uv),
                        "tool_root": str(runner.tool_root),
                        "bin_dir": str(runner.bin_dir),
                        "bootstrap_python": str(runner.root / "python"),
                        "release_version": "0.1.0",
                        "release_tag": "v0.1.0",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return StateStore(state_path)


def test_only_project_upgrades_uv_tool_without_host_steps(tmp_path) -> None:
    source = tmp_path / "release"
    _write_release_source(source, "0.2.0")
    runner = _ToolRunner(tmp_path / "fake-host")
    state_path = tmp_path / "state" / "install-state.json"
    installer = Installer(
        Options(source=source, upgrade=True, non_interactive=True, only_project=True),
        runner=runner,
        state=_owned_uv_state(state_path, runner),
        system="linux",
        machine="x86_64",
    )

    checks = installer.install()

    assert checks and all(item.ok for item in checks)
    assert {item.name for item in checks} == {"polaris", "polaris-health"}
    installs = [call for call in runner.calls if call[1:3] == ["tool", "install"]]
    assert len(installs) == 1
    assert "--force" in installs[0]
    assert "--constraints" in installs[0]
    # The receipt carries no recorded width, so the upgrade preserves [all].
    assert installs[0][-1].endswith(".whl[all]")
    # Host tools, Node, containers, memory models and the scheduler are skipped:
    forbidden = ("winget", "brew", "apt-get", "dnf", "podman", "docker")
    assert not [c for c in runner.calls if Path(c[0]).name in forbidden]
    assert not [c for c in runner.calls if c[1:3] == ["memory", "models"]]
    assert not [c for c in runner.calls if c[1:2] == ["scheduler-service"]]
    assert [c for c in runner.calls if c[1:2] == ["health"]]
    receipt = StateStore(state_path).component("polaris")
    assert receipt["release_version"] == "0.2.0"
    assert receipt["release_tag"] == "v0.2.0"
    assert receipt["install_kind"] == "uv-tool"


def test_only_project_requires_upgrade_and_excludes_dev_and_check(capsys) -> None:
    for argv in (
        ["--only-project"],
        ["--only-project", "--upgrade", "--dev"],
        ["--only-project", "--upgrade", "--check"],
    ):
        with pytest.raises(SystemExit):
            install_module.main(argv)
        assert "--only-project requires --upgrade" in capsys.readouterr().err
