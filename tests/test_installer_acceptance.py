from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

import installer.install as install_module
from installer.install import (
    InstallError,
    Installer,
    Options,
    Runner,
    StateStore,
    download_file,
    managed_runtime_root,
    node_archive_name,
)

NODE_VERSION = install_module.NODE_VERSION


class HermeticRunner(Runner):
    """Inert runner for install-layer acceptance tests.

    Host tools are reported present and healthy, while every mutating command
    (``uv tool install``, the Polaris CLI's own model/scheduler registration) is
    faked inside ``root``. Nothing touches the network, a real package manager,
    or the developer's tools. The installer is always constructed with
    ``system="linux"`` so behavior is identical on every developer host.
    """

    def __init__(self, root: Path, *, node_present: bool = True) -> None:
        super().__init__()
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.calls: list[list[str]] = []
        self.uv = self.root / "uv"
        self.uv.write_text("uv", encoding="utf-8")
        self.tool_root = self.root / "uv-tools"
        self.bin_dir = self.root / "uv-bin"
        self.executable = self.bin_dir / "polaris"
        self.node_present = node_present

    def which(self, command: str) -> str | None:
        if command == "uv":
            return str(self.uv)
        if command == "node" and not self.node_present:
            return None
        if command in {"git", "rg", "node", "npm", "npx", "bash"}:
            return f"/fake-host/{command}"
        return None

    def run(self, argv, **kwargs):
        rendered = [str(item) for item in argv]
        self.calls.append(rendered)
        if rendered[0] == str(self.uv):
            return self._run_uv(rendered)
        if rendered[0] == str(self.executable):
            return self._run_polaris(rendered)
        if rendered[-1:] == ["--version"]:
            name = Path(rendered[0]).name
            version = f"{NODE_VERSION}" if name == "node" else f"{name} 1.0"
            return subprocess.CompletedProcess(rendered, 0, version, "")
        return subprocess.CompletedProcess(rendered, 0, "", "")

    def _run_uv(self, rendered: list[str]) -> subprocess.CompletedProcess[str]:
        args = rendered[1:]
        if args == ["tool", "dir"]:
            return subprocess.CompletedProcess(rendered, 0, str(self.tool_root), "")
        if args == ["tool", "dir", "--bin"]:
            return subprocess.CompletedProcess(rendered, 0, str(self.bin_dir), "")
        if args[:2] == ["tool", "install"]:
            (self.tool_root / "agent-with-llm").mkdir(parents=True, exist_ok=True)
            self.bin_dir.mkdir(parents=True, exist_ok=True)
            self.executable.write_text("launcher", encoding="utf-8")
        return subprocess.CompletedProcess(rendered, 0, "", "")

    def _run_polaris(self, rendered: list[str]) -> subprocess.CompletedProcess[str]:
        args = rendered[1:]
        if args[:2] == ["memory", "models"]:
            root = Path(os.environ["POLARIS_HOME"]) / "models" / "memory" / "bundle-v1"
            root.mkdir(parents=True, exist_ok=True)
            (root / ".polaris-owned").write_text("{}", encoding="utf-8")
            return subprocess.CompletedProcess(rendered, 0, json.dumps({"root": str(root)}), "")
        if args[:1] == ["scheduler-service"]:
            receipt = Path.home() / ".polaris" / "scheduler-service.json"
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(
                json.dumps(
                    {"service_id": "polaris-scheduler", "executable": str(self.executable)}
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(rendered, 0, "", "")
        if args[:1] == ["health"]:
            payload = {
                "status": "ok",
                "checks": [
                    {"name": "configuration", "status": "ok"},
                    {"name": "provider", "status": "ok"},
                ],
            }
            return subprocess.CompletedProcess(rendered, 0, json.dumps(payload), "")
        return subprocess.CompletedProcess(rendered, 0, "", "")


def _make_installer(
    source: Path, state_path: Path, runner: HermeticRunner, **overrides
) -> Installer:
    values = {"source": source, "skip_sandbox": True}
    values.update(overrides)
    return Installer(
        Options(**values),
        runner=runner,
        state=StateStore(state_path),
        system="linux",
        machine="x86_64",
    )


def _write_source(source: Path) -> None:
    source.mkdir(parents=True, exist_ok=True)
    (source / "pyproject.toml").write_text(
        '[project]\nname = "agent-with-llm"\nversion = "0.1.0"\n', encoding="utf-8"
    )


def _patch_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))


def _node_archive_bytes() -> bytes:
    pytest.importorskip("lzma")
    stem = node_archive_name(NODE_VERSION, "linux", "x86_64").removesuffix(".tar.xz")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:xz") as archive:
        for name in ("node", "npm", "npx"):
            payload = f"fake {name}\n".encode()
            info = tarfile.TarInfo(f"{stem}/bin/{name}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def test_full_install_accepts_non_ascii_and_spaced_paths(monkeypatch, tmp_path) -> None:
    root = tmp_path / "中文 目录"
    source = root / "source checkout"
    _write_source(source)
    home = root / "用户 目录"
    _patch_home(monkeypatch, home)
    polaris_home = root / "polaris 数据"
    monkeypatch.setenv("POLARIS_HOME", str(polaris_home))
    state_path = root / "state store" / "install-state.json"
    runner = HermeticRunner(root / "fake host")

    checks = _make_installer(source, state_path, runner).install()

    assert checks and all(item.ok for item in checks)
    names = {item.name for item in checks}
    assert {"polaris", "polaris-health", "scheduler-service"} <= names
    assert state_path.is_file()
    stored = json.loads(state_path.read_text(encoding="utf-8"))
    receipt = stored["components"]["polaris"]
    assert receipt["installed_by_polaris"] is True
    assert receipt["install_kind"] == "uv-tool"
    assert "中文 目录" in receipt["executable"]
    assert Path(receipt["executable"]).is_file()
    bundle = polaris_home / "models" / "memory" / "bundle-v1"
    assert (bundle / ".polaris-owned").is_file()
    assert (home / ".polaris" / "scheduler-service.json").is_file()
    health = next(item for item in checks if item.name == "polaris-health")
    assert health.ok and health.detail == "ok"


def test_repeat_install_is_idempotent_without_duplicate_mutations(
    monkeypatch, tmp_path
) -> None:
    root = tmp_path / "repeat"
    source = root / "source"
    _write_source(source)
    _patch_home(monkeypatch, root / "home")
    monkeypatch.setenv("POLARIS_HOME", str(root / "home" / ".polaris"))
    state_path = root / "state" / "install-state.json"
    runner = HermeticRunner(root / "fake-host")
    path_before = os.environ.get("PATH", "")

    first = _make_installer(source, state_path, runner).install()
    assert all(item.ok for item in first)
    installs = [call for call in runner.calls if call[1:3] == ["tool", "install"]]
    assert len(installs) == 1

    runner.calls.clear()
    second = _make_installer(source, state_path, runner).install()

    assert all(item.ok for item in second)
    assert not [call for call in runner.calls if call[1:3] == ["tool", "install"]]
    assert not [call for call in runner.calls if call[1:3] == ["tool", "update-shell"]]
    assert os.environ.get("PATH", "") == path_before
    state = StateStore(state_path)
    assert state.component_owned("polaris")
    assert state.component("polaris")["executable"] == str(runner.executable)


def test_upgrade_preserves_user_data_in_polaris_home(monkeypatch, tmp_path) -> None:
    root = tmp_path / "upgrade"
    source = root / "source"
    _write_source(source)
    home = root / "home"
    _patch_home(monkeypatch, home)
    polaris_home = home / ".polaris"
    monkeypatch.setenv("POLARIS_HOME", str(polaris_home))
    settings = polaris_home / "settings.toml"
    sessions = polaris_home / "sessions" / "2026-01-01 demo.jsonl"
    sessions.parent.mkdir(parents=True)
    settings.write_text('provider = "fake"\n', encoding="utf-8")
    sessions.write_text('{"event": "用户会话"}\n', encoding="utf-8")
    state_path = root / "state" / "install-state.json"
    runner = HermeticRunner(root / "fake-host")

    _make_installer(source, state_path, runner).install()
    before_settings = settings.read_bytes()
    before_sessions = sessions.read_bytes()

    runner.calls.clear()
    upgraded = _make_installer(source, state_path, runner, upgrade=True).install()

    assert all(item.ok for item in upgraded)
    installs = [call for call in runner.calls if call[1:3] == ["tool", "install"]]
    assert len(installs) == 1 and "--force" in installs[0]
    assert settings.read_bytes() == before_settings
    assert sessions.read_bytes() == before_sessions
    assert StateStore(state_path).component_owned("polaris")


def test_node_download_failure_leaves_no_partial_mutation_and_retry_completes(
    monkeypatch, tmp_path
) -> None:
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "redirected" / variable.lower()))
    monkeypatch.setattr(install_module, "expose_node", lambda node_bin, system: [])
    archive_bytes = _node_archive_bytes()
    digest = hashlib.sha256(archive_bytes).hexdigest()
    archive_name = node_archive_name(NODE_VERSION, "linux", "x86_64")
    attempts = 0

    def flaky_download(url: str, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise InstallError("simulated network failure")
        if url.endswith("SHASUMS256.txt"):
            destination.write_text(f"{digest}  {archive_name}\n", encoding="utf-8")
        else:
            assert url.endswith(archive_name)
            destination.write_bytes(archive_bytes)

    monkeypatch.setattr(install_module, "download_file", flaky_download)
    state_path = tmp_path / "state" / "install-state.json"
    runner = HermeticRunner(tmp_path / "fake-host", node_present=False)
    installer = _make_installer(tmp_path / "source", state_path, runner)

    with pytest.raises(InstallError, match="simulated network failure"):
        installer._install_node()

    runtime_root = managed_runtime_root()
    assert "redirected" in str(runtime_root)
    assert not runtime_root.exists()
    assert not state_path.exists()

    installer._install_node()

    stem = archive_name.removesuffix(".tar.xz")
    assert (runtime_root / stem / "bin" / "node").is_file()
    receipt = installer.state.component("node")
    assert receipt["install_kind"] == "managed-runtime"
    assert Path(receipt["path"]) == (runtime_root / stem).resolve()
    assert installer.state.component_owned("node")


def test_failed_download_writes_no_partial_file_and_caller_retry_succeeds(
    monkeypatch, tmp_path
) -> None:
    target = tmp_path / "中文 目录" / "asset.bin"
    target.parent.mkdir(parents=True)
    attempts = 0

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *exc) -> bool:
            return False

        def read(self) -> bytes:
            return b"payload"

    def flaky_urlopen(request, timeout=0):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("connection reset by peer")
        return FakeResponse()

    monkeypatch.setattr(install_module.urllib.request, "urlopen", flaky_urlopen)

    with pytest.raises(InstallError, match="could not download"):
        download_file("https://releases.invalid/asset", target)
    assert not target.exists()

    download_file("https://releases.invalid/asset", target)
    assert target.read_bytes() == b"payload"


def test_health_command_runs_from_non_ascii_polaris_home(
    monkeypatch, tmp_path, capsys
) -> None:
    import argparse

    from agent_core.cli import health_command

    root = tmp_path / "中文 目录"
    root.mkdir(parents=True)
    monkeypatch.setenv("POLARIS_HOME", str(root / "polaris 数据"))
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        "agent_core.health.collect_dependency_checks",
        lambda profile, **overrides: [],
    )
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
    assert payload["status"] in {"ok", "degraded"}
    assert payload["profile"] == "runtime"
