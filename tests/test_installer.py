from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path

import pytest

from installer.install import (
    EXIT_UNSUPPORTED,
    Check,
    Installer,
    Options,
    RestartRequired,
    Runner,
    StateStore,
    UnsupportedHost,
    checksum_for,
    node_archive_name,
    normalize_machine,
    parse_node_major,
    verify_sha256,
)
from agent_core.sandbox.config import SandboxContainerConfig


class FakeRunner(Runner):
    def __init__(self, present=(), *, outcomes=None, versions=None, dry_run=False) -> None:
        super().__init__(dry_run=dry_run)
        self.present = set(present)
        self.outcomes = outcomes or {}
        self.versions = versions or {}
        self.calls: list[list[str]] = []

    def which(self, command: str) -> str | None:
        return f"/bin/{command}" if command in self.present else None

    def run(self, argv, **kwargs):
        rendered = [str(item) for item in argv]
        self.calls.append(rendered)
        key = tuple(rendered)
        returncode = self.outcomes.get(key, 0)
        stdout = ""
        command = Path(rendered[0]).name
        if rendered[-1:] == ["--version"]:
            stdout = self.versions.get(command, f"{command} 1.0")
        if rendered[-4:] == ["machine", "list", "--format", "json"]:
            stdout = "[]"
        return subprocess.CompletedProcess(rendered, returncode, stdout, "")


def _options(tmp_path, **overrides) -> Options:
    values = {"source": tmp_path, "skip_sandbox": False}
    values.update(overrides)
    return Options(**values)


def test_node_platform_mapping_and_version_parsing() -> None:
    assert normalize_machine("AMD64") == "x86_64"
    assert normalize_machine("aarch64") == "arm64"
    assert parse_node_major("v24.4.1") == 24
    assert node_archive_name("v24.4.1", "windows", "x86_64").endswith("win-x64.zip")
    assert node_archive_name("v24.4.1", "darwin", "arm64").endswith("darwin-arm64.tar.gz")
    assert node_archive_name("v24.4.1", "linux", "x86_64").endswith("linux-x64.tar.xz")


def test_release_manifest_matches_public_bootstraps_and_sandbox_default() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "installer/manifest.json").read_text(encoding="utf-8"))
    assert manifest["uv_version"] in (root / "install.ps1").read_text(encoding="utf-8")
    assert manifest["uv_version"] in (root / "install.sh").read_text(encoding="utf-8")
    assert manifest["sandbox_image"] == SandboxContainerConfig().image


def test_checksum_verification(tmp_path) -> None:
    archive = tmp_path / "asset.zip"
    archive.write_bytes(b"trusted")
    digest = hashlib.sha256(b"trusted").hexdigest()
    assert checksum_for("asset.zip", f"{digest}  *asset.zip\n") == digest
    verify_sha256(archive, digest)
    with pytest.raises(Exception, match="SHA-256 mismatch"):
        verify_sha256(archive, "0" * 64)


def test_state_store_records_only_install_metadata(tmp_path) -> None:
    path = tmp_path / "state.json"
    state = StateStore(path)
    state.mark("host:git", component="git", source="apt")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["components"]["git"]["installed_by_polaris"] is True
    assert StateStore(path).completed("host:git")
    assert StateStore(path).component_owned("git")


def test_check_mode_is_non_mutating_when_runtime_is_ready(tmp_path) -> None:
    runner = FakeRunner(
        {"git", "rg", "node", "npm", "npx", "podman", "polaris"},
        versions={"node": "v24.4.1"},
    )
    installer = Installer(
        _options(tmp_path, check=True),
        runner=runner,
        state=StateStore(tmp_path / "state.json"),
        system="linux",
        machine="x86_64",
    )
    checks = installer.install()
    assert all(isinstance(item, Check) for item in checks)
    assert all(item.ok for item in checks)
    assert not (tmp_path / "state.json").exists()
    assert not any("install" in call for call in runner.calls)


def test_runner_dry_run_executes_probes_but_skips_mutations(monkeypatch) -> None:
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "ready", "")

    monkeypatch.setattr("installer.install.subprocess.run", fake_run)
    runner = Runner(dry_run=True)
    probe = runner.run(["tool", "info"], capture=True)
    mutation = runner.run(["tool", "install"], mutates=True)
    assert probe.stdout == "ready"
    assert mutation.stdout == ""
    assert calls == [["tool", "info"]]


def test_runtime_detection_skips_broken_podman(tmp_path) -> None:
    outcomes = {("podman", "info"): 1, ("docker", "info"): 0}
    runner = FakeRunner({"podman", "docker"}, outcomes=outcomes)
    installer = Installer(
        _options(tmp_path),
        runner=runner,
        state=StateStore(tmp_path / "state.json"),
        system="linux",
        machine="x86_64",
    )
    assert installer._select_usable_runtime() == "docker"


def test_windows_wsl_enable_requires_resumable_restart(tmp_path) -> None:
    runner = FakeRunner(
        {"wsl"},
        outcomes={("wsl", "--status"): 1},
    )
    state = StateStore(tmp_path / "state.json")
    installer = Installer(
        _options(tmp_path),
        runner=runner,
        state=state,
        system="windows",
        machine="x86_64",
    )
    with pytest.raises(RestartRequired):
        installer._ensure_windows_wsl()
    assert state.completed("windows:wsl-enabled")


def test_windows_arm_is_an_explicit_unsupported_host(tmp_path) -> None:
    installer = Installer(
        _options(tmp_path),
        runner=FakeRunner(),
        state=StateStore(tmp_path / "state.json"),
        system="windows",
        machine="arm64",
    )
    with pytest.raises(UnsupportedHost):
        installer._validate_host()
    assert EXIT_UNSUPPORTED == 30
