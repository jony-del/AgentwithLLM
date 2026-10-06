from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest

from installer.install import InstallError, Installer, Options, Runner, StateStore
from tools.build_release_assets import PAYLOAD_FILES, WHEEL_RESOURCES, assemble_assets, sha256


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def release_assets(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text('[project]\nname="agent-with-llm"\nversion="0.1.0"\n')
    for name in (*PAYLOAD_FILES, "install.ps1", "install.sh"):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    # Repository-only config must never be included by the distribution builder.
    (source / ".env").write_text("DO_NOT_DISTRIBUTE=private\n")
    (source / "agent.toml").write_text("# private project configuration\n")
    wheel = tmp_path / "agent_with_llm-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in WHEEL_RESOURCES:
            archive.writestr(name, "# bundled resource\n")
        archive.writestr("agent_with_llm-0.1.0.dist-info/METADATA", "Name: agent-with-llm\nVersion: 0.1.0\n")
        archive.writestr("agent_with_llm-0.1.0.dist-info/entry_points.txt", "[console_scripts]\npolaris = agent_core.cli:main\n")
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("httpx==0.28.1\n")
    output = tmp_path / "assets"
    assemble_assets(source, output, wheel, constraints, "v0.1.0")
    return output


def extract_bundle(assets: Path, destination: Path) -> Path:
    with zipfile.ZipFile(assets / "polaris-installer.zip") as archive:
        archive.extractall(destination)
    return destination


def test_release_assets_are_complete_pinned_and_exclude_repository(release_assets: Path, tmp_path: Path) -> None:
    with zipfile.ZipFile(release_assets / "polaris-installer.zip") as archive:
        names = set(archive.namelist())
    with tarfile.open(release_assets / "polaris-installer.tar.gz") as archive:
        assert {item.name for item in archive.getmembers()} == names
    assert names == {*PAYLOAD_FILES, "release.json", "polaris-constraints.txt", "agent_with_llm-0.1.0-py3-none-any.whl"}
    assert '$ReleaseTag = "v0.1.0"' in (release_assets / "install.ps1").read_text()
    assert 'RELEASE_TAG="v0.1.0"' in (release_assets / "install.sh").read_text()
    for line in (release_assets / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert sha256((release_assets / name).read_bytes()) == digest
    bundle = extract_bundle(release_assets, tmp_path / "bundle")
    for worker in ("installer/install.py", "agent_core/uninstall.py"):
        result = subprocess.run([sys.executable, "-I", str(bundle / worker), "--help"], cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_builder_rejects_version_mismatch_without_publishing_assets(release_assets: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not match project version"):
        assemble_assets(
            tmp_path / "source", tmp_path / "bad-output",
            release_assets / "agent_with_llm-0.1.0-py3-none-any.whl",
            release_assets / "polaris-constraints.txt", "v99.0.0",
        )
    assert not (tmp_path / "bad-output").exists()


def test_version_query_does_not_start_chat(monkeypatch, capsys) -> None:
    from agent_core import cli

    def unexpected_chat(args):
        pytest.fail("--version must not start a provider or an interactive session")

    monkeypatch.setattr(cli, "chat_command", unexpected_chat)
    with pytest.raises(SystemExit) as result:
        cli.main(["--version"])
    assert result.value.code == 0
    assert capsys.readouterr().out.startswith("polaris ")


class DistributionRunner(Runner):
    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.calls: list[list[str]] = []
        self.executable = root / "bin" / ("polaris.exe" if os.name == "nt" else "polaris")

    def which(self, command: str) -> str | None:
        return "uv" if command == "uv" else None

    def run(self, argv, **kwargs):
        args = [str(item) for item in argv]
        self.calls.append(args)
        output = ""
        if args[1:] == ["tool", "dir"]:
            output = str(self.root / "tools")
        elif args[1:] == ["tool", "dir", "--bin"]:
            output = str(self.root / "bin")
        elif args[1:3] == ["tool", "install"]:
            (self.root / "tools" / "agent-with-llm").mkdir(parents=True, exist_ok=True)
            self.executable.parent.mkdir(parents=True, exist_ok=True)
            self.executable.write_text("launcher")
        return subprocess.CompletedProcess(args, 0, output, "")


def test_wheel_install_and_upgrade_keep_ownership_and_constraints(release_assets: Path, tmp_path: Path) -> None:
    bundle = extract_bundle(release_assets, tmp_path / "bundle")
    runner = DistributionRunner(tmp_path / "runtime")
    state = StateStore(tmp_path / "state.json")
    installer = Installer(Options(source=bundle, skip_sandbox=True), runner=runner, state=state)
    installer._install_project()
    command = next(call for call in runner.calls if call[1:3] == ["tool", "install"])
    assert command[-1] == f"{bundle.resolve() / 'agent_with_llm-0.1.0-py3-none-any.whl'}[all]"
    assert command[command.index("--constraints") + 1] == str(bundle.resolve() / "polaris-constraints.txt")
    assert state.component("polaris")["release_tag"] == "v0.1.0"
    assert state.component_owned("polaris")
    installer.options.upgrade = True
    runner.calls.clear()
    installer._install_project()
    assert any("--force" in call for call in runner.calls)
    assert state.component("polaris")["release_version"] == "0.1.0"


@pytest.mark.parametrize("damage", ["wheel", "constraints", "traversal", "identity", "wheel-version"])
def test_invalid_bundle_fails_before_mutating_host(release_assets: Path, tmp_path: Path, damage: str) -> None:
    bundle = extract_bundle(release_assets, tmp_path / "bundle")
    release_path = bundle / "release.json"
    release = json.loads(release_path.read_text())
    if damage in {"wheel", "constraints"}:
        (bundle / release[damage]["filename"]).write_bytes(b"corrupt")
    elif damage == "traversal":
        release["wheel"]["filename"] = "../foreign.whl"
    elif damage == "identity":
        release["package"] = "foreign-app"
    else:
        wheel = bundle / release["wheel"]["filename"]
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("agent_with_llm-0.1.0.dist-info/METADATA", "Name: agent-with-llm\nVersion: 99.0.0\n")
        release["wheel"]["sha256"] = sha256(wheel.read_bytes())
    release_path.write_text(json.dumps(release))
    runner = DistributionRunner(tmp_path / "runtime")
    installer = Installer(Options(source=bundle, skip_sandbox=True), runner=runner, state=StateStore(tmp_path / "state.json"))
    with pytest.raises(InstallError):
        installer.install()
    assert not any("install" in call for call in runner.calls)
    assert not (tmp_path / "state.json").exists()


def _bootstrap_payload(assets: Path) -> None:
    """Use an inert worker so the real remote bootstrap cannot alter this computer."""
    worker = b'import json,os,sys; from pathlib import Path; Path(os.environ["POLARIS_BOOTSTRAP_TRACE"]).write_text(json.dumps(sys.argv[1:])); sys.exit(20)\n'
    with zipfile.ZipFile(assets / "polaris-installer.zip") as archive:
        payload = {name: archive.read(name) for name in archive.namelist()}
    payload["installer/install.py"] = worker
    from tools.build_release_assets import write_archives
    write_archives(assets, payload)
    (assets / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256((assets / name).read_bytes()).hexdigest()}  {name}\n"
        for name in ("polaris-installer.zip", "polaris-installer.tar.gz")
    ))


@pytest.mark.skipif(os.name != "nt", reason="PowerShell pipeline bootstrap on Windows")
@pytest.mark.parametrize("corrupt", [False, True])
def test_powershell_pipeline_without_scriptroot(release_assets: Path, tmp_path: Path, corrupt: bool) -> None:
    _bootstrap_payload(release_assets)
    if corrupt:
        with (release_assets / "polaris-installer.zip").open("ab") as stream:
            stream.write(b"corrupt")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "uv.cmd").write_text(f'@echo off\nif "%2"=="find" echo {sys.executable}\nexit /b 0\n')
    env = os.environ.copy()
    env.update({"POLARIS_BOOTSTRAP_ASSETS": str(release_assets), "POLARIS_BOOTSTRAP_TRACE": str(tmp_path / "trace.json")})
    env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
    script = """
function Invoke-WebRequest {
    param([switch]$UseBasicParsing, [string]$Uri, [string]$OutFile)
    if ($Uri -notmatch '/releases/download/v0.1.0/') { throw 'Unpinned request' }
    $name = [IO.Path]::GetFileName(([Uri]$Uri).AbsolutePath)
    Copy-Item -LiteralPath (Join-Path $env:POLARIS_BOOTSTRAP_ASSETS $name) -Destination $OutFile
}
Get-Content -Raw -LiteralPath (Join-Path $env:POLARIS_BOOTSTRAP_ASSETS 'install.ps1') | Invoke-Expression
"""
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == (10 if corrupt else 20), result.stdout + result.stderr
    trace = tmp_path / "trace.json"
    if corrupt:
        assert not trace.exists()
        assert "SHA-256 mismatch" in result.stderr
    else:
        args = json.loads(trace.read_text())
        assert args[0] == "--source"
        assert not Path(args[1]).exists(), "temporary bundle must be cleaned after the worker exits"


@pytest.mark.skipif(os.name == "nt", reason="Bash pipeline bootstrap on POSIX")
def test_bash_pipeline_ignores_checkout_and_propagates_restart(release_assets: Path, tmp_path: Path) -> None:
    _bootstrap_payload(release_assets)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv = fake_bin / "uv"
    uv.write_text(f'#!/bin/sh\nif [ "$2" = "find" ]; then printf "%s\\n" "{sys.executable}"; fi\n')
    curl = fake_bin / "curl"
    curl.write_text('#!/bin/sh\ncase "$2" in */releases/download/v0.1.0/*) ;; *) exit 90 ;; esac\ncp "$POLARIS_BOOTSTRAP_ASSETS/${2##*/}" "$4"\n')
    uv.chmod(0o755)
    curl.chmod(0o755)
    env = os.environ.copy()
    env.update({"POLARIS_BOOTSTRAP_ASSETS": str(release_assets), "POLARIS_BOOTSTRAP_TRACE": str(tmp_path / "trace.json")})
    env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
    result = subprocess.run(["bash"], input=(release_assets / "install.sh").read_text(), cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 20, result.stdout + result.stderr
    args = json.loads((tmp_path / "trace.json").read_text())
    assert args[1] != str(ROOT)
    assert not Path(args[1]).exists()
