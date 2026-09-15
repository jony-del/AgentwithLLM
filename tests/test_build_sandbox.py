from pathlib import Path
import subprocess
import sys

import pytest
import tomlkit

from tools import build_sandbox


def test_constraint_export_keeps_markers_and_requires_current_lock(monkeypatch, tmp_path):
    (tmp_path / "sandbox").mkdir()
    (tmp_path / "sandbox/mcp-servers").mkdir()
    text = "mcp==2.1.1\ncolorama==0.4.6 ; sys_platform == 'win32'\n"
    def run(argv, **kwargs):
        assert argv in [
            ["uv", "export", "--locked", "--extra", "all", "--extra", "dev", "--no-emit-project", "--no-hashes"],
            ["uv", "export", "--project", str(tmp_path / "sandbox/mcp-servers"), "--locked", "--no-emit-project", "--no-hashes"],
        ]
        assert kwargs["cwd"] == tmp_path and kwargs["check"]
        return subprocess.CompletedProcess(argv, 0, text)
    monkeypatch.setattr(build_sandbox.subprocess, "run", run)
    build_sandbox.export_constraints(tmp_path)
    assert (tmp_path / "sandbox/constraints.txt").read_text(encoding="utf-8") == text
    assert (tmp_path / "sandbox/mcp-servers/requirements.txt").read_text(encoding="utf-8") == text


def test_activation_preserves_startup_settings_and_original(monkeypatch, tmp_path):
    local = tmp_path / "agent.local.toml"
    original = b'# personal settings\n[sandbox.container]\nauto_start_machine=true\npodman_machine_name="mine"\n'
    local.write_bytes(original)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    image = "sha256:" + "a" * 64
    build_sandbox.activate(local, original, image, artifacts)
    result = tomlkit.parse(local.read_text(encoding="utf-8"))
    assert result["sandbox"]["container"]["auto_start_machine"]
    assert result["sandbox"]["container"]["podman_machine_name"] == "mine"
    assert result["sandbox"]["container"]["image"] == image
    assert not result["sandbox"]["container"]["auto_pull"]
    assert (artifacts / "agent.local.toml.before").read_bytes() == original
    assert "# personal settings" in local.read_text(encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed during the build"):
        build_sandbox.activate(local, original, "sha256:" + "b" * 64, artifacts)


@pytest.mark.parametrize("fail_at", ["build", "prepare", "pip", "e2e"])
def test_failed_candidate_never_changes_local_config(monkeypatch, tmp_path, fail_at):
    local = tmp_path / "agent.local.toml"
    original = b'[sandbox.container]\nruntime="podman"\nauto_start_machine=true\n'
    local.write_bytes(original)
    monkeypatch.setattr(build_sandbox, "export_constraints", lambda _: None)
    monkeypatch.setattr(build_sandbox, "ensure_podman_ready", lambda *_: None)
    class Manager:
        backend_name = "container"
        def __init__(self, *_args, **_kwargs):
            pass
        def prepare(self):
            if fail_at == "prepare":
                raise RuntimeError("probe failed")
        def is_enabled(self):
            return True
        def wrap_invocation(self, invocation):
            return ["podman", "run", "--pull=never", "pip-check"], False
        def teardown(self):
            pass
    monkeypatch.setattr(build_sandbox, "SandboxManager", Manager)
    def run(argv, **kwargs):
        stage = "build" if "build" in argv else "e2e" if argv[0] == sys.executable else "pip"
        if stage == "build":
            assert argv[argv.index("--platform") + 1] == "linux/amd64"
            assert argv[argv.index("--build-arg") + 1] == "TARGETARCH=amd64"
            Path(argv[argv.index("--iidfile") + 1]).write_text("sha256:" + "a" * 64, encoding="utf-8")
        if stage == fail_at:
            raise subprocess.CalledProcessError(1, argv)
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(build_sandbox.subprocess, "run", run)
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        build_sandbox.build(root=tmp_path)
    assert local.read_bytes() == original
