"""Tests for the OS sandbox enforcement layer (manager + backends + degradation)."""

import shutil
import sys

import pytest

from agent_core.sandbox import SandboxConfig, SandboxManager, SandboxUnavailableError
from agent_core.sandbox.backends import BubblewrapBackend, NoopBackend, SeatbeltBackend


def _force_linux_with_bwrap(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")


def _force_macos(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")


def _force_windows(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")


# -- config --------------------------------------------------------------------------


def test_config_from_dict_nested_tables() -> None:
    config = SandboxConfig.from_dict(
        {
            "enabled": True,
            "excluded_commands": ["bazel:*"],
            "filesystem": {"deny_read": ["~/.ssh"]},
            "network": {"allowed_domains": ["api.example.com"]},
        }
    )
    assert config.enabled is True
    assert config.excluded_commands == ["bazel:*"]
    assert config.filesystem.deny_read == ["~/.ssh"]
    assert config.network.allowed_domains == ["api.example.com"]


def test_config_defaults_disabled() -> None:
    assert SandboxConfig().enabled is False


# -- backend selection / platform ----------------------------------------------------


def test_backend_selection_linux(monkeypatch) -> None:
    _force_linux_with_bwrap(monkeypatch)
    manager = SandboxManager(SandboxConfig(enabled=True))
    assert isinstance(manager._backend, BubblewrapBackend)
    assert manager.is_supported_platform()


def test_backend_selection_macos(monkeypatch) -> None:
    _force_macos(monkeypatch)
    manager = SandboxManager(SandboxConfig(enabled=True))
    assert isinstance(manager._backend, SeatbeltBackend)


def test_backend_selection_windows_is_noop(monkeypatch) -> None:
    _force_windows(monkeypatch)
    manager = SandboxManager(SandboxConfig(enabled=True))
    assert isinstance(manager._backend, NoopBackend)
    assert not manager.is_supported_platform()
    assert not manager.is_enabled()


# -- degradation ---------------------------------------------------------------------


def test_windows_wrap_is_passthrough(monkeypatch) -> None:
    _force_windows(monkeypatch)
    manager = SandboxManager(SandboxConfig(enabled=True))
    spec, shell = manager.wrap("ls -la", True, command="ls -la")
    assert spec == "ls -la"
    assert shell is True


def test_missing_dependencies_degrades(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", lambda name: None)  # bwrap absent
    manager = SandboxManager(SandboxConfig(enabled=True))
    assert manager.is_supported_platform()  # platform supports it...
    assert not manager.is_enabled()  # ...but the dependency is missing
    assert "missing sandbox dependencies" in (manager.unavailable_reason() or "")
    spec, shell = manager.wrap("ls", True, command="ls")
    assert spec == "ls"  # passthrough


def test_disabled_manager_never_wraps(monkeypatch) -> None:
    _force_linux_with_bwrap(monkeypatch)
    manager = SandboxManager(SandboxConfig(enabled=False))
    assert not manager.is_enabled()
    assert manager.unavailable_reason() is None  # silent when the user didn't ask for it
    spec, _ = manager.wrap("ls", True, command="ls")
    assert spec == "ls"


# -- wrapping ------------------------------------------------------------------------


def test_bubblewrap_wrap_prefixes_command(monkeypatch, tmp_path) -> None:
    _force_linux_with_bwrap(monkeypatch)
    manager = SandboxManager(SandboxConfig(enabled=True), workspace=tmp_path)
    spec, shell = manager.wrap("echo hi", True, command="echo hi")
    assert shell is False
    assert spec[0] == "bwrap"
    assert "--ro-bind" in spec
    assert "--unshare-net" in spec  # network default-deny
    # the workspace is bound writable
    assert str(tmp_path.resolve()) in spec
    # the original command is preserved at the tail via /bin/sh -c
    assert spec[-3:] == ["/bin/sh", "-c", "echo hi"]


def test_bubblewrap_wrap_argv_spec(monkeypatch, tmp_path) -> None:
    _force_linux_with_bwrap(monkeypatch)
    manager = SandboxManager(SandboxConfig(enabled=True), workspace=tmp_path)
    spec, shell = manager.wrap(["pytest", "-q"], False, command=None)
    assert shell is False
    assert spec[0] == "bwrap"
    assert spec[-2:] == ["pytest", "-q"]


def test_seatbelt_wrap_builds_profile(monkeypatch, tmp_path) -> None:
    _force_macos(monkeypatch)
    config = SandboxConfig(enabled=True)
    config.filesystem.deny_read = ["/etc/secret"]
    manager = SandboxManager(config, workspace=tmp_path)
    spec, shell = manager.wrap("echo hi", True, command="echo hi")
    assert shell is False
    assert spec[0] == "sandbox-exec"
    assert spec[1] == "-p"
    profile = spec[2]
    assert "(deny file-write*)" in profile
    assert "secret" in profile  # deny_read path folded into the profile (basename check
    # keeps this cross-platform; absolute-path normalization differs off-macOS)
    assert "(deny network*)" in profile


# -- excluded commands ---------------------------------------------------------------


def test_excluded_command_runs_unsandboxed(monkeypatch) -> None:
    _force_linux_with_bwrap(monkeypatch)
    config = SandboxConfig(enabled=True, excluded_commands=["bazel:*"])
    manager = SandboxManager(config)
    assert manager.is_enabled()
    assert not manager.should_sandbox("bazel build //...")
    assert manager.should_sandbox("echo hi")
    # excluded command is passed through unwrapped
    spec, shell = manager.wrap("bazel build //...", True, command="bazel build //...")
    assert spec == "bazel build //..."


# -- fail_if_unavailable -------------------------------------------------------------


def test_fail_if_unavailable_raises_on_windows(monkeypatch) -> None:
    _force_windows(monkeypatch)
    config = SandboxConfig(enabled=True, fail_if_unavailable=True)
    with pytest.raises(SandboxUnavailableError):
        SandboxManager(config)


def test_fail_if_unavailable_ok_when_available(monkeypatch) -> None:
    _force_linux_with_bwrap(monkeypatch)
    config = SandboxConfig(enabled=True, fail_if_unavailable=True)
    # Should construct without raising.
    manager = SandboxManager(config)
    assert manager.is_enabled()


def test_fail_if_unavailable_ignored_when_disabled(monkeypatch) -> None:
    _force_windows(monkeypatch)
    # enabled=False → fail_if_unavailable is moot, no raise.
    SandboxManager(SandboxConfig(enabled=False, fail_if_unavailable=True))
