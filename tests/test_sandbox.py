"""Tests for the sandbox enforcement layer: pluggable tier backends, selection/downgrade,
lifecycle, and graceful degradation.

Backends are organised by isolation tier (native/container/vm). Nothing here runs a real
sandbox — platform + runtime availability are mocked via ``sys.platform`` and
``shutil.which``, and lifecycle probes via ``subprocess.run``.
"""

import shutil
import subprocess
import sys

import pytest

from agent_core.sandbox import (
    SandboxConfig,
    SandboxManager,
    SandboxTier,
    SandboxUnavailableError,
)
from agent_core.sandbox.backends import (
    BubblewrapBackend,
    ContainerBackend,
    NativeBackend,
    NoopBackend,
    SeatbeltBackend,
    VmBackend,
)


def _only(*present):
    """A shutil.which replacement that reports only ``present`` binaries on PATH."""
    return lambda name: f"/usr/bin/{name}" if name in present else None


def _force_linux(monkeypatch, *present) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", _only(*present))


def _force_macos(monkeypatch, *present) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(shutil, "which", _only(*present))


def _force_windows(monkeypatch, *present) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(shutil, "which", _only(*present))


def _native(**kwargs) -> SandboxConfig:
    return SandboxConfig.from_dict({"enabled": True, "backend": "native", **kwargs})


# -- config --------------------------------------------------------------------------


def test_config_from_dict_nested_tables() -> None:
    config = SandboxConfig.from_dict(
        {
            "enabled": True,
            "backend": "container",
            "excluded_commands": ["bazel:*"],
            "filesystem": {"deny_read": ["~/.ssh"]},
            "network": {"allowed_domains": ["api.example.com"]},
            "container": {"image": "alpine", "runtime": "docker", "memory": "512m"},
            "vm": {"provider": "hyperv", "reset_each_task": False},
        }
    )
    assert config.enabled is True
    assert config.backend == "container"
    assert config.excluded_commands == ["bazel:*"]
    assert config.filesystem.deny_read == ["~/.ssh"]
    assert config.network.allowed_domains == ["api.example.com"]
    assert config.container.image == "alpine"
    assert config.container.runtime == "docker"
    assert config.container.memory == "512m"
    assert config.vm.provider == "hyperv"
    assert config.vm.reset_each_task is False


def test_config_defaults() -> None:
    config = SandboxConfig()
    assert config.enabled is False
    assert config.backend == "auto"


def test_config_bad_backend_degrades_to_auto() -> None:
    assert SandboxConfig.from_dict({"backend": "nonsense"}).backend == "auto"
    assert SandboxConfig.from_dict({"backend": "VM"}).backend == "vm"  # case-normalised


# -- native tier selection -----------------------------------------------------------


def test_native_selection_linux(monkeypatch) -> None:
    _force_linux(monkeypatch, "bwrap")
    manager = SandboxManager(_native())
    assert isinstance(manager._backend, NativeBackend)
    assert isinstance(manager._backend.strategy, BubblewrapBackend)
    assert manager.backend_tier is SandboxTier.NATIVE
    assert manager.is_enabled()


def test_native_selection_macos(monkeypatch) -> None:
    _force_macos(monkeypatch, "sandbox-exec")
    manager = SandboxManager(_native())
    assert isinstance(manager._backend.strategy, SeatbeltBackend)


def test_native_on_windows_does_not_isolate(monkeypatch) -> None:
    _force_windows(monkeypatch)
    manager = SandboxManager(_native())
    # NativeBackend's strategy is the no-op → treated as not isolating → degrades.
    assert isinstance(manager._backend, NoopBackend)
    assert not manager.is_supported_platform()
    assert not manager.is_enabled()


# -- auto selection prefers container -------------------------------------------------


def test_auto_prefers_container_when_runtime_present(monkeypatch) -> None:
    _force_linux(monkeypatch, "bwrap", "podman")
    manager = SandboxManager(SandboxConfig.from_dict({"enabled": True, "backend": "auto"}))
    assert isinstance(manager._backend, ContainerBackend)
    assert manager.backend_tier is SandboxTier.CONTAINER


def test_auto_falls_back_to_native_without_runtime(monkeypatch) -> None:
    _force_linux(monkeypatch, "bwrap")  # no container runtime
    manager = SandboxManager(SandboxConfig.from_dict({"enabled": True, "backend": "auto"}))
    assert isinstance(manager._backend, NativeBackend)


def test_auto_never_selects_vm(monkeypatch) -> None:
    # Everything present; auto must still not pick the heavyweight VM tier.
    _force_linux(monkeypatch, "bwrap", "podman", "kata-runtime")
    manager = SandboxManager(SandboxConfig.from_dict({"enabled": True, "backend": "auto"}))
    assert manager.backend_tier is SandboxTier.CONTAINER


# -- downgrade chain -----------------------------------------------------------------


def test_explicit_vm_degrades_to_container(monkeypatch) -> None:
    # VM requested but no kata runtime; docker present → container wins.
    _force_linux(monkeypatch, "docker")
    manager = SandboxManager(SandboxConfig.from_dict({"enabled": True, "backend": "vm"}))
    assert isinstance(manager._backend, ContainerBackend)
    assert manager._backend.runtime == "docker"


def test_explicit_container_degrades_to_native(monkeypatch) -> None:
    _force_linux(monkeypatch, "bwrap")  # no runtime → native
    manager = SandboxManager(SandboxConfig.from_dict({"enabled": True, "backend": "container"}))
    assert isinstance(manager._backend, NativeBackend)


def test_all_unavailable_degrades_to_noop(monkeypatch) -> None:
    _force_linux(monkeypatch)  # nothing present
    manager = SandboxManager(SandboxConfig.from_dict({"enabled": True, "backend": "container"}))
    assert isinstance(manager._backend, NoopBackend)
    assert not manager.is_enabled()
    reason = manager.unavailable_reason() or ""
    assert "podman" in reason  # actionable: names the missing runtime


# -- degradation / passthrough -------------------------------------------------------


def test_windows_wrap_is_passthrough(monkeypatch) -> None:
    _force_windows(monkeypatch)
    manager = SandboxManager(SandboxConfig(enabled=True))
    spec, shell = manager.wrap("ls -la", True, command="ls -la")
    assert spec == "ls -la"
    assert shell is True


def test_disabled_manager_never_wraps(monkeypatch) -> None:
    _force_linux(monkeypatch, "bwrap", "podman")
    manager = SandboxManager(SandboxConfig(enabled=False))
    assert not manager.is_enabled()
    assert manager.unavailable_reason() is None  # silent when the user didn't ask for it
    spec, _ = manager.wrap("ls", True, command="ls")
    assert spec == "ls"


# -- native wrapping (bwrap / seatbelt strategies) -----------------------------------


def test_bubblewrap_wrap_prefixes_command(monkeypatch, tmp_path) -> None:
    _force_linux(monkeypatch, "bwrap")
    manager = SandboxManager(_native(), workspace=tmp_path)
    spec, shell = manager.wrap("echo hi", True, command="echo hi")
    assert shell is False
    assert spec[0] == "bwrap"
    assert "--ro-bind" in spec
    assert "--unshare-net" in spec  # network default-deny
    assert str(tmp_path.resolve()) in spec
    assert spec[-3:] == ["/bin/sh", "-c", "echo hi"]


def test_bubblewrap_wrap_argv_spec(monkeypatch, tmp_path) -> None:
    _force_linux(monkeypatch, "bwrap")
    manager = SandboxManager(_native(), workspace=tmp_path)
    spec, shell = manager.wrap(["pytest", "-q"], False, command=None)
    assert shell is False
    assert spec[0] == "bwrap"
    assert spec[-2:] == ["pytest", "-q"]


def test_seatbelt_wrap_builds_profile(monkeypatch, tmp_path) -> None:
    _force_macos(monkeypatch, "sandbox-exec")
    config = _native()
    config.filesystem.deny_read = ["/etc/secret"]
    manager = SandboxManager(config, workspace=tmp_path)
    spec, shell = manager.wrap("echo hi", True, command="echo hi")
    assert shell is False
    assert spec[0] == "sandbox-exec"
    assert spec[1] == "-p"
    profile = spec[2]
    assert "(deny file-write*)" in profile
    assert "secret" in profile
    assert "(deny network*)" in profile


# -- container wrapping --------------------------------------------------------------


def _container_manager(monkeypatch, tmp_path, *present, image_present=True, **container):
    _force_linux(monkeypatch, *present)
    # Image inspect / pull probes: succeed only when image_present.
    monkeypatch.setattr(
        "agent_core.sandbox.backends.container._run_probe",
        lambda argv: image_present or "pull" in argv,
    )
    cfg = SandboxConfig.from_dict(
        {"enabled": True, "backend": "container", "container": {"image": "alpine", **container}}
    )
    return SandboxManager(cfg, workspace=tmp_path)


def test_container_wrap_hardening_flags(monkeypatch, tmp_path) -> None:
    manager = _container_manager(
        monkeypatch, tmp_path, "podman", memory="512m", cpus="1.5", pids_limit="128"
    )
    spec, shell = manager.wrap("echo hi", True, command="echo hi")
    assert shell is False
    assert spec[0] == "podman"
    assert spec[1] == "run"
    assert "--rm" in spec
    assert ["--network", "none"] == [spec[i] for i in (_pair(spec, "--network"))]
    assert "--read-only" in spec
    assert "--cap-drop" in spec and "ALL" in spec
    assert "no-new-privileges" in spec
    assert "512m" in spec and "1.5" in spec and "128" in spec
    assert "alpine" in spec
    assert spec[-3:] == ["/bin/sh", "-c", "echo hi"]


def test_container_prefers_podman_over_docker(monkeypatch, tmp_path) -> None:
    manager = _container_manager(monkeypatch, tmp_path, "podman", "docker")
    assert manager._backend.runtime == "podman"


def test_container_network_opt_in(monkeypatch, tmp_path) -> None:
    manager = _container_manager(monkeypatch, tmp_path, "podman")
    manager.config.network.allowed_domains = ["api.example.com"]
    spec, _ = manager.wrap("curl x", True, command="curl x")
    assert "bridge" in spec and "none" not in spec


def test_container_windows_path_mapping(monkeypatch, tmp_path) -> None:
    _force_windows(monkeypatch, "docker")
    monkeypatch.setattr("agent_core.sandbox.backends.container._run_probe", lambda argv: True)
    cfg = SandboxConfig.from_dict(
        {"enabled": True, "backend": "container", "container": {"image": "alpine"}}
    )
    manager = SandboxManager(cfg, workspace="E:/proj/app")
    spec, _ = manager.wrap("ls", True, command="ls")
    # Host path preserved on the host side of -v; guest side mapped to /mnt/e/...
    mount = spec[spec.index("-v") + 1]
    assert mount.endswith(":/mnt/e/proj/app")
    assert spec[spec.index("-w") + 1] == "/mnt/e/proj/app"


def test_container_prepare_pulls_missing_image(monkeypatch, tmp_path) -> None:
    calls = []
    _force_linux(monkeypatch, "podman")

    def fake_probe(argv):
        calls.append(argv)
        # image inspect fails (missing); pull succeeds.
        return "pull" in argv

    monkeypatch.setattr("agent_core.sandbox.backends.container._run_probe", fake_probe)
    cfg = SandboxConfig.from_dict(
        {"enabled": True, "backend": "container", "container": {"image": "alpine", "auto_pull": True}}
    )
    manager = SandboxManager(cfg, workspace=tmp_path)
    manager.prepare()
    assert any("pull" in c for c in calls)
    assert manager.is_enabled()  # still the container backend after a successful pull


def test_container_prepare_missing_image_no_autopull_degrades(monkeypatch, tmp_path) -> None:
    _force_linux(monkeypatch, "podman")
    monkeypatch.setattr("agent_core.sandbox.backends.container._run_probe", lambda argv: False)
    cfg = SandboxConfig.from_dict(
        {"enabled": True, "backend": "container", "container": {"image": "alpine", "auto_pull": False}}
    )
    manager = SandboxManager(cfg, workspace=tmp_path)
    manager.prepare()
    # prepare() could not ready the image → degrade to passthrough (no raise).
    assert isinstance(manager._backend, NoopBackend)
    assert not manager.is_enabled()


def test_container_prepare_fail_if_unavailable_raises(monkeypatch, tmp_path) -> None:
    _force_linux(monkeypatch, "podman")
    monkeypatch.setattr("agent_core.sandbox.backends.container._run_probe", lambda argv: False)
    cfg = SandboxConfig.from_dict(
        {
            "enabled": True,
            "backend": "container",
            "fail_if_unavailable": True,
            "container": {"image": "alpine", "auto_pull": False},
        }
    )
    manager = SandboxManager(cfg, workspace=tmp_path)
    with pytest.raises(SandboxUnavailableError):
        manager.prepare()


# -- vm tier -------------------------------------------------------------------------


def test_vm_kata_selected_on_linux(monkeypatch, tmp_path) -> None:
    _force_linux(monkeypatch, "podman", "kata-runtime")
    monkeypatch.setattr("agent_core.sandbox.backends.container._run_probe", lambda argv: True)
    cfg = SandboxConfig.from_dict({"enabled": True, "backend": "vm"})
    manager = SandboxManager(cfg, workspace=tmp_path)
    assert isinstance(manager._backend, VmBackend)
    assert manager._backend.strategy_name == "kata"
    spec, shell = manager.wrap("echo hi", True, command="echo hi")
    assert shell is False
    # Kata reuses the container launcher with a --runtime override.
    assert spec[0] == "podman"
    assert "--runtime" in spec and "kata-runtime" in spec


def test_vm_hyperv_lifecycle_order(monkeypatch, tmp_path) -> None:
    _force_windows(monkeypatch, "powershell", "ssh")
    calls = []
    monkeypatch.setattr(
        "agent_core.sandbox.backends.vm._run_vm_command",
        lambda argv: calls.append(argv) or True,
    )
    cfg = SandboxConfig.from_dict(
        {
            "enabled": True,
            "backend": "vm",
            "vm": {"provider": "hyperv", "guest_host": "sandbox-vm", "reset_each_task": True},
        }
    )
    manager = SandboxManager(cfg, workspace="E:/proj")
    assert manager._backend.strategy_name == "hyperv"
    manager.prepare()
    manager.reset()
    manager.teardown()
    joined = " ".join(" ".join(c) for c in calls)
    assert "Checkpoint-VM" in joined  # prepare created a base snapshot
    assert "Restore-VMSnapshot" in joined  # reset rolled back
    # wrap runs the command over SSH into the guest.
    spec, _ = manager.wrap("echo hi", True, command="echo hi")
    assert spec[0] == "ssh" and spec[1] == "sandbox-vm"


def test_vm_reset_noop_when_disabled(monkeypatch, tmp_path) -> None:
    _force_windows(monkeypatch, "powershell", "ssh")
    calls = []
    monkeypatch.setattr(
        "agent_core.sandbox.backends.vm._run_vm_command",
        lambda argv: calls.append(argv) or True,
    )
    cfg = SandboxConfig.from_dict(
        {
            "enabled": True,
            "backend": "vm",
            "vm": {"provider": "hyperv", "guest_host": "vm", "reset_each_task": False},
        }
    )
    manager = SandboxManager(cfg, workspace="E:/proj")
    manager.reset()
    assert not any("Restore-VMSnapshot" in " ".join(c) for c in calls)


# -- excluded commands ---------------------------------------------------------------


def test_excluded_command_runs_unsandboxed(monkeypatch) -> None:
    _force_linux(monkeypatch, "bwrap")
    config = _native(excluded_commands=["bazel:*"])
    manager = SandboxManager(config)
    assert manager.is_enabled()
    assert not manager.should_sandbox("bazel build //...")
    assert manager.should_sandbox("echo hi")
    spec, shell = manager.wrap("bazel build //...", True, command="bazel build //...")
    assert spec == "bazel build //..."


# -- fail_if_unavailable (construction-time) -----------------------------------------


def test_fail_if_unavailable_raises_on_windows(monkeypatch) -> None:
    _force_windows(monkeypatch)
    config = SandboxConfig(enabled=True, fail_if_unavailable=True)
    with pytest.raises(SandboxUnavailableError):
        SandboxManager(config)


def test_fail_if_unavailable_ok_when_available(monkeypatch) -> None:
    _force_linux(monkeypatch, "bwrap")
    config = _native(fail_if_unavailable=True)
    manager = SandboxManager(config)
    assert manager.is_enabled()


def test_fail_if_unavailable_ignored_when_disabled(monkeypatch) -> None:
    _force_windows(monkeypatch)
    SandboxManager(SandboxConfig(enabled=False, fail_if_unavailable=True))


def _pair(spec, flag):
    i = spec.index(flag)
    return (i, i + 1)
