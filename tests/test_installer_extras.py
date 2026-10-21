"""Lock the installer's default extra set and the ``--full`` widening flag.

The default end-user install must stay free of benchmark-only dependencies
(``swebench`` and its stack) while ``--full`` restores the complete ``[all]``
extra set.  These tests pin the spec strings handed to uv, the extras recorded
in the install receipt, and the drift between ``DEFAULT_EXTRAS`` and
pyproject.toml.  Upgrades of installer-owned installs must preserve the
originally installed width: a legacy receipt without an ``extras`` field (the
only kind written before the default was narrowed) keeps ``[all]`` so an
upgrade never silently drops already-installed dependencies.  Everything is
hermetic: fake runners, tmp_path sources, no uv or network access.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import installer.install as install_module
from installer.install import DEFAULT_EXTRAS, Installer, Options, Runner, StateStore

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DEFAULT_EXTRAS = ("web", "mcp", "terminal", "memory", "codeintel", "credentials")
DEFAULT_SPEC_SUFFIX = "[web,mcp,terminal,memory,codeintel,credentials]"


def _options(source: Path, **overrides) -> Options:
    values = {"source": source, "skip_sandbox": True}
    values.update(overrides)
    return Options(**values)


def _write_source(source: Path) -> None:
    source.mkdir(parents=True, exist_ok=True)
    (source / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")


def _write_release_bundle(source: Path, version: str = "0.1.1") -> Path:
    """Minimal valid release bundle, mirroring tests/test_upgrade.py helpers."""

    source.mkdir(parents=True, exist_ok=True)
    wheel = source / f"agent_with_llm-{version}-py3-none-any.whl"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            f"agent_with_llm-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: agent-with-llm\nVersion: {version}\n",
        )
    wheel.write_bytes(buffer.getvalue())
    constraints = source / "polaris-constraints.txt"
    constraints.write_text("httpx==0.28.1\n", encoding="utf-8")
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
    return wheel


class _UvToolRunner(Runner):
    """Inert uv stand-in: records argv and fabricates the tool env on install."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.uv = root / ("uv.exe" if os.name == "nt" else "uv")
        self.uv.write_text("uv", encoding="utf-8")
        self.tool_root = root / "uv-tools"
        self.bin_dir = root / "uv-bin"
        self.calls: list[list[str]] = []

    def which(self, command: str) -> str | None:
        return str(self.uv) if command == "uv" else None

    def run(self, argv, **kwargs):
        rendered = [str(item) for item in argv]
        self.calls.append(rendered)
        if rendered[1:] == ["tool", "dir"]:
            return subprocess.CompletedProcess(rendered, 0, str(self.tool_root), "")
        if rendered[1:] == ["tool", "dir", "--bin"]:
            return subprocess.CompletedProcess(rendered, 0, str(self.bin_dir), "")
        if rendered[1:3] == ["tool", "install"]:
            (self.tool_root / "agent-with-llm").mkdir(parents=True, exist_ok=True)
            self.bin_dir.mkdir(parents=True, exist_ok=True)
            executable = self.bin_dir / ("polaris.exe" if os.name == "nt" else "polaris")
            executable.write_text("launcher", encoding="utf-8")
        return subprocess.CompletedProcess(rendered, 0, "", "")

    def install_calls(self) -> list[list[str]]:
        return [call for call in self.calls if call[1:3] == ["tool", "install"]]


def _tool_install(tmp_path: Path, **overrides) -> tuple[_UvToolRunner, StateStore]:
    _write_source(tmp_path)
    runner = _UvToolRunner(tmp_path / "fake-uv")
    state = StateStore(tmp_path / "state.json")
    installer = Installer(
        _options(tmp_path, **overrides),
        runner=runner,
        state=state,
        system="windows" if os.name == "nt" else "linux",
        machine="x86_64",
    )
    installer._install_project()
    return runner, state


def _seed_owned_tool_receipt(state: StateStore, runner: _UvToolRunner, extras: str | None) -> None:
    """An installer-owned uv-tool receipt, with the width recorded only if given."""

    runner.bin_dir.mkdir(parents=True, exist_ok=True)
    executable = runner.bin_dir / ("polaris.exe" if os.name == "nt" else "polaris")
    executable.write_text("launcher", encoding="utf-8")
    (runner.tool_root / "agent-with-llm").mkdir(parents=True, exist_ok=True)
    details = {
        "install_kind": "uv-tool",
        "package": "agent-with-llm",
        "executable": str(executable.resolve()),
        "environment": str((runner.tool_root / "agent-with-llm").resolve()),
        "uv": str(runner.uv.resolve()),
        "tool_root": str(runner.tool_root),
        "bin_dir": str(runner.bin_dir),
        "bootstrap_python": str(Path(sys.executable).resolve()),
        "release_version": "0.1.0",
        "release_tag": "v0.1.0",
    }
    if extras is not None:
        details["extras"] = extras
    state.mark("project:tool", component="polaris", source="test", details=details)


def _tool_upgrade(tmp_path: Path, receipt_extras: str | None, **overrides) -> tuple[_UvToolRunner, StateStore]:
    _write_source(tmp_path)
    runner = _UvToolRunner(tmp_path / "fake-uv")
    state = StateStore(tmp_path / "state.json")
    _seed_owned_tool_receipt(state, runner, receipt_extras)
    installer = Installer(
        _options(tmp_path, upgrade=True, **overrides),
        runner=runner,
        state=state,
        system="windows" if os.name == "nt" else "linux",
        machine="x86_64",
    )
    installer._install_project()
    return runner, state


def test_default_extras_lock() -> None:
    assert tuple(DEFAULT_EXTRAS) == EXPECTED_DEFAULT_EXTRAS


def test_pyproject_all_extra_stays_a_superset_of_the_default_set() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    optional = project["optional-dependencies"]
    default = set(DEFAULT_EXTRAS)
    assert default <= set(optional), "every default extra must exist in pyproject.toml"
    assert "swebench" in optional and "swebench" not in default
    self_references = [
        entry for entry in optional["all"] if entry.startswith(f"{install_module.PACKAGE_NAME}[")
    ]
    assert len(self_references) == 1
    inner = self_references[0].split("[", 1)[1].removesuffix("]")
    all_extras = {item.strip() for item in inner.split(",")}
    assert default < all_extras, "[all] must stay a strict superset of the default set"
    assert "swebench" in all_extras


def test_default_source_spec_excludes_swebench_and_includes_credentials(tmp_path) -> None:
    _write_source(tmp_path)
    installer = Installer(
        _options(tmp_path),
        runner=Runner(),
        state=StateStore(tmp_path / "state.json"),
        system="linux",
        machine="x86_64",
    )

    spec, constraints, release = installer._project_install_spec()

    assert spec == f"{tmp_path.resolve()}{DEFAULT_SPEC_SUFFIX}"
    assert "swebench" not in spec
    assert constraints is None and release == {}


def test_full_source_spec_restores_all_extras(tmp_path) -> None:
    _write_source(tmp_path)
    installer = Installer(
        _options(tmp_path, full=True),
        runner=Runner(),
        state=StateStore(tmp_path / "state.json"),
        system="linux",
        machine="x86_64",
    )

    spec, _, _ = installer._project_install_spec()

    assert spec == f"{tmp_path.resolve()}[all]"


def test_dev_source_spec_keeps_all_dev_regardless_of_full(tmp_path) -> None:
    _write_source(tmp_path)
    for overrides in ({"dev": True}, {"dev": True, "full": True}):
        installer = Installer(
            _options(tmp_path, **overrides),
            runner=Runner(),
            state=StateStore(tmp_path / "state.json"),
            system="linux",
            machine="x86_64",
        )
        spec, _, _ = installer._project_install_spec()
        assert spec == f"{tmp_path.resolve()}[all,dev]"


def test_default_wheel_spec_is_narrowed_and_keeps_constraints(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    wheel = _write_release_bundle(bundle)
    installer = Installer(
        _options(bundle),
        runner=Runner(),
        state=StateStore(tmp_path / "state.json"),
        system="linux",
        machine="x86_64",
    )

    spec, constraints, release = installer._project_install_spec()

    assert spec == f"{wheel.resolve()}{DEFAULT_SPEC_SUFFIX}"
    assert "swebench" not in spec
    assert constraints == (bundle / "polaris-constraints.txt").resolve()
    assert release["tag"] == "v0.1.1"


def test_full_wheel_spec_restores_all_extras(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    wheel = _write_release_bundle(bundle)
    installer = Installer(
        _options(bundle, full=True),
        runner=Runner(),
        state=StateStore(tmp_path / "state.json"),
        system="linux",
        machine="x86_64",
    )

    spec, _, _ = installer._project_install_spec()

    assert spec == f"{wheel.resolve()}[all]"


def test_default_uv_tool_install_records_default_extras(tmp_path) -> None:
    runner, state = _tool_install(tmp_path)

    installs = runner.install_calls()
    assert len(installs) == 1
    assert installs[0][-1] == f"{tmp_path.resolve()}{DEFAULT_SPEC_SUFFIX}"
    assert "swebench" not in installs[0][-1]
    assert state.component("polaris")["extras"] == "default"


def test_full_uv_tool_install_records_full_extras(tmp_path) -> None:
    runner, state = _tool_install(tmp_path, full=True)

    installs = runner.install_calls()
    assert len(installs) == 1
    assert installs[0][-1] == f"{tmp_path.resolve()}[all]"
    assert state.component("polaris")["extras"] == "full"


def test_legacy_receipt_upgrade_preserves_the_all_width(tmp_path) -> None:
    runner, state = _tool_upgrade(tmp_path, None)

    installs = runner.install_calls()
    assert len(installs) == 1
    assert "--force" in installs[0]
    assert installs[0][-1] == f"{tmp_path.resolve()}[all]"
    assert state.component("polaris")["extras"] == "full"


def test_default_receipt_upgrade_keeps_the_narrowed_set(tmp_path) -> None:
    runner, state = _tool_upgrade(tmp_path, "default")

    installs = runner.install_calls()
    assert len(installs) == 1
    assert "--force" in installs[0]
    assert installs[0][-1] == f"{tmp_path.resolve()}{DEFAULT_SPEC_SUFFIX}"
    assert "swebench" not in installs[0][-1]
    assert state.component("polaris")["extras"] == "default"


def test_full_receipt_upgrade_keeps_the_all_width(tmp_path) -> None:
    runner, state = _tool_upgrade(tmp_path, "full")

    installs = runner.install_calls()
    assert len(installs) == 1
    assert installs[0][-1] == f"{tmp_path.resolve()}[all]"
    assert state.component("polaris")["extras"] == "full"


def test_full_flag_overrides_the_recorded_receipt_width(tmp_path) -> None:
    for index, receipt_extras in enumerate((None, "default", "full")):
        source = tmp_path / str(index)
        runner, state = _tool_upgrade(source, receipt_extras, full=True)

        installs = runner.install_calls()
        assert len(installs) == 1
        assert installs[0][-1] == f"{source.resolve()}[all]"
        assert state.component("polaris")["extras"] == "full"


def test_upgrade_without_an_owned_receipt_installs_the_default_set(tmp_path) -> None:
    runner, state = _tool_install(tmp_path, upgrade=True)

    installs = runner.install_calls()
    assert len(installs) == 1
    assert "--force" not in installs[0]
    assert installs[0][-1] == f"{tmp_path.resolve()}{DEFAULT_SPEC_SUFFIX}"
    assert state.component("polaris")["extras"] == "default"


def test_dev_install_records_dev_extras_even_with_full(tmp_path) -> None:
    _write_source(tmp_path)
    scripts = tmp_path / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    scripts.mkdir(parents=True)
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    python.write_text("launcher", encoding="utf-8")
    executable = scripts / ("polaris.exe" if os.name == "nt" else "polaris")
    pip_specs: list[str] = []

    class DevRunner(Runner):
        def which(self, command: str) -> str | None:
            return "uv" if command == "uv" else None

        def run(self, argv, **kwargs):
            rendered = [str(item) for item in argv]
            if rendered[0] == str(python):
                return subprocess.CompletedProcess(rendered, 0, "3.12\n", "")
            if rendered[1:3] == ["pip", "install"]:
                pip_specs.append(rendered[-1])
                executable.write_text("launcher", encoding="utf-8")
            return subprocess.CompletedProcess(rendered, 0, "", "")

    state = StateStore(tmp_path / "state.json")
    installer = Installer(
        _options(tmp_path, dev=True, full=True),
        runner=DevRunner(),
        state=state,
        system="windows" if os.name == "nt" else "linux",
        machine="x86_64",
    )

    installer._install_project()

    assert pip_specs == [f"{tmp_path.resolve()}[all,dev]"]
    assert state.component("polaris")["extras"] == "dev"


def test_full_flag_parses_and_defaults_off() -> None:
    parser = install_module.build_parser()
    assert parser.parse_args([]).full is False
    assert parser.parse_args(["--full"]).full is True


def test_main_maps_full_flag_into_options(monkeypatch, tmp_path) -> None:
    captured: list[Options] = []

    class StubInstaller:
        def __init__(self, options: Options) -> None:
            captured.append(options)

        def install(self) -> list:
            return []

    monkeypatch.setattr(install_module, "Installer", StubInstaller)

    assert install_module.main(["--source", str(tmp_path), "--full"]) == install_module.EXIT_OK
    assert captured[0].full is True
    assert captured[0].dev is False


def test_public_bootstraps_forward_the_full_flag() -> None:
    shell = (ROOT / "install.sh").read_text(encoding="utf-8")
    powershell = (ROOT / "install.ps1").read_text(encoding="utf-8")

    assert "--full)" in shell
    assert "DEV || FULL ||" in shell
    assert "[switch]$Full" in powershell
    assert '$Dev -or $Full' in powershell
    assert 'if ($Full) { $arguments += "--full" }' in powershell


def test_runtime_health_profile_never_requires_benchmark_distributions() -> None:
    from agent_core import health

    for name in ("swebench", "datasets", "docker", "pyarrow", "pandas"):
        assert name not in health._RUNTIME_DISTRIBUTIONS
        assert name not in health._DEV_DISTRIBUTIONS
