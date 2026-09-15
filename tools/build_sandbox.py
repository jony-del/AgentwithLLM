"""Build and accept a local linux/amd64 Podman sandbox before activating it."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_core.config import resolve_sandbox_config  # noqa: E402
from agent_core.sandbox import SandboxInvocation, SandboxManager  # noqa: E402
from agent_core.sandbox.backends.podman_machine import ensure_podman_ready  # noqa: E402
from agent_core.sandbox.config import is_local_image_id  # noqa: E402
from agent_core.tools.base import ExecutionScope  # noqa: E402


def export_constraints(root: Path) -> None:
    result = subprocess.run(
        ["uv", "export", "--locked", "--extra", "all", "--extra", "dev",
         "--no-emit-project", "--no-hashes"],
        cwd=root, check=True, capture_output=True, text=True, encoding="utf-8",
    )
    (root / "sandbox" / "constraints.txt").write_text(result.stdout, encoding="utf-8")
    servers = root / "sandbox" / "mcp-servers"
    result = subprocess.run(
        ["uv", "export", "--project", str(servers), "--locked", "--no-emit-project", "--no-hashes"],
        cwd=root, check=True, capture_output=True, text=True, encoding="utf-8",
    )
    (servers / "requirements.txt").write_text(result.stdout, encoding="utf-8")


def activate(local: Path, original: bytes | None, image: str, artifacts: Path) -> None:
    """Preserve comments and startup settings; never overwrite concurrent edits."""
    import tomlkit

    if (local.read_bytes() if local.exists() else None) != original:
        raise RuntimeError(f"{local} changed during the build; candidate was not activated")
    document = tomlkit.parse(original.decode("utf-8") if original is not None else "")
    sandbox = document.setdefault("sandbox", tomlkit.table())
    sandbox["enabled"] = True
    sandbox["backend"] = "container"
    container = sandbox.setdefault("container", tomlkit.table())
    container["runtime"] = "podman"
    container["auto_pull"] = False
    container["image"] = image
    if original is not None:
        (artifacts / "agent.local.toml.before").write_bytes(original)
    with tempfile.NamedTemporaryFile(dir=local.parent, suffix=".toml.tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(tomlkit.dumps(document).encode("utf-8"))
    try:
        temporary.replace(local)
    finally:
        temporary.unlink(missing_ok=True)


def build(*, root: Path = ROOT, activate_image: bool = True) -> str:
    local = root / "agent.local.toml"
    original = local.read_bytes() if local.exists() else None
    candidate = deepcopy(resolve_sandbox_config(root / "agent.toml"))
    candidate.enabled = True
    candidate.backend = "container"
    candidate.container.runtime = "podman"
    candidate.container.auto_pull = False
    artifacts_root = root / "tmp"
    artifacts_root.mkdir(exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix="sandbox-build-", dir=artifacts_root))
    print(f"Build records: {artifacts}", flush=True)
    if original is not None:
        (artifacts / "agent.local.toml.before").write_bytes(original)
    export_constraints(root)
    if sys.platform == "win32":
        ensure_podman_ready("podman", candidate.container)
    iidfile = artifacts / "image.id"
    subprocess.run(
        ["podman", "build", "--platform", "linux/amd64", "--build-arg", "TARGETARCH=amd64",
         "--layers", "--iidfile", str(iidfile), "--ignorefile", str(root / ".dockerignore"),
         "-f", str(root / "sandbox" / "Containerfile"), str(root)],
        cwd=root, check=True,
    )
    image = iidfile.read_text(encoding="utf-8").strip()
    if not is_local_image_id(image):
        raise RuntimeError(f"Podman iidfile did not contain a complete local image ID: {image!r}")
    candidate.container.image = image
    manager = SandboxManager(candidate, workspace=root)
    try:
        manager.prepare()
        if not manager.is_enabled() or manager.backend_name != "container":
            raise RuntimeError("candidate did not prepare the container sandbox")
        invocation = SandboxInvocation.create(
            ["host-execution-forbidden"], guest_argv=["@python", "-m", "pip", "check"],
            required_guest_capabilities=("python",),
            scope=ExecutionScope.for_workspace(root, network="deny"),
        )
        argv, shell = manager.wrap_invocation(invocation)
        assert isinstance(argv, list) and not shell
        subprocess.run(argv, cwd=root, check=True, timeout=120)
    finally:
        manager.teardown()
    env = dict(os.environ, POLARIS_SANDBOX_E2E_RUNTIME="podman", POLARIS_SANDBOX_E2E_IMAGE=image)
    env.pop("POLARIS_SANDBOX_E2E_EXPECT_STOPPED", None)
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_sandbox_oci_e2e.py"],
        cwd=root, env=env, check=True, timeout=900,
    )
    (artifacts / "result.json").write_text(
        json.dumps({"image": image, "platform": "linux/amd64", "prepared": True,
                    "pip_check": "passed", "oci_e2e": "passed"}, indent=2) + "\n",
        encoding="utf-8",
    )
    if activate_image:
        activate(local, original, image, artifacts)
    print(f"Accepted sandbox image: {image}", flush=True)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-activate", action="store_true", help="accept without changing agent.local.toml")
    args = parser.parse_args()
    try:
        build(activate_image=not args.no_activate)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Sandbox build/acceptance failed; local config was preserved: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
