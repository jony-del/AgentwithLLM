"""Build wheel-based installers without including the repository or user config.

Run with the same uv version as CI. All dependencies are exported from uv.lock;
the bootstrap scripts embedded in each release pin that release's exact tag.
"""

from __future__ import annotations

import argparse
from email.parser import Parser
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PAYLOAD_FILES = (
    "installer/install.py",
    "installer/manifest.json",
    "agent_core/__init__.py",
    "agent_core/uninstall.py",
    "agent_core/scheduler_service.py",
)
WHEEL_RESOURCES = {
    "agent_core/cli.py",
    "agent_core/uninstall.py",
    "agent_core/scheduler_service.py",
    "agent_core/skills/bundled/init.md",
    "agent_core/memory/model_bundle.json",
    "agent_core/sandbox/sandbox-image.lock.json",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_wheel(path: Path, version: str) -> None:
    with zipfile.ZipFile(path) as wheel:
        names = set(wheel.namelist())
        if not WHEEL_RESOURCES <= names:
            raise ValueError(f"wheel is missing bundled resources: {sorted(WHEEL_RESOURCES - names)}")
        if any(not name.startswith(("agent_core/", f"agent_with_llm-{version}.dist-info/")) for name in names):
            raise ValueError("wheel contains files outside the application package")
        metadata = Parser().parsestr(wheel.read(f"agent_with_llm-{version}.dist-info/METADATA").decode())
        if metadata["Name"] != "agent-with-llm" or metadata["Version"] != version:
            raise ValueError("wheel identity does not match project version")
        entrypoints = wheel.read(f"agent_with_llm-{version}.dist-info/entry_points.txt").decode()
        if "polaris = agent_core.cli:main" not in entrypoints:
            raise ValueError("wheel is missing the Polaris console entrypoint")


def write_archives(output: Path, payload: dict[str, bytes]) -> None:
    with zipfile.ZipFile(output / "polaris-installer.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, content in sorted(payload.items()):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(content))
    with (output / "polaris-installer.tar.gz").open("wb") as handle:
        with gzip.GzipFile(filename="", fileobj=handle, mode="wb", mtime=0) as compressed:
            compressed.write(buffer.getvalue())


def assemble_assets(root: Path, output: Path, wheel: Path, constraints: Path, tag: str) -> dict:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]
    if tag != f"v{version}":
        raise ValueError(f"tag {tag!r} does not match project version v{version}")
    validate_wheel(wheel, version)
    output.mkdir(parents=True, exist_ok=True)
    wheel_content = wheel.read_bytes()
    constraints_content = constraints.read_bytes()
    release = {
        "schema": 1,
        "package": project["name"],
        "version": version,
        "tag": tag,
        "wheel": {"filename": wheel.name, "sha256": sha256(wheel_content)},
        "constraints": {"filename": "polaris-constraints.txt", "sha256": sha256(constraints_content)},
    }
    payload = {name: (root / name).read_bytes() for name in PAYLOAD_FILES}
    manifest = json.loads(payload["installer/manifest.json"])
    manifest["release_tag"] = tag
    payload["installer/manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    release_content = (json.dumps(release, indent=2) + "\n").encode()
    payload.update({wheel.name: wheel_content, "polaris-constraints.txt": constraints_content, "release.json": release_content})
    write_archives(output, payload)
    (output / wheel.name).write_bytes(wheel_content)
    (output / "polaris-constraints.txt").write_bytes(constraints_content)
    (output / "release.json").write_bytes(release_content)
    for name, marker, replacement in (
        ("install.ps1", '$ReleaseTag = ""', f'$ReleaseTag = "{tag}"'),
        ("install.sh", 'RELEASE_TAG=""', f'RELEASE_TAG="{tag}"'),
    ):
        content = (root / name).read_text(encoding="utf-8")
        if content.count(marker) != 1:
            raise ValueError(f"{name} must contain exactly one release tag marker")
        (output / name).write_text(content.replace(marker, replacement), encoding="utf-8", newline="\n")
    assets = [wheel.name, "polaris-constraints.txt", "release.json", "polaris-installer.zip", "polaris-installer.tar.gz", "install.ps1", "install.sh"]
    sums = "".join(f"{sha256((output / name).read_bytes())}  {name}\n" for name in sorted(assets))
    (output / "SHA256SUMS").write_text(sums, encoding="utf-8", newline="\n")
    return release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--tag", help="must match v<project.version>")
    # These options also allow offline verification using an already built wheel.
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--constraints", type=Path)
    args = parser.parse_args()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    tag = args.tag or f"v{project['version']}"
    if tag != f"v{project['version']}":
        parser.error("release tag must match the project version")
    uv = shutil.which("uv")
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.wheel is None:
        if uv is None:
            parser.error("uv is required to build a wheel")
        subprocess.run([uv, "build", "--wheel", "--no-build-logs", "--out-dir", str(output)], cwd=ROOT, check=True)
        wheels = list(output.glob(f"agent_with_llm-{project['version']}-*.whl"))
        if len(wheels) != 1:
            parser.error("expected exactly one wheel for this project version")
        wheel = wheels[0]
    else:
        wheel = args.wheel.resolve()
    constraints = args.constraints.resolve() if args.constraints else output / "polaris-constraints.txt"
    if args.constraints is None:
        if uv is None:
            parser.error("uv is required to export locked dependencies")
        subprocess.run([
            uv, "export", "--quiet", "--locked", "--extra", "all", "--no-dev", "--no-default-groups",
            "--no-emit-project", "--no-hashes", "--no-annotate", "--no-header",
            "--output-file", str(constraints),
        ], cwd=ROOT, check=True)
    release = assemble_assets(ROOT, output, wheel, constraints, tag)
    print(json.dumps({"tag": release["tag"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
