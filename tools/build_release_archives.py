"""Build release archives from tracked files while preserving injected lock files."""

from __future__ import annotations

import gzip
import io
from pathlib import Path
import subprocess
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def tracked_files() -> list[Path]:
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [ROOT / item.decode("utf-8") for item in raw.split(b"\0") if item]


def main() -> None:
    files = tracked_files()
    epoch = int(subprocess.check_output(
        ["git", "show", "-s", "--format=%ct", "HEAD"], cwd=ROOT, text=True
    ).strip())
    with zipfile.ZipFile(ROOT / "polaris-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(relative)
            info.date_time = (2020, 1, 1, 0, 0, 0)
            info.external_attr = (0o755 if path.suffix in {".sh", ".py"} else 0o644) << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            info = archive.gettarinfo(str(path), arcname=relative)
            info.mtime = epoch
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as handle:
                archive.addfile(info, handle)
    with (ROOT / "polaris-source.tar.gz").open("wb") as output:
        with gzip.GzipFile(fileobj=output, mode="wb", mtime=epoch) as compressed:
            compressed.write(buffer.getvalue())


if __name__ == "__main__":
    main()
