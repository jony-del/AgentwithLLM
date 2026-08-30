#!/usr/bin/env python3
"""Small, dependency-free OCI image/runtime probe used by Polaris."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import socket
import sys
import uuid


TOOLS = {
    "bash": "/usr/bin/bash",
    "pwsh": "/usr/local/bin/pwsh",
    "python": "/usr/local/bin/python",
    "node": "/usr/local/bin/node",
    "npm": "/usr/local/bin/npm",
    "npx": "/usr/local/bin/npx",
    "pyright-langserver": "/usr/local/bin/pyright-langserver",
}


def manifest() -> int:
    if any(not Path(path).is_file() for path in TOOLS.values()):
        return 3
    print(json.dumps({
        "protocol_version": 1,
        "guest_os": "linux",
        "architecture": platform.machine(),
        "tools": TOOLS,
    }, sort_keys=True))
    return 0


def mount(path: str) -> int:
    root = Path(path).resolve()
    if not root.is_dir() or Path.cwd().resolve() != root:
        return 4
    canary = root / f".polaris-sandbox-canary-{uuid.uuid4().hex}"
    payload = "读写-canary\n"
    try:
        canary.write_text(payload, encoding="utf-8")
        if canary.read_text(encoding="utf-8") != payload:
            return 5
    finally:
        canary.unlink(missing_ok=True)
    return 0


def network_denied() -> int:
    # A successful connection proves that --network none was not enforced.
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=1):
            return 6
    except OSError:
        return 0


def security() -> int:
    if os.geteuid() == 0:
        return 7
    # This image directory is owned by the sandbox uid. A write succeeds without
    # ``--read-only`` and fails only when the root filesystem is actually immutable.
    target = Path("/opt/polaris/rootfs-canary/write-test")
    try:
        target.write_text("unexpected", encoding="utf-8")
    except OSError:
        return 0
    finally:
        try:
            target.unlink()
        except OSError:
            pass
    return 8


def main(argv: list[str]) -> int:
    if argv == ["manifest"]:
        return manifest()
    if len(argv) == 2 and argv[0] == "mount":
        return mount(argv[1])
    if argv == ["network-denied"]:
        return network_denied()
    if argv == ["security"]:
        return security()
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
