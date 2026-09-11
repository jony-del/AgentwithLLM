"""Child-process entry for the process-level crash matrix (audit §9.1).

Runs one real agent turn in this interpreter and dies via ``os._exit(1)``
immediately after the requested durable boundary is fsync'd — no atexit
handlers, no buffer flushing, no cleanup, exactly like a power loss. The parent
test then performs startup recovery against the same POLARIS_HOME/workspace.

Usage: crash_child.py <crash_point> <polaris_home> <workspace> <session_dir>

Exit codes: 1 = died at the requested boundary (expected); 3 = the run
completed without reaching it (harness bug); anything else = unexpected error.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "tests")):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def main() -> int:
    point_name, polaris_home, workspace, session_dir = sys.argv[1:5]
    # Every environment input arrives via argv/env from the parent: this script
    # never sees tests/conftest.py because it runs as a standalone interpreter.
    os.environ["POLARIS_HOME"] = polaris_home
    os.environ.setdefault("AGENT_SANDBOX_ALLOW_UNATTENDED", "1")
    os.chdir(workspace)

    import crashkit

    env = crashkit.CrashWorkspace(
        root=Path(session_dir).parent,
        workspace=Path(workspace).resolve(),
    )
    agent, _tool, _batches = crashkit.build_crash_agent(env)

    def die() -> None:
        os.write(2, f"crash point reached: {point_name}\n".encode())
        os._exit(1)

    crashkit.CrashInjector(crashkit.CrashPoint(point_name), crash=die).install()
    result = asyncio.run(agent.run(crashkit.PROMPT))
    agent.logger.close()
    print(f"run completed without crashing: {result.answer!r}", flush=True)
    return 3


if __name__ == "__main__":
    sys.exit(main())
