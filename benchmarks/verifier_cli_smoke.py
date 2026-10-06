"""Offline public-CLI probes, runnable from a clean verifier source copy.

Uses the fake provider to validate CLI/runtime behavior, not model quality or OS isolation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("functional", "adversarial"))
    kind = parser.parse_args().kind
    source = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="polaris-cli-probe-") as directory:
        root = Path(directory)
        workspace = root / "workspace"
        workspace.mkdir()
        state = root / "state"
        state.mkdir(mode=0o700)
        config = root / "smoke.toml"
        config.write_text('''provider = "fake"
permission = "default"
[sandbox]
enabled = false
[memory]
enabled = false
[codeintel]
enabled = false
[skills]
enabled = false
[capabilities]
mode = "disabled"
[tools.shell.bash]
enabled = false
[tools.shell.powershell]
enabled = false
[hooks]
enabled = false
[compression]
use_llm_summary = false
[verifier]
mode = "off"
check_answer = false
max_repair_attempts = 0
''', encoding="utf-8")
        # Keep only OS/interpreter necessities; do not pass API keys or AGENT overrides.
        allowed = {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP",
                   "TMPDIR", "HOME", "USERPROFILE", "LANG", "LC_ALL", "VIRTUAL_ENV"}
        env = {k: v for k, v in os.environ.items() if k.upper() in allowed}
        env.update(PYTHONPATH=str(source), PYTHONIOENCODING="utf-8", POLARIS_HOME=str(state),
                   AGENT_TRUST_STORE=str(state / "trusted.json"))
        common = ["run", "verifier smoke", "--provider", "fake", "--config", str(config),
                  "--no-memory", "--no-session-persistence", "--quiet", "--no-stream"]
        def invoke(args: list[str]) -> subprocess.CompletedProcess[str]:
            return subprocess.run([sys.executable, "-m", "agent_core", *args], cwd=workspace,
                env=env, capture_output=True, text=True, encoding="utf-8", timeout=45, check=False)

        results = []
        if kind == "functional":
            result = invoke(common)
            assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
            assert "Final answer: verifier smoke" in result.stdout, result.stdout
            assert list((workspace / "runs").glob("*.jsonl")), "missing public run log"
            results.append({"scenario": "plain fake-provider run", "exit_code": result.returncode,
                            "output": result.stdout})
        else:
            malformed = invoke([*common, "--verifier-unknown-option"])
            assert malformed.returncode == 2 and "unrecognized arguments" in malformed.stderr, malformed
            results.append({"scenario": "unknown argument", "exit_code": malformed.returncode,
                            "output": malformed.stderr})
            missing = invoke([*common, "--require-verification"])
            assert missing.returncode == 2, (missing.returncode, missing.stdout, missing.stderr)
            assert "Verification status: unverified" in missing.stdout and "behavioral verifier" in missing.stdout
            logs = [json.loads(line) for path in (workspace / "runs").glob("*.jsonl")
                    for line in path.read_text(encoding="utf-8").splitlines()]
            assert any(row.get("event") == "verifier" and row.get("verdict") == "PARTIAL" for row in logs)
            results.append({"scenario": "required verifier without sandbox", "exit_code": missing.returncode,
                            "output": missing.stdout})
        print(json.dumps({"kind": kind, "checks": results}, ensure_ascii=False))
        print("CLI_PROBE_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
