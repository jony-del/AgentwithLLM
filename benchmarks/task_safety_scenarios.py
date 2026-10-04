"""Offline scripted execution cohort. Measures safety gates, not model coding skill."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_core.codeintel.config import CodeIntelConfig
from agent_core.evaluation import evaluate_logs
from agent_core.models import LLMResult, ToolCall
from agent_core.providers.fake import FakeProvider
from agent_core.react import ReActAgent, ReActConfig
from agent_core.sandbox import SandboxConfig
from agent_core.task_runtime import TaskContract, VerificationCheck
from agent_core.transcript import load_transcript


class Scripted(FakeProvider):
    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)
    def _compute(self, messages):
        return self.responses.pop(0) if self.responses else LLMResult("done", stop_reason="end")


def call(name, arguments):
    return LLMResult("", tool_calls=[ToolCall(name, arguments)])


async def cohort(root: Path) -> dict:
    # Explicit test-only host command execution in disposable workspaces.
    check = VerificationCheck("check", (sys.executable, "-c", "assert open('f.py').read() == 'x=2\\n'"))
    failing = VerificationCheck("check", (sys.executable, "-c", "raise SystemExit(1)"))
    edit = call("edit_file", {"path": "f.py", "old_string": "x=1", "new_string": "x=2"})
    passed = call("run_verification", {"check_id": "check", "argv": list(check.argv)})
    cases = [
        ("verified_review", [edit, passed, LLMResult("fixed"), LLMResult('{"verdict":"passed","findings":[]}')],
         TaskContract("change x to 2", checks=(check,), review_required=True), "completed"),
        ("unchecked_edit", [edit, LLMResult("fixed")], TaskContract("change x to 2"), "unverified"),
        ("failed_check", [edit, call("run_verification", {"check_id": "check", "argv": list(failing.argv)}), LLMResult("fixed")],
         TaskContract("change x to 2", checks=(failing,)), "unverified"),
        ("blocked_review", [edit, passed, LLMResult("fixed"), LLMResult('{"verdict":"blocked","findings":[{"path":"f.py","line":1,"severity":"blocking","message":"Acceptance remains unresolved"}]}')],
         TaskContract("change x to 2", checks=(check,), review_required=True), "unverified"),
        ("repeated_tool_failure", [call("missing_tool", {}) for _ in range(6)], TaskContract("inspect"), "blocked"),
    ]
    outcomes, logs = [], []
    for name, responses, contract, expected in cases:
        workspace = root / "cases" / name
        workspace.mkdir(parents=True)
        (workspace / "f.py").write_bytes(b"x=1\n")
        agent = ReActAgent(Scripted(responses), ReActConfig(
            permission="bypass", project_instructions=False, git_context=False,
            run_dir=str(root / "logs"), session_dir=str(root / "sessions"),
            sandbox=SandboxConfig(enabled=False, allow_unattended_unsandboxed=True),
            codeintel=CodeIntelConfig(enabled=False)), workspace=workspace)
        try:
            result = await agent.run(contract.goal, task_contract=contract)
            events = [json.loads(line) for line in agent.logger.path.read_text(encoding="utf-8").splitlines()]
            final = [event for event in events if event["event"] == "final"][-1]
            assert final["status"] == result.status
            transcript = load_transcript(agent.transcript.path).messages.values()
            pending = set()
            for message in transcript:
                pending.update(call["id"] for call in message.metadata.get("tool_calls", []))
                if message.role == "tool":
                    assert message.metadata["tool_call_id"] in pending
                    pending.remove(message.metadata["tool_call_id"])
            assert not pending
            assert result.status == expected, (name, result.status, expected)
            outcomes.append({"scenario": name, "expected": expected, "actual": result.status,
                             "serialized_final_matches": True, "transcript_tool_pairs_valid": True})
            logs.append(agent.logger.path)
        finally:
            await agent.fire_session_end("benchmark")
            await agent.runtime.close()
            agent.transcript.close()
            agent.logger.close()
    return {"schema_version": 1, "cohort": "synthetic deterministic safety scenarios; no real LLM",
            "scenarios": outcomes, "expected_outcome_match_rate": 1.0, "runtime_metrics": evaluate_logs(logs),
            "limitations": ["Does not measure model coding ability or real task-success improvement",
                            "Host execution explicitly enabled only for disposable fake-provider checks"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="polaris-task-cohort-") as temporary:
        root = Path(temporary).resolve()
        assert root.parent == Path(tempfile.gettempdir()).resolve()
        prior = {key: os.environ.get(key) for key in ("POLARIS_HOME", "AGENT_TRUST_STORE")}
        os.environ["POLARIS_HOME"] = str(root / "private")
        os.environ["AGENT_TRUST_STORE"] = str(root / "trust.json")
        try:
            report = asyncio.run(cohort(root))
        finally:
            for key, value in prior.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scenarios": len(report["scenarios"]), "expected_outcome_match_rate": report["expected_outcome_match_rate"]}))


if __name__ == "__main__":
    main()
