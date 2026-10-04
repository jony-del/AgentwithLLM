from dataclasses import replace
import hashlib
import json
import sys

import pytest

from agent_core.checkpoints import CheckpointStore
from agent_core.models import LLMResult, ToolCall, TokenUsage
from agent_core.providers.base import ProviderConfig
from agent_core.providers.fake import FakeProvider
from agent_core.review import review_task
from agent_core.session import SessionContext
from agent_core.task_runtime import TaskContract, TaskRun, VerificationCheck, VerificationEvidence, capture_revision, verify_completion
from agent_core.tools.transaction import JournalStorage
from tests.test_task_runtime import _scripted_agent


class Reviewer(FakeProvider):
    def __init__(self, response, mutate=None):
        super().__init__()
        self.response, self.mutate, self.requests = response, mutate, []

    async def complete(self, messages, tools, config, **kwargs):
        self.requests.append((messages, tools, config))
        if self.mutate:
            self.mutate()
        return self.response


def setup_review(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "f.py").write_text("x=1\n")
    task = TaskRun(TaskContract("fix", review_required=True), capture_revision(root))
    session = SessionContext(workspace=root, task_run=task,
                             checkpoint_store=CheckpointStore(JournalStorage.local(tmp_path / "state", workspace=root)))
    task.baseline_checkpoint = session.checkpoint_store.capture(task, task.baseline).id
    (root / "f.py").write_text("x=2\n")
    return root, session


async def test_reviewer_has_fresh_context_no_tools_and_version_bound_proof(tmp_path):
    root, session = setup_review(tmp_path)
    provider = Reviewer(LLMResult('{"verdict":"passed","findings":[]}'))
    record = await review_task(session, provider, ProviderConfig(), capture_revision(root))
    assert record["status"] == "passed"
    messages, tools, config = provider.requests[0]
    assert len(messages) == 2 and tools == [] and not config.stream
    assert "x=2" in messages[1].content and "x=1" in messages[1].content
    check = VerificationCheck("test", ("python", "-m", "pytest"))
    session.task_run.contract = replace(session.task_run.contract, checks=(check,))
    current = capture_revision(root)
    session.task_run.evidence = [VerificationEvidence("test", current.digest, current.workspace,
        hashlib.sha256(json.dumps(check.argv).encode()).hexdigest(), 0, "completed")]
    assert verify_completion(session.task_run, current).status == "unverified"  # contract changed
    await review_task(session, provider, ProviderConfig(), current)
    assert verify_completion(session.task_run, current).status == "completed"
    (root / "f.py").write_text("x=3\n")
    assert verify_completion(session.task_run, capture_revision(root)).status == "unverified"


async def test_reviewer_usage_is_reported_to_runtime_accounting(tmp_path):
    root, session = setup_review(tmp_path)
    recorded = []
    session.record_aux_usage = recorded.append
    usage = TokenUsage(input_tokens=12, output_tokens=3)
    await review_task(session, Reviewer(LLMResult('{"verdict":"passed","findings":[]}', usage=usage)),
                      ProviderConfig(), capture_revision(root))
    assert recorded == [usage]


@pytest.mark.parametrize("result", [
    LLMResult("invalid json"), LLMResult('{"verdict":"passed","findings":[]}', termination_proven=False),
    LLMResult('{"verdict":"passed","findings":[]}', stop_reason="max_tokens"),
    LLMResult('{"verdict":"passed","findings":[]}', tool_calls=[ToolCall("write_text_file", {})]),
    LLMResult('{"verdict":"passed","findings":[{"path":"foreign","line":1,"severity":"warning","message":"bad"}]}'),
])
async def test_incomplete_or_malformed_review_cannot_pass(tmp_path, result):
    root, session = setup_review(tmp_path)
    record = await review_task(session, Reviewer(result), ProviderConfig(), capture_revision(root))
    assert record["status"] == "incomplete"


async def test_syntax_and_concurrent_changes_override_model_pass(tmp_path):
    root, session = setup_review(tmp_path)
    (root / "f.py").write_text("def broken(\n")
    provider = Reviewer(LLMResult('{"verdict":"passed","findings":[]}'))
    record = await review_task(session, provider, ProviderConfig(), capture_revision(root))
    assert record["status"] == "blocked" and record["findings"][0]["severity"] == "blocking"
    provider.mutate = lambda: (root / "f.py").write_text("x=9\n")
    record = await review_task(session, provider, ProviderConfig(), capture_revision(root))
    assert record["status"] == "incomplete"


async def test_valid_utf8_bom_does_not_fail_static_review(tmp_path):
    root, session = setup_review(tmp_path)
    (root / "f.py").write_bytes(b"\xef\xbb\xbfx=2\r\n")
    record = await review_task(session, Reviewer(LLMResult('{"verdict":"passed","findings":[]}')),
                               ProviderConfig(), capture_revision(root))
    assert record["status"] == "passed"


async def test_review_findings_survive_context_projection_and_respect_output_budget(tmp_path):
    root, session = setup_review(tmp_path)
    provider = Reviewer(LLMResult(json.dumps({"verdict": "blocked", "findings": [
        {"path": "f.py", "line": 1, "severity": "blocking", "message": "Fix the boundary condition"}]})))
    await review_task(session, provider, ProviderConfig(max_tokens=128), capture_revision(root))
    projection = json.loads(session.task_run.context())
    assert projection["recent_review"]["findings"][0]["message"] == "Fix the boundary condition"
    assert projection["review_required"] is True
    assert provider.requests[0][2].max_tokens == 128


async def test_read_denial_never_reaches_reviewer_and_budget_does_not_truncate_to_success(tmp_path):
    from agent_core.permission_rules import RuleSet
    root, session = setup_review(tmp_path)
    provider = Reviewer(LLMResult('{"verdict":"passed","findings":[]}'))
    session.code_permission_rules = lambda: RuleSet.from_lists(deny=["read_text_file(f.py)"])
    record = await review_task(session, provider, ProviderConfig(), capture_revision(root))
    assert record["status"] == "incomplete" and not provider.requests
    session.code_permission_rules = None
    (root / "f.py").write_text("x=1\n" * 40_000)
    record = await review_task(session, provider, ProviderConfig(), capture_revision(root))
    assert record["status"] == "incomplete" and not provider.requests


async def test_react_required_review_runs_after_checks_and_can_complete(tmp_path):
    (tmp_path / "f.py").write_text("x=1\n")
    check = VerificationCheck("test", (sys.executable, "-c", "assert open('f.py').read() == 'x=2\\n'"))
    agent = _scripted_agent(tmp_path, [
        LLMResult("", tool_calls=[ToolCall("edit_file", {"path": "f.py", "old_string": "x=1", "new_string": "x=2"})]),
        LLMResult("", tool_calls=[ToolCall("run_verification", {"check_id": "test", "argv": list(check.argv)})]),
        LLMResult("fixed"), LLMResult('{"verdict":"passed","findings":[]}'),
    ])
    try:
        result = await agent.run("fix", task_contract=TaskContract("fix", checks=(check,), review_required=True))
        assert result.status == "completed", result.answer
        assert agent.session.task_store.load().reviews[-1]["status"] == "passed"
    finally:
        await agent.fire_session_end("test")
        await agent.runtime.close()
