"""Verifier proofs use real commands; sandbox tests inspect the wrapping contract."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

from agent_core.hook_adapters import AgentHookAdapter, PromptHookAdapter
from agent_core.hooks import ExternalHookSpec, HookContext, HookEvent, HookOutcome
from agent_core.models import LLMResult, Message, ToolCall, ToolResult
from agent_core.permission_rules import RuleSet
from agent_core.providers.base import ProviderConfig
from agent_core.storage import JSONLRunLogger
from agent_core.task_runtime import TaskContract, TaskRun, TaskStore, capture_revision, verify_completion
from agent_core.tools.transaction import JournalStorage
from agent_core.tools.verifier import probe_executor
from agent_core.verifier import (VerifierConfig, completion_issues, matching, observe, requirements,
                                 review_answer, verify_behavior)
from tests.test_independent_review import Reviewer
from tests.test_task_runtime import _scripted_agent


PASS = '{"verdict":"PASS","findings":[]}'


class WrappingSandbox:
    """Test boundary only: real Python commands, deliberately no OS isolation claim."""
    def __init__(self):
        self.invocations = []

    def is_enabled(self):
        return True

    def reset(self):
        pass

    def wrap_invocation(self, invocation):
        self.invocations.append(invocation)
        return list(invocation.host_argv), False


def probe(kind, code, expected="verified", expected_code=0):
    return ToolCall("verifier_probe", {"argv": [sys.executable, "-c", code], "kind": kind,
        "criterion": "public divide API handles " + kind + " input", "expected_exit_code": expected_code,
        "expected_output": expected, "timeout": 5})


def probes():
    return [probe("functional", "from calc import divide; assert divide(6, 2) == 3; print('verified')"),
            probe("adversarial", "from calc import divide\ntry: divide(1, 0)\nexcept ValueError: print('verified')\nelse: raise AssertionError('zero divisor accepted')")]


def setup_agent(tmp_path, responses):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "calc.py").write_text("def divide(a, b):\n    if b == 0: raise ValueError('zero divisor')\n    return a / b\n", encoding="utf-8")
    agent = _scripted_agent(root, responses)
    agent.sandbox = WrappingSandbox()
    current = capture_revision(root)
    agent.session.task_run = TaskRun(TaskContract("verify the divide API", verification_required=True), current)
    return agent, current


async def close(agent):
    await agent.fire_session_end("test")
    await agent.runtime.close()
    agent.logger.close()


async def test_real_functional_and_boundary_probes_in_copy(tmp_path):
    agent, current = setup_agent(tmp_path, [LLMResult("", tool_calls=probes()), LLMResult(PASS)])
    try:
        record = await verify_behavior(agent.session, agent.provider, ProviderConfig(), VerifierConfig(), current,
            lambda copy, revision, output: probe_executor(agent, copy, revision, output))
        assert record["status"] == "passed", record
        assert {p["kind"] for p in record["probes"]} == {"functional", "adversarial"}
        assert all(Path(p["output_path"]).exists() and "verified" in p["output"] for p in record["probes"])
        assert verify_completion(agent.session.task_run, current).status == "completed"
        for invocation in agent.sandbox.invocations:
            assert invocation.scope.workspace != agent.session.workspace
            assert agent.session.workspace in invocation.scope.read_only_roots
            assert invocation.scope.network == "deny"
            assert not invocation.scope.workspace.exists()  # cleaned copy
        assert capture_revision(agent.session.workspace).digest == current.digest
    finally:
        await close(agent)


@pytest.mark.parametrize("response", [LLMResult(PASS), LLMResult("VERDICT: PASS"),
    LLMResult('{"verdict":["PASS"],"findings":[]}'),
    LLMResult(PASS, termination_proven=False), LLMResult(PASS, stop_reason="length"),
    LLMResult("", tool_calls=[ToolCall("write_text_file", {"path": "calc.py", "content": "bad"})])])
async def test_fabricated_or_incomplete_pass_never_certifies(tmp_path, response):
    agent, current = setup_agent(tmp_path, [response])
    try:
        result = await verify_behavior(agent.session, agent.provider, ProviderConfig(), VerifierConfig(), current,
            lambda copy, revision, output: probe_executor(agent, copy, revision, output))
        assert result["status"] == "incomplete"
        assert verify_completion(agent.session.task_run, current).status == "unverified"
    finally:
        await close(agent)


async def test_failed_probe_cannot_be_overridden_by_pass(tmp_path):
    calls = probes()
    calls[0] = probe("functional", "raise AssertionError('broken public behavior')")
    agent, current = setup_agent(tmp_path, [LLMResult("", tool_calls=calls), LLMResult(PASS)])
    try:
        result = await verify_behavior(agent.session, agent.provider, ProviderConfig(), VerifierConfig(), current,
            lambda copy, revision, output: probe_executor(agent, copy, revision, output))
        assert result["status"] == "incomplete" and "failed probe" in result["reason"]
        assert result["probes"][0]["exit_code"] != 0
    finally:
        await close(agent)


async def test_source_mutation_in_copy_invalidates_evidence(tmp_path):
    calls = [probe("functional", "open('calc.py','w').write('broken'); print('verified')")]
    agent, current = setup_agent(tmp_path, [LLMResult("", tool_calls=calls), LLMResult(PASS)])
    try:
        result = await verify_behavior(agent.session, agent.provider, ProviderConfig(), VerifierConfig(), current,
            lambda copy, revision, output: probe_executor(agent, copy, revision, output))
        assert result["status"] != "passed"
        assert result["probes"][0]["state"] == "revision_changed"
        assert capture_revision(agent.session.workspace).digest == current.digest
    finally:
        await close(agent)


async def test_probe_permission_denial_is_not_a_pass(tmp_path):
    agent, current = setup_agent(tmp_path, [LLMResult("", tool_calls=probes()), LLMResult(PASS)])
    agent.permissions._base_rules = RuleSet.from_lists(deny=["verifier_probe"])
    try:
        result = await verify_behavior(agent.session, agent.provider, ProviderConfig(), VerifierConfig(), current,
            lambda copy, revision, output: probe_executor(agent, copy, revision, output))
        assert result["status"] == "incomplete" and not result["probes"]
        assert not agent.sandbox.invocations
    finally:
        await close(agent)


async def test_read_denial_prevents_source_copy_and_model_request(tmp_path):
    agent, current = setup_agent(tmp_path, [LLMResult(PASS)])
    provider = Reviewer(LLMResult(PASS))
    agent.session.code_permission_rules = lambda: RuleSet.from_lists(deny=["read_text_file(calc.py)"])
    try:
        result = await verify_behavior(agent.session, provider, ProviderConfig(), VerifierConfig(), current,
            lambda copy, revision, output: probe_executor(agent, copy, revision, output))
        assert result["status"] == "incomplete" and not provider.requests
    finally:
        await close(agent)


async def test_deadline_kills_and_awaits_probe_process(tmp_path):
    agent, current = setup_agent(tmp_path, [LLMResult("", tool_calls=[
        probe("functional", "import time; time.sleep(30); print('verified')")])])
    try:
        result = await verify_behavior(agent.session, agent.provider, ProviderConfig(), VerifierConfig(timeout=0.25), current,
            lambda copy, revision, output: probe_executor(agent, copy, revision, output))
        assert result["status"] == "incomplete"
        assert not agent.process_supervisor.running()
    finally:
        await close(agent)


async def test_answer_review_has_clean_context_no_tools_and_invalidates_on_new_evidence(tmp_path):
    agent, current = setup_agent(tmp_path, [])
    provider = Reviewer(LLMResult(PASS))
    task = agent.session.task_run
    task.contract = replace(task.contract, verification_required=False)
    observe(task, ToolCall("read_text_file"), ToolResult("read_text_file", "divide validates zero"), current)
    try:
        result = await review_answer(agent.session, provider, ProviderConfig(max_tokens=128),
                                     VerifierConfig(mode="auto"), current, "divide validates zero")
        assert result["status"] == "passed"
        messages, tools, config = provider.requests[0]
        assert len(messages) == 2 and tools == [] and not config.stream and config.max_tokens == 128
        assert matching(task.answer_reviews, task, current, "divide validates zero")
        assert not matching(task.answer_reviews, task, current, "tests passed")
        observe(task, ToolCall("run_tests"), ToolResult("run_tests", "FAIL", ok=False), current)
        assert not matching(task.answer_reviews, task, current, "divide validates zero")
        (agent.session.workspace / "calc.py").write_text("different")
        assert not matching(task.answer_reviews, task, capture_revision(agent.session.workspace), "divide validates zero")
    finally:
        await close(agent)


@pytest.mark.parametrize("response", [LLMResult("bad"), LLMResult(PASS, termination_proven=False),
    LLMResult('{"verdict":{"value":"PASS"},"findings":[]}'),
    LLMResult('{"verdict":"FAIL","findings":[]}'),
    LLMResult('{"verdict":"FAIL","findings":[{"claim":"tests passed","reason":"fabricated", "evidence_ids":["invented"]}]}')])
async def test_invalid_answer_verdict_or_evidence_fails_closed(tmp_path, response):
    agent, current = setup_agent(tmp_path, [])
    try:
        result = await review_answer(agent.session, Reviewer(response), ProviderConfig(), VerifierConfig(), current, "done")
        assert result["status"] == "incomplete"
    finally:
        await close(agent)


async def test_auto_answer_rejection_corrects_main_answer(tmp_path):
    rejected = json.dumps({"verdict": "FAIL", "findings": [{"claim": "tests passed",
        "reason": "Only a file was read; no tests ran", "evidence_ids": []}]})
    (tmp_path / "f.py").write_text("x=1")
    agent = _scripted_agent(tmp_path, [
        LLMResult("", tool_calls=[ToolCall("read_text_file", {"path": "f.py"})]),
        LLMResult("All tests passed"), LLMResult(rejected),
        LLMResult("The file defines x=1; tests were not run"), LLMResult(PASS),
    ])
    agent.config.verifier = VerifierConfig(mode="auto")
    try:
        result = await agent.run("Explain f.py")
        assert result.status == "completed" and "tests were not run" in result.answer
        records = agent.session.task_run.answer_reviews
        assert [r["status"] for r in records] == ["blocked", "passed"]
        assert records[0]["failure_class"] == "verdict"
        repair = next(m for m in result.messages if m.metadata.get("completion_verifier"))
        assert "[answer review] tests passed: Only a file was read; no tests ran" in repair.content
    finally:
        await close(agent)


async def test_required_verifier_without_sandbox_reports_partial(tmp_path):
    agent = _scripted_agent(tmp_path, [LLMResult("done")])
    agent.config.verifier = VerifierConfig(check_answer=False, max_repair_attempts=0)
    try:
        result = await agent.run("Verify it", require_verification=True)
        assert result.status == "unverified"
        assert agent.session.task_run.verifier_runs[-1]["verdict"] == "PARTIAL"
        assert "sandbox" in agent.session.task_run.verifier_runs[-1]["reason"]
    finally:
        await close(agent)


@pytest.mark.parametrize("response,blocked", [
    (LLMResult('{"ok":true}'), False), (LLMResult('{"ok":false,"reason":"missing actual test"}'), True),
    (LLMResult("not JSON"), True), (LLMResult('{"ok":"true"}'), True),
    (LLMResult('{"ok":true}', termination_proven=False), True),
    (LLMResult('{"ok":true}', stop_reason="length"), True),
])
async def test_prompt_gate_structured_results_and_failure(tmp_path, response, blocked):
    provider = Reviewer(response)
    adapter = PromptHookAdapter(ExternalHookSpec("Stop", "prompt", prompt="Check task", decision_mode="gate"),
                                JSONLRunLogger(tmp_path), provider, ProviderConfig())
    result = await adapter.on_stop(HookContext(HookEvent.STOP, [Message("user", "task")]))
    assert result.block is blocked
    assert provider.requests[0][1] == [] and not provider.requests[0][2].stream


async def test_prompt_gate_timeout_never_allows(tmp_path):
    class Slow(Reviewer):
        async def complete(self, *args, **kwargs):
            await asyncio.sleep(10)
    adapter = PromptHookAdapter(ExternalHookSpec("Stop", "prompt", prompt="Check task", decision_mode="gate", timeout=0.01),
                                JSONLRunLogger(tmp_path), Slow(LLMResult('{"ok":true}')), ProviderConfig())
    assert (await adapter.on_stop(HookContext(HookEvent.STOP, []))).block


async def test_agent_gate_uses_structured_result(tmp_path):
    async def factory(task, preset, model):
        assert preset == "hook" and '"ok":false' in task
        return '{"ok":false,"reason":"plan has missing behavior"}'
    adapter = AgentHookAdapter(ExternalHookSpec("Stop", "agent", prompt="Check plan", decision_mode="gate"),
                               JSONLRunLogger(tmp_path), factory)
    outcome = await adapter.on_stop(HookContext(HookEvent.STOP, []))
    assert outcome.block and "missing behavior" in outcome.reason


async def test_exhausted_stop_gate_cannot_claim_completed(tmp_path):
    class Reject:
        async def on_stop(self, ctx):
            return HookOutcome(block=True, reason="required validation missing")
    agent = _scripted_agent(tmp_path, [LLMResult("done")])
    agent.config.max_stop_blocks = 1
    agent.hooks.stop_hooks.append(Reject())
    try:
        result = await agent.run("Say hello")
        assert result.status == "unverified" and "stop gate" in result.answer
    finally:
        await close(agent)


async def test_stop_advisory_is_preserved_and_logged_without_loop(tmp_path):
    class Advisory:
        async def on_stop(self, ctx):
            return HookOutcome(additional_context="Consider documenting limitations")
    agent = _scripted_agent(tmp_path, [LLMResult("hello")])
    agent.hooks.stop_hooks.append(Advisory())
    try:
        result = await agent.run("Say hello")
        assert result.status == "completed" and result.steps == 1
        assert any(m.metadata.get("stop_hook") == "advisory" for m in result.messages)
        assert "hook_advisory" in Path(agent.logger.path).read_text(encoding="utf-8")
    finally:
        await close(agent)


def test_verifier_config_and_trust(tmp_path, monkeypatch):
    from agent_core.config import resolve_verifier_config, resolve_hooks_config
    from agent_core.trust import strip_widening
    path = tmp_path / "agent.toml"
    path.write_text('[verifier]\nmode="auto"\nmax_probes=4\n[[hooks.external]]\nevent="Stop"\ntype="prompt"\nprompt="check"\ndecision_mode="gate"\n')
    assert resolve_verifier_config(path).mode == "auto"
    assert resolve_verifier_config(path).max_probes == 4
    spec = resolve_hooks_config(path).external[0]
    assert spec.decision_mode == "gate" and spec.fail_mode == "closed"
    monkeypatch.setenv("AGENT_VERIFIER_MODE", "required")
    assert resolve_verifier_config(path).mode == "required"
    assert strip_widening({"verifier": {"mode": "off", "check_answer": False, "model": "foreign"}})["verifier"] == {}
    with pytest.raises(ValueError):
        VerifierConfig.from_dict({"mode": "typo"})
    with pytest.raises(ValueError):
        VerifierConfig.from_dict({"mode": ["off"]})
    with pytest.raises(ValueError):
        ExternalHookSpec("PostSampling", "prompt", prompt="x", decision_mode="gate")


def test_state_v3_and_legacy_records(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    task = TaskRun(TaskContract("verify"), capture_revision(root))
    task.answer_reviews = [{"status": "incomplete", "reason": "missing evidence"}]
    store = TaskStore(JournalStorage.local(tmp_path / "state", workspace=root))
    store.save(task)
    assert json.loads(store.path.read_text())["v"] == 3
    assert store.load().answer_reviews == task.answer_reviews
    record = json.loads(store.path.read_text())
    record["v"] = 2
    for field in ("answer_reviews", "verifier_runs", "observations"):
        record["task"].pop(field)
    record["task"]["contract"].pop("verification_required")
    store.path.write_text(json.dumps(record))
    assert store.load().answer_reviews == []
    assert not store.load().contract.verification_required


def test_completion_requires_bound_behavior_and_answer(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    task = TaskRun(TaskContract("verify", verification_required=True), capture_revision(root))
    assert verify_completion(task, task.baseline).status == "unverified"
    assert completion_issues(task, task.baseline, VerifierConfig(), "done")


def test_project_verifier_skill_is_bundled():
    from agent_core.skills.loader import load_skills
    skills = load_skills([Path(__file__).parents[1] / "agent_core" / "skills" / "bundled"])
    assert any(s.name == "init-verifiers" for s in skills)


@pytest.mark.parametrize("manual", [False, True])
async def test_completion_integrates_real_probes_and_answer_review(tmp_path, manual):
    responses = ([LLMResult("", tool_calls=[ToolCall("run_verifier", {})])] if manual else
                 [LLMResult("The divide API handles zero divisors")])
    responses += [LLMResult("", tool_calls=probes()), LLMResult(PASS)]
    if manual:
        responses += [LLMResult("The divide API handles zero divisors")]
    responses += [LLMResult(PASS)]
    agent, _ = setup_agent(tmp_path, responses)
    agent.config.verifier = VerifierConfig(mode="auto" if manual else "required")
    try:
        result = await agent.run("Verify the divide API")
        assert result.status == "completed", result.answer
        task = agent.session.task_run
        assert len(task.verifier_runs) == len(task.answer_reviews) == 1
        assert task.verifier_runs[0]["status"] == task.answer_reviews[0]["status"] == "passed"
        assert len(task.verifier_runs[0]["probes"]) == 2
    finally:
        await close(agent)


async def test_failed_manual_verification_blocks_even_without_source_changes(tmp_path):
    agent, _ = setup_agent(tmp_path, [LLMResult("", tool_calls=[ToolCall("run_verifier", {})]),
                                    LLMResult(PASS), LLMResult("done")])
    agent.config.verifier = VerifierConfig(mode="off", check_answer=False, max_repair_attempts=0)
    try:
        result = await agent.run("Run the verifier")
        assert result.status == "unverified"
        assert agent.session.task_run.verifier_runs[-1]["status"] == "incomplete"
    finally:
        await close(agent)


@pytest.mark.parametrize("stalled", [False, True])
async def test_stop_contract_change_invalidates_completion_without_source_change(tmp_path, stalled):
    agent = _scripted_agent(tmp_path, [LLMResult("done")])
    agent.config.verifier = VerifierConfig(check_answer=False, max_repair_attempts=0)
    class RequireVerification:
        async def on_stop(self, ctx):
            task = agent.session.task_run
            task.contract = replace(task.contract, verification_required=True)
            if stalled:
                for _ in range(3):
                    task.record_failure("broken", {}, "ToolFailed", task.baseline.digest)
            return HookOutcome()
    agent.hooks.stop_hooks.append(RequireVerification())
    try:
        result = await agent.run("Say hello")
        assert result.status == ("blocked" if stalled else "unverified") and "behavioral verifier" in result.answer
    finally:
        await close(agent)


async def test_stale_file_read_is_not_rebound_to_new_source_version(tmp_path):
    agent, old = setup_agent(tmp_path, [])
    task = agent.session.task_run
    (agent.session.workspace / "calc.py").write_text("def divide(a,b): return -1")
    current = capture_revision(agent.session.workspace)
    observe(task, ToolCall("read_text_file", {"path": "calc.py"}),
        ToolResult("read_text_file", "old divide implementation", metadata={
            "file_version": {"path": "calc.py", "sha256": old.files["calc.py"]}}), current)
    provider = Reviewer(LLMResult(PASS))
    try:
        await review_answer(agent.session, provider, ProviderConfig(), VerifierConfig(), current, "done")
        assert task.observations[-1]["path"] == "calc.py"
        assert not json.loads(provider.requests[0][0][1].content)["observations"]
    finally:
        await close(agent)


@pytest.mark.parametrize("type", ["prompt", "agent"])
def test_missing_gate_dependency_never_silently_drops_check(tmp_path, type):
    from agent_core.hook_adapters import build_external_adapter
    with pytest.raises(RuntimeError):
        build_external_adapter(ExternalHookSpec("Stop", type, prompt="check", decision_mode="gate"),
                               logger=JSONLRunLogger(tmp_path))


def test_malformed_configured_gate_is_an_error(tmp_path):
    from agent_core.config import resolve_hooks_config
    config = tmp_path / "agent.toml"
    config.write_text('[[hooks.external]]\nevent="Stop"\ntype="prompt"\ndecision_mode="gate"\n')
    with pytest.raises(ValueError, match="requires prompt"):
        resolve_hooks_config(config)


@pytest.mark.parametrize("denied", [False, True])
async def test_browser_probes_use_connected_mcp_and_original_permission_gate(tmp_path, denied):
    from types import SimpleNamespace
    from agent_core.mcp.adapter import MCPTool
    from agent_core.mcp.config import MCPServerConfig
    called = []
    class Browser:
        def call_tool(self, server, name, arguments, timeout, **kwargs):
            called.append((server, name, arguments))
            return SimpleNamespace(content=[SimpleNamespace(text="button is disabled")], isError=False)
    calls = [ToolCall("verifier_browser_probe", {"tool": "playwright__browser_snapshot", "arguments": {},
        "kind": kind, "criterion": "invalid input disables submit", "expected_output": "button is disabled"})
        for kind in ("functional", "adversarial")]
    agent, current = setup_agent(tmp_path, [LLMResult("", tool_calls=calls), LLMResult(PASS)])
    tool = MCPTool(Browser(), MCPServerConfig(name="playwright", risk="read"),
                   SimpleNamespace(name="browser_snapshot", description="Inspect browser", inputSchema={"type": "object"}))
    agent.registry.register(tool)
    if denied:
        agent.permissions._base_rules = RuleSet.from_lists(deny=[tool.name])
    try:
        record = await verify_behavior(agent.session, agent.provider, ProviderConfig(), VerifierConfig(), current,
            lambda copy, revision, output: probe_executor(agent, copy, revision, output))
        assert (record["status"] == "passed") is not denied
        assert bool(called) is not denied
        assert all(p["browser_tool"] == tool.name for p in record["probes"])
        assert all(p["ok"] is not denied for p in record["probes"])
    finally:
        await close(agent)


class Recorder:
    """Scripted provider that records every request payload."""
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    async def complete(self, messages, tools, config, **kwargs):
        self.requests.append(messages)
        return self.responses.pop(0)


async def test_environment_failure_skips_repair_loop(tmp_path):
    agent = _scripted_agent(tmp_path, [LLMResult("done")])
    agent.config.verifier = VerifierConfig(check_answer=False)  # max_repair_attempts defaults to 2
    try:
        result = await agent.run("Verify it", require_verification=True)
        assert result.status == "unverified"
        task = agent.session.task_run
        assert len(task.verifier_runs) == 1
        assert task.verifier_runs[-1]["failure_class"] == "environment"
        assert not any(m.metadata.get("completion_verifier") for m in result.messages)
    finally:
        await close(agent)


async def test_probe_budget_failure_is_budget_class(tmp_path):
    calls = [probe("functional", "print('verified')") for _ in range(3)]
    agent, current = setup_agent(tmp_path, [LLMResult("", tool_calls=calls)])
    try:
        result = await verify_behavior(agent.session, agent.provider, ProviderConfig(),
                                       VerifierConfig(max_probes=2), current,
                                       lambda copy, revision, output: probe_executor(agent, copy, revision, output))
        assert result["status"] == "incomplete" and result["failure_class"] == "budget"
    finally:
        await close(agent)


async def test_malformed_verdict_failure_is_transient_and_repairable(tmp_path):
    from agent_core.verifier import repair_worthwhile
    agent, current = setup_agent(tmp_path, [LLMResult("VERDICT: PASS")])
    try:
        result = await verify_behavior(agent.session, agent.provider, ProviderConfig(), VerifierConfig(), current,
            lambda copy, revision, output: probe_executor(agent, copy, revision, output))
        assert result["status"] == "incomplete" and result["failure_class"] == "transient"
        issues = completion_issues(agent.session.task_run, current, VerifierConfig(mode="required"), "done")
        assert issues[0].startswith("[transient] behavioral verifier") and repair_worthwhile(issues)
        environment = ("[environment] behavioral verifier has not passed on the final revision: no sandbox",)
        assert not repair_worthwhile(environment)
        mixed = environment + ("plan has incomplete steps",)
        assert repair_worthwhile(mixed)
    finally:
        await close(agent)


async def test_previous_attempts_injected_on_retry(tmp_path):
    agent, current = setup_agent(tmp_path, [])
    fail = json.dumps({"verdict": "FAIL", "findings": [{"claim": "zero divisor",
        "reason": "zero divisor accepted", "evidence_ids": []}]})
    provider = Recorder([LLMResult("", tool_calls=probes()), LLMResult(fail),
                         LLMResult("", tool_calls=probes()), LLMResult(PASS)])
    try:
        first = await verify_behavior(agent.session, provider, ProviderConfig(), VerifierConfig(), current,
            lambda copy, revision, output: probe_executor(agent, copy, revision, output))
        assert first["status"] == "blocked" and first["failure_class"] == "verdict"
        second = await verify_behavior(agent.session, provider, ProviderConfig(), VerifierConfig(), current,
            lambda copy, revision, output: probe_executor(agent, copy, revision, output))
        assert second["status"] == "passed"
        payload = json.loads(provider.requests[2][1].content)
        previous = payload["previous_attempts_untrusted"]
        assert previous and previous[-1]["verdict"] == "FAIL"
        assert previous[-1]["findings"][0]["reason"] == "zero divisor accepted"
    finally:
        await close(agent)


async def test_stale_observations_available_to_answer_review(tmp_path):
    agent, current = setup_agent(tmp_path, [])
    task = agent.session.task_run
    task.contract = replace(task.contract, verification_required=False)
    observe(task, ToolCall("run_tests"), ToolResult("run_tests", "3 passed"), current)
    (agent.session.workspace / "calc.py").write_text("def divide(a, b):\n    return a / b\n", encoding="utf-8")
    later = capture_revision(agent.session.workspace)
    provider = Reviewer(LLMResult(PASS))
    try:
        await review_answer(agent.session, provider, ProviderConfig(), VerifierConfig(), later, "tests passed")
        payload = json.loads(provider.requests[0][0][1].content)
        assert not payload["observations"]
        stale = payload["stale_observations"]
        assert len(stale) == 1 and stale[0]["stale"] is True and "3 passed" in stale[0]["output"]
    finally:
        await close(agent)


def test_min_changed_files_threshold_and_trust(tmp_path):
    from agent_core.trust import strip_widening
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("a=1\n", encoding="utf-8")
    task = TaskRun(TaskContract("change"), capture_revision(root))
    (root / "a.py").write_text("a=2\n", encoding="utf-8")
    current = capture_revision(root)
    relaxed = VerifierConfig(mode="auto", min_changed_files=3)
    assert not requirements(task, current, relaxed)[0]
    assert requirements(task, current, VerifierConfig(mode="auto"))[0]
    task.contract = replace(task.contract, acceptance=("divide works",))
    assert requirements(task, current, relaxed)[0]
    assert strip_widening({"verifier": {"min_changed_files": 3}})["verifier"] == {}
    assert strip_widening({"verifier": {"min_changed_files": 0}})["verifier"] == {"min_changed_files": 0}
    assert VerifierConfig.from_dict({"min_changed_files": -5}).min_changed_files == 0
