"""Tests for the built-in programmatic hooks, the external adapters, and pipeline assembly.

Three layers:
* ``builtin_hooks.py`` — Stop completion gate, prompt validation/injection, observers.
* ``hook_adapters.py`` — context projection, output→outcome mapping, the command transport
  (success / block-by-exit-code / timeout-kill / garbled output) and the matcher gate.
* ``ReActAgent._build_hook_pipeline`` — that config toggles and external specs land in the
  right pipeline lists, and that the default Stop completion hook blocks open to-dos.
"""

import sys
from pathlib import Path

from agent_core.builtin_hooks import (
    PostSamplingObserverHook,
    StopCompletionHook,
    UserPromptContextHook,
)
from agent_core.hook_adapters import (
    CommandHookAdapter,
    build_external_adapter,
    outcome_from_output,
    project_hook_input,
)
from agent_core.hooks import (
    ExternalHookSpec,
    HookContext,
    HookEvent,
    HooksConfig,
)
from agent_core.models import Message
from agent_core.providers.fake import FakeProvider
from agent_core.react import AgentRunResult, ReActAgent, ReActConfig
from agent_core.session import SessionContext
from agent_core.storage import JSONLRunLogger


def _hermetic_config(tmp_path: Path, **overrides) -> ReActConfig:
    base = dict(
        run_dir=str(tmp_path),
        project_instructions=False,
        git_context=False,
        session_dir="",
    )
    base.update(overrides)
    return ReActConfig(**base)


def _command(script: Path) -> str:
    """A shell command string that runs ``script`` with the current interpreter."""
    return f'"{sys.executable}" "{script}"'


# --- built-in hooks -----------------------------------------------------------


async def test_stop_completion_blocks_open_todos_once() -> None:
    session = SessionContext()
    session.todos.replace([{"content": "do x", "status": "in_progress"}])
    hook = StopCompletionHook(session)

    blocked = await hook.on_stop(HookContext(event=HookEvent.STOP, messages=[], stop_hook_active=False))
    assert blocked.block is True
    assert "do x" in (blocked.additional_context or "")

    # Already blocked once this run → don't pin the agent.
    again = await hook.on_stop(HookContext(event=HookEvent.STOP, messages=[], stop_hook_active=True))
    assert again.block is False


async def test_stop_completion_allows_when_all_done() -> None:
    session = SessionContext()
    session.todos.replace([{"content": "do x", "status": "completed"}])
    hook = StopCompletionHook(session)
    outcome = await hook.on_stop(HookContext(event=HookEvent.STOP, messages=[], stop_hook_active=False))
    assert outcome.block is False


async def test_user_prompt_context_blocks_empty_and_injects() -> None:
    hook = UserPromptContextHook()
    empty = await hook.on_user_prompt(HookContext(event=HookEvent.USER_PROMPT_SUBMIT, messages=[], prompt="   "))
    assert empty.block is True
    ok = await hook.on_user_prompt(HookContext(event=HookEvent.USER_PROMPT_SUBMIT, messages=[], prompt="hi"))
    assert ok.block is False
    assert ok.additional_context and "submitted at" in ok.additional_context


async def test_post_sampling_observer_writes_log(tmp_path: Path) -> None:
    logger = JSONLRunLogger(tmp_path)
    hook = PostSamplingObserverHook(logger)
    ctx = HookContext(
        event=HookEvent.POST_SAMPLING,
        messages=[Message("user", "hi"), Message("assistant", "yo")],
        last_assistant_message="yo",
    )
    await hook.after_sampling(ctx)
    text = Path(logger.path).read_text(encoding="utf-8")
    assert "hook_observe" in text and "PostSampling" in text


# --- projection + output mapping (pure) ---------------------------------------


def test_project_hook_input_is_bounded() -> None:
    messages = [Message("user", "x" * 5000) for _ in range(50)]
    ctx = HookContext(
        event=HookEvent.STOP, messages=messages, session_id="s1", last_assistant_message="done"
    )
    data = project_hook_input(ctx, max_messages=10, max_content_chars=100)
    assert data["hook_event_name"] == "Stop"
    assert data["session_id"] == "s1"
    assert data["last_assistant_message"] == "done"
    assert len(data["messages"]) == 10  # tail only
    assert all(len(m["content"]) <= 101 for m in data["messages"])  # truncated (+ ellipsis)


def test_outcome_from_output_block_signals() -> None:
    # Exit code 2 blocks even with empty stdout.
    assert outcome_from_output("", 2).block is True
    # JSON continue:false blocks and injects context.
    out = outcome_from_output(
        '{"continue": false, "stopReason": "no", "hookSpecificOutput": {"additionalContext": "ctx"}}',
        0,
    )
    assert out.block is True and out.reason == "no" and out.additional_context == "ctx"
    # Plain success with top-level additionalContext, no block.
    out2 = outcome_from_output('{"additionalContext": "grounding"}', 0)
    assert out2.block is False and out2.additional_context == "grounding"
    # Non-JSON stdout is ignored except for the exit code.
    assert outcome_from_output("just some text", 0).block is False


# --- command adapter (subprocess) ---------------------------------------------


async def test_command_adapter_injects_context(tmp_path: Path) -> None:
    script = tmp_path / "echo_ctx.py"
    script.write_text(
        "import sys, json\n"
        "json.load(sys.stdin)\n"
        "print(json.dumps({'hookSpecificOutput': {'additionalContext': 'from-hook'}}))\n",
        encoding="utf-8",
    )
    spec = ExternalHookSpec(event="Stop", type="command", command=_command(script), timeout=10)
    adapter = CommandHookAdapter(spec, JSONLRunLogger(tmp_path))
    outcome = await adapter.on_stop(HookContext(event=HookEvent.STOP, messages=[]))
    assert outcome.block is False
    assert outcome.additional_context == "from-hook"


async def test_command_adapter_exit_2_blocks(tmp_path: Path) -> None:
    script = tmp_path / "block.py"
    script.write_text("import sys; sys.stdin.read(); sys.exit(2)\n", encoding="utf-8")
    spec = ExternalHookSpec(event="Stop", type="command", command=_command(script), timeout=10)
    adapter = CommandHookAdapter(spec, JSONLRunLogger(tmp_path))
    outcome = await adapter.on_stop(HookContext(event=HookEvent.STOP, messages=[]))
    assert outcome.block is True


async def test_command_adapter_timeout_degrades_to_allow(tmp_path: Path) -> None:
    script = tmp_path / "slow.py"
    script.write_text("import time; time.sleep(5)\n", encoding="utf-8")
    spec = ExternalHookSpec(event="Stop", type="command", command=_command(script), timeout=0.3)
    adapter = CommandHookAdapter(spec, JSONLRunLogger(tmp_path))
    outcome = await adapter.on_stop(HookContext(event=HookEvent.STOP, messages=[]))
    # Killed on timeout → no block, run proceeds.
    assert outcome.block is False


async def test_command_adapter_matcher_gates_compaction(tmp_path: Path) -> None:
    script = tmp_path / "block.py"
    script.write_text("import sys; sys.stdin.read(); sys.exit(2)\n", encoding="utf-8")
    spec = ExternalHookSpec(
        event="PreCompact", type="command", command=_command(script), matcher="auto", timeout=10
    )
    adapter = CommandHookAdapter(spec, JSONLRunLogger(tmp_path))
    # trigger "reactive" doesn't match "auto" → command never runs → no block.
    miss = await adapter.before_compact(
        HookContext(event=HookEvent.PRE_COMPACT, messages=[], trigger="reactive")
    )
    assert miss.block is False
    # trigger "auto" matches → command runs and blocks.
    hit = await adapter.before_compact(
        HookContext(event=HookEvent.PRE_COMPACT, messages=[], trigger="auto")
    )
    assert hit.block is True


# --- adapter builder ----------------------------------------------------------


def test_build_external_adapter_dependency_gating(tmp_path: Path) -> None:
    logger = JSONLRunLogger(tmp_path)
    cmd = ExternalHookSpec(event="Stop", type="command", command="x")
    assert isinstance(build_external_adapter(cmd, logger=logger), CommandHookAdapter)
    # prompt without a provider, agent without a factory → skipped (None).
    prompt = ExternalHookSpec(event="Stop", type="prompt", prompt="?")
    assert build_external_adapter(prompt, logger=logger) is None
    agent_spec = ExternalHookSpec(event="Stop", type="agent", prompt="?")
    assert build_external_adapter(agent_spec, logger=logger) is None


# --- pipeline assembly (integration) ------------------------------------------


def test_default_pipeline_has_builtin_hooks(tmp_path: Path) -> None:
    agent = ReActAgent(FakeProvider(), _hermetic_config(tmp_path), logger=JSONLRunLogger(tmp_path))
    assert any(isinstance(h, StopCompletionHook) for h in agent.hooks.stop_hooks)
    assert any(isinstance(h, PostSamplingObserverHook) for h in agent.hooks.post_sampling_hooks)
    # compaction logger is the same instance on both lists.
    assert agent.hooks.pre_compact_hooks and agent.hooks.post_compact_hooks
    # injection hook is off by default.
    assert not any(isinstance(h, UserPromptContextHook) for h in agent.hooks.user_prompt_hooks)


def test_disabled_subsystem_has_no_lifecycle_hooks(tmp_path: Path) -> None:
    cfg = _hermetic_config(tmp_path, hooks=HooksConfig(enabled=False))
    agent = ReActAgent(FakeProvider(), cfg, logger=JSONLRunLogger(tmp_path))
    assert agent.hooks.stop_hooks == []
    assert agent.hooks.post_sampling_hooks == []
    # The MaxOutput tool post-hook is still present regardless.
    assert agent.hooks.post_hooks


def test_external_spec_lands_in_right_list(tmp_path: Path) -> None:
    cfg = _hermetic_config(
        tmp_path,
        hooks=HooksConfig(
            external=[ExternalHookSpec(event="Stop", type="command", command="echo hi")]
        ),
    )
    agent = ReActAgent(FakeProvider(), cfg, logger=JSONLRunLogger(tmp_path))
    assert any(isinstance(h, CommandHookAdapter) for h in agent.hooks.stop_hooks)


async def test_builtin_stop_completion_forces_continuation(tmp_path: Path) -> None:
    agent = ReActAgent(FakeProvider(), _hermetic_config(tmp_path), logger=JSONLRunLogger(tmp_path))
    agent.session.todos.replace([{"content": "finish the work", "status": "pending"}])
    result = await agent.run("hello")
    assert isinstance(result, AgentRunResult)
    # Blocked the first stop, injected one continuation, then allowed the next stop.
    continues = [m for m in result.messages if m.metadata.get("stop_hook") == "continue"]
    assert len(continues) == 1
    assert "finish the work" in continues[0].content
