"""Tests for the lifecycle (non-tool) hooks and their ordering in the run loop.

Covers the reference-aligned hook surface added on top of the original pre/post
*tool* hooks: UserPromptSubmit, PostSampling, PreCompact, PostCompact, and the
blockable/continuable Stop hook ("可阻断/可续跑").
"""

from pathlib import Path

from agent_core.compression import CompressionConfig
from agent_core.hooks import HookContext, HookEvent, HookOutcome, HookPipeline
from agent_core.models import Message
from agent_core.providers.fake import FakeProvider
from agent_core.react import AgentRunResult, ReActAgent, ReActConfig
from agent_core.storage import JSONLRunLogger


# --- stub hooks ---------------------------------------------------------------


class RecordingUserPromptHook:
    def __init__(self, block: bool = False, context: str | None = None, reason: str | None = None):
        self.block = block
        self.context = context
        self.reason = reason
        self.prompts: list[str] = []

    async def on_user_prompt(self, ctx: HookContext) -> HookOutcome:
        assert ctx.event is HookEvent.USER_PROMPT_SUBMIT
        self.prompts.append(ctx.prompt or "")
        return HookOutcome(block=self.block, additional_context=self.context, reason=self.reason)


class RecordingPostSamplingHook:
    def __init__(self) -> None:
        self.calls = 0
        self.last_texts: list[str | None] = []

    async def after_sampling(self, ctx: HookContext) -> None:
        assert ctx.event is HookEvent.POST_SAMPLING
        self.calls += 1
        self.last_texts.append(ctx.last_assistant_message)


class BlockingStopHook:
    def __init__(self, block_times: int) -> None:
        self.block_times = block_times
        self.calls = 0
        self.active_flags: list[bool] = []

    async def on_stop(self, ctx: HookContext) -> HookOutcome:
        assert ctx.event is HookEvent.STOP
        self.calls += 1
        self.active_flags.append(ctx.stop_hook_active)
        if self.calls <= self.block_times:
            return HookOutcome(block=True, reason="not done yet", additional_context="Keep working.")
        return HookOutcome(block=False)


class RecordingCompactHook:
    def __init__(self) -> None:
        self.pre_triggers: list[str | None] = []
        self.post: list[tuple[str | None, str | None]] = []

    async def before_compact(self, ctx: HookContext) -> HookOutcome:
        assert ctx.event is HookEvent.PRE_COMPACT
        self.pre_triggers.append(ctx.trigger)
        return HookOutcome()

    async def after_compact(self, ctx: HookContext) -> HookOutcome:
        assert ctx.event is HookEvent.POST_COMPACT
        self.post.append((ctx.trigger, ctx.summary))
        return HookOutcome()


class BlockingPreCompactHook:
    def __init__(self) -> None:
        self.calls = 0

    async def before_compact(self, ctx: HookContext) -> HookOutcome:
        self.calls += 1
        return HookOutcome(block=True, reason="not now")


def _agent(tmp_path: Path, hooks: HookPipeline, config: ReActConfig | None = None) -> ReActAgent:
    cfg = config or ReActConfig(run_dir=str(tmp_path))
    return ReActAgent(
        FakeProvider(), cfg, hooks=hooks, logger=JSONLRunLogger(tmp_path)
    )


def _hermetic_config(tmp_path: Path, **overrides) -> ReActConfig:
    """A config with the external context sources off, for deterministic loop tests."""
    base = dict(
        run_dir=str(tmp_path),
        project_instructions=False,
        git_context=False,
        session_dir="",
    )
    base.update(overrides)
    return ReActConfig(**base)


# --- HookPipeline aggregation (unit) ------------------------------------------


async def test_pipeline_user_prompt_folds_context_and_blocks() -> None:
    first = RecordingUserPromptHook(context="A")
    second = RecordingUserPromptHook(block=True, reason="stop")
    pipeline = HookPipeline(user_prompt_hooks=[first, second])
    ctx = HookContext(event=HookEvent.USER_PROMPT_SUBMIT, messages=[], prompt="hi")
    outcome = await pipeline.run_user_prompt(ctx)
    assert outcome.block is True
    assert outcome.reason == "stop"
    # The blocking hook carried no context of its own, so the fold surfaces the
    # context gathered from earlier hooks.
    assert outcome.additional_context == "A"


async def test_pipeline_stop_short_circuits_on_first_block() -> None:
    blocking = BlockingStopHook(block_times=1)
    never = BlockingStopHook(block_times=0)
    pipeline = HookPipeline(stop_hooks=[blocking, never])
    ctx = HookContext(event=HookEvent.STOP, messages=[])
    outcome = await pipeline.run_stop(ctx)
    assert outcome.block is True
    assert blocking.calls == 1
    assert never.calls == 0  # short-circuited before the second hook


async def test_pipeline_post_sampling_runs_each_hook() -> None:
    a, b = RecordingPostSamplingHook(), RecordingPostSamplingHook()
    pipeline = HookPipeline(post_sampling_hooks=[a, b])
    ctx = HookContext(event=HookEvent.POST_SAMPLING, messages=[], last_assistant_message="x")
    await pipeline.run_post_sampling(ctx)
    assert a.calls == 1 and b.calls == 1


# --- UserPromptSubmit (integration) -------------------------------------------


async def test_user_prompt_block_aborts_before_model_call(tmp_path: Path) -> None:
    fake = FakeProvider()
    hook = RecordingUserPromptHook(block=True, reason="not allowed")
    agent = ReActAgent(
        fake, _hermetic_config(tmp_path), hooks=HookPipeline(user_prompt_hooks=[hook]),
        logger=JSONLRunLogger(tmp_path),
    )
    result = await agent.run("hello")
    assert isinstance(result, AgentRunResult)
    assert result.answer == "not allowed"
    assert result.steps == 0
    assert fake.calls == 0  # the model was never called
    assert hook.prompts == ["hello"]


async def test_user_prompt_context_is_injected(tmp_path: Path) -> None:
    hook = RecordingUserPromptHook(context="Extra grounding for the model.")
    agent = ReActAgent(
        FakeProvider(), _hermetic_config(tmp_path),
        hooks=HookPipeline(user_prompt_hooks=[hook]), logger=JSONLRunLogger(tmp_path),
    )
    result = await agent.run("hello")
    injected = [m for m in result.messages if m.metadata.get("hook") == "user_prompt_context"]
    assert len(injected) == 1
    assert "Extra grounding for the model." in injected[0].content


# --- PostSampling (integration) -----------------------------------------------


async def test_post_sampling_fires_with_assistant_text(tmp_path: Path) -> None:
    hook = RecordingPostSamplingHook()
    agent = ReActAgent(
        FakeProvider(), _hermetic_config(tmp_path),
        hooks=HookPipeline(post_sampling_hooks=[hook]), logger=JSONLRunLogger(tmp_path),
    )
    await agent.run("hello")
    # One assistant turn -> exactly one PostSampling fire, reaped at the terminal return.
    assert hook.calls == 1
    assert hook.last_texts[0] and "Final answer" in hook.last_texts[0]


# --- Stop hook: blockable / continuable ---------------------------------------


async def test_stop_hook_blocks_then_allows(tmp_path: Path) -> None:
    hook = BlockingStopHook(block_times=2)
    agent = ReActAgent(
        FakeProvider(), _hermetic_config(tmp_path),
        hooks=HookPipeline(stop_hooks=[hook]), logger=JSONLRunLogger(tmp_path),
    )
    result = await agent.run("hello")
    # Blocked twice, then allowed the third stop -> 3 firings.
    assert hook.calls == 3
    # stop_hook_active is False on the first fire, True afterwards (parity w/ reference).
    assert hook.active_flags == [False, True, True]
    # Two continuation directives were injected back into the conversation.
    continues = [m for m in result.messages if m.metadata.get("stop_hook") == "continue"]
    assert len(continues) == 2
    assert "Keep working." in continues[0].content


async def test_stop_hook_block_is_capped(tmp_path: Path) -> None:
    hook = BlockingStopHook(block_times=99)  # always wants to continue
    agent = ReActAgent(
        FakeProvider(), _hermetic_config(tmp_path, max_stop_blocks=2),
        hooks=HookPipeline(stop_hooks=[hook]), logger=JSONLRunLogger(tmp_path),
    )
    result = await agent.run("hello")
    # The cap stops the loop after 2 blocks even though the hook keeps asking.
    assert hook.calls == 3
    continues = [m for m in result.messages if m.metadata.get("stop_hook") == "continue"]
    assert len(continues) == 2
    assert "Final answer" in result.answer


async def test_stop_hook_disabled_by_zero_cap(tmp_path: Path) -> None:
    hook = BlockingStopHook(block_times=99)
    agent = ReActAgent(
        FakeProvider(), _hermetic_config(tmp_path, max_stop_blocks=0),
        hooks=HookPipeline(stop_hooks=[hook]), logger=JSONLRunLogger(tmp_path),
    )
    result = await agent.run("hello")
    # Hook still observes the stop once, but can never force a continuation.
    assert hook.calls == 1
    assert not [m for m in result.messages if m.metadata.get("stop_hook") == "continue"]


# --- PreCompact / PostCompact (integration) -----------------------------------


def _big_history(n: int = 16, size: int = 1500) -> list[Message]:
    history: list[Message] = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        history.append(Message(role, f"turn {i} " + "x" * size))
    return history


def _force_compact_config(tmp_path: Path) -> ReActConfig:
    # A tiny effective window with no buffer/reserve makes the auto-compact gate trip
    # on any non-trivial history, so a real prefix fold happens on the first turn.
    compression = CompressionConfig(
        context_window_tokens=200,
        autocompact_buffer_tokens=0,
        reserved_output_tokens_for_summary=0,
    )
    return _hermetic_config(tmp_path, compression=compression)


async def test_pre_and_post_compact_fire_on_fold(tmp_path: Path) -> None:
    hook = RecordingCompactHook()
    agent = ReActAgent(
        FakeProvider(), _force_compact_config(tmp_path),
        hooks=HookPipeline(pre_compact_hooks=[hook], post_compact_hooks=[hook]),
        logger=JSONLRunLogger(tmp_path),
    )
    result = await agent.run("hello", history=_big_history())
    assert "auto" in hook.pre_triggers
    assert hook.post, "PostCompact should fire after a real fold"
    trigger, summary = hook.post[0]
    assert trigger == "auto"
    assert summary  # a summary message was produced by the fold
    assert isinstance(result, AgentRunResult)


async def test_pre_compact_block_skips_compaction(tmp_path: Path) -> None:
    hook = BlockingPreCompactHook()
    agent = ReActAgent(
        FakeProvider(), _force_compact_config(tmp_path),
        hooks=HookPipeline(pre_compact_hooks=[hook]), logger=JSONLRunLogger(tmp_path),
    )
    result = await agent.run("hello", history=_big_history())
    assert hook.calls >= 1
    # Blocking PreCompact means no prefix fold ever ran, so no summary message exists.
    from agent_core.compression import is_summary_message

    assert not any(is_summary_message(m) for m in result.messages)
