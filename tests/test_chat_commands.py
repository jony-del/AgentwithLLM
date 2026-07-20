"""Tests for the interactive chat slash-command dispatcher (``agent_core.chat_commands``)."""

from __future__ import annotations

from agent_core.chat_commands import ChatTurn, dispatch, is_immediate_command
from agent_core.memory import MemoryConfig
from agent_core.models import Message
from agent_core.providers import FakeProvider
from agent_core.react import ReActAgent, ReActConfig
from agent_core.skills import Skill, SkillContext, SkillRegistry
from agent_core.ui import NullUI


def _agent(skills=None, **config_kwargs) -> ReActAgent:
    # Memory defaults to on in ReActConfig; disable it so tests don't touch the real
    # on-disk store (callers that test /memory pass their own memory config).
    config_kwargs.setdefault("memory", MemoryConfig(enabled=False))
    agent = ReActAgent(provider=FakeProvider(), config=ReActConfig(**config_kwargs))
    if skills is not None:
        agent.skills = SkillRegistry(skills)
        agent.session.skills = agent.skills
    return agent


class PickingUI(NullUI):
    def __init__(self, choice: tuple[str, str | None] | None) -> None:
        self.choice = choice
        self.seen_specs = []

    async def pick_model(self, current_model, current_effort, spec):
        self.seen_specs.append((current_model, current_effort, spec))
        return self.choice


class PickingPermissionUI(NullUI):
    def __init__(self, choice: str | None) -> None:
        self.choice = choice
        self.seen: list[str] = []

    async def pick_permission_mode(self, current_mode: str) -> str | None:
        self.seen.append(current_mode)
        return self.choice


# --- plain text & skills -----------------------------------------------------


async def test_plain_message_passes_through() -> None:
    turn = await dispatch("hello there", _agent(skills=[]), NullUI(), [])
    assert turn == ChatTurn(prompt="hello there")


async def test_path_like_slash_is_not_a_command() -> None:
    turn = await dispatch("/path/to/file", _agent(skills=[]), NullUI(), [])
    assert turn.prompt == "/path/to/file"


async def test_exit_quits() -> None:
    assert (await dispatch("/exit", _agent(skills=[]), NullUI(), [])).quit is True
    assert (await dispatch("/quit", _agent(skills=[]), NullUI(), [])).quit is True


async def test_inline_skill_returns_rendered_prompt() -> None:
    skill = Skill(name="note", description="d", body="Do: $ARGUMENTS", context=SkillContext.INLINE)
    turn = await dispatch("/note buy milk", _agent(skills=[skill]), NullUI(), [])
    assert turn.prompt == "Do: buy milk"


async def test_fork_skill_runs_via_subagent_and_handles(capsys) -> None:
    skill = Skill(name="explore", description="d", body="Explore the repo", context=SkillContext.FORK)
    turn = await dispatch("/explore now", _agent(skills=[skill]), NullUI(), [])
    assert turn == ChatTurn()  # fully handled
    assert "Final answer: Explore the repo\n\nnow" in capsys.readouterr().out


async def test_unknown_command_reported(capsys) -> None:
    turn = await dispatch("/bogus", _agent(skills=[]), NullUI(), [])
    assert turn == ChatTurn()
    assert "Unknown command" in capsys.readouterr().out


async def test_user_invocable_false_skill_is_unknown(capsys) -> None:
    skill = Skill(name="modelonly", description="d", body="b", user_invocable=False)
    await dispatch("/modelonly", _agent(skills=[skill]), NullUI(), [])
    assert "Unknown command" in capsys.readouterr().out


# --- built-in commands -------------------------------------------------------


async def test_help_lists_commands(capsys) -> None:
    await dispatch("/help", _agent(skills=[]), NullUI(), [])
    out = capsys.readouterr().out
    assert "/clear" in out and "/compact" in out and "/model" in out


async def test_skills_lists_user_invocable(capsys) -> None:
    visible = Skill(name="visible", description="Shows", body="b")
    hidden = Skill(name="hidden", description="d", body="b", user_invocable=False)
    await dispatch("/skills", _agent(skills=[visible, hidden]), NullUI(), [])
    out = capsys.readouterr().out
    assert "/visible" in out and "/hidden" not in out


async def test_clear_returns_empty_history(capsys) -> None:
    turn = await dispatch("/clear", _agent(skills=[]), NullUI(), [Message("user", "x")])
    assert turn.history == []
    assert "cleared" in capsys.readouterr().out.lower()


async def test_status_shows_model_and_counts(capsys) -> None:
    await dispatch("/status", _agent(model="claude-opus-4-8"), NullUI(), [])
    out = capsys.readouterr().out
    assert "claude-opus-4-8" in out and "skills" in out and "tools" in out


async def test_context_shows_window_and_threshold(capsys) -> None:
    await dispatch("/context", _agent(), NullUI(), [Message("user", "x" * 4000)])
    out = capsys.readouterr().out
    assert "context window" in out and "auto-compact threshold" in out


async def test_cost_shows_usage(capsys) -> None:
    agent = _agent()
    agent._session_input_tokens = 1234
    agent._session_output_tokens = 56
    await dispatch("/cost", agent, NullUI(), [])
    out = capsys.readouterr().out
    assert "1,234" in out and "total tokens" in out


async def test_model_shows_current_without_args(capsys) -> None:
    await dispatch("/model", _agent(model="claude-opus-4-8"), NullUI(), [])
    assert "claude-opus-4-8" in capsys.readouterr().out


async def test_model_switches_to_supported(capsys) -> None:
    agent = _agent(model="claude-opus-4-8")
    await dispatch("/model claude-sonnet-4-6", agent, NullUI(), [])
    assert agent.config.model == "claude-sonnet-4-6"
    assert "switched" in capsys.readouterr().out.lower()


async def test_model_rejects_unsupported(capsys) -> None:
    agent = _agent(provider="claude", model="claude-opus-4-8")
    await dispatch("/model gpt-4", agent, NullUI(), [])
    assert agent.config.model == "claude-opus-4-8"  # unchanged
    assert "Unsupported" in capsys.readouterr().out


async def test_model_accepts_openai_model_without_switching_provider(capsys) -> None:
    agent = _agent(provider="openai", model="gpt-4.1-mini")
    await dispatch("/model gpt-5.1", agent, NullUI(), [])
    assert agent.config.provider == "openai"
    assert agent.config.model == "gpt-5.1"
    assert "switched" in capsys.readouterr().out.lower()


async def test_model_opens_openai_picker_and_updates_effort(capsys) -> None:
    agent = _agent(provider="openai", model="gpt-4.1-nano", effort="high")
    ui = PickingUI(("gpt-5.6", "max"))

    await dispatch("/model", agent, ui, [])

    assert agent.config.provider == "openai"
    assert agent.config.model == "gpt-5.6"
    assert agent.config.effort == "max"
    assert len(ui.seen_specs) == 1
    current_model, current_effort, spec = ui.seen_specs[0]
    assert current_model == "gpt-4.1-nano"
    assert current_effort == "high"
    assert "OpenAI Responses" in spec.title
    assert spec.efforts_fn("gpt-5.6") == ("none", "low", "medium", "high", "xhigh", "max")
    assert "switched" in capsys.readouterr().out.lower()


async def test_model_openai_picker_clears_effort_for_no_effort_model(capsys) -> None:
    agent = _agent(provider="openai", model="gpt-5.6", effort="max")
    ui = PickingUI(("gpt-4.1-nano", None))

    await dispatch("/model", agent, ui, [])

    assert agent.config.provider == "openai"
    assert agent.config.model == "gpt-4.1-nano"
    assert agent.config.effort is None
    assert "(no effort levels)" in capsys.readouterr().out


async def test_model_openai_picker_cancel_leaves_config_unchanged(capsys) -> None:
    agent = _agent(provider="openai", model="gpt-4.1-nano", effort="high")
    ui = PickingUI(None)

    await dispatch("/model", agent, ui, [])

    assert agent.config.provider == "openai"
    assert agent.config.model == "gpt-4.1-nano"
    assert agent.config.effort == "high"
    out = capsys.readouterr().out
    assert "Current provider/model: openai / gpt-4.1-nano" in out
    assert "Known model families:" in out


async def test_model_openai_compat_without_args_stays_non_interactive(capsys) -> None:
    agent = _agent(provider="openai-compat", model="local-model")
    ui = PickingUI(("should-not-be-used", "high"))

    await dispatch("/model", agent, ui, [])

    assert agent.config.model == "local-model"
    assert ui.seen_specs == []
    assert "Switch with: /model <non-empty model id>" in capsys.readouterr().out


async def test_model_explicit_openai_switch_preserves_effort(capsys) -> None:
    agent = _agent(provider="openai", model="gpt-5.6", effort="max")

    await dispatch("/model custom-openai-model", agent, NullUI(), [])

    assert agent.config.provider == "openai"
    assert agent.config.model == "custom-openai-model"
    assert agent.config.effort == "max"
    assert "switched" in capsys.readouterr().out.lower()


async def test_permissions_direct_switch_updates_live_agent(capsys) -> None:
    agent = _agent(permission="default")
    await dispatch("/permissions acceptedits", agent, NullUI(), [])
    assert agent.config.permission.value == "acceptedits"
    assert agent.permissions.mode.value == "acceptedits"
    assert "accept edits on" in capsys.readouterr().out


async def test_permissions_bare_command_uses_six_mode_picker(capsys) -> None:
    agent = _agent(permission="default")
    ui = PickingPermissionUI("plan")
    await dispatch("/permissions", agent, ui, [])
    assert ui.seen == ["default"]
    assert agent.permissions.mode.value == "plan"
    assert "plan mode on" in capsys.readouterr().out


async def test_permissions_cancel_lists_modes_and_invalid_keeps_state(capsys) -> None:
    agent = _agent(permission="default")
    await dispatch("/permissions", agent, NullUI(), [])
    listed = capsys.readouterr().out
    assert "manual mode on" in listed and "bypass permissions on" in listed

    await dispatch("/permissions invalid", agent, NullUI(), [])
    assert agent.permissions.mode.value == "default"
    assert "Unknown permission mode" in capsys.readouterr().out


async def test_misspelled_permission_commands_are_not_registered(capsys) -> None:
    agent = _agent()
    await dispatch("/permmision", agent, NullUI(), [])
    await dispatch("/permission", agent, NullUI(), [])
    out = capsys.readouterr().out
    assert out.count("Unknown command") == 2


async def test_model_accepts_openai_compat_custom_model(capsys) -> None:
    agent = _agent(provider="openai-compat", model="local-model")
    await dispatch("/model qwen3-coder", agent, NullUI(), [])
    assert agent.config.provider == "openai-compat"
    assert agent.config.model == "qwen3-coder"
    assert "switched" in capsys.readouterr().out.lower()


async def test_memory_disabled_message(capsys) -> None:
    await dispatch("/memory", _agent(), NullUI(), [])
    assert "disabled" in capsys.readouterr().out.lower()


async def test_memory_enabled_empty(capsys, tmp_path) -> None:
    agent = _agent(memory=MemoryConfig(enabled=True, dir=str(tmp_path)))
    await dispatch("/memory", agent, NullUI(), [])
    assert "No memories stored yet" in capsys.readouterr().out


async def test_compact_empty_history(capsys) -> None:
    turn = await dispatch("/compact", _agent(), NullUI(), [])
    assert turn == ChatTurn()
    assert "Nothing to compact" in capsys.readouterr().out


async def test_compact_folds_large_history(capsys) -> None:
    # Long tool/user messages exceed the snip threshold so a fold genuinely saves chars.
    big = [Message("user", "task")]
    for i in range(8):
        big.append(Message("assistant", "calling tool"))
        big.append(Message("tool", "X" * 9000))
    turn = await dispatch("/compact", _agent(), NullUI(), big)
    assert turn.history is not None
    assert sum(len(m.content) for m in turn.history) < sum(len(m.content) for m in big)
    assert "Compacted" in capsys.readouterr().out


async def test_resume_disabled_when_no_session_dir(capsys) -> None:
    await dispatch("/resume", _agent(session_dir=""), NullUI(), [])
    assert "disabled" in capsys.readouterr().out.lower()


# --- agent support: cumulative usage feeds /cost -----------------------------


async def test_session_usage_accumulates_after_run(tmp_path) -> None:
    agent = _agent(run_dir=str(tmp_path), session_dir="")
    assert agent._session_input_tokens == 0
    await agent.run("hello")
    # FakeProvider reports usage, so the session counters move off zero.
    assert agent._session_input_tokens > 0
    assert agent._session_output_tokens > 0


async def test_compact_now_returns_input_when_nothing_to_fold() -> None:
    agent = _agent()
    small = [Message("user", "hi")]
    compacted, saved = await agent.compact_now(small)
    assert saved == 0
    assert compacted == small


def test_command_immediacy_is_explicit() -> None:
    for command in (
        "/exit",
        "/status",
        "/mcp",
        "/plugin manage",
        "/sandbox",
        "/rename title",
        "/model claude-opus-4-6",
        "/fast",
        "/effort high",
    ):
        assert is_immediate_command(command)
    for command in (
        "/help",
        "/clear",
        "/context",
        "/compact",
        "/permissions",
        "/memory",
        "/resume",
        "/reload-plugins",
        "/unknown",
    ):
        assert not is_immediate_command(command)


async def test_effort_validates_current_model_capabilities(capsys) -> None:
    agent = _agent(provider="claude", model="claude-sonnet-4-6", effort="high")
    await dispatch("/effort xhigh", agent, NullUI(), [])
    assert agent.config.effort == "high"
    assert "unsupported" in capsys.readouterr().out
    await dispatch("/effort max", agent, NullUI(), [])
    assert agent.config.effort == "max"
    await dispatch("/effort auto", agent, NullUI(), [])
    assert agent.config.effort is None


async def test_fast_switches_to_exact_supported_model_and_model_disables_it(capsys) -> None:
    agent = _agent(provider="claude", model="claude-sonnet-4-6")
    await dispatch("/fast on", agent, NullUI(), [])
    assert agent.fast_mode is True
    assert agent.config.model == "claude-opus-4-6"
    await dispatch("/model claude-sonnet-4-6", agent, NullUI(), [])
    assert agent.fast_mode is False


async def test_rename_persists_custom_title(tmp_path, capsys) -> None:
    agent = _agent(
        run_dir=str(tmp_path / "runs"),
        session_dir=str(tmp_path / "sessions"),
    )
    await dispatch("/rename My useful session", agent, NullUI(), [])
    assert agent.session_title == "My useful session"
    assert '"type": "custom-title"' in agent.transcript.path.read_text(encoding="utf-8")
    assert "renamed" in capsys.readouterr().out.lower()
