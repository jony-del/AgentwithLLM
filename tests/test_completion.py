"""Tests for the chat slash-command completer (``agent_core.terminal.completion``)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from agent_core.skills import Skill, SkillRegistry
from agent_core.terminal import completion as completion_mod
from agent_core.terminal.completion import SlashCompleter
from agent_core.transcript import SessionInfo


def _agent(skills=(), session_dir="sessions", workspace=None):
    """Minimal stand-in exposing only what the completer reads off the agent."""
    return SimpleNamespace(
        skills=SkillRegistry(list(skills)),
        config=SimpleNamespace(session_dir=session_dir, model="claude-opus-4-8"),
        session=SimpleNamespace(workspace=workspace or Path.cwd()),
    )


def _complete(agent, text: str):
    completer = SlashCompleter(agent)
    return list(completer.get_completions(Document(text), CompleteEvent()))


def _session(**kw) -> SessionInfo:
    base = dict(
        session_id="abc123def456",
        path=Path("abc123def456.jsonl"),
        modified=1_700_000_000.0,
        first_prompt="do the thing",
        message_count=5,
        title=None,
        tag=None,
        git_branch=None,
    )
    base.update(kw)
    return SessionInfo(**base)


# --- name completion ---------------------------------------------------------


def test_slash_lists_commands_and_skills() -> None:
    skill = Skill(name="myskill", description="A skill", body="b")
    displays = {c.display_text for c in _complete(_agent(skills=[skill]), "/")}
    assert {"/help", "/resume", "/model", "/skills"} <= displays
    assert "/myskill" in displays


def test_prefix_filters_to_matching_names() -> None:
    review = Skill(name="review", description="Review code", body="b")
    comps = _complete(_agent(skills=[review]), "/re")
    displays = {c.display_text for c in comps}
    assert "/resume" in displays  # built-in starting with "re"
    assert "/review" in displays  # skill starting with "re"
    assert "/model" not in displays  # filtered out


def test_command_completion_inserts_name_with_trailing_space() -> None:
    (resume,) = [c for c in _complete(_agent(), "/resume") if c.display_text == "/resume"]
    assert resume.text == "/resume "
    assert resume.start_position == -len("/resume")


def test_skill_shadowed_by_builtin_is_not_listed() -> None:
    # A skill named like a built-in must not produce a duplicate completion.
    clash = Skill(name="model", description="clash", body="b")
    comps = [c for c in _complete(_agent(skills=[clash]), "/model") if c.display_text == "/model"]
    assert len(comps) == 1
    assert comps[0].display_meta_text == "Show or switch the model."  # the built-in, not the skill


def test_multiline_buffer_yields_nothing() -> None:
    assert _complete(_agent(), "/cmd\nsecond line") == []


def test_non_slash_yields_nothing() -> None:
    assert _complete(_agent(), "hello world") == []


# --- /resume argument completion ---------------------------------------------


def test_resume_lists_sessions_by_summary_phrase(monkeypatch) -> None:
    info = _session(title="Implementing the Skills Subsystem", git_branch="main", message_count=7)
    monkeypatch.setattr(completion_mod, "list_sessions", lambda _d: [info])
    (comp,) = _complete(_agent(), "/resume ")
    assert comp.text == info.session_id  # inserts the id
    assert comp.display_text == "Implementing the Skills Subsystem"  # shows the phrase
    assert "7 msgs" in comp.display_meta_text and "[main]" in comp.display_meta_text


def test_resume_falls_back_to_first_prompt(monkeypatch) -> None:
    info = _session(title=None, first_prompt="git commit the changes")
    monkeypatch.setattr(completion_mod, "list_sessions", lambda _d: [info])
    (comp,) = _complete(_agent(), "/resume ")
    assert comp.display_text == "git commit the changes"


def test_resume_filters_by_typed_partial(monkeypatch) -> None:
    a = _session(session_id="aaa", title="Implement skills")
    b = _session(session_id="bbb", title="Fix the parser")
    monkeypatch.setattr(completion_mod, "list_sessions", lambda _d: [a, b])
    texts = {c.text for c in _complete(_agent(), "/resume parser")}
    assert texts == {"bbb"}


def test_resume_continue_alias_also_completes(monkeypatch) -> None:
    monkeypatch.setattr(completion_mod, "list_sessions", lambda _d: [_session()])
    assert len(_complete(_agent(), "/continue ")) == 1


def test_resume_no_session_dir_yields_nothing(monkeypatch) -> None:
    called = False

    def _boom(_d):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(completion_mod, "list_sessions", _boom)
    assert _complete(_agent(session_dir=""), "/resume ") == []
    assert called is False  # guarded before touching disk


def test_resume_listing_error_yields_nothing(monkeypatch) -> None:
    def _raise(_d):
        raise OSError("disk gone")

    monkeypatch.setattr(completion_mod, "list_sessions", _raise)
    assert _complete(_agent(), "/resume ") == []  # never propagates


def test_model_has_no_inline_completion() -> None:
    # `/model` opens the interactive picker instead of text-completing model names.
    assert _complete(_agent(), "/model ") == []
