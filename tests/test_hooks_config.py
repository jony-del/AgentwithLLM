"""Tests for resolving the ``[hooks]`` toml table into a ``HooksConfig``.

Covers defaults (no table), builtin toggles, external-spec parsing, the "degrade, don't
crash" guards (unknown event/type, missing required field), and the ``AGENT_HOOKS`` env
override.
"""

from pathlib import Path

from agent_core.config import resolve_hooks_config
from agent_core.hooks import HooksConfig


def _write_toml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "agent.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_missing_table_yields_defaults(tmp_path: Path) -> None:
    config = resolve_hooks_config(tmp_path / "absent.toml")
    assert isinstance(config, HooksConfig)
    assert config.enabled is True
    # Observation/control built-ins default on; injection off.
    assert config.builtin.stop_completion is True
    assert config.builtin.post_sampling_observer is True
    assert config.builtin.compaction_logger is True
    assert config.external == []
    # The prompt-input firewall is on by default with its baseline thresholds.
    assert config.prompt_validation.enabled is True
    assert config.prompt_validation.max_chars == 100_000
    assert config.prompt_validation.reject_control_chars is True
    assert config.prompt_validation.neutralize_framing is True


def test_builtin_toggles_and_enabled(tmp_path: Path) -> None:
    path = _write_toml(
        tmp_path,
        """
        [hooks]
        enabled = false
        [hooks.builtin]
        stop_completion = false
        compaction_logger = false
        """,
    )
    config = resolve_hooks_config(path)
    assert config.enabled is False
    assert config.builtin.stop_completion is False
    assert config.builtin.compaction_logger is False
    # Unmentioned toggles keep their defaults.
    assert config.builtin.post_sampling_observer is True


def test_prompt_validation_table_resolves(tmp_path: Path) -> None:
    path = _write_toml(
        tmp_path,
        """
        [hooks.prompt_validation]
        enabled = false
        max_chars = 5000
        neutralize_framing = false
        bogus_key = "ignored"
        """,
    )
    config = resolve_hooks_config(path)
    assert config.prompt_validation.enabled is False
    assert config.prompt_validation.max_chars == 5000
    assert config.prompt_validation.neutralize_framing is False
    # Unmentioned fields keep defaults; unknown keys are ignored (don't crash).
    assert config.prompt_validation.reject_control_chars is True


def test_external_specs_parse(tmp_path: Path) -> None:
    path = _write_toml(
        tmp_path,
        """
        [[hooks.external]]
        event = "Stop"
        type = "command"
        command = "python check.py"
        timeout = 7

        [[hooks.external]]
        event = "PreCompact"
        type = "command"
        matcher = "auto"
        command = "./pre.sh"
        """,
    )
    config = resolve_hooks_config(path)
    assert len(config.external) == 2
    stop, pre = config.external
    assert stop.event == "Stop" and stop.type == "command"
    assert stop.command == "python check.py" and stop.timeout == 7.0
    assert pre.matcher == "auto"


def test_bad_specs_are_dropped(tmp_path: Path) -> None:
    path = _write_toml(
        tmp_path,
        """
        [[hooks.external]]
        event = "NotAnEvent"
        type = "command"
        command = "x"

        [[hooks.external]]
        event = "Stop"
        type = "bogus"
        command = "x"

        [[hooks.external]]
        event = "Stop"
        type = "command"
        # missing command → dropped

        [[hooks.external]]
        event = "UserPromptSubmit"
        type = "command"
        command = "ok.sh"
        """,
    )
    config = resolve_hooks_config(path)
    # Only the last, fully-valid spec survives.
    assert len(config.external) == 1
    assert config.external[0].event == "UserPromptSubmit"


def test_http_and_prompt_required_fields(tmp_path: Path) -> None:
    path = _write_toml(
        tmp_path,
        """
        [[hooks.external]]
        event = "Stop"
        type = "http"
        # missing url → dropped

        [[hooks.external]]
        event = "Stop"
        type = "http"
        url = "http://localhost/hook"

        [[hooks.external]]
        event = "Stop"
        type = "prompt"
        prompt = "is it done?"
        model = "claude-haiku-4-5-20251001"
        """,
    )
    config = resolve_hooks_config(path)
    types = [(s.type, s.url, s.prompt) for s in config.external]
    assert types == [("http", "http://localhost/hook", None), ("prompt", None, "is it done?")]


def test_agent_hooks_env_overrides_enabled(tmp_path: Path, monkeypatch) -> None:
    path = _write_toml(tmp_path, "[hooks]\nenabled = true\n")
    monkeypatch.setenv("AGENT_HOOKS", "0")
    assert resolve_hooks_config(path).enabled is False
    monkeypatch.setenv("AGENT_HOOKS", "yes")
    assert resolve_hooks_config(path).enabled is True
