import os
from pathlib import Path

from agent_core.config import (
    load_dotenv,
    resolve_concurrency_config,
    resolve_config,
    resolve_permission_rules,
    resolve_sandbox_config,
)


def test_load_dotenv_sets_missing_environment_variables(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
        # Local agent settings
        ANTHROPIC_API_KEY="test-key"
        AGENT_MODEL=claude-test-model # inline comment
        """,
        encoding="utf-8",
    )

    load_dotenv(env_file)

    assert os.environ["ANTHROPIC_API_KEY"] == "test-key"
    assert os.environ["AGENT_MODEL"] == "claude-test-model"


def test_load_dotenv_does_not_override_existing_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_PROVIDER", "fake")
    env_file = tmp_path / ".env"
    env_file.write_text("AGENT_PROVIDER=claude\n", encoding="utf-8")

    load_dotenv(env_file)

    assert os.environ["AGENT_PROVIDER"] == "fake"


def test_resolve_config_loads_dotenv_before_reading_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("AGENT_MODEL=claude-from-env-file\n", encoding="utf-8")

    values = resolve_config({"model": None, "permission": None, "provider": None}, env_file=env_file)

    assert values["model"] == "claude-from-env-file"


def test_cli_values_override_dotenv_values(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("AGENT_MODEL=claude-from-env-file\n", encoding="utf-8")

    values = resolve_config(
        {"model": "claude-from-cli", "permission": None, "provider": None},
        env_file=env_file,
    )

    assert values["model"] == "claude-from-cli"


def test_resolve_concurrency_config_from_toml(tmp_path: Path) -> None:
    config_file = tmp_path / "agent.toml"
    config_file.write_text(
        """
        [concurrency]
        parallel_tools = false
        max_tool_workers = 0
        """,
        encoding="utf-8",
    )

    values = resolve_concurrency_config(config_file)

    assert values == {
        "parallel_tools": False,
        "max_tool_workers": 1,
        "max_api_concurrency": 8,
        "api_rate_limit_per_min": 0,
    }


def test_resolve_sandbox_config_defaults_disabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AGENT_SANDBOX", raising=False)
    config = resolve_sandbox_config(tmp_path / "absent.toml")
    assert config.enabled is False


def test_resolve_sandbox_config_from_toml(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AGENT_SANDBOX", raising=False)
    config_file = tmp_path / "agent.toml"
    config_file.write_text(
        """
        [sandbox]
        enabled = true
        excluded_commands = ["bazel:*"]
        [sandbox.filesystem]
        deny_read = ["~/.ssh"]
        [sandbox.network]
        allowed_domains = ["api.example.com"]
        """,
        encoding="utf-8",
    )
    config = resolve_sandbox_config(config_file)
    assert config.enabled is True
    assert config.excluded_commands == ["bazel:*"]
    assert config.filesystem.deny_read == ["~/.ssh"]
    assert config.network.allowed_domains == ["api.example.com"]


def test_resolve_sandbox_config_env_overrides_toml(tmp_path: Path, monkeypatch) -> None:
    config_file = tmp_path / "agent.toml"
    config_file.write_text("[sandbox]\nenabled = false\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_SANDBOX", "true")
    assert resolve_sandbox_config(config_file).enabled is True


def test_resolve_permission_rules_from_toml(tmp_path: Path) -> None:
    config_file = tmp_path / "agent.toml"
    config_file.write_text(
        """
        [permissions]
        allow = ["run_command(git *)"]
        deny = ["run_command(rm *)", "bad("]
        """,
        encoding="utf-8",
    )
    rules = resolve_permission_rules(config_file)
    assert rules.allow_matches("run_command", {"command": "git status"})
    assert rules.deny_matches("run_command", {"command": "rm x"})
    # The unparseable "bad(" entry was dropped, not raised.
    assert len(rules.deny) == 1


def test_resolve_permission_rules_absent_table(tmp_path: Path) -> None:
    rules = resolve_permission_rules(tmp_path / "absent.toml")
    assert rules.is_empty
