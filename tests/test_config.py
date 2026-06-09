import os
from pathlib import Path

from agent_core.config import load_dotenv, resolve_concurrency_config, resolve_config


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

    assert values == {"parallel_tools": False, "max_tool_workers": 1}
