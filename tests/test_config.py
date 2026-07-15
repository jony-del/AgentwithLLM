import os
from pathlib import Path

from agent_core.config import (
    load_dotenv,
    resolve_concurrency_config,
    resolve_config,
    resolve_permission_rules,
    resolve_sandbox_config,
)
from agent_core.permission_types import PermissionRuleSource


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


def test_resolve_config_accepts_openai_compat_provider_from_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_PROVIDER", "openai-compat")
    values = resolve_config(
        {"model": None, "permission": None, "provider": None},
        config_file=tmp_path / "absent.toml",
        env_file=tmp_path / "absent.env",
    )
    assert values["provider"] == "openai-compat"


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


def test_resolve_sandbox_config_backend_tables(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AGENT_SANDBOX", raising=False)
    monkeypatch.delenv("AGENT_SANDBOX_BACKEND", raising=False)
    config_file = tmp_path / "agent.toml"
    config_file.write_text(
        """
        [sandbox]
        enabled = true
        backend = "container"
        [sandbox.container]
        runtime = "docker"
        image = "alpine"
        memory = "512m"
        [sandbox.vm]
        provider = "hyperv"
        guest_host = "sandbox-vm"
        reset_each_task = false
        """,
        encoding="utf-8",
    )
    config = resolve_sandbox_config(config_file)
    assert config.backend == "container"
    assert config.container.runtime == "docker"
    assert config.container.image == "alpine"
    assert config.container.memory == "512m"
    assert config.vm.provider == "hyperv"
    assert config.vm.guest_host == "sandbox-vm"
    assert config.vm.reset_each_task is False


def test_resolve_sandbox_config_backend_env_overrides_and_degrades(tmp_path: Path, monkeypatch) -> None:
    config_file = tmp_path / "agent.toml"
    config_file.write_text('[sandbox]\nbackend = "native"\n', encoding="utf-8")
    monkeypatch.setenv("AGENT_SANDBOX_BACKEND", "vm")
    assert resolve_sandbox_config(config_file).backend == "vm"
    # An unknown backend degrades to "auto" (parse-failure-degrade invariant).
    monkeypatch.setenv("AGENT_SANDBOX_BACKEND", "nonsense")
    assert resolve_sandbox_config(config_file).backend == "auto"


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
    matched = rules.allow_match("run_command", {"command": "git status"})
    assert matched is not None and matched.source is PermissionRuleSource.USER


def test_resolve_permission_rules_absent_table(tmp_path: Path) -> None:
    rules = resolve_permission_rules(tmp_path / "absent.toml")
    assert rules.is_empty
