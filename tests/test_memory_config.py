from pathlib import Path

from agent_core.config import resolve_memory_config
from agent_core.memory.config import MemoryConfig


def test_from_dict_applies_known_fields_and_coerces() -> None:
    config = MemoryConfig.from_dict(
        {"enabled": "true", "recall_k": "9", "merge_threshold": "0.4", "unknown": "x"}
    )
    assert config.enabled is True
    assert config.recall_k == 9
    assert config.merge_threshold == 0.4
    # Unknown keys are ignored; absent fields keep their defaults.
    assert config.dir == MemoryConfig().dir


def test_from_dict_handles_none_and_empty() -> None:
    assert MemoryConfig.from_dict(None) == MemoryConfig()
    assert MemoryConfig.from_dict({}) == MemoryConfig()


def _write_toml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "agent.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_resolve_reads_memory_table(tmp_path: Path) -> None:
    toml = _write_toml(tmp_path, "[memory]\nenabled = true\nrecall_k = 7\n")
    config = resolve_memory_config(None, toml)
    assert config.enabled is True
    assert config.recall_k == 7


def test_resolve_env_overrides_toml(tmp_path: Path, monkeypatch) -> None:
    toml = _write_toml(tmp_path, "[memory]\nenabled = true\n")
    monkeypatch.setenv("AGENT_MEMORY", "0")
    config = resolve_memory_config(None, toml)
    assert config.enabled is False


def test_resolve_cli_overrides_env(tmp_path: Path, monkeypatch) -> None:
    toml = _write_toml(tmp_path, "[memory]\nenabled = false\n")
    monkeypatch.setenv("AGENT_MEMORY", "0")
    config = resolve_memory_config(True, toml)  # explicit --memory wins
    assert config.enabled is True


def test_resolve_without_table_is_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AGENT_MEMORY", raising=False)
    toml = _write_toml(tmp_path, "model = \"x\"\n")  # no [memory] table
    assert resolve_memory_config(None, toml) == MemoryConfig()
