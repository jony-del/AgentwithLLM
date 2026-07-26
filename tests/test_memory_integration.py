import json
from pathlib import Path

from agent_core.memory.config import MemoryConfig, MemoryRetrievalConfig
from agent_core.providers import FakeProvider
from agent_core.react import ReActAgent, ReActConfig


def _agent(tmp_path: Path, *, memory: bool) -> ReActAgent:
    config = ReActConfig(
        run_dir=str(tmp_path / "runs"),
        # This suite exercises agent integration, not optional local model
        # installation. Lexical mode keeps it deterministic on developer hosts
        # that may already have a production embedding/reranker bundle active.
        memory=MemoryConfig(
            enabled=memory,
            dir=str(tmp_path / "memory"),
            retrieval=MemoryRetrievalConfig(mode="lexical"),
        ),
    )
    return ReActAgent(provider=FakeProvider(), config=config)


def _events(agent: ReActAgent) -> list[dict]:
    return [json.loads(line) for line in agent.logger.path.read_text(encoding="utf-8").splitlines()]


async def test_memory_enabled_extracts_after_run(tmp_path: Path) -> None:
    agent = _agent(tmp_path, memory=True)
    await agent.run("I prefer dark mode and Python 3.11")

    assert agent.memory_store is not None
    assert len(agent.memory_store) == 1
    assert any(e["event"] == "memory_extract" for e in _events(agent))


async def test_memory_enabled_recalls_relevant_memory(tmp_path: Path) -> None:
    agent = _agent(tmp_path, memory=True)
    await agent.run("I prefer dark mode and Python 3.11")  # seeds the store

    result = await agent.run("remind me about my dark mode preference")
    recall_blocks = [m for m in result.messages if m.metadata.get("memory") == "recall"]
    assert len(recall_blocks) == 1, agent.retriever.last_trace.to_dict()
    assert "dark mode" in recall_blocks[0].content
    assert any(e["event"] == "memory_recall" for e in _events(agent))


async def test_memory_disabled_is_a_no_op(tmp_path: Path) -> None:
    agent = _agent(tmp_path, memory=False)
    result = await agent.run("I prefer dark mode and Python 3.11")

    assert agent.memory_store is None
    assert agent.retriever is None
    assert agent.extractor is None
    assert not any(m.metadata.get("memory") for m in result.messages)
    # No memory directory is created when the feature is off.
    assert not (tmp_path / "memory").exists()
