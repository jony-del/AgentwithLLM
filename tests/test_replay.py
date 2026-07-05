"""``polaris replay <run_id>`` — turn a recorded runs/*.jsonl into a readable timeline.

Reads only: no agent construction, no API calls. Tolerates unknown event types and
corrupt lines (forward compatibility for future schema versions).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_core.cli import replay_command
from agent_core.providers.fake import FakeProvider
from agent_core.react import ReActAgent, ReActConfig
from agent_core.storage import read_events


def _args(run_id: str, run_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(run_id=run_id, run_dir=str(run_dir))


async def test_replay_renders_a_real_run(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "runs"
    agent = ReActAgent(
        provider=FakeProvider(),
        config=ReActConfig(run_dir=str(run_dir), session_dir="", git_context=False,
                           project_instructions=False),
    )
    result = await agent.run("tool: echo replay-me")
    assert "echo" in result.answer

    assert replay_command(_args(agent.logger.run_id, run_dir)) == 0
    out = capsys.readouterr().out
    assert "user" in out and "tool: echo replay-me" in out
    assert "tool_result" in out and "permission" in out
    assert "final" in out
    assert "event(s)." in out


async def test_replay_unique_prefix_and_ambiguity(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    (run_dir / "20260101-aaa.jsonl").write_text(
        json.dumps({"ts": 1.0, "v": 1, "event": "user", "content": "x"}) + "\n", encoding="utf-8"
    )
    (run_dir / "20260202-bbb.jsonl").write_text("", encoding="utf-8")

    assert replay_command(_args("20260101", run_dir)) == 0  # unique prefix resolves
    capsys.readouterr()
    assert replay_command(_args("2026", run_dir)) == 1  # ambiguous
    assert "ambiguous" in capsys.readouterr().err
    assert replay_command(_args("nope", run_dir)) == 1  # no match, actionable
    assert "no run matching" in capsys.readouterr().err


def test_replay_tolerates_unknown_events_and_corrupt_lines(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    lines = [
        json.dumps({"ts": 1.0, "v": 99, "event": "from_the_future", "widget": "x" * 500}),
        "{this is not json",
        json.dumps({"event": "no_ts_no_version"}),  # pre-v1 record shape
    ]
    (run_dir / "weird.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    records = list(read_events(run_dir / "weird.jsonl"))
    assert [r["event"] for r in records] == ["from_the_future", "_unparseable", "no_ts_no_version"]

    assert replay_command(_args("weird", run_dir)) == 0
    out = capsys.readouterr().out
    assert "from_the_future" in out and "_unparseable" in out
    assert "3 event(s)." in out
