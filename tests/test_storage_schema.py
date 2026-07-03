"""Schema-version stamping on persisted records (runs/*.jsonl + transcripts).

Replay/analysis tooling needs to know which record shape it is reading; every
record therefore carries a ``"v"`` field from day one, and loaders stay tolerant
of records without it (pre-v1).
"""

import json
from pathlib import Path

from agent_core.models import Message
from agent_core.storage import SCHEMA_VERSION as RUN_SCHEMA_VERSION, JSONLRunLogger
from agent_core.transcript import SCHEMA_VERSION as TRANSCRIPT_SCHEMA_VERSION, TranscriptStore, load_transcript


async def test_run_logger_stamps_schema_version(tmp_path: Path) -> None:
    logger = JSONLRunLogger(tmp_path)
    await logger.write("unit_test", {"payload": 1})

    record = json.loads(logger.path.read_text(encoding="utf-8").splitlines()[0])
    assert record["v"] == RUN_SCHEMA_VERSION
    assert record["event"] == "unit_test"


async def test_transcript_stamps_schema_version(tmp_path: Path) -> None:
    store = TranscriptStore(tmp_path, tmp_path, "sess-schema")
    await store.append_message(Message("user", "hello"))
    await store.append_meta("tag", {"tag": "x"})

    lines = [json.loads(line) for line in store.path.read_text(encoding="utf-8").splitlines()]
    assert all(entry["v"] == TRANSCRIPT_SCHEMA_VERSION for entry in lines)


async def test_loader_tolerates_records_without_version(tmp_path: Path) -> None:
    # Pre-v1 records (no "v" field) must still load — the version stamp is additive.
    store = TranscriptStore(tmp_path, tmp_path, "sess-legacy")
    legacy = {"type": "message", "role": "user", "content": "old", "uuid": "u1", "parent_uuid": None}
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    loaded = load_transcript(store.path)
    assert [m.content for m in loaded.messages.values()] == ["old"]
