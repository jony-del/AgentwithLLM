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


async def test_run_logger_write_after_close_reopens(tmp_path: Path) -> None:
    # close() is idempotent and never bricks the logger: a later write reopens.
    logger = JSONLRunLogger(tmp_path)
    await logger.write("first", {})
    logger.close()
    logger.close()  # idempotent
    await logger.write("second", {})
    logger.close()

    events = [json.loads(line)["event"] for line in logger.path.read_text(encoding="utf-8").splitlines()]
    assert events == ["first", "second"]


async def test_run_logger_concurrent_writes_stay_line_atomic(tmp_path: Path) -> None:
    # Overlapping to_thread writers share one held handle; every line must parse.
    import asyncio

    logger = JSONLRunLogger(tmp_path)
    await asyncio.gather(*(logger.write("evt", {"i": i, "pad": "x" * 200}) for i in range(50)))
    logger.close()

    lines = logger.path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]  # raises on a torn line
    assert sorted(record["i"] for record in records) == list(range(50))


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
