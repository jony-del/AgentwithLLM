"""Unified candidate records: merged evidence, roles, split validation, persistence (S3)."""
from __future__ import annotations

import json

import pytest

from agent_core.codeintel.config import CodeIntelConfig
from agent_core.codeintel.models import CodeHit, FileVersion, StaleEvidence
from agent_core.codeintel.runtime import ensure_working_set, get_service, notify_committed, release_service
from agent_core.codeintel.snapshots import worktree_id
from agent_core.codeintel.working_set import MAX_CANDIDATES, TaskWorkingSet
from agent_core.session import SessionContext
from agent_core.task_runtime import TaskContract, TaskRun, capture_revision
from agent_core.tools.codeintel import CodeSearchTool, CodeValidateTool
from agent_core.tools.task_control import TaskStateTool


def make_working(tmp_path):
    working = TaskWorkingSet()
    working.binding = (worktree_id(tmp_path), "")
    return working


def hit(path, *, sha="a" * 64, symbol=None, lines=(1, 3), root_id=None):
    return CodeHit(path, lines[0], lines[1], "body", FileVersion(root_id or "wt", path, sha),
                   kind="symbol", symbol=symbol)


def test_text_symbol_lsp_and_relation_evidence_merge_into_one_candidate(tmp_path):
    working = make_working(tmp_path)
    identity = working.binding[0]
    symbol = {"qualified_name": "a.needle", "kind": "function"}
    working.add_candidate(hit("a.py", symbol=symbol, root_id=identity), origin="code_search",
                          evidence_ref="code_search:symbol")
    working.add_candidate(hit("a.py", symbol=symbol, root_id=identity), origin="lsp",
                          evidence_ref="lsp:definition")
    working.add_candidate(hit("a.py", symbol=symbol, root_id=identity), origin="code_relations",
                          evidence_ref="code_relations:calls:d2", role="target")
    assert len(working.candidates) == 1
    record = next(iter(working.candidates.values()))
    assert record["evidence_refs"] == ["code_search:symbol", "lsp:definition", "code_relations:calls:d2"]
    assert record["role"] == "target"
    assert record["validation_state"] == "needs_revalidation"
    assert "body" not in json.dumps(record)  # no raw source persisted


def test_contradicting_evidence_is_visible_and_blocks_use(tmp_path):
    working = make_working(tmp_path)
    record = working.add_candidate(hit("a.py", root_id=working.binding[0]), origin="code_search",
                                   evidence_ref="code_search:text")
    working.record_validation("a.py", ok=False, version=record["expected_version"],
                              detail="candidate version changed; retrieve current evidence")
    assert record["validation_state"] == "invalid"
    assert record["anchor_validation"]["status"] == "invalid"
    assert record["contradicting"] and "version changed" in record["contradicting"][0]


def test_anchor_and_relevance_validation_stay_separate(tmp_path):
    working = make_working(tmp_path)
    record = working.add_candidate(hit("a.py", root_id=working.binding[0]), origin="code_search",
                                   evidence_ref="code_search:text")
    working.record_validation("a.py", ok=True, version=record["expected_version"], anchor_sha256="b" * 64)
    assert record["validation_state"] == "anchor_valid"
    assert record["anchor_validation"]["status"] == "valid"
    assert record["relevance_validation"] is None  # mechanical validation never proves relevance
    newer_sha = "c" * 64
    working.add_candidate(hit("a.py", sha=newer_sha, root_id=working.binding[0]),
                          origin="code_search", evidence_ref="code_search:text")
    assert record["validation_state"] == "needs_revalidation"  # version moved under the candidate


def test_committed_paths_invalidate_candidates_without_editing_authority(tmp_path):
    working = make_working(tmp_path)
    record = working.add_candidate(hit("a.py", root_id=working.binding[0]), origin="code_search",
                                   evidence_ref="code_search:text")
    working.record_validation("a.py", ok=True, version=record["expected_version"], anchor_sha256="b" * 64)
    working.invalidate(("a.py",))
    assert record["validation_state"] == "stale"
    assert record["invalidation_reason"] == "committed paths changed"


def test_candidates_persist_bounded_and_restore_as_needs_revalidation(tmp_path):
    working = make_working(tmp_path)
    identity = working.binding[0]
    for index in range(MAX_CANDIDATES + 10):
        working.add_candidate(hit(f"src/{index}.py", root_id=identity), origin="code_search",
                              evidence_ref="code_search:text")
    assert len(working.candidates) == MAX_CANDIDATES
    saved = working.to_state()
    assert saved["v"] == 2
    assert len(json.dumps(saved, ensure_ascii=False).encode()) <= 128 * 1024
    assert "body" not in json.dumps(saved) and "code_search:text" in json.dumps(saved)
    restored = make_working(tmp_path)
    restored.restore(saved, allowed=lambda path: not path.endswith("9.py"))
    assert restored.candidates
    assert all(r["validation_state"] == "needs_revalidation" for r in restored.candidates.values())
    assert all(r["invalidation_reason"] == "restored from history" for r in restored.candidates.values())
    denied = [r for r in restored.candidates.values() if r["path"].endswith("9.py")]
    assert not denied


def test_v1_state_restores_and_malformed_candidates_fail_closed(tmp_path):
    working = make_working(tmp_path)
    identity = working.binding[0]
    working.add(hit("a.py", root_id=identity))
    saved_v1 = working.to_state()
    assert saved_v1["v"] == 1
    restored = make_working(tmp_path)
    restored.restore(saved_v1, allowed=lambda _: True)
    assert restored.entries and not restored.candidates
    broken = make_working(tmp_path)
    broken.add_candidate(hit("a.py", root_id=identity), origin="code_search", evidence_ref="code_search:text")
    saved_v2 = broken.to_state()
    for change in ("schema_version", "path", "expected_version", "role"):
        damaged = json.loads(json.dumps(saved_v2))
        damaged["candidates"][0][change] = {"not": "valid"} if change != "role" else "owner"
        with pytest.raises(ValueError, match="candidate"):
            make_working(tmp_path).restore(damaged, allowed=lambda _: True)
    damaged = json.loads(json.dumps(saved_v2))
    damaged["candidates"][0]["path"] = "../escape.py"
    with pytest.raises(ValueError, match="candidate"):
        make_working(tmp_path).restore(damaged, allowed=lambda _: True)


async def test_validate_tool_records_anchor_outcome_on_candidate(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def needle():\n    return 1\n", encoding="utf-8")
    session = SessionContext(workspace=root,
                             task_run=TaskRun(TaskContract("validate"), capture_revision(root)),
                             codeintel_config=CodeIntelConfig(maintain=False, watch=False))
    service = await get_service(session)
    try:
        for _ in range(20):
            status = await service.ensure_index()
            if status["catalog_complete"] and not status["pending"]:
                break
        search = json.loads((await CodeSearchTool(session).run({"query": "needle", "kind": "symbol", "path": "."})).content)
        version = search["hits"][0]["version"]
        working = ensure_working_set(session)
        assert working.candidates  # search created a candidate
        ok = json.loads((await CodeValidateTool(session).run(
            {"path": "a.py", "expected_version": version, "anchor": "def needle():"})).content)
        assert ok["status"] == "validated_anchor"
        record = working.candidate_for("a.py")
        assert record["validation_state"] == "anchor_valid"
        assert record["anchor_validation"]["anchor_sha256"] == ok["anchor_sha256"]
        page = json.loads(TaskStateTool(session)._invoke({"section": "code_candidates"}).content)
        assert page["total"] >= 1 and page["page"][0]["validation_state"] == "anchor_valid"
        (root / "a.py").write_text("def needle():\n    return 2\n", encoding="utf-8")
        await notify_committed(session, ("a.py",), "edit")
        assert record["validation_state"] == "stale"
        with pytest.raises(StaleEvidence):
            await CodeValidateTool(session).run({"path": "a.py", "expected_version": version, "anchor": "def needle():"})
        assert record["validation_state"] == "stale"  # stale stays stale; the failure is visible
        assert any("version changed" in note for note in record["contradicting"])
    finally:
        await release_service(session)
