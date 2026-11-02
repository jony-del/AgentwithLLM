from __future__ import annotations

from dataclasses import asdict
import json

from agent_core.codeintel.models import CodeHit, FileVersion, SearchPage
from agent_core.compression import CompressionConfig, CompressionPipeline, shrink_oversize_messages
from agent_core.models import Message


def code_message(name="code_search"):
    hits = [CodeHit(f"src/{i}.py", 1, 1, "x" * 1000, FileVersion("worktree", f"src/{i}.py", "a" * 64),
                    excerpt_start_line=1, excerpt_end_line=1) for i in range(8)]
    page = SearchPage(hits=hits, cursor="NEXT_PAGE_MUST_NOT_SKIP_OMITTED_HITS").to_dict()
    page["coverage"]["complete"] = True
    body = asdict(hits[0]) if name == "code_context" else page
    recovery = {"tool": name, "arguments": {"query": "needle", "kind": "text", "cursor": "original-page"}}
    return Message("tool", f"{name}: {json.dumps(body)}", name=name,
                   metadata={"codeintel": body, "code_recovery": recovery, "tool_call_id": "call-1"})


def body(message):
    return json.loads(message.content.removeprefix(f"{message.name}: "))


def test_snip_preserves_json_references_and_page_recovery():
    original = code_message()
    original_metadata = json.dumps(original.metadata, sort_keys=True)
    pipeline = CompressionPipeline(CompressionConfig(max_message_chars=1200))
    results, _ = pipeline.run_cheap_stages([original])
    result = results[0]
    payload = body(result)
    assert len(result.content) <= 1200
    assert not payload["coverage"]["complete"] and "context_compacted" in payload["coverage"]["reasons"]
    assert payload["hits"] and payload["hits"][0]["version"]["sha256"] == "a" * 64
    assert payload["hits"][0]["truncated"] and payload["hits"][0]["excerpt_end_line"] is None
    assert payload["hits"][0]["next_start_line"] == 1
    assert payload["omitted_hits"] > 0 and payload["cursor"] is None
    assert payload["recovery"]["arguments"]["cursor"] == "original-page"
    assert "NEXT_PAGE_MUST_NOT_SKIP" not in result.content
    assert result.uuid == original.uuid and result.version == original.version + 1
    assert result.metadata["tool_call_id"] == "call-1"
    assert json.dumps(original.metadata, sort_keys=True) == original_metadata
    assert pipeline.run_cheap_stages(results)[0] == results


def test_context_region_is_compressed_as_versioned_reference():
    original = code_message("code_context")
    results, _ = CompressionPipeline(CompressionConfig(max_message_chars=1100)).run_cheap_stages([original])
    payload = body(results[0])
    assert payload["hits"][0]["path"] == "src/0.py"
    assert payload["hits"][0]["text"] == ""
    assert payload["hits"][0]["version"]["path"] == "src/0.py"


def test_last_resort_shrink_does_not_break_structured_evidence():
    original = code_message()
    results = shrink_oversize_messages([original], tokens_to_drop=1000)
    assert results is not None and len(results[0].content) < len(original.content)
    payload = body(results[0])
    assert payload["compacted"] and payload["recovery"]
    assert payload["cursor"] is None and not payload["coverage"]["complete"]


def test_microcompact_retains_explicit_loss_and_recovery():
    original = code_message()
    latest = Message("tool", "ok", name="echo")
    pipeline = CompressionPipeline(CompressionConfig(max_message_chars=100000, microcompact_keep_recent_tool_results=1))
    results, _ = pipeline.run_cheap_stages([original, latest])
    payload = body(results[0])
    assert payload["compacted"] and payload["recovery"]
    assert payload["omitted_hits"] == 8 and payload["cursor"] is None
    assert results[1] is latest


def test_compression_never_resurrects_spilled_metadata():
    original = code_message()
    original.content = "code_search: <tool_output_ref>" + "safe preview " * 1000 + "</tool_output_ref>"
    original.metadata["spilled"] = True
    original.metadata["codeintel"]["hits"][0]["text"] = "MUST_NOT_RESURRECT"
    results, _ = CompressionPipeline(CompressionConfig(max_message_chars=600)).run_cheap_stages([original])
    assert "MUST_NOT_RESURRECT" not in results[0].content
    assert results[0].metadata["spilled"]


def test_pinned_code_evidence_is_preserved():
    original = code_message()
    original.metadata["pinned"] = True
    results, _ = CompressionPipeline(CompressionConfig(max_message_chars=600)).run_cheap_stages([original])
    assert results[0] is original


def test_small_emergency_budget_still_produces_valid_json():
    results, _ = CompressionPipeline(CompressionConfig(max_message_chars=80)).run_cheap_stages([code_message()])
    assert body(results[0])["compacted"]
    assert len(results[0].content) <= 80
