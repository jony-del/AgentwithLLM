from copy import deepcopy

import pytest

from agent_core.benchmarks.repository import compare, evaluate, fingerprint


def inputs():
    manifest = {"schema_version": 1, "tasks": [dict(id=str(i), repository="fixture", base_commit="a" * 40,
        problem="fix", acceptance_id="hidden-v1", category=category, relevant_files=["a.py", "b.py"])
        for i, category in enumerate(("single_file", "cross_file", "cross_module"))]}
    observations = {"schema_version": 1, "manifest_sha256": fingerprint(manifest),
        "environment": {"model": "scripted", "configuration_sha256": "b" * 64, "hardware": "fixture"},
        "runs": [dict(task_id="0", base_commit="a" * 40, acceptance_id="hidden-v1", status="finished",
                      retrieved_files=["unrelated", "a.py", "a.py"], patch_applied=True, target_tests_passed=True)]}
    return manifest, observations


def test_retrieval_patch_and_local_tests_do_not_imply_resolution():
    manifest, observed = inputs()
    result = evaluate(manifest, observed, k=2)
    assert result["not_run"] == 2
    assert result["rates"]["resolved"] == 0
    assert result["rates"]["patch_applied"] == 1 / 3
    assert result["results"][0]["recall_at_k"] == .5
    assert result["results"][0]["precision_at_k"] == .5
    assert result["results"][0]["reciprocal_rank"] == .5
    assert result["results"][1]["seconds"] is None


@pytest.mark.parametrize("change", [
    {"base_commit": "wrong"}, {"acceptance_id": "wrong"}, {"seconds": float("nan")},
    {"tool_calls": -1}, {"hidden_tests_passed": True}, {"patch_applied": "true"},
    {"status": "completed"}, {"retrieved_files": "a.py"},
])
def test_malformed_or_unbound_results_fail_closed(change):
    manifest, observed = inputs()
    observed["runs"][0].update(change)
    with pytest.raises(ValueError):
        evaluate(manifest, observed)


def test_success_is_patch_bound_and_pairing_is_fixed():
    manifest, observed = inputs()
    before = evaluate(manifest, observed)
    observed["runs"][0].update(hidden_tests_passed=True, evaluated_patch_sha256="a" * 64)
    after = evaluate(manifest, observed)
    assert compare(before, after)["rate_deltas"]["resolved"] == 1 / 3
    other = deepcopy(after)
    other["environment"] = {"model": "different"}
    with pytest.raises(ValueError, match="same manifest"):
        compare(before, other)
    observed["runs"].append(dict(observed["runs"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(manifest, observed)


def test_missing_timeout_and_environment_failures_stay_in_denominator():
    manifest, observed = inputs()
    observed["runs"][0]["status"] = "timeout"
    assert evaluate(manifest, observed)["rates"]["resolved"] == 0
    observed["manifest_sha256"] = "wrong"
    with pytest.raises(ValueError, match="fingerprint"):
        evaluate(manifest, observed)


def v2_inputs():
    manifest = {"schema_version": 2, "tasks": [{
        "id": "t1", "repository": "fixture", "base_commit": "a" * 40,
        "problem": "fix", "acceptance_id": "hidden-v1", "category": "cross_file",
        "relevant_files": ["src/a.py", "src/b.py"],
        "relevant_symbols": [
            {"path": "src/a.py", "qualified": "a.run", "kind": "function", "range": [10, 20]},
            {"path": "src/b.py", "qualified": "b.Helper", "kind": "class"}],
        "edit_ranges": [{"path": "src/a.py", "start_line": 12, "end_line": 18}],
        "roles": {"src/a.py": "target", "src/b.py": "dependency"},
        "gold_sources": {"src/a.py": "maintainer-review"}},
        {"id": "t2", "repository": "fixture", "base_commit": "a" * 40,
         "problem": "fix", "acceptance_id": "hidden-v1", "category": "single_file",
         "relevant_files": ["src/c.py"]}]}
    observations = {"schema_version": 2, "manifest_sha256": fingerprint(manifest),
        "environment": {"model": "scripted", "configuration_sha256": "b" * 64, "hardware": "fixture"},
        "runs": [{
            "task_id": "t1", "base_commit": "a" * 40, "acceptance_id": "hidden-v1", "status": "finished",
            "retrieved_files": ["src/a.py", "src/z.py"],
            "retrieved_candidates": [
                {"path": "src/z.py", "symbol": {"qualified": "z.run", "kind": "function"}},
                {"path": "src/a.py", "symbol": {"qualified": "a.run", "kind": "function", "range": [10, 20],
                                                "id": "worktree-scoped-id-ignored"}}],
            "selected_candidates": [
                {"path": "src/a.py", "symbol": {"qualified": "a.run", "kind": "function", "range": [10, 20]}}],
            "validated_candidates": [
                {"path": "src/a.py", "valid": True}, {"path": "src/z.py", "valid": False}],
            "patch_locations": [{"path": "src/a.py", "start_line": 14, "end_line": 16}],
            "patch_applied": True, "target_tests_passed": True}]}
    return manifest, observations


def test_v2_symbol_selection_and_edit_range_metrics():
    manifest, observed = v2_inputs()
    result = evaluate(manifest, observed, k=2)
    row = result["results"][0]
    assert row["symbol_recall_at_k"] == 0.5       # one of two gold symbols matched
    assert row["symbol_precision_at_k"] == 0.5    # one of K=2 ranked candidates hit
    assert row["symbol_reciprocal_rank"] == 0.5   # first hit at rank 2
    assert row["selected_file_recall_at_k"] == 0.5
    assert row["selected_symbol_recall_at_k"] == 0.5
    assert row["edit_range_recall_at_k"] == 1.0   # patch location overlaps the allowed range
    assert row["edit_range_precision_at_k"] == 0.5
    assert row["validation_rate"] == 0.5
    assert row["candidates_validated"] == 2
    assert row["role_recall_at_k"] == {"target": 1.0, "dependency": 0.0}
    assert result["roles"]["target"] == {"recall_at_k": 1.0, "tasks": 1}
    assert result["rates"]["symbol_recall_at_k"] == 0.5  # t2 has no symbol gold: excluded from the mean
    assert result["metric_observation_counts"]["symbol_recall_at_k"] == 1


def test_missing_annotations_and_sections_are_null_neither_success_nor_failure():
    manifest, observed = v2_inputs()
    result = evaluate(manifest, observed, k=2)
    row = result["results"][1]  # t2: no symbol/edit-range gold, no candidate observations
    for key in ("symbol_recall_at_k", "selected_file_recall_at_k", "edit_range_recall_at_k", "validation_rate"):
        assert row[key] is None
    assert result["metric_observation_counts"]["edit_range_recall_at_k"] == 1  # only t1 observed
    # Reported empty candidate lists score zero rather than null.
    observed["runs"][0]["retrieved_candidates"] = []
    row = evaluate(manifest, observed, k=2)["results"][0]
    assert row["symbol_recall_at_k"] == 0 and row["symbol_precision_at_k"] == 0


def test_symbol_matching_uses_normalized_identity_never_worktree_ids():
    manifest, observed = v2_inputs()
    candidate = observed["runs"][0]["retrieved_candidates"][1]
    candidate["symbol"]["id"] = "completely-different-worktree-id"
    assert evaluate(manifest, observed, k=2)["results"][0]["symbol_recall_at_k"] == 0.5
    candidate["symbol"]["qualified"] = "a.other"  # id now matches nothing; identity must not rescue it
    candidate["symbol"]["id"] = "some-id"
    assert evaluate(manifest, observed, k=2)["results"][0]["symbol_recall_at_k"] == 0


def test_v2_malformed_annotations_and_observations_fail_closed():
    manifest, observed = v2_inputs()
    for change in ({"relevant_symbols": [{"path": "src/a.py"}]},          # no identity at all
                   {"edit_ranges": [{"path": "src/a.py", "start_line": 5, "end_line": 2}]},
                   {"roles": {"src/a.py": "owner"}},
                   {"roles": {"src/other.py": "target"}}):
        broken = deepcopy(manifest)
        broken["tasks"][0].update(change)
        with pytest.raises(ValueError):
            evaluate(broken, {**observed, "manifest_sha256": fingerprint(broken)})
    for change in ({"retrieved_candidates": [{"path": "src/a.py", "symbol": {"range": [3, 1]}}]},
                   {"validated_candidates": [{"path": "src/a.py", "valid": "yes"}]},
                   {"patch_locations": [{"path": "src/a.py", "start_line": 0, "end_line": 1}]}):
        broken_run = deepcopy(observed)
        broken_run["runs"][0].update(change)
        with pytest.raises(ValueError):
            evaluate(manifest, broken_run)


def test_compare_declares_treatment_and_supports_unpaired():
    manifest, observed = v2_inputs()
    before = evaluate(manifest, observed, k=2)
    changed = deepcopy(observed)
    changed["environment"]["model"] = "scripted-v2"
    changed["runs"][0]["retrieved_candidates"] = [
        {"path": "src/a.py", "symbol": {"qualified": "a.run", "kind": "function", "range": [10, 20]}},
        {"path": "src/b.py", "symbol": {"qualified": "b.Helper", "kind": "class"}}]
    after = evaluate(manifest, changed, k=2)
    with pytest.raises(ValueError, match="same manifest"):
        compare(before, after)  # model differs but is not declared
    paired = compare(before, after, treatment=("model",))
    assert paired["treatment"] == ["model"] and paired["paired"] is True
    assert paired["rate_deltas"]["symbol_recall_at_k"] == 0.5  # 0.5 -> 1.0 on the single observed task
    assert paired["paired_outcomes"]["symbol_recall_at_k"] == {"improved": 1, "regressed": 0, "unchanged": 0}
    unpaired = compare(before, after, treatment=("model",), paired=False)
    assert "task_deltas" not in unpaired and unpaired["rate_deltas"]["symbol_recall_at_k"] == 0.5


def test_v1_manifest_and_observations_keep_legacy_output_shape():
    manifest, observed = inputs()
    result = evaluate(manifest, observed, k=2)
    assert result["schema_version"] == 1
    assert "symbol_recall_at_k" not in result["results"][0]
    assert "metric_observation_counts" not in result
    comparison = compare(evaluate(manifest, observed), evaluate(manifest, observed))
    assert comparison["rate_deltas"]["resolved"] == 0
