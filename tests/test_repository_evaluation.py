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
