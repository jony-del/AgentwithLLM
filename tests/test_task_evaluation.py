import json

import pytest

from agent_core.evaluation import evaluate_logs


def test_resumed_task_counted_once_and_outcomes_not_model_claims(tmp_path):
    first, second, third = [tmp_path / name for name in ("a.jsonl", "b.jsonl", "c.jsonl")]
    first.write_text(json.dumps({"event": "task_metrics", "task_id": "same", "ts": 1, "status": "unverified"}) + "\n")
    second.write_text(json.dumps({"event": "task_metrics", "task_id": "same", "ts": 2, "status": "completed",
                                  "verification_attempts": 1, "review_passed": True}) + "\n")
    third.write_text(json.dumps({"event": "final", "answer": "I succeeded"}) + "\n")
    report = evaluate_logs([first, second, third])
    assert report["tasks"] == 1 and report["completed"] == 1
    assert report["runs_without_metrics"] == 1 and report["reviewed_completions"] == 1
    assert report["completed_with_checks"] == 1


@pytest.mark.parametrize("record", [
    {"status": "running"}, {"status": "completed", "duration_seconds": -1},
    {"status": "completed", "input_tokens": "secret"},
])
def test_invalid_metrics_fail_closed(tmp_path, record):
    path = tmp_path / "log.jsonl"
    path.write_text(json.dumps({"event": "task_metrics", **record}) + "\n")
    with pytest.raises(ValueError):
        evaluate_logs([path])


def test_log_budgets_and_missing_data(tmp_path):
    assert evaluate_logs([])["completion_rate"] is None
    path = tmp_path / "huge.jsonl"
    path.write_bytes(b"x" * (1024 * 1024 + 2))
    with pytest.raises(ValueError, match="budget"):
        evaluate_logs([path])
