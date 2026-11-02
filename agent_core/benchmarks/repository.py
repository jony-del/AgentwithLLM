"""Offline, paired repository-task evaluation. Never executes input commands."""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def evaluate(manifest: dict, observations: dict, *, k: int = 20) -> dict:
    """Aggregate evaluator-supplied observations, not the agent's completion claim.

    The caller owns the trustworthiness of hidden-test results. Hash binding catches
    accidental dataset/revision mixing; it is not a signed evaluator attestation.
    """
    if manifest.get("schema_version") != 1 or observations.get("schema_version") != 1:
        raise ValueError("unsupported repository evaluation schema")
    if type(k) is not int or not 1 <= k <= 100:
        raise ValueError("invalid retrieval K")
    tasks = manifest.get("tasks")
    runs = observations.get("runs")
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 1000 or not isinstance(runs, list) or len(runs) > 1000:
        raise ValueError("invalid task/run budget")
    dataset = fingerprint(manifest)
    if observations.get("manifest_sha256") != dataset:
        raise ValueError("manifest fingerprint mismatch")
    environment = observations.get("environment")
    if not isinstance(environment, dict) or any(not isinstance(environment.get(n), str) or not environment[n]
            for n in ("model", "configuration_sha256", "hardware")):
        raise ValueError("evaluation requires explicit model/configuration/hardware identity")
    if re.fullmatch(r"[0-9a-f]{64}", environment["configuration_sha256"]) is None:
        raise ValueError("invalid configuration fingerprint")
    indexed = {}
    for task in tasks:
        if not isinstance(task, dict) or any(not isinstance(task.get(n), str) or not task[n] for n in
                ("id", "repository", "base_commit", "problem", "acceptance_id")):
            raise ValueError("invalid task identity")
        if task["id"] in indexed or task.get("category") not in {"single_file", "cross_file", "cross_module"}:
            raise ValueError("duplicate task or invalid category")
        if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", task["base_commit"]) is None:
            raise ValueError("task must pin a complete Git commit identity")
        gold = task.get("relevant_files")
        if not isinstance(gold, list) or not gold or len(gold) > 1000 or any(not isinstance(p, str) or not p for p in gold):
            raise ValueError("invalid relevant files")
        indexed[task["id"]] = task
    records = {}
    for run in runs:
        if not isinstance(run, dict) or run.get("task_id") not in indexed or run["task_id"] in records:
            raise ValueError("unknown or duplicate run; use separate files for repetitions")
        task = indexed[run["task_id"]]
        if run.get("base_commit") != task["base_commit"] or run.get("acceptance_id") != task["acceptance_id"]:
            raise ValueError("run revision/acceptance mismatch")
        if run.get("status") not in {"finished", "failed", "timeout", "environment_error"}:
            raise ValueError("invalid run status")
        for field in ("patch_applied", "target_tests_passed", "hidden_tests_passed"):
            if run.get(field) is not None and type(run[field]) is not bool:
                raise ValueError("invalid outcome")
        for field in ("seconds", "input_tokens", "output_tokens", "tool_calls", "llm_calls"):
            value = run.get(field)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError("invalid metric")
        retrieved = run.get("retrieved_files", [])
        if not isinstance(retrieved, list) or len(retrieved) > 10000 or any(not isinstance(p, str) for p in retrieved):
            raise ValueError("invalid retrieval observations")
        if run.get("hidden_tests_passed") is True:
            if run.get("status") != "finished" or run.get("patch_applied") is not True:
                raise ValueError("inconsistent hidden-test success")
            digest = run.get("evaluated_patch_sha256", "")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("hidden test success requires evaluated patch identity")
        records[run["task_id"]] = run
    results = []
    for key, task in indexed.items():
        run = records.get(key, {})
        ranked = list(dict.fromkeys(run.get("retrieved_files", [])))[:k]
        gold = set(task["relevant_files"])
        relevant = [i + 1 for i, path in enumerate(ranked) if path in gold]
        results.append({"task_id": key, "category": task["category"], "status": run.get("status", "not_run"),
            "recall_at_k": len(relevant) / len(gold), "precision_at_k": len(relevant) / k,
            "reciprocal_rank": 1 / relevant[0] if relevant else 0,
            "patch_applied": run.get("patch_applied") is True,
            "target_tests_passed": run.get("target_tests_passed") is True,
            "resolved": run.get("hidden_tests_passed") is True,
            **{field: run.get(field) for field in ("seconds", "input_tokens", "output_tokens", "tool_calls", "llm_calls")}})
    return {"schema_version": 1, "manifest_sha256": dataset, "k": k,
        "environment": environment, "tasks": len(results), "results": results,
        "rates": {field: sum(r[field] for r in results) / len(results) for field in
                  ("recall_at_k", "precision_at_k", "reciprocal_rank", "patch_applied", "target_tests_passed", "resolved")},
        "not_run": sum(r["status"] == "not_run" for r in results),
        "costs": {field: {"observed_tasks": sum(r[field] is not None for r in results),
                         "total": sum(r[field] for r in results if r[field] is not None)}
                  for field in ("seconds", "input_tokens", "output_tokens", "tool_calls", "llm_calls")},
        "limitations": ["Offline aggregation of evaluator-supplied data, not independent execution or attestation.",
                        "Missing tasks count as unresolved; missing cost metrics remain null.",
                        "Relevant files are evaluator-only labels; never include them in solver prompts.",
                        "Precision@K uses K as denominator; duplicate retrieved paths are removed."]}


def compare(before: dict, after: dict) -> dict:
    if (before["manifest_sha256"], before["k"], before["environment"]) != (
            after["manifest_sha256"], after["k"], after["environment"]):
        raise ValueError("paired comparison requires the same manifest, K and environment")
    return {"schema_version": 1, "manifest_sha256": after["manifest_sha256"],
            "rate_deltas": {key: value - before["rates"][key] for key, value in after["rates"].items()}}
