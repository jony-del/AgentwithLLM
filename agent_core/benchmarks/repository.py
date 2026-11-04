"""Offline, paired repository-task evaluation. Never executes input commands.

Schema v2 adds symbol/edit-range gold annotations and separately recorded
retrieval, selection, mechanical-validation and patch-location observations.
Symbol identity is the normalized (path, qualified, kind, range, content
version) tuple; worktree-scoped symbol_reference.id values are never compared
across workspaces. Missing gold annotations or unreported observation sections
produce null metrics that leave both the success and the failure denominator.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

SUPPORTED_SCHEMAS = (1, 2)
ROLES = {"target", "dependency", "test"}
RATE_FIELDS = ("recall_at_k", "precision_at_k", "reciprocal_rank",
               "patch_applied", "target_tests_passed", "resolved")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _bounded_str(value: Any, limit: int = 1024) -> bool:
    return isinstance(value, str) and 0 < len(value) <= limit


def _line_range(value: Any) -> bool:
    return (isinstance(value, list) and len(value) == 2
            and all(type(v) is int and v >= 1 for v in value) and value[0] <= value[1])


def _valid_symbol_gold(entry: Any) -> bool:
    """Normalized gold symbol identity; no worktree-scoped reference ids."""
    if not isinstance(entry, dict) or not _bounded_str(entry.get("path")):
        return False
    for key in ("qualified", "kind", "content_version"):
        if entry.get(key) is not None and not _bounded_str(entry[key], 4096):
            return False
    if entry.get("range") is not None and not _line_range(entry["range"]):
        return False
    return any(entry.get(key) is not None for key in ("qualified", "kind", "range", "content_version"))


def _valid_symbol_view(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    for key in ("qualified", "kind", "content_version"):
        if value.get(key) is not None and not _bounded_str(value[key], 4096):
            return False
    return value.get("range") is None or _line_range(value["range"])


def _valid_candidate(entry: Any) -> bool:
    if not isinstance(entry, dict) or not _bounded_str(entry.get("path")):
        return False
    if entry.get("symbol") is not None and not _valid_symbol_view(entry["symbol"]):
        return False
    return all(entry.get(key) is None or (type(entry[key]) is int and entry[key] >= 1)
               for key in ("start_line", "end_line"))


def _valid_patch_location(entry: Any) -> bool:
    return (isinstance(entry, dict) and _bounded_str(entry.get("path"))
            and type(entry.get("start_line")) is int and type(entry.get("end_line")) is int
            and 1 <= entry["start_line"] <= entry["end_line"])


def _symbol_matches(gold: dict, candidate: dict) -> bool:
    if candidate.get("path") != gold["path"]:
        return False
    symbol = candidate.get("symbol")
    if not isinstance(symbol, dict):
        symbol = {}
    return all(symbol.get(key) == gold[key]
               for key in ("qualified", "kind", "range", "content_version") if gold.get(key) is not None)


def _location_matches(gold_range: dict, location: dict) -> bool:
    return (location.get("path") == gold_range["path"]
            and location["start_line"] <= gold_range["end_line"]
            and gold_range["start_line"] <= location["end_line"])


def _ranked_metrics(ranked: list[dict] | None, gold: list[dict] | None, k: int,
                    match: Any, identity: Any) -> dict[str, float] | None:
    """Precision/Recall/MRR at K; None when annotation or observation is missing."""
    if gold is None or ranked is None:
        return None
    deduped: list[dict] = []
    seen = set()
    for item in ranked:
        key = identity(item)
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    top = deduped[:k]
    hits = [i + 1 for i, item in enumerate(top) if any(match(g, item) for g in gold)]
    covered = {id(g) for g in gold if any(match(g, item) for item in top)}
    return {"precision": len(hits) / k, "recall": len(covered) / len(gold),
            "mrr": 1 / hits[0] if hits else 0}


def _candidate_identity(item: dict) -> str:
    return json.dumps([item.get("path"), item.get("symbol")], sort_keys=True, ensure_ascii=False)


def _location_identity(item: dict) -> str:
    return json.dumps([item.get("path"), item.get("start_line"), item.get("end_line")])


def _rate(value: dict[str, float] | None, key: str) -> float | None:
    return None if value is None else value[key]


def evaluate(manifest: dict, observations: dict, *, k: int = 20) -> dict:
    """Aggregate evaluator-supplied observations, not the agent's completion claim.

    The caller owns the trustworthiness of hidden-test results. Hash binding catches
    accidental dataset/revision mixing; it is not a signed evaluator attestation.
    """
    manifest_version = manifest.get("schema_version")
    observation_version = observations.get("schema_version")
    if manifest_version not in SUPPORTED_SCHEMAS or observation_version not in SUPPORTED_SCHEMAS:
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
        symbols = task.get("relevant_symbols")
        if symbols is not None and (not isinstance(symbols, list) or not symbols or len(symbols) > 1000
                                    or any(not _valid_symbol_gold(s) for s in symbols)):
            raise ValueError("invalid relevant symbols")
        edit_ranges = task.get("edit_ranges")
        if edit_ranges is not None and (not isinstance(edit_ranges, list) or not edit_ranges
                                        or len(edit_ranges) > 1000 or any(not _valid_patch_location(r) for r in edit_ranges)):
            raise ValueError("invalid edit ranges")
        roles = task.get("roles")
        if roles is not None and (not isinstance(roles, dict) or len(roles) > 1000
                                  or any(not _bounded_str(p) or p not in gold for p in roles)
                                  or any(r not in ROLES for r in roles.values())):
            raise ValueError("invalid gold roles")
        sources = task.get("gold_sources")
        if sources is not None and (not isinstance(sources, dict) or len(sources) > 1000
                                    or any(not _bounded_str(p) or not _bounded_str(s, 4096) for p, s in sources.items())):
            raise ValueError("invalid gold evidence sources")
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
        for field in ("retrieved_candidates", "selected_candidates"):
            value = run.get(field)
            if value is not None and (not isinstance(value, list) or len(value) > 10000
                                      or any(not _valid_candidate(c) for c in value)):
                raise ValueError("invalid candidate observations")
        validated = run.get("validated_candidates")
        if validated is not None and (not isinstance(validated, list) or len(validated) > 10000 or any(
                not isinstance(v, dict) or not _bounded_str(v.get("path")) or type(v.get("valid")) is not bool
                or (v.get("symbol") is not None and not _valid_symbol_view(v["symbol"]))
                or (v.get("detail") is not None and not _bounded_str(v["detail"], 512)) for v in validated)):
            raise ValueError("invalid candidate validation observations")
        locations = run.get("patch_locations")
        if locations is not None and (not isinstance(locations, list) or len(locations) > 10000
                                      or any(not _valid_patch_location(p) for p in locations)):
            raise ValueError("invalid patch location observations")
        if run.get("hidden_tests_passed") is True:
            if run.get("status") != "finished" or run.get("patch_applied") is not True:
                raise ValueError("inconsistent hidden-test success")
            digest = run.get("evaluated_patch_sha256", "")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("hidden test success requires evaluated patch identity")
        records[run["task_id"]] = run
    version = max(manifest_version, observation_version)
    results = []
    for key, task in indexed.items():
        run = records.get(key, {})
        ranked = list(dict.fromkeys(run.get("retrieved_files", [])))[:k]
        gold = set(task["relevant_files"])
        relevant = [i + 1 for i, path in enumerate(ranked) if path in gold]
        result = {"task_id": key, "category": task["category"], "status": run.get("status", "not_run"),
            "recall_at_k": len(relevant) / len(gold), "precision_at_k": len(relevant) / k,
            "reciprocal_rank": 1 / relevant[0] if relevant else 0,
            "patch_applied": run.get("patch_applied") is True,
            "target_tests_passed": run.get("target_tests_passed") is True,
            "resolved": run.get("hidden_tests_passed") is True,
            **{field: run.get(field) for field in ("seconds", "input_tokens", "output_tokens", "tool_calls", "llm_calls")}}
        if version == 2:
            symbol_gold = task.get("relevant_symbols")
            edit_gold = task.get("edit_ranges")
            retrieved_candidates = run.get("retrieved_candidates")
            selected = run.get("selected_candidates")
            locations = run.get("patch_locations")
            retrieval_symbols = _ranked_metrics(retrieved_candidates, symbol_gold, k, _symbol_matches, _candidate_identity)
            file_gold = [{"path": p} for p in task["relevant_files"]]
            selection_files = _ranked_metrics(selected, file_gold, k,
                lambda g, c: c.get("path") == g["path"], _candidate_identity)
            selection_symbols = _ranked_metrics(selected, symbol_gold, k, _symbol_matches, _candidate_identity)
            patch_ranges = _ranked_metrics(locations, edit_gold, k, _location_matches, _location_identity)
            validated = run.get("validated_candidates")
            roles = task.get("roles")
            result.update({
                "symbol_recall_at_k": _rate(retrieval_symbols, "recall"),
                "symbol_precision_at_k": _rate(retrieval_symbols, "precision"),
                "symbol_reciprocal_rank": _rate(retrieval_symbols, "mrr"),
                "selected_file_recall_at_k": _rate(selection_files, "recall"),
                "selected_file_precision_at_k": _rate(selection_files, "precision"),
                "selected_file_reciprocal_rank": _rate(selection_files, "mrr"),
                "selected_symbol_recall_at_k": _rate(selection_symbols, "recall"),
                "selected_symbol_precision_at_k": _rate(selection_symbols, "precision"),
                "selected_symbol_reciprocal_rank": _rate(selection_symbols, "mrr"),
                "edit_range_recall_at_k": _rate(patch_ranges, "recall"),
                "edit_range_precision_at_k": _rate(patch_ranges, "precision"),
                "edit_range_reciprocal_rank": _rate(patch_ranges, "mrr"),
                "candidates_validated": None if validated is None else len(validated),
                "validation_rate": None if not validated else
                    sum(1 for v in validated if v["valid"]) / len(validated),
            })
            if roles:
                observed_paths = set(ranked)
                result["role_recall_at_k"] = {role: len(observed_paths & {p for p, r in roles.items() if r == role})
                    / len({p for p, r in roles.items() if r == role}) for role in sorted(set(roles.values()))}
            if task.get("gold_sources"):
                result["gold_sources"] = task["gold_sources"]
        results.append(result)
    rates: dict[str, float] = {field: sum(r[field] for r in results) / len(results) for field in RATE_FIELDS}
    report = {"schema_version": version, "manifest_sha256": dataset, "k": k,
        "environment": environment, "tasks": len(results), "results": results,
        "rates": rates,
        "not_run": sum(r["status"] == "not_run" for r in results),
        "costs": {field: {"observed_tasks": sum(r[field] is not None for r in results),
                         "total": sum(r[field] for r in results if r[field] is not None)}
                  for field in ("seconds", "input_tokens", "output_tokens", "tool_calls", "llm_calls")},
        "limitations": ["Offline aggregation of evaluator-supplied data, not independent execution or attestation.",
                        "Missing tasks count as unresolved; missing cost metrics remain null.",
                        "Relevant files are evaluator-only labels; never include them in solver prompts.",
                        "Precision@K uses K as denominator; duplicate retrieved paths are removed."]}
    if version == 2:
        observed: dict[str, int] = {}
        for field in ("symbol_recall_at_k", "symbol_precision_at_k", "symbol_reciprocal_rank",
                      "selected_file_recall_at_k", "selected_file_precision_at_k", "selected_file_reciprocal_rank",
                      "selected_symbol_recall_at_k", "selected_symbol_precision_at_k", "selected_symbol_reciprocal_rank",
                      "edit_range_recall_at_k", "edit_range_precision_at_k", "edit_range_reciprocal_rank",
                      "validation_rate"):
            values = [r[field] for r in results if r.get(field) is not None]
            observed[field] = len(values)
            if values:
                rates[field] = sum(values) / len(values)
        report["metric_observation_counts"] = observed
        role_values: dict[str, list[float]] = {}
        for r in results:
            for role, value in (r.get("role_recall_at_k") or {}).items():
                role_values.setdefault(role, []).append(value)
        if role_values:
            report["roles"] = {role: {"recall_at_k": sum(v) / len(v), "tasks": len(v)}
                               for role, v in sorted(role_values.items())}
        report["limitations"] += [
            "Symbol/edit-range metrics are null when gold annotations or observation sections are absent; "
            "null metrics join neither success nor failure denominators.",
            "Symbol identity is the normalized (path, qualified, kind, range, content version) tuple; "
            "worktree-scoped symbol ids are never compared.",
            "Edit-range matching is interval overlap on one-based inclusive lines."]
    return report


def compare(before: dict, after: dict, *, treatment: tuple[str, ...] | list[str] = (),
            paired: bool = True) -> dict:
    """Paired or unpaired comparison with explicitly declared treatment variables.

    Environment fields may differ only when declared in ``treatment`` (model,
    hardware, configuration, task order, ...); undeclared differences fail.
    """
    if (before["manifest_sha256"], before["k"]) != (after["manifest_sha256"], after["k"]):
        raise ValueError("paired comparison requires the same manifest, K and declared treatment scope")
    declared = set(treatment)
    differing = {key for key in set(before["environment"]) | set(after["environment"])
                 if before["environment"].get(key) != after["environment"].get(key)}
    undeclared = differing - declared
    if undeclared:
        raise ValueError("paired comparison requires the same manifest, K and environment; "
                         f"undeclared environment differences {sorted(undeclared)} must be declared as treatment")
    keys = [key for key in after["rates"] if key in before["rates"]]
    report = {"schema_version": 2, "manifest_sha256": after["manifest_sha256"],
            "treatment": sorted(declared), "paired": paired,
            "rate_deltas": {key: after["rates"][key] - before["rates"][key] for key in keys}}
    if paired:
        earlier = {r["task_id"]: r for r in before["results"]}
        later = {r["task_id"]: r for r in after["results"]}
        if set(earlier) != set(later):
            raise ValueError("paired comparison requires identical task sets")
        deltas = []
        for task_id in sorted(earlier):
            entry = {"task_id": task_id}
            for key in keys:
                old, new = earlier[task_id].get(key), later[task_id].get(key)
                if old is not None and new is not None:
                    entry[key] = new - old
            deltas.append(entry)
        report["task_deltas"] = deltas
        report["paired_outcomes"] = {key: {
            "improved": sum(1 for d in deltas if d.get(key, 0) > 0),
            "regressed": sum(1 for d in deltas if d.get(key, 0) < 0),
            "unchanged": sum(1 for d in deltas if key in d and d[key] == 0)} for key in keys}
    return report
