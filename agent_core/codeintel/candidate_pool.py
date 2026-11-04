"""Unified candidate records: roles, merged evidence and split validation state.

A CandidateRecord is bounded task-local evidence metadata. It never stores raw
source, raw query text or edit credentials, and it never grants an edit
precondition: editing still requires a fresh code_validate/expected-version
check. Anchor validation (mechanical) and relevance validation (semantic) are
recorded separately; workspace or version changes only invalidate a candidate,
and invalidated candidates are excluded until revalidated.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

CANDIDATE_SCHEMA = 1
ROLES = frozenset({"target", "dependency", "test", "unknown"})
VALIDATION_STATES = frozenset({"needs_revalidation", "anchor_valid", "stale", "invalid"})
MAX_EVIDENCE = 16
MAX_TEXT = 512


def candidate_id(path: str, symbol: dict[str, Any] | None, line_range: tuple[int, int] | None) -> str:
    """Task-local identity on the normalized tuple; never a worktree-scoped id."""
    symbol_key = [symbol.get("qualified"), symbol.get("kind"), symbol.get("range")] if isinstance(symbol, dict) else None
    return hashlib.sha256(json.dumps([path, symbol_key, line_range], ensure_ascii=False).encode()).hexdigest()[:24]


def _bounded_text(value: Any) -> str | None:
    return value[:MAX_TEXT] if isinstance(value, str) and value.strip() else None


def _bounded_refs(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        if isinstance(value, str) and value.strip() and value[:MAX_TEXT] not in result:
            result.append(value[:MAX_TEXT])
        if len(result) >= MAX_EVIDENCE:
            break
    return result


def new_candidate(path: str, expected_version: dict[str, Any], *, symbol: dict[str, Any] | None = None,
                  line_range: tuple[int, int] | None = None, origin: str = "unknown",
                  role: str = "unknown") -> dict[str, Any]:
    if role not in ROLES:
        raise ValueError("invalid candidate role")
    return {
        "schema_version": CANDIDATE_SCHEMA,
        "id": candidate_id(path, symbol, line_range),
        "path": path,
        "expected_version": expected_version,
        "symbol": symbol,
        "range": list(line_range) if line_range else None,
        "role": role,
        "evidence_refs": [],
        "supporting": [],
        "contradicting": [],
        "assessment": "",
        "anchor_validation": None,
        "relevance_validation": None,
        "validation_state": "needs_revalidation",
        "invalidation_reason": None,
        "origin": origin[:MAX_TEXT],
    }


def validate_record(record: Any) -> bool:
    """Structural validation for restored records; fail-closed on any malformed field."""
    if not isinstance(record, dict) or record.get("schema_version") != CANDIDATE_SCHEMA:
        return False
    if (not isinstance(record.get("id"), str) or not 0 < len(record["id"]) <= 64
            or not isinstance(record.get("path"), str) or not 0 < len(record["path"]) <= 4096):
        return False
    version = record.get("expected_version")
    if (not isinstance(version, dict) or version.get("path") != record["path"]
            or not isinstance(version.get("worktree_id"), str)
            or not isinstance(version.get("sha256"), str) or len(version.get("sha256", "")) != 64):
        return False
    if record.get("role") not in ROLES or record.get("validation_state") not in VALIDATION_STATES:
        return False
    for key in ("evidence_refs", "supporting", "contradicting"):
        if not isinstance(record.get(key), list) or len(record[key]) > MAX_EVIDENCE:
            return False
    for key in ("assessment", "invalidation_reason", "origin"):
        if record.get(key) is not None and not isinstance(record[key], str):
            return False
    symbol = record.get("symbol")
    if symbol is not None and not isinstance(symbol, dict):
        return False
    line_range = record.get("range")
    if line_range is not None and (not isinstance(line_range, list) or len(line_range) != 2
                                   or any(type(v) is not int or v < 1 for v in line_range)):
        return False
    for key in ("anchor_validation", "relevance_validation"):
        if record.get(key) is not None and not isinstance(record[key], dict):
            return False
    return True


def merge_evidence(record: dict[str, Any], *, evidence_ref: str, note: str | None = None,
                   contradicts: bool = False, role: str | None = None,
                   expected_version: dict[str, Any] | None = None) -> bool:
    """Merge one evidence observation; returns True when the record changed."""
    changed = False
    ref = evidence_ref[:MAX_TEXT]
    if ref not in record["evidence_refs"] and len(record["evidence_refs"]) < MAX_EVIDENCE:
        record["evidence_refs"].append(ref)
        changed = True
    note_text = _bounded_text(note)
    bucket = "contradicting" if contradicts else "supporting"
    if note_text and note_text not in record[bucket] and len(record[bucket]) < MAX_EVIDENCE:
        record[bucket].append(note_text)
        changed = True
    if role is not None and role in ROLES and record["role"] == "unknown":
        record["role"] = role
        changed = True
    if expected_version is not None and expected_version != record["expected_version"]:
        record["expected_version"] = expected_version
        if record["validation_state"] == "anchor_valid":
            record["validation_state"] = "needs_revalidation"
            record["invalidation_reason"] = "observed version changed"
        changed = True
    if contradicts:
        record["validation_state"] = "invalid" if record["validation_state"] != "stale" else "stale"
        record["invalidation_reason"] = note_text or "contradicting evidence"
        changed = True
    return changed


def record_anchor_validation(record: dict[str, Any], *, ok: bool, version: dict[str, Any],
                             anchor_sha256: str | None = None, detail: str | None = None) -> None:
    """Mechanical anchor outcome only; relevance stays a separate, unproven field."""
    record["anchor_validation"] = {"status": "valid" if ok else "invalid",
                                   "version_sha256": version.get("sha256", "")[:64],
                                   "anchor_sha256": (anchor_sha256 or "")[:64],
                                   "detail": _bounded_text(detail)}
    if ok:
        record["expected_version"] = version
        record["validation_state"] = "anchor_valid"
        record["invalidation_reason"] = None
    else:
        # A stale candidate stays stale: the superseding change, not the anchor, is why.
        if record["validation_state"] != "stale":
            record["validation_state"] = "invalid"
        record["invalidation_reason"] = _bounded_text(detail) or "anchor validation failed"
        if detail:
            merge_evidence(record, evidence_ref="code_validate", note=detail, contradicts=True)


def invalidate(record: dict[str, Any], reason: str) -> None:
    record["validation_state"] = "stale"
    record["invalidation_reason"] = reason[:MAX_TEXT]
