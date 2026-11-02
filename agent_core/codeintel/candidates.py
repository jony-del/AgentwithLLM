"""Revalidate a proposed edit anchor against exact current source bytes."""
from __future__ import annotations

import hashlib

from agent_core.codeintel.backends import parser_fingerprint, python_structure
from agent_core.codeintel.models import StaleEvidence, symbol_reference
from agent_core.codeintel.snapshots import read_snapshot


def validate_candidate(root, path, expected_version, anchor, symbol, budget):
    data, version = read_snapshot(root, path, budget)
    if not isinstance(expected_version, dict) or any(expected_version.get(k) != getattr(version, k)
                                                     for k in ("worktree_id", "path", "sha256")):
        raise StaleEvidence("candidate version changed; retrieve current evidence")
    text = data.decode("utf-8")
    if not isinstance(anchor, str) or not anchor or text.find(anchor) < 0 or text.find(anchor, text.find(anchor) + 1) >= 0:
        raise ValueError("candidate anchor must match exactly once in the file")
    offset = text.index(anchor)
    first = text.count("\n", 0, offset) + 1
    last = first + anchor.count("\n") - int(anchor.endswith("\n"))
    if symbol is not None:
        if not path.endswith(".py") or not isinstance(symbol, dict):
            raise ValueError("symbol validation currently requires a Python definition")
        definitions, _ = python_structure(text)
        matches = [s for s in definitions if symbol_reference(version, s["qualified"], s["kind"],
                   s["line"], s["end_line"], parser_fingerprint())["id"] == symbol.get("id")]
        if len(matches) != 1 or not matches[0]["line"] <= first <= last <= matches[0]["end_line"]:
            raise ValueError("anchor does not belong to the supplied current symbol")
    budget.check()
    result = {"schema_version": 1, "status": "validated_anchor", "path": version.path,
              "expected_version": version.to_dict(), "start_line": first, "end_line": last,
              "anchor_sha256": hashlib.sha256(anchor.encode()).hexdigest(),
              "scope": "unique literal anchor in current file; behavioral relevance is not proven"}
    return result
