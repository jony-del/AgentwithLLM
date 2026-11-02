"""Loss-aware, bounded projections of code evidence for conversation compression."""
from __future__ import annotations

import json
from typing import Any


def compact_code_result(content: str, name: str | None, max_chars: int, *,
                        recovery: Any = None, keep_text: bool = True) -> str | None:
    """Keep complete records or versioned references, never slices of JSON.

    Parse only the visible result, not the full result cached in message metadata:
    that would resurrect content removed by output hooks or another compressor.
    None means this is not an intact structured code-tool result.
    """
    if name not in {"code_search", "code_relations", "code_context", "lsp"}:
        return None
    prefix = f"{name}: " if content.startswith(f"{name}: ") else ""
    try:
        body = json.loads(content[len(prefix):])
    except (ValueError, TypeError):
        return None
    if not isinstance(body, dict):
        return None
    hits = body.get("hits") if "hits" in body else [body] if "version" in body else None
    if not isinstance(hits, list) or any(not isinstance(h, dict) or not isinstance(h.get("version"), dict) for h in hits):
        return None

    def dump(value: Any) -> str:
        return prefix + json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    # A normal page cursor points past ALL hits in the original page. Forwarding
    # that cursor after dropping hits would silently skip evidence. Recovery must
    # repeat the original page or read the retained references, not the next page.
    fallback = {"tool": name, "repeat_original_call": True}
    recover = recovery if isinstance(recovery, dict) else body.get("recovery", fallback)
    coverage = {**body.get("coverage", {}), "complete": False}
    coverage["reasons"] = list(dict.fromkeys([*coverage.get("reasons", []), "context_compacted"]))
    result: dict[str, Any] = {
        "schema_version": 1, "compacted": True, "coverage": coverage,
        "hits": [], "omitted_hits": len(hits) + int(body.get("omitted_hits", 0)),
        "cursor": None, "recovery": recover,
    }
    if isinstance(body.get("exploration"), dict):
        result["exploration"] = {k: body["exploration"][k] for k in
            ("query_id", "no_gain_repeats", "next_action") if k in body["exploration"]}
        result["exploration"]["freshness"] = "historical; recover omitted evidence before continuing"
        result["exploration"]["next_action"] = "recover_original_page"
        if len(dump(result)) > max_chars:
            result.pop("exploration")
    if len(dump(result)) > max_chars:
        result["coverage"] = {"complete": False, "reasons": ["context_compacted"]}
    if len(dump(result)) > max_chars:
        result["recovery"] = fallback
    if len(dump(result)) > max_chars:
        # Very small emergency budgets cannot retain a hash/path or tool args.
        # Still disclose the loss and preserve a valid JSON value.
        for minimal in ({"compacted": True, "recovery": "repeat original tool call"}, {"compacted": True}, {}):
            if len(dump(minimal)) <= max_chars:
                return dump(minimal)
        return "{}"
    for hit in hits:
        reference = {key: value for key, value in hit.items() if key != "text"}
        reference.update(text="", truncated=True, excerpt_start_line=None, excerpt_end_line=None,
                         next_start_line=hit.get("start_line"))
        for candidate in ([hit, reference] if keep_text else [reference]):
            result["hits"].append(candidate)
            result["omitted_hits"] -= 1
            if len(dump(result)) <= max_chars:
                break
            result["hits"].pop()
            result["omitted_hits"] += 1
        else:
            break
    return dump(result)
