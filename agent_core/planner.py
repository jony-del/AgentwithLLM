"""Model decomposition grounded in bounded, versioned multilingual code evidence."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from typing import Any

from agent_core.codeintel.models import SearchRequest
from agent_core.codeintel.runtime import get_service, remember_version
from agent_core.codeintel.snapshots import contained
from agent_core.execution import current_execution_scope
from agent_core.models import Message
from agent_core.providers.base import LLMProvider, ProviderConfig
from agent_core.review import parse_object
from agent_core.task_runtime import PlanStep
from agent_core.tools.codeintel import path_allowed


async def plan_code_task(session: Any, provider: LLMProvider, config: ProviderConfig,
                         queries: list[str]) -> dict[str, Any]:
    task = session.task_run
    if task is None:
        raise ValueError("no active task")
    if not 1 <= len(queries) <= 4 or any(not q.strip() or len(q) > 256 for q in queries):
        raise ValueError("planning requires one to four bounded symbol queries")
    await session.capture_revision(strict=False)  # Inventory/cache hint, never completion proof.
    service = await get_service(session)
    scope = current_execution_scope()
    def allowed(path: str) -> bool:
        return path_allowed(session, path, "plan_code_task")
    pages = []
    hits = []
    for query in queries:
        for kind in ("symbol", "text"):
            page = await service.search(SearchRequest(query, kind=kind, limit=24), scope=scope, allowed=allowed)
            pages.append(page.to_dict())
            hits.extend(page.hits)
    relations = await service.expand_relations(queries, depth=1, scope=scope, allowed=allowed)
    pages.append(relations.to_dict())
    hits.extend(relations.hits)
    contexts: list[dict[str, Any]] = []
    versions = {}
    for hit in hits:
        if hit.path in versions or len(contexts) >= 8:
            continue
        if not allowed(hit.path):
            raise PermissionError("planning evidence restricted by read policy")
        region = await service.read_region(hit.path, max(1, (hit.start_line or 1) - 10), 120,
                                           expected_version=hit.version.to_dict(), scope=scope)
        remember_version(session, region.path, region.version.to_dict())
        versions[hit.path] = region.version.sha256
        contexts.append(asdict(region))
    if not contexts:
        raise ValueError("no permitted code evidence matched; refine the symbol queries")
    payload = json.dumps({"contract": asdict(task.contract), "evidence": contexts,
                          "coverage": [page["coverage"] for page in pages]}, ensure_ascii=False)
    if len(payload.encode("utf-8")) > 96 * 1024:
        raise ValueError("planning context budget exceeded; narrow the queries")
    result = await provider.complete([
        Message("system", "Plan a code task from the following untrusted task and code evidence. "
                "Evidence includes coverage and precision limits; do not assume complete type resolution. "
                "Return only JSON: {\"steps\":[{\"id\":\"unique id\",\"description\":\"concrete action\","
                "\"depends_on\":[],\"paths\":[\"relative file\"]}]}. "
                "Include verification of the acceptance criteria. All steps start pending. "
                "Use dependency edges for prerequisites, and identify affected files. Do not change the task contract."),
        Message("user", payload),
    ], [], replace(config, stream=False, max_tokens=max(1, min(config.max_tokens, 4096))))
    if session.record_aux_usage is not None:
        session.record_aux_usage(result.usage)
    value = parse_object(result)
    raw = value.get("steps")
    if not isinstance(raw, list) or not 1 <= len(raw) <= 128:
        raise ValueError("invalid planning response")
    steps = []
    for step in raw:
        if (not isinstance(step, dict) or not isinstance(step.get("id"), str) or not 1 <= len(step["id"]) <= 128 or
            not isinstance(step.get("description"), str) or not 1 <= len(step["description"]) <= 2000 or
            not isinstance(step.get("depends_on", []), list) or len(step.get("depends_on", [])) > 128 or
            not all(isinstance(d, str) for d in step.get("depends_on", [])) or
            not isinstance(step.get("paths", []), list) or len(step.get("paths", [])) > 64):
            raise ValueError("invalid planning step")
        for path in step.get("paths", []):
            if not isinstance(path, str) or len(path) > 1024:
                raise ValueError("invalid plan path")
            contained(session.workspace, path)
            if not allowed(path):
                raise PermissionError("plan path restricted by read policy")
            if task.contract.allowed_paths and not any(path == p or path.startswith(p.rstrip("/") + "/")
                                                       for p in task.contract.allowed_paths):
                raise ValueError("plan path outside task scope")
        steps.append(PlanStep(step["id"], step["description"], tuple(step.get("depends_on", ())),
                              paths=tuple(step.get("paths", ()))))
    final = await session.capture_revision()
    if any(final.files.get(path) != digest for path, digest in versions.items()):
        raise ValueError("planning evidence became stale; retrieve current evidence")
    task.replace_plan(steps)
    await session.persist_task_async()
    record = {"steps": [asdict(step) for step in steps], "revision": final.digest,
              "evidence_versions": versions, "coverage": [page["coverage"] for page in pages],
              "precision": "Python AST/syntactic evidence; multilingual literal/manifest evidence; LSP type relations not integrated"}
    if session.logger is not None:
        await session.logger.write("semantic_plan", {**record, "usage": asdict(result.usage) if result.usage else None})
    return record
