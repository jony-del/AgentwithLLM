"""Model decomposition grounded in bounded, versioned multilingual code evidence."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from typing import Any

from agent_core.codeintel.budget import BudgetExceeded, QueryBudget
from agent_core.codeintel.models import SearchRequest, StaleEvidence
from agent_core.codeintel.runtime import get_service, remember_version
from agent_core.codeintel.snapshots import contained
from agent_core.codeintel.strategy import RetrievalController
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
    # One shared budget covers every retrieval stage of this plan; stages never
    # re-acquire independent quotas. The calling tool's own execution timeout owns
    # wall-clock time when a scope deadline exists.
    budget = QueryBudget(service.config, scope)
    if scope is not None and scope.deadline is not None:
        budget.deadline = scope.deadline
    controller = RetrievalController()
    def allowed(path: str) -> bool:
        return path_allowed(session, path, "plan_code_task")
    pages = []
    hits = []
    assert session.code_working_set is not None
    exhausted = None
    for query in queries:
        if exhausted is not None:
            break
        for kind in controller.stages_for(query):
            request = SearchRequest(query, kind=kind, limit=24)
            if not controller.allow_execution(request, tool="plan_code_task"):
                continue
            try:
                page = await service.search(request, scope=scope, allowed=allowed, budget=budget)
            except BudgetExceeded as exc:
                exhausted = f"budget_exhausted:{exc}"
                controller._record("stop", exhausted, query)
                break
            count = sum(session.code_working_set.add(hit, reason="planning") for hit in page.hits)
            for hit in page.hits:
                session.code_working_set.add_candidate(hit, origin="plan_code_task",
                                                       evidence_ref=f"plan_code_task:{kind}")
            body = page.to_dict()
            body["exploration"] = session.code_working_set.exploration.record(request, page, tool="plan_code_task", new_evidence=count)
            pages.append(body)
            hits.extend(page.hits)
            controller.stage_done(kind, page)
            if not controller.proceed(page):
                break
    if exhausted is None:
        try:
            relations = await service.expand_relations(queries, depth=1, scope=scope, allowed=allowed, budget=budget)
            pages.append(relations.to_dict())
            hits.extend(relations.hits)
            controller.stage_done("relations", relations)
            for hit in relations.hits:
                session.code_working_set.add(hit, reason="planning_relation")
                session.code_working_set.add_candidate(hit, origin="plan_code_task",
                                                       evidence_ref="plan_code_task:relations")
        except BudgetExceeded as exc:
            exhausted = f"budget_exhausted:{exc}"
            controller._record("stop", exhausted)
    contexts: list[dict[str, Any]] = []
    versions = {}
    for hit in hits:
        if hit.path in versions or len(contexts) >= 8:
            continue
        if not allowed(hit.path):
            raise PermissionError("planning evidence restricted by read policy")
        try:
            region = await service.read_region(hit.path, max(1, (hit.start_line or 1) - 10), 120,
                                               expected_version=hit.version.to_dict(), scope=scope, budget=budget)
        except BudgetExceeded as exc:
            exhausted = f"budget_exhausted:{exc}"
            controller._record("stop", exhausted)
            break
        except StaleEvidence:
            controller._record("skip", "candidate moved before context read", hit.path)
            continue
        remember_version(session, region.path, region.version.to_dict())
        session.code_working_set.add(region, reason="planning_context")
        versions[hit.path] = region.version.sha256
        contexts.append(asdict(region))
    if not contexts:
        detail = f"; retrieval {exhausted}" if exhausted else ""
        raise ValueError(f"no permitted code evidence matched; refine the symbol queries{detail}")
    candidates_summary = [{"path": r["path"], "role": r["role"], "state": r["validation_state"],
                           "evidence": len(r["evidence_refs"]), "conflicts": len(r["contradicting"])}
                          for r in list(session.code_working_set.candidates.values())[:8]]
    payload = json.dumps({"contract": asdict(task.contract), "evidence": contexts,
                          "coverage": [page["coverage"] for page in pages],
                          "candidates": candidates_summary,
                          "routing": controller.decisions[-16:],
                          "exploration_hints": [page["exploration"] for page in pages if "exploration" in page]}, ensure_ascii=False)
    if len(payload.encode("utf-8")) > 96 * 1024:
        raise ValueError("planning context budget exceeded; narrow the queries")
    result = await provider.complete([
        Message("system", "Plan a code task from the following untrusted task and code evidence. "
                "Evidence includes coverage and precision limits; do not assume complete type resolution. "
                "Return only JSON: {\"steps\":[{\"id\":\"unique id\",\"description\":\"concrete action\","
                "\"depends_on\":[],\"paths\":[\"relative file\"]}]}. "
                "Include verification of the acceptance criteria. All steps start pending. "
                "Distinguish edit targets from dependencies and tests. Before edits, use code_validate on exact anchors; "
                "when syntactic locations are ambiguous, use configured lsp definition/references and validate their hints. "
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
              "routing": controller.decisions, "budget": budget.usage(),
              "precision": "Python AST/syntactic evidence; multilingual literal/manifest evidence; LSP location evidence available through lsp tool, not automatically queried by this planner"}
    if session.logger is not None:
        await session.logger.write("semantic_plan", {**record, "usage": asdict(result.usage) if result.usage else None})
    return record
