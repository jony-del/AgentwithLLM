"""Staged retrieval controller: routing, convergence and one shared budget (S4)."""
from __future__ import annotations


import pytest

from agent_core.codeintel.budget import QueryBudget
from agent_core.codeintel.models import SearchCoverage, SearchPage, SearchRequest
from agent_core.codeintel.runtime import get_service, release_service
from agent_core.codeintel.strategy import RetrievalController
from agent_core.planner import plan_code_task
from agent_core.providers.base import ProviderConfig
from tests.test_semantic_planner import Planner, setup_planner


def test_controller_routes_by_query_shape():
    controller = RetrievalController()
    assert controller.stages_for("target_function") == ["symbol", "text"]
    assert controller.stages_for("src/client.py") == ["path", "text"]
    assert controller.stages_for("fix the crash on startup") == ["text"]


def test_controller_converges_repeats_and_caps_stages():
    controller = RetrievalController()
    request = SearchRequest("needle", kind="symbol")
    assert controller.allow_execution(request, tool="plan_code_task")
    assert not controller.allow_execution(request, tool="plan_code_task")  # identical repeat converges
    assert controller.decisions[-1]["decision"] == "dedupe"
    assert controller.allow_execution(SearchRequest("needle", kind="text"), tool="plan_code_task")
    empty_complete = SearchPage(hits=[], scan_complete=True, coverage=SearchCoverage(complete=True))
    assert not controller.proceed(empty_complete)  # complete zero-hit coverage stops the route
    assert SearchPage()  # sanity
    controller.stages = controller.MAX_STAGES
    assert controller.stages_for("another") == []
    assert controller.decisions[-1]["decision"] == "stop"


async def test_plan_code_task_dedupes_identical_queries(tmp_path):
    session = await setup_planner(tmp_path)
    provider = Planner({"steps": [{"id": "a", "description": "a"}]})
    try:
        result = await plan_code_task(session, provider, ProviderConfig(), ["target", "target"])
        dedupes = [d for d in result["routing"] if d["decision"] == "dedupe"]
        assert len(dedupes) == 2  # both stages of the repeated query converged
        queries = session.code_working_set.exploration.queries
        assert len(queries) == 2  # symbol+text once, not twice
        assert all(q["attempts"] == 1 for q in queries.values())
        assert result["budget"]["candidates"] > 0
    finally:
        await release_service(session)


async def test_one_shared_budget_covers_every_stage(tmp_path):
    session = await setup_planner(tmp_path)
    service = await get_service(session)
    try:
        config = service.config
        budget = QueryBudget(config)
        first = await service.search(SearchRequest("target", kind="symbol"), budget=budget)
        used_after_first = budget.candidates
        assert first.usage["candidates"] == used_after_first
        second = await service.search(SearchRequest("target", kind="text"), budget=budget)
        assert budget.candidates == second.usage["candidates"] >= used_after_first
        # Exhaustion is shared too: a drained budget cannot refresh between stages.
        drained = QueryBudget(config)
        drained.candidates = config.max_candidates
        page = await service.search(SearchRequest("target", kind="text"), budget=drained)
        assert any("max_candidates" in reason for reason in page.coverage.reasons)
    finally:
        await release_service(session)


async def test_exhausted_budget_fails_closed_without_fake_evidence(tmp_path):
    session = await setup_planner(tmp_path)
    service = await get_service(session)
    object.__setattr__(service.config, "max_candidates", 0)
    provider = Planner({"steps": [{"id": "a", "description": "a"}]})
    try:
        with pytest.raises(ValueError, match="no permitted code evidence"):
            await plan_code_task(session, provider, ProviderConfig(), ["target"])
        assert session.task_run.plan == []
    finally:
        await release_service(session)
