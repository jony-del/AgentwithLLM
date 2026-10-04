import json

import pytest

from agent_core.codeintel.config import CodeIntelConfig
from agent_core.codeintel.runtime import get_service, release_service
from agent_core.models import LLMResult
from agent_core.planner import plan_code_task
from agent_core.providers.base import ProviderConfig
from agent_core.providers.fake import FakeProvider
from agent_core.session import SessionContext
from agent_core.task_runtime import TaskContract, TaskRun, capture_revision


class Planner(FakeProvider):
    def __init__(self, response, mutate=None):
        super().__init__()
        self.response, self.mutate, self.requests = response, mutate, []
    async def complete(self, messages, tools, config, **kwargs):
        self.requests.append((messages, tools))
        self.config = config
        if self.mutate:
            self.mutate()
        return LLMResult(json.dumps(self.response))


async def setup_planner(tmp_path):
    (tmp_path / "a.py").write_text("def target():\n    return 1\n")
    (tmp_path / "client.ts").write_text("export function target(): number { return 1; }\n")
    session = SessionContext(workspace=tmp_path, task_run=TaskRun(TaskContract("change target"), capture_revision(tmp_path)),
                             codeintel_config=CodeIntelConfig(watch=False))
    service = await get_service(session)
    for _ in range(20):
        state = await service.ensure_index()
        if state["catalog_complete"] and not state["pending"]:
            break
    return session


async def test_multilingual_evidence_plan_is_versioned_and_dependency_checked(tmp_path):
    session = await setup_planner(tmp_path)
    provider = Planner({"steps": [{"id": "edit", "description": "update implementations", "paths": ["a.py", "client.ts"]},
                                  {"id": "test", "description": "verify outputs", "depends_on": ["edit"]}]})
    try:
        result = await plan_code_task(session, provider, ProviderConfig(max_tokens=128), ["target"])
        assert set(result["evidence_versions"]) == {"a.py", "client.ts"}
        assert session.task_run.plan[1].depends_on == ("edit",)
        assert session.task_run.plan[0].status == "pending"
        assert result["coverage"] and "syntactic" in result["precision"]
        assert provider.requests[0][1] == []
        assert provider.config.max_tokens == 128
    finally:
        await release_service(session)


@pytest.mark.parametrize("response", [
    {"steps": [{"id": "a", "description": "a", "depends_on": ["b"]}, {"id": "b", "description": "b", "depends_on": ["a"]}]},
    {"steps": [{"id": "a", "description": "a", "paths": ["../escape"]}]},
    {"steps": [{"id": "a", "description": "a", "paths": [".env"]}]},
    {"steps": "invalid"},
])
async def test_invalid_plans_do_not_replace_previous_state(tmp_path, response):
    session = await setup_planner(tmp_path)
    try:
        with pytest.raises(ValueError):
            await plan_code_task(session, Planner(response), ProviderConfig(), ["target"])
        assert session.task_run.plan == []
    finally:
        await release_service(session)


async def test_planning_rejects_stale_evidence(tmp_path):
    session = await setup_planner(tmp_path)
    provider = Planner({"steps": [{"id": "a", "description": "a"}]},
                       mutate=lambda: (tmp_path / "a.py").write_text("def target():\n    return 2\n"))
    try:
        with pytest.raises(ValueError, match="stale"):
            await plan_code_task(session, provider, ProviderConfig(), ["target"])
        assert session.task_run.plan == []
    finally:
        await release_service(session)
