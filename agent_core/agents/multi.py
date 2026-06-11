"""Run several sub-agents over the same task and collect their answers.

Used by the sub-agent machinery (see ``agent_core.tools.subagent``) when more than one
child should work in parallel — e.g. investigating several areas at once. A single child
is dispatched directly via ``SessionContext.subagent_factory``; this coordinator is the
fan-out-to-many path.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol


class SubAgent(Protocol):
    name: str

    async def run(self, task: str) -> str:
        ...


@dataclass(slots=True)
class MultiAgentCoordinator:
    agents: list[SubAgent]

    async def run_all(self, task: str) -> dict[str, str]:
        """Run every agent on ``task`` concurrently; return ``{name: answer}``.

        Each agent's API calls overlap on one event loop, bounded by the shared
        provider gate. A child raising is captured as an ``[error] ...`` string
        rather than failing the whole batch, so one bad sub-agent doesn't sink
        the others. Answers keep the agents' declared order.
        """
        if not self.agents:
            return {}

        async def run_one(agent: SubAgent) -> str:
            try:
                return await agent.run(task)
            except Exception as exc:  # noqa: BLE001 - isolate one child's failure
                return f"[error] {type(exc).__name__}: {exc}"

        answers = await asyncio.gather(*(run_one(agent) for agent in self.agents))
        return {agent.name: answer for agent, answer in zip(self.agents, answers)}
