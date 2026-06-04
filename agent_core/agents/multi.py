from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SubAgent(Protocol):
    name: str

    def run(self, task: str) -> str:
        ...


@dataclass(slots=True)
class MultiAgentCoordinator:
    agents: list[SubAgent]

    def run_all(self, task: str) -> dict[str, str]:
        return {agent.name: agent.run(task) for agent in self.agents}

