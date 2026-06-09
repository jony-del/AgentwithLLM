"""Run several sub-agents over the same task and collect their answers.

Used by the sub-agent machinery (see ``agent_core.tools.subagent``) when more than one
child should work in parallel — e.g. investigating several areas at once. A single child
is dispatched directly via ``SessionContext.subagent_factory``; this coordinator is the
fan-out-to-many path.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Protocol


class SubAgent(Protocol):
    name: str

    def run(self, task: str) -> str:
        ...


@dataclass(slots=True)
class MultiAgentCoordinator:
    agents: list[SubAgent]
    max_workers: int = 4

    def run_all(self, task: str) -> dict[str, str]:
        """Run every agent on ``task`` concurrently; return ``{name: answer}``.

        A child raising is captured as an ``[error] ...`` string rather than failing the
        whole batch, so one bad sub-agent doesn't sink the others.
        """
        if not self.agents:
            return {}
        results: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(self.agents))) as pool:
            futures = {pool.submit(agent.run, task): agent.name for agent in self.agents}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as exc:  # noqa: BLE001 - isolate one child's failure
                    results[name] = f"[error] {type(exc).__name__}: {exc}"
        return results
