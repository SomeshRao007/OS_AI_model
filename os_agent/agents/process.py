"""Process specialist — processes, CPU/memory, cron, systemd, users/groups."""

from __future__ import annotations

from os_agent.agents.base import AgentResponse, BaseAgent
from os_agent.inference.engine import InferenceEngine
from os_agent.memory.agent_memory import AgentMemory


class ProcessAgent(BaseAgent):

    def __init__(self, memory: AgentMemory | None = None):
        super().__init__("process", memory)

    def handle(self, query: str, engine: InferenceEngine) -> AgentResponse:
        prompt = self._augmented_prompt(query)
        hits = self._memory.search(query) if self._memory else []
        response = engine.infer(prompt, query)
        if self._memory:
            self._memory.store(query, response)
        return AgentResponse(
            domain=self._domain,
            response=response,
            memory_hits=[h.response for h in hits],
        )
