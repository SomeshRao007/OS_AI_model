"""Network specialist — SSH, ports, firewall, DNS, remote transfers."""

from os_agent.agents.base import AgentResponse, BaseAgent
from os_agent.inference.engine import InferenceEngine


class NetworkAgent(BaseAgent):

    def __init__(self):
        super().__init__("network")

    def handle(self, query: str, engine: InferenceEngine) -> AgentResponse:
        response = engine.infer(self._system_prompt, query)
        return AgentResponse(domain=self._domain, response=response)
