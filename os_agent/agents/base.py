"""Base agent interface and response type for all domain specialists."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from os_agent.inference.engine import InferenceEngine
from os_agent.inference.prompt import get_prompt


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """Immutable response from a domain specialist.

    Future steps will add fields:
      - Step 5: memory_hits (FAISS results that informed the answer)
      - Step 7: action_type (safe/moderate/dangerous), command (extracted)
      - Step 8: needs_confirmation (for desktop notification flow)
    """

    domain: str
    response: str


class BaseAgent(ABC):
    """Abstract base for all domain specialist agents.

    Each specialist inherits this and implements handle(). The domain name
    selects the system prompt from prompt.py. The engine is passed per-call
    (not stored) because one shared engine serves all agents.
    """

    def __init__(self, domain: str):
        self._domain = domain
        self._system_prompt = get_prompt(domain)

    @property
    def domain(self) -> str:
        return self._domain

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @abstractmethod
    def handle(self, query: str, engine: InferenceEngine) -> AgentResponse:
        ...
