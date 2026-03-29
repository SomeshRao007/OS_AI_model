"""Base agent interface and response type for all domain specialists."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from os_agent.inference.engine import InferenceEngine
from os_agent.inference.prompt import get_prompt
from os_agent.memory.agent_memory import AgentMemory


# max chars per memory hit injected into the system prompt
_MAX_HIT_CHARS = 200


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """Immutable response from a domain specialist.

    action_type and command are populated by the AI mode handler after
    extracting commands from the response text and classifying risk.
    Future: Step 8 adds needs_confirmation for desktop notification flow.
    """

    domain: str
    response: str
    memory_hits: list[str] = field(default_factory=list)
    action_type: str | None = None
    command: str | None = None


class BaseAgent(ABC):
    """Abstract base for all domain specialist agents.

    Each specialist inherits this and implements handle(). The domain name
    selects the system prompt from prompt.py. The engine is passed per-call
    (not stored) because one shared engine serves all agents.

    Optional AgentMemory enables FAISS-backed retrieval: search before
    inference (augment the prompt), store after (learn from the interaction).
    """

    def __init__(self, domain: str, memory: AgentMemory | None = None):
        self._domain = domain
        self._system_prompt = get_prompt(domain)
        self._memory = memory

    @property
    def domain(self) -> str:
        return self._domain

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def _augmented_prompt(self, query: str) -> str:
        """Prepend FAISS memory hits to the system prompt (3 max, truncated)."""
        if not self._memory:
            return self._system_prompt

        hits = self._memory.search(query, top_k=3)
        if not hits:
            return self._system_prompt

        lines = []
        for h in hits:
            truncated = h.response[:_MAX_HIT_CHARS]
            lines.append(f"- Q: {h.query[:_MAX_HIT_CHARS]} -> A: {truncated}")
        context = "\n".join(lines)
        return f"{self._system_prompt}\n\nRelevant prior solutions:\n{context}"

    def augmented_prompt(self, query: str) -> str:
        """Public accessor for the memory-augmented system prompt."""
        return self._augmented_prompt(query)

    def augmented_prompt_with_context(
        self, query: str, env_context: str, session_context: str = ""
    ) -> str:
        """Memory-augmented prompt with environment and session context for AI mode."""
        base = self._augmented_prompt(query)
        prompt = f"{base}\n\nEnvironment:\n{env_context}"
        if session_context:
            prompt = f"{prompt}\n\n{session_context}"
        return prompt

    @abstractmethod
    def handle(self, query: str, engine: InferenceEngine) -> AgentResponse:
        ...
