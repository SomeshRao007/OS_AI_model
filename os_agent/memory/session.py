"""Tier 3: Ephemeral session context — in-memory, not persisted.

Tracks conversation turns and arbitrary metadata for the current session.
Agents use this for multi-step reasoning and context injection into prompts.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class Turn:
    """A single interaction turn in the current session."""
    query: str
    domain: str
    response: str
    memory_hits: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class SessionContext:
    """Ephemeral session state — resets when the shell or daemon restarts."""

    def __init__(self, max_turns: int = 20) -> None:
        self._turns: list[Turn] = []
        self._metadata: dict = {}
        self._max_turns = max_turns

    def add_turn(
        self, query: str, domain: str, response: str,
        memory_hits: list[str] | None = None,
    ) -> None:
        self._turns.append(Turn(
            query=query,
            domain=domain,
            response=response,
            memory_hits=memory_hits or [],
        ))
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns:]

    def recent_turns(self, n: int = 5) -> list[Turn]:
        return self._turns[-n:]

    def get_context_string(self, n: int = 3, max_chars_per_turn: int = 250) -> str:
        """Format recent turns for compact prompt injection.

        Each turn is truncated to fit within the token budget. Domain tags
        give the model critical context (e.g. "domain: process" tells it the
        user was working with system tools, not databases).
        """
        turns = self.recent_turns(n)
        if not turns:
            return ""
        lines = ["Recent session:"]
        for i, t in enumerate(turns, 1):
            q = t.query[:100]
            # Extract just the command or first line of response
            resp = t.response.strip().split("\n")[0][:150]
            line = f"[{i}] Q: {q} | A: {resp} (domain: {t.domain})"
            lines.append(line[:max_chars_per_turn])
        return "\n".join(lines)

    def set_meta(self, key: str, value) -> None:
        self._metadata[key] = value

    def get_meta(self, key: str, default=None):
        return self._metadata.get(key, default)

    def clear(self) -> None:
        self._turns.clear()
        self._metadata.clear()

    @property
    def turn_count(self) -> int:
        return len(self._turns)


# --- built-in test ---

if __name__ == "__main__" and "--test" in sys.argv:
    print("=== SessionContext Test ===")

    sc = SessionContext(max_turns=5)

    # Test 1: add turns and verify cap
    for i in range(7):
        sc.add_turn(f"query_{i}", "files", f"response_{i}")
    status = "PASS" if sc.turn_count == 5 else "FAIL"
    print(f"  Turn cap:      {status} (count={sc.turn_count})")

    # oldest should be query_2 (0 and 1 were evicted)
    oldest = sc.recent_turns(5)[0].query
    status = "PASS" if oldest == "query_2" else "FAIL"
    print(f"  Oldest turn:   {status} (got={oldest!r})")

    # Test 2: context string
    ctx = sc.get_context_string(n=2)
    status = "PASS" if "query_5" in ctx and "query_6" in ctx else "FAIL"
    print(f"  Context str:   {status}")

    # Test 3: metadata
    sc.set_meta("cwd", "/home")
    got = sc.get_meta("cwd")
    status = "PASS" if got == "/home" else "FAIL"
    print(f"  Metadata:      {status} (got={got!r})")

    # Test 4: clear
    sc.clear()
    status = "PASS" if sc.turn_count == 0 and sc.get_meta("cwd") is None else "FAIL"
    print(f"  Clear:         {status}")

    # Test 5: empty context string
    ctx = sc.get_context_string()
    status = "PASS" if ctx == "" else "FAIL"
    print(f"  Empty ctx:     {status}")

    print("=== Done ===")
