"""Annotated command history for neurosh shell.

Tracks both direct and AI commands with metadata (domain, exit code).
Separate from prompt_toolkit FileHistory which handles up-arrow recall.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class HistoryEntry:
    command: str
    mode: str  # "direct" or "ai"
    timestamp: float = field(default_factory=time.time)
    domain: str | None = None
    exit_code: int | None = None


class ShellHistory:
    """In-memory annotated history for the /history meta-command."""

    def __init__(self, max_entries: int = 500):
        self._entries: list[HistoryEntry] = []
        self._max = max_entries

    def add_direct(self, cmd: str, exit_code: int) -> None:
        self._append(HistoryEntry(command=cmd, mode="direct", exit_code=exit_code))

    def add_ai(self, query: str, domain: str) -> None:
        self._append(HistoryEntry(command=query, mode="ai", domain=domain))

    def recent(self, n: int = 20) -> list[HistoryEntry]:
        return self._entries[-n:]

    def format_display(self, n: int = 20) -> str:
        entries = self.recent(n)
        if not entries:
            return "  (no history)"

        lines = []
        for e in entries:
            ts = time.strftime("%H:%M:%S", time.localtime(e.timestamp))
            if e.mode == "direct":
                status = f"exit={e.exit_code}" if e.exit_code is not None else ""
                lines.append(f"  {ts}  [BASH]  {e.command}  {status}")
            else:
                domain_tag = f"[{e.domain.upper()}]" if e.domain else "[AI]"
                lines.append(f"  {ts}  {domain_tag:12s}  {e.command}")
        return "\n".join(lines)

    def _append(self, entry: HistoryEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self._max:
            self._entries = self._entries[-self._max :]
