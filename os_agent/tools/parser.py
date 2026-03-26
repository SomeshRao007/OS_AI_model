"""Parse model responses to extract executable commands from code blocks."""

from __future__ import annotations

import re

# Match fenced code blocks optionally tagged bash/sh/shell
_CODE_BLOCK_RE = re.compile(r"```(?:bash|sh|shell)?\n(.+?)```", re.DOTALL)


def extract_command(response: str) -> str | None:
    """Extract the first bash code block from a model response.

    Returns the command string or None if no code block found.
    """
    match = _CODE_BLOCK_RE.search(response)
    if not match:
        return None
    cmd = match.group(1).strip()
    return cmd if cmd else None


def extract_all_commands(response: str) -> list[str]:
    """Extract all bash code blocks from a model response."""
    return [m.strip() for m in _CODE_BLOCK_RE.findall(response) if m.strip()]
