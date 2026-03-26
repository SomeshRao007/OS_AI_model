"""Tab completion for neurosh shell.

Direct mode: meta commands + common Linux commands.
AI mode: meta commands only (user types natural language).
AI-powered completion deferred to a later step.
"""

from __future__ import annotations

from prompt_toolkit.completion import WordCompleter

META_COMMANDS = [
    "/ai", "/direct", "/history", "/memory", "/agents",
    "/clear", "/help", "/exit", "/quit",
]

COMMON_COMMANDS = [
    "ls", "cd", "pwd", "cat", "head", "tail", "grep", "find", "chmod",
    "chown", "cp", "mv", "rm", "mkdir", "rmdir", "touch", "df", "du",
    "ps", "top", "kill", "ssh", "scp", "ping", "curl", "wget", "apt",
    "systemctl", "journalctl", "uname", "dmesg", "ip", "ss", "tar",
    "gzip", "zip", "unzip", "sudo", "man", "echo", "sort", "awk", "sed",
]

_direct_completer = WordCompleter(META_COMMANDS + COMMON_COMMANDS, sentence=True)
_ai_completer = WordCompleter(META_COMMANDS, sentence=True)


def create_completer(mode: str) -> WordCompleter:
    """Return the appropriate completer for the given shell mode."""
    if mode == "ai":
        return _ai_completer
    return _direct_completer
