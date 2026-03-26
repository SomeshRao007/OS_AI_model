"""Tab completion for neurosh shell.

Terminal mode: meta commands + common Linux commands.
Chatbot mode: meta commands only (user types natural language).
AI-powered completion deferred to a later step.
"""

from __future__ import annotations

from prompt_toolkit.completion import WordCompleter

META_COMMANDS = [
    "/chatbot", "/terminal", "/ai", "/history", "/memory", "/agents",
    "/clear", "/help", "/exit", "/quit",
]

COMMON_COMMANDS = [
    "ls", "cd", "pwd", "cat", "head", "tail", "grep", "find", "chmod",
    "chown", "cp", "mv", "rm", "mkdir", "rmdir", "touch", "df", "du",
    "ps", "top", "kill", "ssh", "scp", "ping", "curl", "wget", "apt",
    "systemctl", "journalctl", "uname", "dmesg", "ip", "ss", "tar",
    "gzip", "zip", "unzip", "sudo", "man", "echo", "sort", "awk", "sed",
]

_terminal_completer = WordCompleter(META_COMMANDS + COMMON_COMMANDS, sentence=True)
_chatbot_completer = WordCompleter(META_COMMANDS, sentence=True)


def create_completer(mode: str) -> WordCompleter:
    """Return the appropriate completer for the given shell mode."""
    if mode == "chatbot":
        return _chatbot_completer
    return _terminal_completer
