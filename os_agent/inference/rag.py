"""RAG layer: detect command from query, inject help context into system prompt.

Plugs in between agent.augmented_prompt_with_context() and engine.infer()
to give the model ground-truth flag information before generation.
"""

from __future__ import annotations

import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_HELP_DB_PATH = _PROJECT_ROOT / "os_agent" / "config" / "help_db.json"

_help_db: dict | None = None


def _load_db() -> dict:
    global _help_db
    if _help_db is None:
        if not _HELP_DB_PATH.exists():
            _help_db = {}
        else:
            with open(_HELP_DB_PATH, encoding="utf-8") as f:
                _help_db = json.load(f)
    return _help_db


# Keyword → command mapping. Longer/more specific keywords are matched first
# (detect_command sorts by length descending before iterating).
KEYWORD_MAP: dict[str, str] = {
    # ── ssh-keygen ───────────────────────────────────────────
    "ssh-keygen":          "ssh-keygen",
    "generate ssh key":    "ssh-keygen",
    "generate rsa":        "ssh-keygen",
    "generate ed25519":    "ssh-keygen",
    "ssh key pair":        "ssh-keygen",
    "rsa key":             "ssh-keygen",
    "ed25519":             "ssh-keygen",
    "ssh key":             "ssh-keygen",
    "public key":          "ssh-keygen",

    # ── nohup ────────────────────────────────────────────────
    "run in background":   "nohup",
    "keep after logout":   "nohup",
    "survive logout":      "nohup",
    "survive disconnect":  "nohup",
    "background process":  "nohup",
    "nohup":               "nohup",

    # ── find ─────────────────────────────────────────────────
    "find setuid":         "find",
    "setuid binaries":     "find",
    "find suid":           "find",
    "find files modified": "find",
    "find files larger":   "find",
    "find files older":    "find",
    "search filesystem":   "find",
    "find -perm":          "find",

    # ── wc ───────────────────────────────────────────────────
    "count lines":         "wc",
    "line count":          "wc",
    "word count":          "wc",
    "count words":         "wc",

    # ── chage ────────────────────────────────────────────────
    "password expiry":     "chage",
    "password aging":      "chage",
    "account expiry":      "chage",
    "expire password":     "chage",
    "password expire":     "chage",
    "chage":               "chage",

    # ── useradd ──────────────────────────────────────────────
    "create user":         "useradd",
    "add user":            "useradd",
    "new user account":    "useradd",
    "useradd":             "useradd",

    # ── gdb ──────────────────────────────────────────────────
    "open core dump":      "gdb",
    "debug core dump":     "gdb",
    "core dump":           "gdb",
    "attach debugger":     "gdb",
    "debug binary":        "gdb",
    "gdb":                 "gdb",

    # ── perf ─────────────────────────────────────────────────
    "profile cpu":         "perf",
    "cpu profile":         "perf",
    "profile process":     "perf",
    "profile nginx":       "perf",
    "profile apache":      "perf",
    "perf record":         "perf",
    "flame graph":         "perf",
    "sampling frequency":  "perf",
    "profiling":           "perf",
    "cpu usage":           "perf",
    "perf ":               "perf",
    "profile ":            "perf",

    # ── valgrind ─────────────────────────────────────────────
    "memory leak":         "valgrind",
    "heap check":          "valgrind",
    "valgrind":            "valgrind",

    # ── lsof ─────────────────────────────────────────────────
    "which process is":    "lsof",
    "process using file":  "lsof",
    "open file handle":    "lsof",
    "file descriptor":     "lsof",
    "lsof":                "lsof",

    # ── nmap ─────────────────────────────────────────────────
    "port scan":           "nmap",
    "scan host":           "nmap",
    "scan ports":          "nmap",
    "nmap":                "nmap",

    # ── iperf3 ───────────────────────────────────────────────
    "bandwidth test":      "iperf3",
    "network speed test":  "iperf3",
    "throughput test":     "iperf3",
    "iperf":               "iperf3",

    # ── strace ───────────────────────────────────────────────
    "system call":         "strace",
    "trace syscall":       "strace",
    "strace":              "strace",

    # ── rsync ────────────────────────────────────────────────
    "sync files":          "rsync",
    "copy remote files":   "rsync",
    "rsync":               "rsync",

    # ── docker ───────────────────────────────────────────────
    "docker container":    "docker",
    "docker image":        "docker",
    "docker":              "docker",

    # ── kubectl ──────────────────────────────────────────────
    "kubernetes pods":     "kubectl",
    "kubectl":             "kubectl",
    "k8s":                 "kubectl",

    # ── terraform ────────────────────────────────────────────
    "terraform apply":     "terraform",
    "terraform plan":      "terraform",
    "terraform":           "terraform",

    # ── certbot ──────────────────────────────────────────────
    "ssl certificate":     "certbot",
    "tls certificate":     "certbot",
    "letsencrypt":         "certbot",
    "certbot":             "certbot",

    # ── nginx ────────────────────────────────────────────────
    "nginx config":        "nginx",
    "nginx reload":        "nginx",
    "test nginx":          "nginx",

    # ── ansible ──────────────────────────────────────────────
    "ansible playbook":    "ansible-playbook",
    "run playbook":        "ansible-playbook",
    "ansible":             "ansible",

    # ── dpkg-query ───────────────────────────────────────────
    "installed packages":  "dpkg-query",
    "list packages":       "dpkg-query",
    "dpkg-query":          "dpkg-query",
}

# Pre-sorted by keyword length descending — more specific matches win.
_SORTED_KEYWORDS: list[tuple[str, str]] = sorted(
    KEYWORD_MAP.items(), key=lambda kv: len(kv[0]), reverse=True
)


def detect_command(query: str) -> str | None:
    """Map query keywords to a command name. Returns None if no match."""
    q = query.lower()
    for keyword, command in _SORTED_KEYWORDS:
        if keyword in q:
            return command
    return None


def get_help_context(command: str) -> str:
    """Return the one-line flag summary for a command from help_db.json.

    Returns empty string if the command is not in the db or has no summary.
    """
    db = _load_db()
    entry = db.get(command)
    if not entry:
        return ""
    summary = entry.get("summary", "")
    return f"{command}: {summary}" if summary and not summary.startswith(command) else summary


def build_rag_context(query: str) -> str:
    """Detect command from query and return formatted context for prompt injection.

    Returns empty string if no command detected or not in help_db.
    """
    command = detect_command(query)
    if not command:
        return ""
    return get_help_context(command)
