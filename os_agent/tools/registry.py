"""Per-domain tool whitelists and risk classification data.

Each agent domain has a set of allowed command basenames. Commands outside
the whitelist are blocked (out-of-domain). Risk classification uses
dangerous patterns (regex) and a safe-commands set to categorize commands.
"""

from __future__ import annotations

import re
import shlex

# ── Domain whitelists ─────────────────────────────────────────────────────

DOMAIN_WHITELIST: dict[str, frozenset[str]] = {
    "files": frozenset({
        "find", "ls", "stat", "du", "df", "cat", "head", "tail",
        "cp", "mv", "mkdir", "chmod", "chown", "ln", "tar", "gzip",
        "rm", "rmdir", "touch", "wc", "grep", "sort", "awk", "sed",
        "uniq", "file", "diff", "tree", "realpath", "basename", "dirname",
    }),
    "network": frozenset({
        "ip", "ss", "ping", "traceroute", "nslookup", "dig", "curl",
        "wget", "nc", "iptables", "nft", "ssh", "scp", "rsync",
        "netstat", "host", "whois",
    }),
    "process": frozenset({
        "ps", "top", "kill", "killall", "nice", "renice", "systemctl",
        "journalctl", "crontab", "htop", "free", "uptime", "nohup",
        "pkill", "pgrep", "lsof", "strace",
    }),
    "packages": frozenset({
        "apt", "dpkg", "snap", "flatpak", "pip", "npm", "apt-get",
        "apt-cache", "pip3",
    }),
    "kernel": frozenset({
        "uname", "lsmod", "modprobe", "dmesg", "sysctl", "lspci",
        "lsusb", "lsblk", "modinfo", "lscpu",
    }),
}

# ── Safe commands (read-only, auto-execute without confirmation) ──────────

SAFE_COMMANDS: frozenset[str] = frozenset({
    "ls", "cat", "head", "tail", "wc", "grep", "find", "stat", "df",
    "du", "ps", "top", "free", "uptime", "uname", "lsmod", "dmesg",
    "lspci", "lsusb", "lsblk", "ip", "ss", "ping", "traceroute",
    "nslookup", "dig", "file", "sort", "uniq", "id", "whoami",
    "hostname", "date", "journalctl", "modinfo", "lscpu", "pgrep",
    "host", "whois", "tree", "realpath", "basename", "dirname",
    "printenv", "env", "which", "type", "netstat", "diff",
})

# ── Dangerous patterns (regex on full command string) ─────────────────────

DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\brm\s+.*-[a-zA-Z]*[rf]"), "rm with -r or -f flag"),
    (re.compile(r"\brm\s+.*--(?:recursive|force)"), "rm with --recursive/--force"),
    (re.compile(r"\bmkfs\b"), "filesystem format"),
    (re.compile(r"\bdd\s+.*\bif="), "raw disk copy"),
    (re.compile(r"\bkill\s+.*-(?:9|KILL)\b"), "force kill"),
    (re.compile(r"\biptables\s+-F\b"), "flush firewall rules"),
    (re.compile(r"\biptables\s+-X\b"), "delete firewall chains"),
    (re.compile(r"\bchmod\s+777\b"), "world-writable permissions"),
    (re.compile(r"\bshutdown\b"), "system shutdown"),
    (re.compile(r"\breboot\b"), "system reboot"),
    (re.compile(r"\bhalt\b"), "system halt"),
    (re.compile(r"\binit\s+[06]\b"), "runlevel change"),
    (re.compile(r">\s*/dev/sd"), "write to raw device"),
    (re.compile(r"\bfdisk\b"), "disk partitioning"),
    (re.compile(r"\bparted\b"), "disk partitioning"),
    (re.compile(r"\bwipefs\b"), "wipe filesystem signatures"),
    (re.compile(r"\bfind\b.*-delete\b"), "find with -delete"),
    (re.compile(r"\bfind\b.*-exec\s+rm\b"), "find with -exec rm"),
]

# ── Wrappers to strip before extracting base command ──────────────────────

_WRAPPER_COMMANDS: frozenset[str] = frozenset({
    "sudo", "env", "nice", "nohup", "time", "strace",
})


def extract_base_commands(command: str) -> list[str]:
    """Extract base command names from a (possibly piped/chained) command.

    Splits on |, &&, ||, ; then strips wrappers like sudo/env/nohup
    to find the actual command basename.
    """
    # Split on shell operators
    segments = re.split(r"\s*(?:\|\||&&|[|;])\s*", command)
    base_commands: list[str] = []

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # Tokenize and skip wrappers / env var assignments
        parts = segment.split()
        for part in parts:
            # Skip env var assignments (e.g., VAR=1)
            if "=" in part:
                continue
            # Strip path prefix (e.g., /usr/bin/ls -> ls)
            basename = part.split("/")[-1]
            if basename in _WRAPPER_COMMANDS:
                continue
            base_commands.append(basename)
            break

    return base_commands


def is_command_allowed(command: str, domain: str) -> bool:
    """Check if all base commands in the string are allowed for this domain."""
    whitelist = DOMAIN_WHITELIST.get(domain)
    if whitelist is None:
        return False

    base_cmds = extract_base_commands(command)
    if not base_cmds:
        return False

    return all(cmd in whitelist for cmd in base_cmds)
