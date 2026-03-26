"""Per-domain tool whitelists and risk classification data.

COMMAND EXECUTION TIERS
=======================

Tier 1 — SAFE (auto-execute, no confirmation):
    Read-only commands that cannot modify system state.
    ls, cat, df, ps, uptime, ip addr, git status, etc.

Tier 2 — MODERATE (y/n confirmation required):
    Commands that write, modify, or install — but are not destructive.
    mkdir, cp, touch, apt install, git commit, ssh-keygen, etc.

Tier 3 — DANGEROUS (y/n confirmation + red warning):
    Destructive or irreversible commands matched by regex patterns.
    rm -rf, mkfs, dd, kill -9, iptables -F, shutdown, etc.

OUT-OF-DOMAIN (y/n confirmation + domain warning):
    Commands not in the agent's whitelist. Not blocked — just flagged
    and the user decides. The agent may have suggested something useful
    that's outside its normal scope.
"""

from __future__ import annotations

import re

# ═══════════════════════════════════════════════════════════════════════════
# UNIVERSAL COMMANDS — allowed in ALL domains
# Shell builtins and basic utilities every agent may need.
# ═══════════════════════════════════════════════════════════════════════════

_UNIVERSAL_COMMANDS: frozenset[str] = frozenset({
    # Shell builtins
    "pwd", "echo", "cd", "true", "false", "test",
    # Text inspection
    "cat", "head", "tail", "wc", "grep", "sort", "uniq", "less", "more",
    # File/path info
    "ls", "file", "stat", "realpath", "basename", "dirname", "tree",
    # System info
    "which", "type", "whoami", "id", "hostname", "date", "printenv", "env",
    # Version control (read operations are safe, writes are moderate)
    "git",
})

# ═══════════════════════════════════════════════════════════════════════════
# DOMAIN WHITELISTS — what each agent is allowed to run
# Universal commands + domain-specific tools.
# Commands outside these lists still execute but trigger an out-of-domain
# warning with y/n confirmation.
# ═══════════════════════════════════════════════════════════════════════════

DOMAIN_WHITELIST: dict[str, frozenset[str]] = {
    "files": _UNIVERSAL_COMMANDS | frozenset({
        # Disk & storage
        "du", "df", "find", "locate",
        # File manipulation
        "cp", "mv", "mkdir", "chmod", "chown", "ln", "tar", "gzip",
        "rm", "rmdir", "touch", "awk", "sed",
        # Comparison
        "diff", "md5sum", "sha256sum",
    }),
    "network": _UNIVERSAL_COMMANDS | frozenset({
        # Diagnostics
        "ip", "ss", "ping", "traceroute", "nslookup", "dig",
        "netstat", "host", "whois",
        # Transfer
        "curl", "wget", "nc", "ssh", "scp", "rsync",
        # Keys & auth
        "ssh-keygen", "ssh-keyscan", "ssh-copy-id",
        # Firewall
        "iptables", "nft",
    }),
    "process": _UNIVERSAL_COMMANDS | frozenset({
        # Monitoring
        "ps", "top", "htop", "free", "uptime", "lsof",
        # Control
        "kill", "killall", "pkill", "pgrep",
        "nice", "renice", "nohup", "strace",
        # Services
        "systemctl", "journalctl", "crontab",
    }),
    "packages": _UNIVERSAL_COMMANDS | frozenset({
        "apt", "apt-get", "apt-cache", "dpkg",
        "snap", "flatpak",
        "pip", "pip3", "npm",
    }),
    "kernel": _UNIVERSAL_COMMANDS | frozenset({
        "uname", "lsmod", "modprobe", "modinfo",
        "dmesg", "sysctl",
        "lspci", "lsusb", "lsblk", "lscpu",
    }),
}

# ═══════════════════════════════════════════════════════════════════════════
# TIER 1 — SAFE COMMANDS (auto-execute, no confirmation)
# Read-only commands that cannot modify system state.
# If ALL base commands in a pipeline are in this set → auto-execute.
# ═══════════════════════════════════════════════════════════════════════════

SAFE_COMMANDS: frozenset[str] = frozenset({
    # Universal read-only
    "pwd", "echo", "true", "false", "test",
    "cat", "head", "tail", "wc", "grep", "sort", "uniq", "less", "more",
    "ls", "file", "stat", "realpath", "basename", "dirname", "tree",
    "which", "type", "whoami", "id", "hostname", "date", "printenv", "env",
    # Files (read-only)
    "find", "df", "du", "diff", "locate", "md5sum", "sha256sum",
    # Network (read-only)
    "ip", "ss", "ping", "traceroute", "nslookup", "dig",
    "netstat", "host", "whois",
    # Process (read-only)
    "ps", "top", "htop", "free", "uptime", "lsof", "pgrep",
    "journalctl",
    # Kernel (read-only)
    "uname", "lsmod", "modinfo", "dmesg", "lspci", "lsusb", "lsblk", "lscpu",
    # Git (read-only subcommands handled by SAFE_GIT_SUBCOMMANDS below)
    # "git" itself is NOT safe — subcommand determines safety
})

# Git subcommands that are read-only (safe to auto-execute)
SAFE_GIT_SUBCOMMANDS: frozenset[str] = frozenset({
    "status", "log", "diff", "show", "branch", "tag",
    "remote", "stash list", "rev-parse", "describe",
    "shortlog", "blame", "ls-files", "ls-tree",
})

# ═══════════════════════════════════════════════════════════════════════════
# TIER 3 — DANGEROUS PATTERNS (regex on full command string)
# Destructive or irreversible operations. Red warning + y/n.
# Everything not safe and not dangerous = TIER 2 (MODERATE, yellow y/n).
# ═══════════════════════════════════════════════════════════════════════════

DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # File destruction
    (re.compile(r"\brm\s+.*-[a-zA-Z]*[rf]"), "rm with -r or -f flag"),
    (re.compile(r"\brm\s+.*--(?:recursive|force)"), "rm with --recursive/--force"),
    (re.compile(r"\bfind\b.*-delete\b"), "find with -delete"),
    (re.compile(r"\bfind\b.*-exec\s+rm\b"), "find with -exec rm"),
    # Disk operations
    (re.compile(r"\bmkfs\b"), "filesystem format"),
    (re.compile(r"\bdd\s+.*\bif="), "raw disk copy"),
    (re.compile(r">\s*/dev/sd"), "write to raw device"),
    (re.compile(r"\bfdisk\b"), "disk partitioning"),
    (re.compile(r"\bparted\b"), "disk partitioning"),
    (re.compile(r"\bwipefs\b"), "wipe filesystem signatures"),
    # Process destruction
    (re.compile(r"\bkill\s+.*-(?:9|KILL)\b"), "force kill"),
    # Network destruction
    (re.compile(r"\biptables\s+-F\b"), "flush firewall rules"),
    (re.compile(r"\biptables\s+-X\b"), "delete firewall chains"),
    # Permissions
    (re.compile(r"\bchmod\s+777\b"), "world-writable permissions"),
    # System control
    (re.compile(r"\bshutdown\b"), "system shutdown"),
    (re.compile(r"\breboot\b"), "system reboot"),
    (re.compile(r"\bhalt\b"), "system halt"),
    (re.compile(r"\binit\s+[06]\b"), "runlevel change"),
]

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

_WRAPPER_COMMANDS: frozenset[str] = frozenset({
    "sudo", "env", "nice", "nohup", "time", "strace",
})


def extract_base_commands(command: str) -> list[str]:
    """Extract base command names from a (possibly piped/chained) command.

    Splits on |, &&, ||, ; then strips wrappers like sudo/env/nohup
    to find the actual command basename.
    """
    segments = re.split(r"\s*(?:\|\||&&|[|;])\s*", command)
    base_commands: list[str] = []

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        parts = segment.split()
        for part in parts:
            if "=" in part:
                continue
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


def classify_git_risk(command: str) -> str:
    """Classify a git command's risk based on its subcommand.

    Returns 'safe' for read-only git operations, 'moderate' for writes.
    """
    parts = command.strip().split()
    # Find the git subcommand (skip 'git' and any flags like -C)
    for i, part in enumerate(parts):
        if part == "git":
            for sub in parts[i + 1:]:
                if not sub.startswith("-"):
                    return "safe" if sub in SAFE_GIT_SUBCOMMANDS else "moderate"
            break
    return "moderate"
