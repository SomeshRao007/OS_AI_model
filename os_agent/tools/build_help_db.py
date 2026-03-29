"""One-time script: run --help on ~160 Linux commands, build help_db.json.

Usage:
    python os_agent/tools/build_help_db.py

Output:
    os_agent/config/help_db.json

Re-runnable: overwrites the JSON each time. Safe to re-run after installing
more commands — OVERRIDES are always preserved.
"""

import json
import re
import subprocess
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_OUTPUT = _PROJECT_ROOT / "os_agent" / "config" / "help_db.json"

COMMANDS = [
    # ── Networking (core) ───────────────────────────────────
    "ssh", "ssh-keygen", "ssh-copy-id", "ssh-agent", "ssh-add",
    "scp", "rsync", "curl", "wget", "nc",
    "dig", "nslookup", "host", "ping", "traceroute",
    "ip", "ss", "netstat", "tcpdump", "iptables", "nft", "ufw",
    "hostname", "hostnamectl",

    # ── Advanced Networking & Troubleshooting ───────────────
    "nmap", "mtr", "iperf3", "resolvectl", "socat", "telnet",

    # ── Files & Text ────────────────────────────────────────
    "find", "chmod", "chown", "chgrp", "ln", "stat", "file",
    "cp", "mv", "rm", "mkdir", "rmdir", "touch", "tree",
    "tar", "gzip", "gunzip", "bzip2", "xz", "zip", "unzip",
    "grep", "sed", "awk", "sort", "uniq", "cut", "tr",
    "wc", "head", "tail", "diff", "tee", "xargs",
    "cat", "less", "more", "od", "hexdump", "strings",
    "du", "df",

    # ── Data Parsing & Encoding ─────────────────────────────
    "jq", "yq", "base64", "column", "xxd",

    # ── Process & Resource ──────────────────────────────────
    "ps", "top", "htop", "kill", "killall", "pkill", "pgrep",
    "nice", "renice", "nohup", "disown",
    "strace", "ltrace", "lsof", "watch", "timeout",
    "systemctl", "journalctl", "crontab",

    # ── Users & Permissions ─────────────────────────────────
    "useradd", "usermod", "userdel", "adduser", "deluser",
    "passwd", "chage", "id", "groups", "who", "w",
    "sudo", "su", "visudo",
    "setfacl", "getfacl", "chattr", "lsattr",

    # ── Package Management ──────────────────────────────────
    "apt", "apt-get", "apt-cache", "apt-mark",
    "dpkg", "dpkg-query", "dpkg-reconfigure",
    "snap", "pip", "pip3",

    # ── Storage & Filesystems ───────────────────────────────
    "mount", "umount", "lsblk", "blkid", "fdisk", "parted",
    "mkfs.ext4", "mkfs.xfs", "fsck", "tune2fs", "xfs_repair",
    "swapon", "swapoff", "mkswap", "fallocate",
    "lvcreate", "lvextend", "vgcreate", "pvdisplay",

    # ── Kernel & System ─────────────────────────────────────
    "modprobe", "lsmod", "modinfo", "rmmod",
    "sysctl", "dmesg", "uname", "uptime", "free",
    "lscpu", "lspci", "lsusb", "lshw",

    # ── Debugging & Profiling ───────────────────────────────
    "gdb", "perf", "valgrind",
    "ltrace", "objdump", "nm", "ldd", "addr2line",

    # ── Build Tools ─────────────────────────────────────────
    "gcc", "g++", "make", "cmake", "pkg-config",
    "ar", "ld", "strip",

    # ── Git ─────────────────────────────────────────────────
    "git",

    # ── Containers & Orchestration ──────────────────────────
    "docker", "docker-compose", "kubectl", "helm",
    "minikube", "kubeadm", "crictl",

    # ── Web Servers, Proxies & Certificates ─────────────────
    "nginx", "apachectl", "certbot", "haproxy",

    # ── IaC & Config Management ─────────────────────────────
    "terraform", "ansible", "ansible-playbook",

    # ── Cloud CLI ───────────────────────────────────────────
    "aws",

    # ── Database ────────────────────────────────────────────
    "psql", "mysqldump", "mysql", "sqlite3",

    # ── Security ────────────────────────────────────────────
    "openssl", "gpg", "fail2ban-client",
    "aa-status", "auditctl", "ausearch",

    # ── Monitoring / Misc ───────────────────────────────────
    "nethogs", "iotop", "iftop", "vmstat", "iostat", "sar",
    "timedatectl", "localectl", "loginctl",
    "envsubst", "logrotate",
]

# Hand-curated overrides — bypass --help parsing for critical commands.
# These are the commands the model most frequently gets wrong.
OVERRIDES = {
    "ssh-keygen": {
        "summary": "-t keytype  -b bits  -C comment(string/email)  -f output_filepath  -N passphrase",
        "flags": {
            "-t": {"expects": "enum",     "values": ["rsa", "ed25519", "ecdsa", "dsa"]},
            "-b": {"expects": "number",   "hint": "key size in bits"},
            "-C": {"expects": "string",   "hint": "comment or email — NOT a filename"},
            "-f": {"expects": "filepath", "hint": "output key file path"},
            "-N": {"expects": "string",   "hint": "passphrase (empty string for none)"},
        },
    },
    "nohup": {
        "summary": "nohup COMMAND [ARGS]  — no flags except --help/--version",
        "flags": {},
        "note": "nohup accepts NO flags before the command. Correct: nohup cmd &",
    },
    "find": {
        "summary": "-name  -type f/d  -size +NNM  -mtime  -perm /MODE  -exec CMD {} \\;",
        "flags": {
            "-name":     {"expects": "string"},
            "-iname":    {"expects": "string"},
            "-type":     {"expects": "enum",            "values": ["f", "d", "l", "b", "c", "p", "s"]},
            "-perm":     {"expects": "permission_mode", "note": "use /4000 not +4000 (deprecated)"},
            "-size":     {"expects": "size_spec",       "hint": "e.g. +100M, -1k"},
            "-mtime":    {"expects": "number",          "hint": "days; -1 = last 24h"},
            "-atime":    {"expects": "number"},
            "-ctime":    {"expects": "number"},
            "-exec":     {"expects": "command"},
            "-maxdepth": {"expects": "number"},
            "-mindepth": {"expects": "number"},
            "-user":     {"expects": "string"},
            "-group":    {"expects": "string"},
            "-newer":    {"expects": "filepath"},
            "-not":      {"expects": None},
            "-o":        {"expects": None},
            "-delete":   {"expects": None},
            "-print":    {"expects": None},
            "-ls":       {"expects": None},
        },
    },
    "chage": {
        "summary": "-M max_days  -m min_days  -W warn_days  -I inactive_days  -E expire_date  -d last_change  -l (list)",
        "flags": {
            "-M": {"expects": "number", "hint": "max days between password changes"},
            "-m": {"expects": "number", "hint": "min days between changes"},
            "-W": {"expects": "number", "hint": "warn N days before expiry"},
            "-I": {"expects": "number", "hint": "inactive days after expiry"},
            "-E": {"expects": "string", "hint": "account expiry date (YYYY-MM-DD or -1)"},
            "-d": {"expects": "number", "hint": "last password change date (0 = force change now)"},
            "-l": {"expects": None,     "hint": "list aging info for user"},
        },
    },
    "useradd": {
        "summary": "-m (create home)  -s shell  -d homedir  -g group  -G groups  -u uid  -c comment",
        "flags": {
            "-m": {"expects": None,       "hint": "create home directory"},
            "-M": {"expects": None,       "hint": "do NOT create home directory"},
            "-s": {"expects": "filepath", "hint": "login shell path e.g. /bin/bash"},
            "-d": {"expects": "filepath", "hint": "home directory path"},
            "-g": {"expects": "string",   "hint": "primary group"},
            "-G": {"expects": "string",   "hint": "supplementary groups (comma-separated)"},
            "-u": {"expects": "number",   "hint": "UID"},
            "-c": {"expects": "string",   "hint": "comment/GECOS field"},
            "-e": {"expects": "string",   "hint": "account expiry date"},
            "-r": {"expects": None,       "hint": "create system account"},
        },
    },
    "wc": {
        "summary": "-l (lines)  -w (words)  -c (bytes)  -m (chars)  -L (longest line) — no -n flag",
        "flags": {
            "-l": {"expects": None, "hint": "count lines"},
            "-w": {"expects": None, "hint": "count words"},
            "-c": {"expects": None, "hint": "count bytes"},
            "-m": {"expects": None, "hint": "count characters"},
            "-L": {"expects": None, "hint": "length of longest line"},
        },
        "invalid_flags": ["-n"],
    },
    "gdb": {
        "summary": "gdb BINARY [COREFILE|PID]  or  gdb -p PID",
        "flags": {
            "-p":      {"expects": "number",   "hint": "attach to running process PID"},
            "--batch": {"expects": None},
            "-ex":     {"expects": "string",   "hint": "execute gdb command on start"},
            "--args":  {"expects": "command",  "hint": "pass args to debugged program"},
            "-q":      {"expects": None,       "hint": "quiet mode"},
            "--core":  {"expects": "filepath"},
        },
        "note": "To open a core dump: gdb /path/to/binary /path/to/corefile",
    },
    "perf": {
        "summary": "perf record -F Hz -g [-p PID] [-- COMMAND]",
        "flags": {
            "-F": {"expects": "number",   "hint": "sampling frequency in Hz — use 99 or 999, NOT 999999"},
            "-g": {"expects": None,       "hint": "enable call-graph recording"},
            "-p": {"expects": "number",   "hint": "target process PID"},
            "-a": {"expects": None,       "hint": "system-wide profiling"},
            "-o": {"expects": "filepath"},
            "-e": {"expects": "string",   "hint": "event selector"},
        },
    },
}


def _infer_type(arg_name: str, desc: str) -> str:
    """Infer expected argument type from placeholder name and description text."""
    combined = f"{arg_name} {desc}".lower()
    if any(k in combined for k in ("file", "path", "dir", "output", "input", "dest", "src")):
        return "filepath"
    if any(k in combined for k in ("number", "count", "size", "bits", "port", "num",
                                    "pid", "freq", "int", "hz", "sec", "time", "limit")):
        return "number"
    if any(k in combined for k in ("type", "algo", "format", "alg", "cipher", "mode")):
        return "enum"
    return "string"


def _parse_help(cmd: str) -> dict:
    """Run `cmd --help 2>&1`, parse flags, return {flags, summary}.

    On FileNotFoundError (command not installed) or timeout, returns an empty
    entry — the validator treats unknown commands as soft-warn only, never blocking.
    """
    try:
        proc = subprocess.run(
            [cmd, "--help"],
            capture_output=True, text=True, timeout=5,
        )
        output = proc.stdout + proc.stderr
    except FileNotFoundError:
        return {"flags": {}, "summary": f"{cmd}: not installed"}
    except subprocess.TimeoutExpired:
        return {"flags": {}, "summary": f"{cmd}: --help timed out"}

    flags: dict[str, dict] = {}
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("-"):
            continue

        # Primary pattern: -f FILE   description
        #                  --flag=FILE  description
        m = re.match(r"^(-{1,2}[\w-]+)(?:[=\s]+([A-Z][A-Z0-9_-]*))?(?:\s{2,}(.*?))?$", stripped)
        if m:
            flag = m.group(1)
            arg_name = m.group(2)
            desc = (m.group(3) or "").strip()
        else:
            # Fallback: grab flag token and check for ARG placeholder immediately after
            m2 = re.match(r"^(-{1,2}[\w-]+)(.*)", stripped)
            if not m2:
                continue
            flag = m2.group(1)
            rest = m2.group(2).strip()
            arg_m = re.match(r"^[=\s]+([A-Z][A-Z0-9_-]*)\s*(.*)", rest)
            if arg_m:
                arg_name = arg_m.group(1)
                desc = arg_m.group(2).strip()
            else:
                arg_name = None
                desc = rest

        expects = _infer_type(arg_name, desc) if arg_name else None
        flags[flag] = {"expects": expects}

    # Compact summary: first 8 flags with their expected arg type
    parts = []
    for f, info in list(flags.items())[:8]:
        parts.append(f"{f} {info['expects']}" if info["expects"] else f)
    summary = f"{cmd}: {' '.join(parts)}" if parts else f"{cmd}: (no flags parsed)"
    return {"flags": flags, "summary": summary}


def main() -> None:
    total = len(COMMANDS)
    db: dict[str, dict] = {}

    for i, cmd in enumerate(COMMANDS, 1):
        if cmd in OVERRIDES:
            db[cmd] = OVERRIDES[cmd]
            n = len(OVERRIDES[cmd]["flags"])
            print(f"[{i:3d}/{total}] {cmd:30s}  OVERRIDE  ({n} flags)")
        else:
            result = _parse_help(cmd)
            db[cmd] = result
            n = len(result["flags"])
            tag = "parsed" if n > 0 else "not installed / no flags"
            print(f"[{i:3d}/{total}] {cmd:30s}  {tag}  ({n} flags)")

    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)

    installed = sum(1 for v in db.values() if v.get("flags"))
    print(f"\nDone. {installed}/{total} commands have flags. Written → {_OUTPUT}")
    print("Re-run any time after installing more commands.")


if __name__ == "__main__":
    main()
