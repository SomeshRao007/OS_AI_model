"""
Generate synthetic training data using Claude API (Haiku).

Generates:
  1. Direct-answer examples (~1500): natural question -> one command + one-line explanation
  2. Clarification dialogues (~500): ambiguous request -> assistant asks clarifying question

Requires: ANTHROPIC_API_KEY env var, anthropic Python package
Estimated cost: ~$0.50-1.00

Usage:
  python generate_synthetic.py                      # generate all
  python generate_synthetic.py --direct-only         # only direct-answer examples
  python generate_synthetic.py --clarification-only  # only clarification examples
  python generate_synthetic.py --dry-run             # show prompts without API calls
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"

MODEL = "claude-haiku-4-5-20251001"
BATCH_SIZE = 50

NEW_SYSTEM_PROMPT = (
    "You are an AI assistant built into a Linux-based operating system. "
    "When asked to perform a task, respond with one correct command followed "
    "by a brief one-line explanation. If the request is ambiguous, ask one "
    "clarifying question instead of guessing. For conceptual questions, give "
    "a clear and focused explanation. Never list multiple alternative commands. "
    "Never restate the question."
)

# ---------------------------------------------------------------------------
# Domain definitions
# ---------------------------------------------------------------------------

DIRECT_DOMAINS = {
    "file_operations": {
        "count": 200,
        "commands": (
            "find, chmod, chown, chgrp, ln, stat, file, locate, cp, mv, rm, "
            "mkdir, rmdir, touch, tar, zip, unzip, gzip, xz, bzip2"
        ),
        "description": "File/directory operations, permissions, links, archiving",
    },
    "networking": {
        "count": 200,
        "commands": (
            "ss, netstat, ip, curl, wget, ssh, scp, rsync, iptables, nft, "
            "ping, traceroute, dig, nslookup, nc, tcpdump, hostname, ifconfig"
        ),
        "description": "Network diagnostics, file transfers, firewalls, DNS, SSH",
    },
    "process_management": {
        "count": 200,
        "commands": (
            "ps, kill, killall, pkill, top, htop, systemctl, journalctl, "
            "crontab, nohup, screen, tmux, bg, fg, jobs, nice, renice, "
            "strace, lsof, wait"
        ),
        "description": "Process control, systemd services, scheduling, debugging",
    },
    "user_permissions": {
        "count": 150,
        "commands": (
            "useradd, usermod, userdel, passwd, groups, groupadd, groupdel, "
            "sudoers, visudo, chage, id, who, w, last, setfacl, getfacl"
        ),
        "description": "User/group management, sudo, account policies, ACLs",
    },
    "package_management": {
        "count": 150,
        "commands": (
            "apt, apt-get, apt-cache, dpkg, snap, pip, cargo, dnf, yum, "
            "pacman, flatpak, dpkg-query"
        ),
        "description": "Package install/remove/search/update across distros",
    },
    "text_processing": {
        "count": 150,
        "commands": (
            "grep, egrep, sed, awk, sort, uniq, cut, tr, wc, head, tail, "
            "jq, diff, comm, paste, tee, xargs, column"
        ),
        "description": "Text search, transformation, filtering, JSON processing",
    },
    "storage_filesystems": {
        "count": 150,
        "commands": (
            "mount, umount, df, du, lsblk, fdisk, mkfs, blkid, fstab, "
            "dd, lvs, pvs, vgs, lvcreate, mdadm, e2fsck, tune2fs"
        ),
        "description": "Disk management, mounting, LVM, RAID, space analysis",
    },
    "shell_scripting": {
        "count": 150,
        "commands": (
            "for loops, while loops, if/elif/else, case statements, functions, "
            "getopts, $1/$@/$#, set -euo pipefail, trap, pipes, subshells, "
            "variable expansion, read, arrays, here-documents"
        ),
        "description": "Bash scripting patterns, control flow, error handling",
    },
    "kernel_os_concepts": {
        "count": 150,
        "commands": (
            "modprobe, lsmod, rmmod, insmod, dmesg, sysctl, uname, uptime, "
            "free, vmstat, iostat, sar, /proc, /sys, cgroups, namespaces, "
            "signals, ulimit"
        ),
        "description": "Kernel modules, /proc & /sys, monitoring, cgroups, signals",
    },
}

CLARIFICATION_CATEGORIES = {
    "missing_path": {
        "count": 100,
        "description": "User gives a task but doesn't specify which directory or file",
        "examples": (
            "Delete old logs, Clean up temp files, Back up the config, "
            "Compress the data, Check permissions on the folder"
        ),
    },
    "missing_target": {
        "count": 100,
        "description": "User references 'the service', 'the process', 'the container' without naming it",
        "examples": (
            "Restart the service, Kill the process, Check the container logs, "
            "Stop the database, Enable the daemon"
        ),
    },
    "missing_scope": {
        "count": 100,
        "description": "User asks for configuration but scope/parameters are undefined",
        "examples": (
            "Set up firewall rules, Configure the network, Set up SSH access, "
            "Create a cron job, Set up log rotation"
        ),
    },
    "dangerous_ambiguity": {
        "count": 100,
        "description": "User asks something destructive without specifying target precisely",
        "examples": (
            "Format the disk, Delete everything, Wipe the partition, "
            "Reset all permissions, Remove the old kernel"
        ),
    },
    "multi_distro": {
        "count": 100,
        "description": "Task varies by Linux distribution and user hasn't specified which",
        "examples": (
            "Install nginx, Set up Docker, Add a PPA, Install the latest Python, "
            "Configure the firewall"
        ),
    },
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

DIRECT_PROMPT = """\
Generate exactly {count} Linux command training examples for the "{domain}" domain.

Domain: {description}
Commands to cover: {commands}

Each example is a JSON object with "user" and "assistant" fields.

Format:
- "user": Natural human question. Vary phrasing: "how do I...", "show me...", \
"what command...", "list the...", "I need to...", "can you...", etc.
- "assistant": A bash code block with ONE command, followed by exactly one brief \
explanation sentence.

Example:
{{"user": "How do I find all PDF files in my home directory?", \
"assistant": "```bash\\nfind ~/ -name '*.pdf' -type f\\n```\\n\
Searches your home directory recursively for all files ending in .pdf."}}

Rules:
1. ONE command per example — never list alternatives or variations
2. Use realistic values: real paths (/home/user, /var/log, /etc/nginx), \
real usernames (admin, deploy, jsmith), real IPs (192.168.1.50, 10.0.0.1)
3. Never use {{{{placeholder}}}} or template syntax
4. The explanation must be exactly ONE sentence — do not restate the question
5. Cover diverse use cases — avoid repeating similar examples
6. Include both simple and intermediate-complexity commands
7. Commands must be correct and production-ready

Return ONLY a valid JSON array of {count} objects. No markdown fences, no extra text."""

CLARIFICATION_PROMPT = """\
Generate exactly {count} examples where a user asks an ambiguous Linux task \
and the assistant asks ONE clarifying question instead of guessing.

Category: {category} — {description}
Example requests like: {examples}

Each example is a JSON object with "user" and "assistant" fields.

Format:
- "user": A natural but ambiguous request missing critical information
- "assistant": ONE clarifying question, optionally suggesting a diagnostic command

Examples:
{{"user": "Delete the old logs", \
"assistant": "Which directory should I look in? For system logs check /var/log/, \
or specify your application's log path."}}
{{"user": "Restart the service", \
"assistant": "Which service? Run `systemctl list-units --type=service --state=running` \
to see active services."}}

Rules:
1. The request must be genuinely ambiguous — missing a critical detail
2. Ask exactly ONE focused clarifying question
3. Optionally suggest a diagnostic command to help the user answer
4. Never guess or provide a command based on assumptions
5. Vary the phrasing — don't start every response the same way
6. Cover diverse scenarios within the category

Return ONLY a valid JSON array of {count} objects. No markdown fences, no extra text."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_chatml(user_msg: str, assistant_msg: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": NEW_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg.strip()},
            {"role": "assistant", "content": assistant_msg.strip()},
        ]
    }


def parse_json_response(text: str) -> list[dict] | None:
    """Extract JSON array from API response, handling markdown fences."""
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    try:
        result = json.loads(text.strip())
        if isinstance(result, list):
            return result
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def save_jsonl(data: list[dict], filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} examples to {filepath}")


def append_jsonl(data: list[dict], filepath: Path):
    """Append examples to JSONL file (for incremental saving)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def call_api(client, prompt: str, label: str, max_retries: int = 2) -> list[dict]:
    """Call Claude API and parse the JSON response."""
    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** (attempt + 1)
                print(f"\n    API error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
                continue
            print(f"\n    ERROR: API call failed for {label}: {e}")
            return []

        parsed = parse_json_response(text)
        if parsed:
            return parsed

        if attempt < max_retries:
            print(f"\n    Retry {attempt + 1}/{max_retries} for {label} (failed to parse JSON)")
            time.sleep(1)

    print(f"\n    WARNING: Failed to parse response for {label} after {max_retries + 1} attempts")
    return []


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

def generate_direct_examples(client, dry_run: bool = False) -> list[dict]:
    """Generate direct-answer training examples across all domains."""
    print("\n[1/2] Generating direct-answer examples...")
    all_examples = []
    output_path = RAW_DIR / "synthetic_direct.jsonl"

    # Clear file for fresh run
    if not dry_run and output_path.exists():
        output_path.unlink()

    for domain_name, info in DIRECT_DOMAINS.items():
        total = info["count"]
        batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx in range(batches):
            count = min(BATCH_SIZE, total - batch_idx * BATCH_SIZE)
            prompt = DIRECT_PROMPT.format(
                count=count,
                domain=domain_name.replace("_", " "),
                description=info["description"],
                commands=info["commands"],
            )

            label = f"{domain_name} batch {batch_idx + 1}/{batches}"

            if dry_run:
                print(f"  [DRY RUN] {label}: would generate {count} examples")
                continue

            print(f"  Generating {label} ({count} examples)...", end=" ", flush=True)
            examples = call_api(client, prompt, label)

            chatml_examples = []
            for ex in examples:
                user_msg = ex.get("user", "")
                asst_msg = ex.get("assistant", "")
                if user_msg and asst_msg:
                    chatml_examples.append(to_chatml(user_msg, asst_msg))

            all_examples.extend(chatml_examples)
            # Save incrementally so progress isn't lost on crash
            if chatml_examples:
                append_jsonl(chatml_examples, output_path)
            print(f"got {len(chatml_examples)}")

            time.sleep(0.5)

    print(f"  Total direct-answer examples: {len(all_examples)}")
    return all_examples


def generate_clarification_examples(client, dry_run: bool = False) -> list[dict]:
    """Generate clarification dialogue training examples."""
    print("\n[2/2] Generating clarification dialogues...")
    all_examples = []
    output_path = RAW_DIR / "synthetic_clarification.jsonl"

    # Clear file for fresh run
    if not dry_run and output_path.exists():
        output_path.unlink()

    for cat_name, info in CLARIFICATION_CATEGORIES.items():
        total = info["count"]
        batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx in range(batches):
            count = min(BATCH_SIZE, total - batch_idx * BATCH_SIZE)
            prompt = CLARIFICATION_PROMPT.format(
                count=count,
                category=cat_name.replace("_", " "),
                description=info["description"],
                examples=info["examples"],
            )

            label = f"{cat_name} batch {batch_idx + 1}/{batches}"

            if dry_run:
                print(f"  [DRY RUN] {label}: would generate {count} examples")
                continue

            print(f"  Generating {label} ({count} examples)...", end=" ", flush=True)
            examples = call_api(client, prompt, label)

            chatml_examples = []
            for ex in examples:
                user_msg = ex.get("user", "")
                asst_msg = ex.get("assistant", "")
                if user_msg and asst_msg:
                    chatml_examples.append(to_chatml(user_msg, asst_msg))

            all_examples.extend(chatml_examples)
            # Save incrementally
            if chatml_examples:
                append_jsonl(chatml_examples, output_path)
            print(f"got {len(chatml_examples)}")

            time.sleep(0.5)

    print(f"  Total clarification examples: {len(all_examples)}")
    return all_examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data via Claude API"
    )
    parser.add_argument("--direct-only", action="store_true",
                        help="Only generate direct-answer examples")
    parser.add_argument("--clarification-only", action="store_true",
                        help="Only generate clarification examples")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated without API calls")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    client = None
    if not args.dry_run:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)

    direct_examples = []
    clarification_examples = []

    if not args.clarification_only:
        direct_examples = generate_direct_examples(client, args.dry_run)

    if not args.direct_only:
        clarification_examples = generate_clarification_examples(client, args.dry_run)

    if args.dry_run:
        total_direct = sum(d["count"] for d in DIRECT_DOMAINS.values())
        total_clarification = sum(c["count"] for c in CLARIFICATION_CATEGORIES.values())
        total_batches = (
            sum((d["count"] + BATCH_SIZE - 1) // BATCH_SIZE for d in DIRECT_DOMAINS.values())
            + sum((c["count"] + BATCH_SIZE - 1) // BATCH_SIZE for c in CLARIFICATION_CATEGORIES.values())
        )
        print(f"\n{'='*60}")
        print(f"DRY RUN SUMMARY")
        print(f"  Direct-answer examples: {total_direct}")
        print(f"  Clarification examples: {total_clarification}")
        print(f"  Total API calls: {total_batches}")
        print(f"  Estimated cost: ~$0.50-1.00")
        print(f"{'='*60}")
        return

    total = len(direct_examples) + len(clarification_examples)
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"  Direct-answer: {len(direct_examples)} examples")
    print(f"  Clarification: {len(clarification_examples)} examples")
    print(f"  Total: {total} examples")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
