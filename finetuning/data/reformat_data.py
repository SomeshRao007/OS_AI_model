"""
Reformat existing training data for direct-answer response style.

Transformations:
  1. Swap system prompt (non-kernel) to new direct-answer style
  2. Drop comprehensive multi-variation tldr examples (~6,270)
  3. Replace {{placeholder}} with realistic values in remaining tldr
  4. Strip "This uses X to..." restatements from tldr responses
  5. Combine all sources (including synthetic) into one JSONL

Usage:
  python reformat_data.py                # reformat + combine all
  python reformat_data.py --stats-only   # show what would change without writing
"""

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
OUTPUT_FILE = RAW_DIR / "combined_directstyle.jsonl"

OLD_SYSTEM_PROMPT = (
    "You are a Linux system expert. You provide accurate shell commands, "
    "system administration guidance, and kernel development assistance. "
    "Always explain the implications of commands that modify system state. "
    "Flag destructive operations with warnings."
)

NEW_SYSTEM_PROMPT = (
    "You are an AI assistant built into a Linux-based operating system. "
    "When asked to perform a task, respond with one correct command followed "
    "by a brief one-line explanation. If the request is ambiguous, ask one "
    "clarifying question instead of guessing. For conceptual questions, give "
    "a clear and focused explanation. Never list multiple alternative commands. "
    "Never restate the question."
)

KERNEL_SYSTEM_PROMPT_PREFIX = "You are a Linux kernel development expert"

# Pattern matching comprehensive tldr examples
# These ask "How do I use the `X` command...Show practical examples"
COMPREHENSIVE_PATTERN = re.compile(
    r"^How do I use the `[^`]+` command in Linux\? Show practical examples\.$"
)

# Pattern matching the restatement suffix in individual tldr examples
# Matches: "\n\nThis uses `X` to description" or "\n\nThis uses `X` to description:"
RESTATEMENT_PATTERN = re.compile(
    r"\n\nThis uses `[^`]+` to .+$", re.DOTALL
)

# Pattern matching {{placeholder}} syntax
PLACEHOLDER_PATTERN = re.compile(r"\{\{([^}]+)\}\}")

# ---------------------------------------------------------------------------
# Placeholder replacement maps
# ---------------------------------------------------------------------------

PLACEHOLDER_MAP = {
    "file": ["report.txt", "data.csv", "config.yaml", "notes.md", "output.log"],
    "filename": ["report.txt", "data.csv", "backup.tar.gz", "script.sh", "app.conf"],
    "path": ["/home/user/projects", "./src", "/var/log", "/opt/app", "/tmp/workspace"],
    "directory": ["/home/user/documents", "/var/www/html", "/etc/nginx", "/opt/data", "./build"],
    "dir": ["/home/user/projects", "/var/log/app", "/tmp/output", "./dist", "/srv/data"],
    "folder": ["/home/user/downloads", "/var/backups", "/opt/releases", "./assets"],
    "hostname": ["server01", "web-prod", "db-primary", "gateway", "monitor"],
    "host": ["server01", "192.168.1.100", "web-prod.local", "db-01", "10.0.0.5"],
    "username": ["admin", "deploy", "jsmith", "appuser", "sysop"],
    "user": ["admin", "deploy", "jsmith", "webadmin", "dbuser"],
    "port": ["8080", "3000", "443", "5432", "6379"],
    "ip": ["192.168.1.50", "10.0.0.1", "172.16.0.10", "192.168.0.100"],
    "ip_address": ["192.168.1.50", "10.0.0.1", "172.16.0.10", "192.168.0.100"],
    "address": ["192.168.1.50", "10.0.0.1", "172.16.0.10"],
    "url": ["https://example.com", "https://api.example.com/v1", "http://localhost:8080"],
    "pattern": ['"error"', '"TODO"', '"WARN"', '"failed"', '"timeout"'],
    "string": ['"error"', '"hello"', '"search_term"', '"match_this"'],
    "number": ["5", "10", "100", "42", "3"],
    "count": ["5", "10", "20", "50"],
    "size": ["100M", "1G", "500K", "50M"],
    "name": ["myapp", "backup", "webserver", "data-pipeline", "monitor"],
    "command": ["ls -la", "ps aux", "df -h", "free -m", "uptime"],
    "process": ["nginx", "postgres", "redis-server", "node", "python3"],
    "service": ["nginx", "postgresql", "redis", "docker", "sshd"],
    "package": ["curl", "vim", "htop", "nginx", "git"],
    "group": ["developers", "www-data", "docker", "sudo", "staff"],
    "device": ["/dev/sda1", "/dev/sdb", "/dev/nvme0n1p1", "/dev/vda"],
    "interface": ["eth0", "ens33", "wlan0", "enp0s3"],
    "key": ["id_rsa", "deploy_key", "backup_key"],
    "domain": ["example.com", "api.internal", "mail.company.org"],
    "email": ["admin@example.com", "user@company.org"],
    "remote": ["origin", "upstream", "production"],
    "branch": ["main", "develop", "feature/auth"],
    "tag": ["v1.0.0", "v2.3.1", "latest"],
    "message": ["Initial commit", "Fix bug in auth", "Update config"],
    "n": ["5", "10", "3", "20"],
    "m": ["3", "5", "10"],
    "seconds": ["30", "60", "5", "120"],
    "minutes": ["5", "10", "30"],
    "days": ["7", "30", "90", "365"],
    "permission": ["755", "644", "700", "600"],
    "signal": ["SIGTERM", "SIGKILL", "SIGHUP", "SIGUSR1"],
    "pid": ["1234", "5678", "42", "9999"],
    "source": ["/home/user/data", "/var/log/app.log", "./input.txt"],
    "destination": ["/backup/data", "/tmp/output", "./result.txt"],
    "target": ["/opt/app", "/var/www", "/home/deploy/release"],
    "regex": ['"^error"', '"[0-9]+"', '"WARNING|ERROR"'],
    "expression": ['"s/old/new/g"', '"1,5p"', '"/pattern/d"'],
    "text": ["Hello World", "Error occurred", "Task complete"],
    "label": ["app=web", "env=prod", "tier=frontend"],
    "mount_point": ["/mnt/data", "/media/usb", "/mnt/backup"],
    "filesystem": ["ext4", "xfs", "btrfs"],
    "log_file": ["/var/log/syslog", "/var/log/app/error.log", "./debug.log"],
    "config_file": ["/etc/nginx/nginx.conf", "/etc/ssh/sshd_config", "./config.yaml"],
    "script": ["./deploy.sh", "/opt/scripts/backup.sh", "./run.sh"],
}


def get_replacement(placeholder_name: str, rng: random.Random) -> str:
    """Get a realistic replacement for a placeholder name."""
    name_lower = placeholder_name.lower().replace(" ", "_").replace("/", "_")

    # Try exact match first
    if name_lower in PLACEHOLDER_MAP:
        return rng.choice(PLACEHOLDER_MAP[name_lower])

    # Try substring match — check if any key is contained in the placeholder name
    for key, values in PLACEHOLDER_MAP.items():
        if key in name_lower:
            return rng.choice(values)

    # Handle composite placeholders like "path/to/file" or "path/to/directory"
    if "path" in name_lower and "file" in name_lower:
        return rng.choice(["/home/user/report.txt", "/var/log/app.log", "./data.csv"])
    if "path" in name_lower and ("dir" in name_lower or "folder" in name_lower):
        return rng.choice(["/home/user/projects", "/var/www/html", "/opt/app"])
    if "path" in name_lower:
        return rng.choice(["/home/user/data", "/var/log/app", "/opt/config"])

    # Fallback
    return "value"


def replace_placeholders(text: str, seed: int) -> str:
    """Replace all {{placeholder}} occurrences with realistic values.

    Uses a seeded RNG so the same placeholder gets the same value within one example.
    """
    rng = random.Random(seed)
    # Track replacements within this example so same placeholder → same value
    seen = {}

    def replacer(match):
        placeholder = match.group(1)
        if placeholder not in seen:
            seen[placeholder] = get_replacement(placeholder, rng)
        return seen[placeholder]

    return PLACEHOLDER_PATTERN.sub(replacer, text)


def strip_restatement(response: str) -> str:
    """Remove 'This uses `X` to ...' suffix from tldr individual responses."""
    return RESTATEMENT_PATTERN.sub("", response)


def is_comprehensive_tldr(user_content: str) -> bool:
    """Check if this is a comprehensive multi-variation tldr example."""
    return bool(COMPREHENSIVE_PATTERN.match(user_content))


def is_kernel_example(example: dict) -> bool:
    """Check if this example uses the kernel system prompt."""
    for msg in example.get("messages", []):
        if msg["role"] == "system":
            return msg["content"].startswith(KERNEL_SYSTEM_PROMPT_PREFIX)
    return False


def swap_system_prompt(example: dict) -> dict:
    """Replace old system prompt with new direct-answer prompt."""
    messages = []
    for msg in example["messages"]:
        if msg["role"] == "system" and msg["content"] == OLD_SYSTEM_PROMPT:
            messages.append({"role": "system", "content": NEW_SYSTEM_PROMPT})
        else:
            messages.append(msg)
    return {"messages": messages}


def load_jsonl(filepath: Path) -> list[dict]:
    """Load a JSONL file, return empty list if not found."""
    if not filepath.exists():
        print(f"  SKIP (not found): {filepath}")
        return []
    examples = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def save_jsonl(data: list[dict], filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} examples to {filepath}")


def get_user_content(example: dict) -> str:
    for msg in example.get("messages", []):
        if msg["role"] == "user":
            return msg["content"]
    return ""


def get_assistant_content(example: dict) -> str:
    for msg in example.get("messages", []):
        if msg["role"] == "assistant":
            return msg["content"]
    return ""


def set_assistant_content(example: dict, new_content: str) -> dict:
    messages = []
    for msg in example["messages"]:
        if msg["role"] == "assistant":
            messages.append({"role": "assistant", "content": new_content})
        else:
            messages.append(msg)
    return {"messages": messages}


def set_user_content(example: dict, new_content: str) -> dict:
    messages = []
    for msg in example["messages"]:
        if msg["role"] == "user":
            messages.append({"role": "user", "content": new_content})
        else:
            messages.append(msg)
    return {"messages": messages}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def reformat_all(stats_only: bool = False):
    stats = {
        "tldr_total": 0,
        "tldr_comprehensive_dropped": 0,
        "tldr_placeholders_replaced": 0,
        "tldr_restatements_stripped": 0,
        "tldr_kept": 0,
        "nl2bash_total": 0,
        "kernel_docs_total": 0,
        "kernel_samples_total": 0,
        "manpages_total": 0,
        "synthetic_direct": 0,
        "synthetic_clarification": 0,
        "system_prompts_swapped": 0,
    }

    all_output = []

    # --- 1. Process tldr data ---
    print("\n[1/6] Processing tldr data...")
    tldr_data = load_jsonl(RAW_DIR / "tldr_chatml.jsonl")
    stats["tldr_total"] = len(tldr_data)

    for i, ex in enumerate(tldr_data):
        user_content = get_user_content(ex)
        assistant_content = get_assistant_content(ex)

        # 1a. Drop comprehensive multi-variation examples
        if is_comprehensive_tldr(user_content):
            stats["tldr_comprehensive_dropped"] += 1
            continue

        # 1b. Replace {{placeholder}} with realistic values
        seed = int(hashlib.md5(user_content.encode()).hexdigest()[:8], 16)
        if PLACEHOLDER_PATTERN.search(assistant_content):
            assistant_content = replace_placeholders(assistant_content, seed)
            stats["tldr_placeholders_replaced"] += 1
        if PLACEHOLDER_PATTERN.search(user_content):
            user_content = replace_placeholders(user_content, seed)

        # 1c. Strip restatements from individual examples
        original_len = len(assistant_content)
        assistant_content = strip_restatement(assistant_content)
        if len(assistant_content) < original_len:
            stats["tldr_restatements_stripped"] += 1

        # Apply changes
        ex = set_assistant_content(ex, assistant_content)
        ex = set_user_content(ex, user_content)

        # 1d. Swap system prompt
        ex = swap_system_prompt(ex)
        stats["system_prompts_swapped"] += 1

        all_output.append(ex)
        stats["tldr_kept"] += 1

    print(f"  tldr: {stats['tldr_total']} total, {stats['tldr_comprehensive_dropped']} comprehensive dropped, "
          f"{stats['tldr_kept']} kept")
    print(f"  Placeholders replaced: {stats['tldr_placeholders_replaced']}, "
          f"Restatements stripped: {stats['tldr_restatements_stripped']}")

    # --- 2. Process NL2Bash data ---
    print("\n[2/6] Processing NL2Bash data...")
    nl2bash_data = load_jsonl(RAW_DIR / "nl2bash_chatml.jsonl")
    stats["nl2bash_total"] = len(nl2bash_data)

    for ex in nl2bash_data:
        ex = swap_system_prompt(ex)
        stats["system_prompts_swapped"] += 1
        all_output.append(ex)

    print(f"  NL2Bash: {stats['nl2bash_total']} examples (system prompt swapped)")

    # --- 3. Process manpages data ---
    print("\n[3/6] Processing manpages data...")
    manpages_data = load_jsonl(RAW_DIR / "manpages_chatml.jsonl")
    stats["manpages_total"] = len(manpages_data)

    for ex in manpages_data:
        ex = swap_system_prompt(ex)
        stats["system_prompts_swapped"] += 1
        all_output.append(ex)

    print(f"  Manpages: {stats['manpages_total']} examples (system prompt swapped)")

    # --- 4. Process kernel data (keep original system prompt) ---
    print("\n[4/6] Processing kernel data...")
    kernel_docs = load_jsonl(RAW_DIR / "kernel_docs_chatml.jsonl")
    kernel_samples = load_jsonl(RAW_DIR / "kernel_samples_chatml.jsonl")
    stats["kernel_docs_total"] = len(kernel_docs)
    stats["kernel_samples_total"] = len(kernel_samples)

    all_output.extend(kernel_docs)
    all_output.extend(kernel_samples)

    print(f"  Kernel docs: {stats['kernel_docs_total']}, samples: {stats['kernel_samples_total']} "
          f"(kernel prompt kept unchanged)")

    # --- 5. Add synthetic data ---
    print("\n[5/6] Adding synthetic data...")
    synthetic_direct = load_jsonl(RAW_DIR / "synthetic_direct.jsonl")
    synthetic_clarification = load_jsonl(RAW_DIR / "synthetic_clarification.jsonl")
    stats["synthetic_direct"] = len(synthetic_direct)
    stats["synthetic_clarification"] = len(synthetic_clarification)

    all_output.extend(synthetic_direct)
    all_output.extend(synthetic_clarification)

    print(f"  Synthetic direct: {stats['synthetic_direct']}, "
          f"clarification: {stats['synthetic_clarification']}")

    # --- 6. Summary ---
    print(f"\n{'='*60}")
    print("REFORMAT SUMMARY")
    print(f"{'='*60}")
    for key, val in stats.items():
        print(f"  {key:35s}: {val}")
    print(f"  {'combined_total':35s}: {len(all_output)}")
    print(f"{'='*60}")

    if stats_only:
        print("\n  [STATS ONLY] No files written.")
        return

    # --- Write output ---
    print(f"\n[6/6] Writing combined output...")
    save_jsonl(all_output, OUTPUT_FILE)
    print(f"\nDone. Next step:")
    print(f"  python filter_data.py --input {OUTPUT_FILE} --output data/output/ --split")


def main():
    parser = argparse.ArgumentParser(
        description="Reformat training data for direct-answer style"
    )
    parser.add_argument("--stats-only", action="store_true",
                        help="Show statistics without writing files")
    args = parser.parse_args()

    reformat_all(stats_only=args.stats_only)


if __name__ == "__main__":
    main()
