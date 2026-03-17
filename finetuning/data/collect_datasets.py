"""
Collect and format training datasets for Linux OS task fine-tuning.

Sources:
  1. NL2Bash (HuggingFace) — natural language to bash command pairs
  2. Local man pages — parsed into instruction/response pairs
  3. tldr-pages (GitHub) — simplified command examples
  4. User-provided raw data (bash history, configs, etc.)

Usage:
  python collect_datasets.py --all              # collect everything
  python collect_datasets.py --nl2bash          # NL2Bash only
  python collect_datasets.py --manpages         # man pages only
  python collect_datasets.py --tldr             # tldr-pages only
  python collect_datasets.py --user-data DIR    # process user-provided data
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
OUTPUT_DIR = Path(__file__).parent / "output"

SYSTEM_PROMPT = (
    "You are a Linux system expert. You provide accurate shell commands, "
    "system administration guidance, and kernel development assistance. "
    "Always explain the implications of commands that modify system state. "
    "Flag destructive operations with warnings."
)


def to_chatml(instruction: str, response: str, system: str = SYSTEM_PROMPT) -> dict:
    """Convert an instruction/response pair to Qwen ChatML format."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction.strip()},
            {"role": "assistant", "content": response.strip()},
        ]
    }


def save_jsonl(data: list[dict], filepath: Path):
    """Save list of dicts as JSONL."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} examples to {filepath}")


# ---------------------------------------------------------------------------
# 1. NL2Bash
# ---------------------------------------------------------------------------

def collect_nl2bash() -> list[dict]:
    """Download and parse NL2Bash dataset from HuggingFace."""
    print("\n[1/4] Collecting NL2Bash dataset...")

    nl2bash_dir = RAW_DIR / "nl2bash"
    nl2bash_dir.mkdir(parents=True, exist_ok=True)

    # Try HuggingFace datasets library first
    examples = []

    # Method 1: HuggingFace datasets (try multiple dataset IDs)
    hf_datasets = [
        ("GWHed/nl2bash", "nl", "bash"),
        ("AnishJoshi/nl2bash-custom", "nl_command", "bash_code"),
        ("aelhalili/bash-commands-dataset", "prompt", "response"),
    ]
    try:
        from datasets import load_dataset
        for ds_name, nl_col, bash_col in hf_datasets:
            loaded_count = 0
            for split in ["train", "dev", "test"]:
                try:
                    ds = load_dataset(ds_name, split=split)
                    for row in ds:
                        nl = row.get(nl_col, "")
                        cmd = row.get(bash_col, "")
                        if nl and cmd and isinstance(nl, str) and isinstance(cmd, str):
                            response = f"```bash\n{cmd.strip()}\n```"
                            examples.append(to_chatml(nl, response))
                            loaded_count += 1
                except Exception:
                    continue
            if loaded_count:
                print(f"  Loaded {loaded_count} pairs from {ds_name}")
        if examples:
            return examples
    except ImportError:
        print(f"  HuggingFace datasets library not installed")

    # Method 2: Direct download from the original NL2Bash GitHub repo
    print("  Trying direct download from NL2Bash GitHub...")
    base_url = "https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data"

    for split in ["train", "dev", "test"]:
        nl_file = nl2bash_dir / f"{split}.nl.filtered"
        cm_file = nl2bash_dir / f"{split}.cm.filtered"

        for fname, url_suffix in [(nl_file, f"{split}.nl.filtered"), (cm_file, f"{split}.cm.filtered")]:
            if not fname.exists():
                url = f"{base_url}/{url_suffix}"
                result = subprocess.run(
                    ["curl", "-sL", "-o", str(fname), url],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    print(f"  Warning: Failed to download {url_suffix}")

        if nl_file.exists() and cm_file.exists():
            nls = nl_file.read_text().strip().split("\n")
            cms = cm_file.read_text().strip().split("\n")
            for nl, cmd in zip(nls, cms):
                if nl.strip() and cmd.strip():
                    response = f"```bash\n{cmd.strip()}\n```"
                    examples.append(to_chatml(nl.strip(), response))

    if examples:
        print(f"  Loaded {len(examples)} pairs from GitHub")
    else:
        print("  WARNING: Could not download NL2Bash. Install `datasets` library or check network.")
        print("  Run: pip install datasets")

    return examples


# ---------------------------------------------------------------------------
# 2. Man Pages
# ---------------------------------------------------------------------------

# Common commands worth documenting
IMPORTANT_COMMANDS = [
    # File operations
    "ls", "cp", "mv", "rm", "mkdir", "rmdir", "find", "locate", "chmod",
    "chown", "chgrp", "ln", "stat", "file", "touch", "dd", "tar", "zip",
    "unzip", "gzip", "bzip2", "xz",
    # Text processing
    "grep", "sed", "awk", "sort", "uniq", "cut", "paste", "tr", "wc",
    "head", "tail", "cat", "tee", "diff", "comm", "join",
    # Process management
    "ps", "top", "htop", "kill", "killall", "pkill", "nice", "renice",
    "nohup", "bg", "fg", "jobs", "wait", "strace", "lsof",
    # System info
    "uname", "uptime", "whoami", "id", "hostname", "dmesg", "lscpu",
    "lsblk", "lspci", "lsusb", "free", "df", "du", "mount", "umount",
    # Networking
    "ip", "ss", "ping", "traceroute", "dig", "nslookup", "curl", "wget",
    "netstat", "iptables", "nft", "tcpdump", "nc",
    # User/group management
    "useradd", "usermod", "userdel", "groupadd", "passwd", "su", "sudo",
    # Package management
    "apt", "apt-get", "dpkg", "dnf", "yum", "pacman", "snap", "flatpak",
    # Systemd
    "systemctl", "journalctl", "timedatectl", "hostnamectl", "loginctl",
    # Disk/filesystem
    "fdisk", "parted", "mkfs", "fsck", "blkid", "lvm", "mdadm",
    # Misc
    "crontab", "at", "ssh", "scp", "rsync", "git", "make", "gcc",
    "xargs", "envsubst", "watch", "tmux", "screen",
]


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)


def collect_manpages() -> list[dict]:
    """Parse local man pages into instruction/response pairs."""
    print("\n[2/4] Collecting man pages...")

    examples = []
    failed = []

    for cmd in IMPORTANT_COMMANDS:
        result = subprocess.run(
            ["man", "-P", "cat", cmd],
            capture_output=True, text=True, timeout=10,
            env={**os.environ, "COLUMNS": "120", "MAN_KEEP_FORMATTING": "0"}
        )

        if result.returncode != 0:
            failed.append(cmd)
            continue

        man_text = strip_ansi(result.stdout)
        if not man_text.strip():
            failed.append(cmd)
            continue

        # Sections are uppercase words at the start of a line (no leading spaces)
        # Split into sections
        sections = {}
        current_section = ""
        current_content = []

        for line in man_text.split("\n"):
            # Section headers: all-caps word(s) at start of line, no leading whitespace
            if re.match(r"^[A-Z][A-Z ]{1,30}$", line.strip()) and not line.startswith(" "):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = line.strip()
                current_content = []
            else:
                current_content.append(line)

        if current_section:
            sections[current_section] = "\n".join(current_content).strip()

        name_desc = sections.get("NAME", cmd).strip()
        synopsis = sections.get("SYNOPSIS", "").strip()
        description = sections.get("DESCRIPTION", "").strip()[:800]
        examples_text = sections.get("EXAMPLES", "").strip()[:600]

        instruction = f"Explain the `{cmd}` Linux command and show common usage examples."
        response_parts = [f"## `{cmd}` — {name_desc}"]

        if synopsis:
            response_parts.append(f"\n**Synopsis:**\n```\n{synopsis}\n```")
        if description:
            response_parts.append(f"\n**Description:**\n{description}")
        if examples_text:
            response_parts.append(f"\n**Examples:**\n{examples_text}")

        response = "\n".join(response_parts)

        if len(response) > 100:
            examples.append(to_chatml(instruction, response))

    if failed:
        print(f"  Skipped {len(failed)} commands (no man page): {', '.join(failed[:10])}...")

    print(f"  Parsed {len(examples)} man pages")
    return examples


# ---------------------------------------------------------------------------
# 3. tldr-pages
# ---------------------------------------------------------------------------

def collect_tldr() -> list[dict]:
    """Clone tldr-pages and parse into instruction/response pairs."""
    print("\n[3/4] Collecting tldr-pages...")

    tldr_dir = RAW_DIR / "tldr"

    if not (tldr_dir / "pages").exists():
        print("  Cloning tldr-pages repository...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/tldr-pages/tldr.git", str(tldr_dir)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ERROR: Failed to clone tldr-pages: {result.stderr}")
            return []

    examples = []
    pages_dir = tldr_dir / "pages"

    # Process all platform directories (common, linux, osx, etc.)
    for platform_dir in sorted(pages_dir.iterdir()):
        if not platform_dir.is_dir():
            continue
        # Prioritize linux and common
        if platform_dir.name not in ("linux", "common"):
            continue

        for md_file in sorted(platform_dir.glob("*.md")):
            content = md_file.read_text()

            # Parse tldr format: # title, > description, - explanation, `command`
            lines = content.strip().split("\n")

            title = ""
            description = ""
            entries = []
            current_desc = ""

            for line in lines:
                line = line.strip()
                if line.startswith("# "):
                    title = line[2:].strip()
                elif line.startswith("> "):
                    description += line[2:].strip() + " "
                elif line.startswith("- "):
                    current_desc = line[2:].strip()
                elif line.startswith("`") and line.endswith("`") and current_desc:
                    cmd = line.strip("`")
                    entries.append((current_desc, cmd))
                    current_desc = ""

            if not entries:
                continue

            # Create one comprehensive example per command
            instruction = f"How do I use the `{title}` command in Linux? Show practical examples."
            response_parts = [f"`{title}` — {description.strip()}\n"]

            for desc, cmd in entries:
                response_parts.append(f"**{desc}**\n```bash\n{cmd}\n```\n")

            response = "\n".join(response_parts)
            examples.append(to_chatml(instruction, response))

            # Also create individual examples for each usage
            for desc, cmd in entries:
                instruction_single = desc.rstrip(".:")
                if not instruction_single.endswith("?"):
                    instruction_single = f"How to {instruction_single.lower()}?"
                response_single = f"```bash\n{cmd}\n```\n\nThis uses `{title}` to {desc.lower()}"
                examples.append(to_chatml(instruction_single, response_single))

    print(f"  Parsed {len(examples)} examples from tldr-pages")
    return examples


# ---------------------------------------------------------------------------
# 4. Kernel Documentation
# ---------------------------------------------------------------------------

# Focus on the most relevant kernel doc directories for OS-level tasks
KERNEL_DOC_PRIORITY_DIRS = [
    "driver-api", "core-api", "filesystems", "networking",
    "process", "admin-guide", "security", "mm", "scheduler",
    "locking", "trace", "bpf", "block", "devicetree",
]


def collect_kernel_docs(docs_path: str | None = None) -> list[dict]:
    """Parse Linux kernel documentation into Q&A pairs."""
    print("\n[K] Collecting kernel documentation...")

    if docs_path:
        doc_root = Path(docs_path)
    else:
        doc_root = RAW_DIR / "kernel_docs" / "Documentation"

    if not doc_root.exists():
        print(f"  ERROR: Kernel docs not found at {doc_root}")
        print("  Run: git clone --depth 1 https://github.com/torvalds/linux.git /tmp/linux-src")
        return []

    examples = []

    # Prioritize the most relevant directories
    rst_files = []
    for priority_dir in KERNEL_DOC_PRIORITY_DIRS:
        target = doc_root / priority_dir
        if target.exists():
            rst_files.extend(target.rglob("*.rst"))
            rst_files.extend(target.rglob("*.txt"))

    # Also grab top-level docs
    for f in doc_root.glob("*.rst"):
        rst_files.append(f)

    skipped = 0
    for rst_file in rst_files:
        content = rst_file.read_text(errors="replace")

        # Skip index files and very short docs
        if rst_file.name == "index.rst" or len(content) < 200:
            skipped += 1
            continue

        # Skip very long docs (>8000 chars) — trim them
        if len(content) > 8000:
            content = content[:8000] + "\n\n[... truncated for brevity]"

        # Extract title from RST (first line with underline)
        lines = content.split("\n")
        title = rst_file.stem.replace("-", " ").replace("_", " ").title()
        for i, line in enumerate(lines[:10]):
            if i + 1 < len(lines) and lines[i + 1].strip() and all(
                c in "=-~^" for c in lines[i + 1].strip()
            ):
                title = line.strip()
                break

        # Derive the topic from the file path
        rel_path = rst_file.relative_to(doc_root)
        topic = str(rel_path.parent).replace("/", " > ")

        instruction = f"Explain the Linux kernel concept: {title} (from kernel docs: {topic})"
        response = f"## {title}\n\n{content}"

        examples.append(to_chatml(
            instruction,
            response,
            system="You are a Linux kernel development expert. You explain kernel internals, "
                   "APIs, subsystems, and module development with accuracy and clarity. "
                   "Include relevant code examples and safety considerations."
        ))

    print(f"  Parsed {len(examples)} kernel docs (skipped {skipped} index/short files)")
    return examples


# ---------------------------------------------------------------------------
# 4b. Kernel Source Samples (actual code from samples/ and small drivers)
# ---------------------------------------------------------------------------

# Most educational kernel sample directories
KERNEL_SAMPLE_DIRS = [
    "kobject", "kprobes", "kfifo", "workqueue", "configfs",
    "connector", "trace_events", "livepatch", "seccomp",
    "hw_breakpoint", "fprobe", "ftrace", "timers", "pidfd",
    "watchdog", "watch_queue", "bpf", "cgroup", "landlock",
    "hidraw", "hid", "vfs",
]


def collect_kernel_samples(kernel_src: str = "/tmp/linux-src") -> list[dict]:
    """Parse kernel samples/ directory for example modules and code."""
    print("\n[KS] Collecting kernel source samples...")

    src_root = Path(kernel_src)
    samples_dir = src_root / "samples"

    if not samples_dir.exists():
        print(f"  ERROR: Kernel source not found at {kernel_src}")
        print("  The full kernel source (not just Documentation) is needed.")
        return []

    examples = []

    # 1. Collect from samples/ — well-commented example code
    for sample_dir_name in KERNEL_SAMPLE_DIRS:
        sample_dir = samples_dir / sample_dir_name
        if not sample_dir.exists():
            continue

        for c_file in sample_dir.glob("*.c"):
            content = c_file.read_text(errors="replace")
            if len(content) < 100 or len(content) > 12000:
                continue

            filename = c_file.name
            topic = sample_dir_name.replace("_", " ")

            instruction = (
                f"Show me a Linux kernel example of {topic} "
                f"(based on kernel samples/{sample_dir_name}/{filename})"
            )
            response = (
                f"Here's an example kernel module for **{topic}** "
                f"from the official Linux kernel samples:\n\n"
                f"```c\n{content}\n```\n\n"
                f"This example demonstrates {topic} usage in the Linux kernel. "
                f"Build it with `make -C /lib/modules/$(uname -r)/build M=$(pwd) modules`."
            )

            examples.append(to_chatml(
                instruction, response,
                system="You are a Linux kernel development expert. You explain kernel internals, "
                       "APIs, subsystems, and module development with accuracy and clarity. "
                       "Include relevant code examples and safety considerations."
            ))

    # 2. Collect Makefiles from samples (shows how to build modules)
    for makefile in samples_dir.rglob("Makefile"):
        content = makefile.read_text(errors="replace")
        if len(content) < 20 or len(content) > 3000:
            continue
        rel = makefile.relative_to(samples_dir)
        topic = str(rel.parent)
        if topic == ".":
            continue

        instruction = f"How do I write a Makefile for a kernel module related to {topic}?"
        response = (
            f"Here's an example kernel module Makefile for **{topic}** "
            f"from the Linux kernel source:\n\n"
            f"```makefile\n{content}\n```"
        )
        examples.append(to_chatml(
            instruction, response,
            system="You are a Linux kernel development expert. You explain kernel internals, "
                   "APIs, subsystems, and module development with accuracy and clarity. "
                   "Include relevant code examples and safety considerations."
        ))

    # 3. Collect key kernel headers that define commonly-used APIs
    key_headers = [
        "include/linux/module.h", "include/linux/init.h",
        "include/linux/kernel.h", "include/linux/fs.h",
        "include/linux/cdev.h", "include/linux/ioctl.h",
        "include/linux/slab.h", "include/linux/mutex.h",
        "include/linux/spinlock.h", "include/linux/workqueue.h",
        "include/linux/kthread.h", "include/linux/proc_fs.h",
        "include/linux/seq_file.h", "include/linux/netfilter.h",
        "include/linux/sysfs.h", "include/linux/kobject.h",
        "include/linux/device.h", "include/linux/platform_device.h",
    ]

    for header_path in key_headers:
        full_path = src_root / header_path
        if not full_path.exists():
            continue
        content = full_path.read_text(errors="replace")
        # Only keep the first 6000 chars — headers can be very long
        if len(content) > 6000:
            content = content[:6000] + "\n/* ... truncated ... */"

        header_name = full_path.name
        instruction = f"What functions and macros does the Linux kernel header <linux/{header_name}> provide?"
        response = (
            f"The kernel header `<linux/{header_name}>` defines key APIs:\n\n"
            f"```c\n{content}\n```\n\n"
            f"This header is included via `#include <linux/{header_name}>` in kernel modules."
        )
        examples.append(to_chatml(
            instruction, response,
            system="You are a Linux kernel development expert. You explain kernel internals, "
                   "APIs, subsystems, and module development with accuracy and clarity. "
                   "Include relevant code examples and safety considerations."
        ))

    print(f"  Collected {len(examples)} kernel source examples")
    return examples


# ---------------------------------------------------------------------------
# 5. User-provided data
# ---------------------------------------------------------------------------

def collect_user_data(user_dir: str) -> list[dict]:
    """Process user-provided raw data (bash history, configs, etc.)."""
    print(f"\n[4/4] Processing user-provided data from {user_dir}...")

    user_path = Path(user_dir)
    if not user_path.exists():
        print(f"  ERROR: Directory {user_dir} does not exist")
        return []

    examples = []

    # Process bash history if present
    history_files = list(user_path.glob("*history*")) + list(user_path.glob("*.bash_history"))
    for hfile in history_files:
        lines = hfile.read_text().strip().split("\n")
        # Filter out trivial commands
        meaningful_cmds = []
        trivial = {"ls", "cd", "pwd", "clear", "exit", "history", ""}
        for line in lines:
            line = line.strip()
            # Skip timestamp lines (bash history with timestamps)
            if line.startswith("#"):
                continue
            base_cmd = line.split()[0] if line.split() else ""
            if base_cmd not in trivial and len(line) > 5:
                meaningful_cmds.append(line)

        # Deduplicate while preserving order
        seen = set()
        unique_cmds = []
        for cmd in meaningful_cmds:
            if cmd not in seen:
                seen.add(cmd)
                unique_cmds.append(cmd)

        for cmd in unique_cmds[:2000]:  # cap at 2000
            instruction = f"What does this Linux command do: `{cmd}`?"
            response = f"```bash\n{cmd}\n```\n\nThis command needs to be explained based on its components."
            examples.append(to_chatml(instruction, response))
        print(f"  Processed {len(unique_cmds)} unique commands from {hfile.name}")

    # Process config files if present
    config_files = list(user_path.glob("*.conf")) + list(user_path.glob("*.service"))
    for cfile in config_files:
        content = cfile.read_text()
        if len(content) < 20 or len(content) > 5000:
            continue
        instruction = f"Explain this Linux configuration file ({cfile.name}):"
        response = f"```\n{content}\n```\n\nThis configuration needs explanation based on its directives."
        examples.append(to_chatml(instruction, response))
    if config_files:
        print(f"  Processed {len(config_files)} config files")

    # Process any pre-formatted JSONL files
    jsonl_files = list(user_path.glob("*.jsonl"))
    for jfile in jsonl_files:
        for line in jfile.read_text().strip().split("\n"):
            if line.strip():
                item = json.loads(line)
                # Accept if it has messages array or instruction/response keys
                if "messages" in item:
                    examples.append(item)
                elif "instruction" in item and "response" in item:
                    examples.append(to_chatml(item["instruction"], item["response"]))
        print(f"  Loaded pre-formatted data from {jfile.name}")

    print(f"  Total user data: {len(examples)} examples")
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect training datasets for Linux OS fine-tuning")
    parser.add_argument("--all", action="store_true", help="Collect all auto-downloadable datasets")
    parser.add_argument("--nl2bash", action="store_true", help="Collect NL2Bash dataset")
    parser.add_argument("--manpages", action="store_true", help="Parse local man pages")
    parser.add_argument("--tldr", action="store_true", help="Collect tldr-pages")
    parser.add_argument("--kernel-docs", type=str, nargs="?", const="auto",
                        help="Parse kernel Documentation (path or 'auto' to use raw/kernel_docs)")
    parser.add_argument("--kernel-samples", type=str, nargs="?", const="/tmp/linux-src",
                        help="Parse kernel samples/ and key headers (path to kernel source root)")
    parser.add_argument("--user-data", type=str, help="Path to user-provided raw data directory")
    args = parser.parse_args()

    if not any([args.all, args.nl2bash, args.manpages, args.tldr, args.kernel_docs, args.kernel_samples, args.user_data]):
        parser.print_help()
        sys.exit(1)

    all_examples = []

    if args.all or args.nl2bash:
        nl2bash_data = collect_nl2bash()
        all_examples.extend(nl2bash_data)
        save_jsonl(nl2bash_data, RAW_DIR / "nl2bash_chatml.jsonl")

    if args.all or args.manpages:
        manpage_data = collect_manpages()
        all_examples.extend(manpage_data)
        save_jsonl(manpage_data, RAW_DIR / "manpages_chatml.jsonl")

    if args.all or args.tldr:
        tldr_data = collect_tldr()
        all_examples.extend(tldr_data)
        save_jsonl(tldr_data, RAW_DIR / "tldr_chatml.jsonl")

    if args.all or args.kernel_docs:
        docs_path = None if args.kernel_docs in ("auto", None) else args.kernel_docs
        kernel_data = collect_kernel_docs(docs_path)
        all_examples.extend(kernel_data)
        save_jsonl(kernel_data, RAW_DIR / "kernel_docs_chatml.jsonl")

    if args.all or args.kernel_samples:
        ks_path = args.kernel_samples if args.kernel_samples and args.kernel_samples != "auto" else "/tmp/linux-src"
        kernel_sample_data = collect_kernel_samples(ks_path)
        all_examples.extend(kernel_sample_data)
        save_jsonl(kernel_sample_data, RAW_DIR / "kernel_samples_chatml.jsonl")

    if args.user_data:
        user_data = collect_user_data(args.user_data)
        all_examples.extend(user_data)
        save_jsonl(user_data, RAW_DIR / "user_data_chatml.jsonl")

    # Save combined output
    if all_examples:
        save_jsonl(all_examples, RAW_DIR / "combined_raw.jsonl")
        print(f"\n{'='*60}")
        print(f"Total collected: {len(all_examples)} examples")
        print(f"Saved to: {RAW_DIR / 'combined_raw.jsonl'}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
