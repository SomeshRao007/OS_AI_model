"""Shared evaluation question bank for the OS AI agent.

Contains ~130 questions organized by category, difficulty, and test type.
Used by test_inference.py, eval_gguf.py, and master.py routing tests.

Each question is a dict with:
  - q: the question string
  - eval_domain: category for grouping results (files, networking, process, etc.)
  - route_domain: expected harness routing domain (files, network, process, packages, kernel)
  - difficulty: basic | intermediate | advanced | developer
  - test_type: command | conceptual | routing | format | adversarial
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    q: str
    eval_domain: str
    route_domain: str
    difficulty: str
    test_type: str


# ═══════════════════════════════════════════════════════════════════════════
# ORIGINAL 44 QUESTIONS (preserved exactly)
# ═══════════════════════════════════════════════════════════════════════════

_ORIGINAL_44: list[EvalQuestion] = [
    # --- File operations (6) ---
    EvalQuestion("Find all files larger than 100MB on Linux", "files", "files", "basic", "command"),
    EvalQuestion("Find files modified in the last 24 hours in /var/log", "files", "files", "basic", "command"),
    EvalQuestion("Recursively search for the string 'ERROR' in all .log files under /var", "files", "files", "basic", "command"),
    EvalQuestion("What does chmod 755 do and when would you use it?", "files", "files", "basic", "conceptual"),
    EvalQuestion("How do I change the owner of a directory and all its contents?", "files", "files", "basic", "command"),
    EvalQuestion("How do I create a symbolic link?", "files", "files", "basic", "command"),

    # --- Networking & SSH (7) ---
    EvalQuestion("List all open TCP ports on the system", "networking", "network", "basic", "command"),
    EvalQuestion("Generate an SSH key pair and add it to authorized_keys", "networking", "network", "basic", "command"),
    EvalQuestion("How do I copy a file to a remote server using SCP?", "networking", "network", "basic", "command"),
    EvalQuestion("How do I check my current IP address on Linux?", "networking", "network", "basic", "command"),
    EvalQuestion("How do I test if a remote port is open without telnet?", "networking", "network", "basic", "command"),
    EvalQuestion("How do I block port 22 with iptables?", "networking", "network", "basic", "command"),
    EvalQuestion("How do I use rsync to sync a local folder to a remote server?", "networking", "network", "basic", "command"),

    # --- Process & resource management (6) ---
    EvalQuestion("How do I check disk usage broken down by directory?", "process", "process", "basic", "command"),
    EvalQuestion("How do I kill a process by name without knowing its PID?", "process", "process", "basic", "command"),
    EvalQuestion("Show me how to find which process is using the most memory", "process", "process", "basic", "command"),
    EvalQuestion("How do I run a process in the background and keep it after SSH logout?", "process", "process", "basic", "command"),
    EvalQuestion("How do I schedule a cron job to run a script every day at midnight?", "process", "process", "basic", "command"),
    EvalQuestion("How do I check CPU and memory usage in real time?", "process", "process", "basic", "command"),

    # --- User & permission management (4) ---
    EvalQuestion("How do I add a user to the sudo group?", "users", "process", "basic", "command"),
    EvalQuestion("How do I create a new user with a home directory?", "users", "process", "basic", "command"),
    EvalQuestion("How do I lock a user account without deleting it?", "users", "process", "basic", "command"),
    EvalQuestion("How do I view all groups a user belongs to?", "users", "process", "basic", "command"),

    # --- Package & service management (4) ---
    EvalQuestion("How do I install a .deb package manually?", "packages", "packages", "basic", "command"),
    EvalQuestion("How do I start, stop, and restart a systemd service?", "packages", "process", "basic", "command"),
    EvalQuestion("How do I check if a service is enabled on boot with systemd?", "packages", "process", "basic", "command"),
    EvalQuestion("How do I find which package owns a specific file on Debian/Ubuntu?", "packages", "packages", "basic", "command"),

    # --- Text processing (4) ---
    EvalQuestion("How do I extract the 3rd column from a space-separated file using awk?", "text", "files", "basic", "command"),
    EvalQuestion("How do I replace all occurrences of 'foo' with 'bar' in a file using sed?", "text", "files", "basic", "command"),
    EvalQuestion("How do I count lines, words, and characters in a file?", "text", "files", "basic", "command"),
    EvalQuestion("How do I sort a file and remove duplicate lines?", "text", "files", "basic", "command"),

    # --- Storage & archiving (4) ---
    EvalQuestion("How do I mount a USB drive on Linux?", "storage", "files", "basic", "command"),
    EvalQuestion("How do I check available disk space on all mounted filesystems?", "storage", "files", "basic", "command"),
    EvalQuestion("How do I create a compressed tar.gz archive of a directory?", "storage", "files", "basic", "command"),
    EvalQuestion("How do I find and delete files older than 30 days?", "storage", "files", "basic", "command"),

    # --- Kernel / OS concepts (5) ---
    EvalQuestion("What is a Linux kernel module and how do you load one?", "kernel", "kernel", "basic", "conceptual"),
    EvalQuestion("Explain the difference between a process and a thread in Linux", "kernel", "kernel", "basic", "conceptual"),
    EvalQuestion("How does virtual memory paging work in Linux?", "kernel", "kernel", "basic", "conceptual"),
    EvalQuestion("What is the purpose of the /proc filesystem?", "kernel", "kernel", "basic", "conceptual"),
    EvalQuestion("How do I check the current kernel version and build info?", "kernel", "kernel", "basic", "command"),

    # --- Shell scripting (4) ---
    EvalQuestion("Write a bash script that checks if a file exists and prints a message", "scripting", "files", "basic", "command"),
    EvalQuestion("How do I loop over all .log files in a directory in bash?", "scripting", "files", "basic", "command"),
    EvalQuestion("How do I capture the output of a command into a variable in bash?", "scripting", "files", "basic", "command"),
    EvalQuestion("How do I pass arguments to a bash script and validate them?", "scripting", "files", "basic", "command"),
]

# ═══════════════════════════════════════════════════════════════════════════
# NEW: INTERMEDIATE/ADVANCED QUESTIONS PER EXISTING DOMAIN (~44 new)
# ═══════════════════════════════════════════════════════════════════════════

_DOMAIN_EXPANDED: list[EvalQuestion] = [
    # --- Files: advanced (6) ---
    EvalQuestion("Find all setuid files on the system", "files", "files", "intermediate", "command"),
    EvalQuestion("How do I find duplicate files by content across two directories?", "files", "files", "intermediate", "command"),
    EvalQuestion("How do I recursively change permissions on only directories, not files?", "files", "files", "intermediate", "command"),
    EvalQuestion("How do I watch a directory for new files being created in real time?", "files", "files", "advanced", "command"),
    EvalQuestion("What is the difference between hard links and soft links?", "files", "files", "intermediate", "conceptual"),
    EvalQuestion("How do I find all files owned by a specific user across the system?", "files", "files", "intermediate", "command"),

    # --- Networking: advanced (6) ---
    EvalQuestion("How do I set up an SSH tunnel to forward a remote port to localhost?", "networking", "network", "intermediate", "command"),
    EvalQuestion("How do I flush the DNS cache on Linux?", "networking", "network", "intermediate", "command"),
    EvalQuestion("How do I capture network traffic on a specific interface with tcpdump?", "networking", "network", "advanced", "command"),
    EvalQuestion("What is the difference between TCP and UDP?", "networking", "network", "intermediate", "conceptual"),
    EvalQuestion("How do I check which process is listening on a specific port?", "networking", "network", "intermediate", "command"),
    EvalQuestion("How do I add a static route on Linux?", "networking", "network", "advanced", "command"),

    # --- Process: advanced (6) ---
    EvalQuestion("How do I send a SIGHUP signal to a process to reload its config?", "process", "process", "intermediate", "command"),
    EvalQuestion("How do I limit the CPU usage of a process using cgroups?", "process", "process", "advanced", "command"),
    EvalQuestion("What are zombie processes and how do I clean them up?", "process", "process", "intermediate", "conceptual"),
    EvalQuestion("How do I trace system calls made by a running process?", "process", "process", "advanced", "command"),
    EvalQuestion("How do I find all child processes of a given PID?", "process", "process", "intermediate", "command"),
    EvalQuestion("How do I set process priority with nice and renice?", "process", "process", "intermediate", "command"),

    # --- Users: advanced (4) ---
    EvalQuestion("How do I set up passwordless sudo for a specific command?", "users", "process", "intermediate", "command"),
    EvalQuestion("How do I set a password expiration policy for a user?", "users", "process", "intermediate", "command"),
    EvalQuestion("What is PAM and how does it handle authentication in Linux?", "users", "process", "advanced", "conceptual"),
    EvalQuestion("How do I set file ACLs with setfacl to give a specific user read access?", "users", "files", "advanced", "command"),

    # --- Packages: advanced (4) ---
    EvalQuestion("How do I pin a package to prevent it from being upgraded?", "packages", "packages", "intermediate", "command"),
    EvalQuestion("How do I add a third-party PPA repository on Ubuntu?", "packages", "packages", "intermediate", "command"),
    EvalQuestion("How do I list all installed packages sorted by size?", "packages", "packages", "intermediate", "command"),
    EvalQuestion("How do I build and install a package from source using configure, make, make install?", "packages", "packages", "advanced", "command"),

    # --- Text processing: advanced (4) ---
    EvalQuestion("How do I use awk to sum a numeric column in a CSV file?", "text", "files", "intermediate", "command"),
    EvalQuestion("How do I extract JSON values from a file using jq?", "text", "files", "intermediate", "command"),
    EvalQuestion("How do I use sed to delete all blank lines from a file?", "text", "files", "intermediate", "command"),
    EvalQuestion("How do I find lines matching a regex pattern that spans multiple words?", "text", "files", "intermediate", "command"),

    # --- Storage: advanced (4) ---
    EvalQuestion("How do I create and manage LVM logical volumes?", "storage", "files", "advanced", "command"),
    EvalQuestion("How do I add an entry to /etc/fstab to auto-mount a partition at boot?", "storage", "files", "advanced", "command"),
    EvalQuestion("What is the difference between ext4 and XFS filesystems?", "storage", "files", "advanced", "conceptual"),
    EvalQuestion("How do I check and repair a filesystem with fsck?", "storage", "files", "advanced", "command"),

    # --- Kernel: advanced (5) ---
    EvalQuestion("What are Linux namespaces and how are they used for containers?", "kernel", "kernel", "advanced", "conceptual"),
    EvalQuestion("How do I read kernel ring buffer messages with dmesg?", "kernel", "kernel", "intermediate", "command"),
    EvalQuestion("What is eBPF and what is it used for in modern Linux?", "kernel", "kernel", "advanced", "conceptual"),
    EvalQuestion("How do I change a kernel parameter at runtime using sysctl?", "kernel", "kernel", "intermediate", "command"),
    EvalQuestion("What is the difference between user space and kernel space?", "kernel", "kernel", "advanced", "conceptual"),

    # --- Scripting: advanced (5) ---
    EvalQuestion("How do I write a bash function that returns a value?", "scripting", "files", "intermediate", "command"),
    EvalQuestion("How do I use trap to handle signals in a bash script?", "scripting", "files", "advanced", "command"),
    EvalQuestion("How do I use bash arrays to iterate over a list of items?", "scripting", "files", "intermediate", "command"),
    EvalQuestion("How do I redirect both stdout and stderr to a file in bash?", "scripting", "files", "intermediate", "command"),
    EvalQuestion("How do I write a bash script with a case statement for menu options?", "scripting", "files", "intermediate", "command"),
]

# ═══════════════════════════════════════════════════════════════════════════
# NEW: DEVELOPER-LEVEL QUESTIONS (~50)
# These are intentionally harder and many fall outside the fine-tuning data.
# They test whether base knowledge survives fine-tuning AND whether the
# agent harness routes/handles them gracefully.
# ═══════════════════════════════════════════════════════════════════════════

_DEVELOPER: list[EvalQuestion] = [
    # --- Git Advanced (8) ---
    EvalQuestion("How do I use git bisect to find the commit that introduced a bug?", "git", "files", "developer", "command"),
    EvalQuestion("How do I cherry-pick a commit from another branch?", "git", "files", "developer", "command"),
    EvalQuestion("How do I recover a dropped stash using git reflog?", "git", "files", "developer", "command"),
    EvalQuestion("How do I set up a git pre-commit hook that runs linting?", "git", "files", "developer", "command"),
    EvalQuestion("How do I squash the last 3 commits into one?", "git", "files", "developer", "command"),
    EvalQuestion("How do I clone only a specific branch with shallow history?", "git", "files", "developer", "command"),
    EvalQuestion("How do I use git worktree to work on multiple branches simultaneously?", "git", "files", "developer", "command"),
    EvalQuestion("How do I resolve a merge conflict in git from the command line?", "git", "files", "developer", "command"),

    # --- Docker / Containers (8) ---
    EvalQuestion("How do I build a Docker image from a Dockerfile?", "docker", "files", "developer", "command"),
    EvalQuestion("How do I run a Docker container with a mounted volume?", "docker", "files", "developer", "command"),
    EvalQuestion("How do I list all running Docker containers and their resource usage?", "docker", "process", "developer", "command"),
    EvalQuestion("How do I exec into a running Docker container to debug it?", "docker", "process", "developer", "command"),
    EvalQuestion("How do I view logs from a Docker container?", "docker", "process", "developer", "command"),
    EvalQuestion("How do I remove all stopped containers and unused images to free disk space?", "docker", "process", "developer", "command"),
    EvalQuestion("How do I create a Docker network for container-to-container communication?", "docker", "network", "developer", "command"),
    EvalQuestion("What is the difference between CMD and ENTRYPOINT in a Dockerfile?", "docker", "files", "developer", "conceptual"),

    # --- Debugging & Profiling (8) ---
    EvalQuestion("How do I use strace to trace system calls of a command?", "debugging", "process", "developer", "command"),
    EvalQuestion("How do I attach gdb to a running process for debugging?", "debugging", "process", "developer", "command"),
    EvalQuestion("How do I check for memory leaks using valgrind?", "debugging", "process", "developer", "command"),
    EvalQuestion("How do I use perf to profile CPU performance of a program?", "debugging", "process", "developer", "command"),
    EvalQuestion("How do I generate and analyze a core dump from a crashed program?", "debugging", "process", "developer", "command"),
    EvalQuestion("How do I use lsof to find which process has a file open?", "debugging", "process", "developer", "command"),
    EvalQuestion("How do I inspect the memory map of a process using /proc?", "debugging", "kernel", "developer", "command"),
    EvalQuestion("What is the difference between static and dynamic analysis tools?", "debugging", "kernel", "developer", "conceptual"),

    # --- Build Systems & Compilation (6) ---
    EvalQuestion("How do I compile a C program with gcc and enable all warnings?", "build", "files", "developer", "command"),
    EvalQuestion("How do I create a Makefile with a clean target and dependency tracking?", "build", "files", "developer", "command"),
    EvalQuestion("How do I compile a shared library (.so) from C source files?", "build", "files", "developer", "command"),
    EvalQuestion("How do I use pkg-config to find compiler flags for a library?", "build", "files", "developer", "command"),
    EvalQuestion("How do I cross-compile a binary for ARM on an x86 machine?", "build", "files", "developer", "command"),
    EvalQuestion("What is the difference between static and dynamic linking?", "build", "kernel", "developer", "conceptual"),

    # --- Systemd Advanced (5) ---
    EvalQuestion("How do I create a systemd timer that runs a script every hour?", "systemd", "process", "developer", "command"),
    EvalQuestion("How do I write a custom systemd service unit file?", "systemd", "process", "developer", "command"),
    EvalQuestion("How do I view systemd journal logs for a specific service since yesterday?", "systemd", "process", "developer", "command"),
    EvalQuestion("How do I set up systemd service dependencies so service B starts after service A?", "systemd", "process", "developer", "command"),
    EvalQuestion("What is systemd socket activation and when would you use it?", "systemd", "process", "developer", "conceptual"),

    # --- Security & Hardening (5) ---
    EvalQuestion("How do I configure UFW to allow only SSH and HTTP traffic?", "security", "network", "developer", "command"),
    EvalQuestion("How do I harden SSH by disabling root login and password auth?", "security", "network", "developer", "command"),
    EvalQuestion("How do I use fail2ban to protect against brute-force SSH attacks?", "security", "network", "developer", "command"),
    EvalQuestion("How do I check AppArmor profiles and their enforcement status?", "security", "kernel", "developer", "command"),
    EvalQuestion("How do I audit file access using auditd and ausearch?", "security", "kernel", "developer", "command"),

    # --- Database CLI (5) ---
    EvalQuestion("How do I connect to a PostgreSQL database and list all tables?", "database", "files", "developer", "command"),
    EvalQuestion("How do I dump a MySQL database to a SQL file for backup?", "database", "files", "developer", "command"),
    EvalQuestion("How do I create a SQLite database and run a query from the command line?", "database", "files", "developer", "command"),
    EvalQuestion("How do I restore a PostgreSQL database from a pg_dump backup?", "database", "files", "developer", "command"),
    EvalQuestion("How do I check the size of all databases in PostgreSQL?", "database", "files", "developer", "command"),

    # --- CI/CD & DevOps (5) ---
    EvalQuestion("How do I use ssh-agent to avoid typing my SSH passphrase repeatedly?", "devops", "network", "developer", "command"),
    EvalQuestion("How do I set up log rotation with logrotate for a custom application?", "devops", "files", "developer", "command"),
    EvalQuestion("How do I create a cron job that logs its output with timestamps?", "devops", "process", "developer", "command"),
    EvalQuestion("How do I use envsubst to substitute environment variables in a config template?", "devops", "files", "developer", "command"),
    EvalQuestion("How do I write a health check script that curls an endpoint and alerts on failure?", "devops", "files", "developer", "command"),
]

# ═══════════════════════════════════════════════════════════════════════════
# NEW: HARNESS STRESS TESTS (~10)
# Designed to break routing, parser, or classification.
# ═══════════════════════════════════════════════════════════════════════════

_HARNESS_STRESS: list[EvalQuestion] = [
    # Cross-domain: keyword tie-breakers (process + network keywords)
    EvalQuestion("Which process is using the most network bandwidth?", "harness", "process", "advanced", "routing"),
    EvalQuestion("Show me network connections grouped by process", "harness", "network", "advanced", "routing"),

    # Cross-domain: file + process overlap
    EvalQuestion("Find all open files for a specific process", "harness", "process", "advanced", "routing"),
    EvalQuestion("Which process is writing to /var/log/syslog?", "harness", "process", "advanced", "routing"),

    # Zero-keyword: force model fallback (no strong domain signals)
    EvalQuestion("How do I make my system faster?", "harness", "process", "basic", "adversarial"),
    EvalQuestion("Something is wrong with my computer", "harness", "process", "basic", "adversarial"),

    # Format test: should give explanation, NOT a command
    EvalQuestion("What are inodes?", "harness", "kernel", "intermediate", "format"),
    EvalQuestion("Explain how pipes work in Unix", "harness", "kernel", "intermediate", "format"),

    # Ambiguous: could be command or conceptual
    EvalQuestion("Tell me about Linux permissions", "harness", "files", "basic", "format"),
    EvalQuestion("How does cron work?", "harness", "process", "basic", "format"),

    # Out-of-domain: no specialist covers this well
    EvalQuestion("How do I set up a Python virtual environment?", "harness", "packages", "intermediate", "adversarial"),
    EvalQuestion("How do I configure nginx as a reverse proxy?", "harness", "network", "advanced", "adversarial"),
]

# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

ALL_QUESTIONS: list[EvalQuestion] = _ORIGINAL_44 + _DOMAIN_EXPANDED + _DEVELOPER + _HARNESS_STRESS

# Backward-compatible tuple format: (question_text, eval_domain)
# Used by test_inference.py and eval_gguf.py
LEGACY_TUPLES: list[tuple[str, str]] = [(q.q, q.eval_domain) for q in ALL_QUESTIONS]

# Original 44 only (for regression testing against existing benchmarks)
ORIGINAL_44_TUPLES: list[tuple[str, str]] = [(q.q, q.eval_domain) for q in _ORIGINAL_44]

# Routing test cases: (question_text, expected_route_domain)
# Used by master.py keyword/routing tests
ROUTING_TEST_CASES: list[tuple[str, str]] = [(q.q, q.route_domain) for q in ALL_QUESTIONS]


def filter_questions(
    *,
    difficulty: str | None = None,
    test_type: str | None = None,
    eval_domain: str | None = None,
) -> list[EvalQuestion]:
    """Filter the full question bank by any combination of criteria."""
    result = ALL_QUESTIONS
    if difficulty:
        result = [q for q in result if q.difficulty == difficulty]
    if test_type:
        result = [q for q in result if q.test_type == test_type]
    if eval_domain:
        result = [q for q in result if q.eval_domain == eval_domain]
    return result


# Quick stats for verification
if __name__ == "__main__":
    total = len(ALL_QUESTIONS)
    by_difficulty = {}
    by_eval_domain = {}
    by_test_type = {}
    by_route_domain = {}

    for q in ALL_QUESTIONS:
        by_difficulty[q.difficulty] = by_difficulty.get(q.difficulty, 0) + 1
        by_eval_domain[q.eval_domain] = by_eval_domain.get(q.eval_domain, 0) + 1
        by_test_type[q.test_type] = by_test_type.get(q.test_type, 0) + 1
        by_route_domain[q.route_domain] = by_route_domain.get(q.route_domain, 0) + 1

    print(f"Total questions: {total}")
    print(f"  Original 44: {len(_ORIGINAL_44)}")
    print(f"  Domain expanded: {len(_DOMAIN_EXPANDED)}")
    print(f"  Developer: {len(_DEVELOPER)}")
    print(f"  Harness stress: {len(_HARNESS_STRESS)}")
    print()
    print("By difficulty:")
    for k, v in sorted(by_difficulty.items()):
        print(f"  {k}: {v}")
    print()
    print("By eval domain:")
    for k, v in sorted(by_eval_domain.items()):
        print(f"  {k}: {v}")
    print()
    print("By test type:")
    for k, v in sorted(by_test_type.items()):
        print(f"  {k}: {v}")
    print()
    print("By routing domain:")
    for k, v in sorted(by_route_domain.items()):
        print(f"  {k}: {v}")
