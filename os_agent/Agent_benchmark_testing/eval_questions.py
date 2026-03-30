"""Shared evaluation question bank for the OS AI agent.

Contains ~170 questions organized by category, difficulty, and test type.
Used by test_inference.py, test_harness.py, and master.py routing tests.

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
# ORIGINAL 44 QUESTIONS — natural dev/DevOps phrasing
# ═══════════════════════════════════════════════════════════════════════════

_ORIGINAL_44: list[EvalQuestion] = [
    # --- File operations (6) ---
    EvalQuestion("what files are taking the most space in this folder", "files", "files", "basic", "command"),
    EvalQuestion("show me log files changed in the last 24 hours under /var/log", "files", "files", "basic", "command"),
    EvalQuestion("grep for ERROR in all .log files under /var recursively", "files", "files", "basic", "command"),
    EvalQuestion("when should I use 755 vs 644 permissions", "files", "files", "basic", "conceptual"),
    EvalQuestion("change owner of /opt/app and everything inside it to deploy user", "files", "files", "basic", "command"),
    EvalQuestion("create a symlink from /etc/nginx/sites-enabled/myapp to sites-available/myapp", "files", "files", "basic", "command"),

    # --- Networking & SSH (7) ---
    EvalQuestion("list all open TCP ports on this box", "networking", "network", "basic", "command"),
    EvalQuestion("generate an ed25519 SSH key for deploy@prod", "networking", "network", "basic", "command"),
    EvalQuestion("scp the config.tar.gz to 10.0.1.5:/tmp/", "networking", "network", "basic", "command"),
    EvalQuestion("what's my IP", "networking", "network", "basic", "command"),
    EvalQuestion("check if port 443 is open on 10.0.1.5 without telnet", "networking", "network", "basic", "command"),
    EvalQuestion("block incoming traffic on port 22 with iptables", "networking", "network", "basic", "command"),
    EvalQuestion("rsync /var/www/ to backup@10.0.1.5:/backups/www/", "networking", "network", "basic", "command"),

    # --- Process & resource management (6) ---
    EvalQuestion("show me disk usage by directory", "process", "process", "basic", "command"),
    EvalQuestion("kill the node process that's stuck", "process", "process", "basic", "command"),
    EvalQuestion("which process is using the most memory right now", "process", "process", "basic", "command"),
    EvalQuestion("run this script in the background so it survives SSH logout", "process", "process", "basic", "command"),
    EvalQuestion("set up a cron that runs backup.sh every night at midnight", "process", "process", "basic", "command"),
    EvalQuestion("show me CPU and memory usage in real time", "process", "process", "basic", "command"),

    # --- User & permission management (4) ---
    EvalQuestion("add user jenkins to the sudo group", "users", "process", "basic", "command"),
    EvalQuestion("create a new user deploy with a home directory", "users", "process", "basic", "command"),
    EvalQuestion("lock the intern account without deleting it", "users", "process", "basic", "command"),
    EvalQuestion("what groups does the deploy user belong to", "users", "process", "basic", "command"),

    # --- Package & service management (4) ---
    EvalQuestion("install this .deb package I downloaded", "packages", "packages", "basic", "command"),
    EvalQuestion("restart the nginx service", "packages", "process", "basic", "command"),
    EvalQuestion("check if postgres is set to start on boot", "packages", "process", "basic", "command"),
    EvalQuestion("which package owns /usr/bin/curl", "packages", "packages", "basic", "command"),

    # --- Text processing (4) ---
    EvalQuestion("grab the 3rd column from this space-separated file with awk", "text", "files", "basic", "command"),
    EvalQuestion("replace all occurrences of localhost with 0.0.0.0 in config.yaml", "text", "files", "basic", "command"),
    EvalQuestion("count lines, words, and characters in access.log", "text", "files", "basic", "command"),
    EvalQuestion("sort this file and remove duplicate lines", "text", "files", "basic", "command"),

    # --- Storage & archiving (4) ---
    EvalQuestion("mount the USB drive at /dev/sdb1 to /mnt/usb", "storage", "files", "basic", "command"),
    EvalQuestion("show free disk space on all mounted filesystems", "storage", "files", "basic", "command"),
    EvalQuestion("tar and gzip the /opt/app directory for backup", "storage", "files", "basic", "command"),
    EvalQuestion("find and delete log files older than 30 days under /var/log", "storage", "files", "basic", "command"),

    # --- Kernel / OS concepts (5) ---
    EvalQuestion("what is a kernel module and how do I load one", "kernel", "kernel", "basic", "conceptual"),
    EvalQuestion("what's the difference between a process and a thread", "kernel", "kernel", "basic", "conceptual"),
    EvalQuestion("how does virtual memory paging work in Linux", "kernel", "kernel", "basic", "conceptual"),
    EvalQuestion("what is /proc used for", "kernel", "kernel", "basic", "conceptual"),
    EvalQuestion("show me the kernel version and build info", "kernel", "kernel", "basic", "command"),

    # --- Shell scripting (4) ---
    EvalQuestion("write a bash snippet that checks if /tmp/lock.pid exists before proceeding", "scripting", "files", "basic", "command"),
    EvalQuestion("loop over all .log files in /var/log and print their sizes", "scripting", "files", "basic", "command"),
    EvalQuestion("capture the output of date into a variable in bash", "scripting", "files", "basic", "command"),
    EvalQuestion("validate that a bash script got exactly 2 arguments", "scripting", "files", "basic", "command"),
]

# ═══════════════════════════════════════════════════════════════════════════
# INTERMEDIATE/ADVANCED QUESTIONS PER EXISTING DOMAIN (~44)
# ═══════════════════════════════════════════════════════════════════════════

_DOMAIN_EXPANDED: list[EvalQuestion] = [
    # --- Files: advanced (6) ---
    EvalQuestion("find all setuid binaries on the system", "files", "files", "intermediate", "command"),
    EvalQuestion("find duplicate files by content in /opt/uploads and /var/data", "files", "files", "intermediate", "command"),
    EvalQuestion("recursively change permissions on directories only, leave files alone", "files", "files", "intermediate", "command"),
    EvalQuestion("watch /var/spool/incoming for new files being created", "files", "files", "advanced", "command"),
    EvalQuestion("what's the difference between hard links and symlinks", "files", "files", "intermediate", "conceptual"),
    EvalQuestion("find all files owned by the old admin user across the system", "files", "files", "intermediate", "command"),

    # --- Networking: advanced (6) ---
    EvalQuestion("tunnel port 5432 from the remote db server to my localhost", "networking", "network", "intermediate", "command"),
    EvalQuestion("flush the DNS cache on this machine", "networking", "network", "intermediate", "command"),
    EvalQuestion("capture HTTP traffic on eth0 with tcpdump, only port 80", "networking", "network", "advanced", "command"),
    EvalQuestion("what's the difference between TCP and UDP", "networking", "network", "intermediate", "conceptual"),
    EvalQuestion("which process is listening on port 8080", "networking", "network", "intermediate", "command"),
    EvalQuestion("add a static route to 10.10.0.0/16 via 192.168.1.1", "networking", "network", "advanced", "command"),

    # --- Process: advanced (6) ---
    EvalQuestion("send SIGHUP to nginx to reload its config", "process", "process", "intermediate", "command"),
    EvalQuestion("cap the ffmpeg process to 50% CPU using cgroups", "process", "process", "advanced", "command"),
    EvalQuestion("what are zombie processes and how do I clean them up", "process", "process", "intermediate", "conceptual"),
    EvalQuestion("trace system calls from the running redis process", "process", "process", "advanced", "command"),
    EvalQuestion("show all child processes of PID 1234", "process", "process", "intermediate", "command"),
    EvalQuestion("lower the priority of this backup job with nice", "process", "process", "intermediate", "command"),

    # --- Users: advanced (4) ---
    EvalQuestion("let the deploy user run systemctl restart nginx without a password", "users", "process", "intermediate", "command"),
    EvalQuestion("set password expiry to 90 days for user alice", "users", "process", "intermediate", "command"),
    EvalQuestion("what is PAM and how does it handle authentication", "users", "process", "advanced", "conceptual"),
    EvalQuestion("give user bob read-only access to /var/log/app using ACLs", "users", "files", "advanced", "command"),

    # --- Packages: advanced (4) ---
    EvalQuestion("pin the postgresql package so it doesn't get upgraded", "packages", "packages", "intermediate", "command"),
    EvalQuestion("add the deadsnakes PPA for Python 3.12 on Ubuntu", "packages", "packages", "intermediate", "command"),
    EvalQuestion("list all installed packages sorted by size", "packages", "packages", "intermediate", "command"),
    EvalQuestion("build and install this project from source using configure and make", "packages", "packages", "advanced", "command"),

    # --- Text processing: advanced (4) ---
    EvalQuestion("sum the values in column 3 of this CSV with awk", "text", "files", "intermediate", "command"),
    EvalQuestion("extract the status field from response.json using jq", "text", "files", "intermediate", "command"),
    EvalQuestion("delete all blank lines from this config file with sed", "text", "files", "intermediate", "command"),
    EvalQuestion("grep for lines matching 'error.*timeout' in the logs", "text", "files", "intermediate", "command"),

    # --- Storage: advanced (4) ---
    EvalQuestion("create an LVM logical volume from the free space on vg0", "storage", "files", "advanced", "command"),
    EvalQuestion("add an fstab entry to auto-mount /dev/sdb1 at /data on boot", "storage", "files", "advanced", "command"),
    EvalQuestion("what's the difference between ext4 and XFS", "storage", "files", "advanced", "conceptual"),
    EvalQuestion("run fsck on /dev/sda2 to check for errors", "storage", "files", "advanced", "command"),

    # --- Kernel: advanced (5) ---
    EvalQuestion("what are Linux namespaces and how do containers use them", "kernel", "kernel", "advanced", "conceptual"),
    EvalQuestion("show kernel ring buffer messages since last boot with dmesg", "kernel", "kernel", "intermediate", "command"),
    EvalQuestion("what is eBPF and what's it used for", "kernel", "kernel", "advanced", "conceptual"),
    EvalQuestion("set vm.swappiness to 10 at runtime with sysctl", "kernel", "kernel", "intermediate", "command"),
    EvalQuestion("what's the difference between user space and kernel space", "kernel", "kernel", "advanced", "conceptual"),

    # --- Scripting: advanced (5) ---
    EvalQuestion("write a bash function that returns the disk usage percentage", "scripting", "files", "intermediate", "command"),
    EvalQuestion("use trap to clean up temp files on script exit", "scripting", "files", "advanced", "command"),
    EvalQuestion("iterate over a bash array of server hostnames and ping each", "scripting", "files", "intermediate", "command"),
    EvalQuestion("redirect both stdout and stderr to /var/log/deploy.log", "scripting", "files", "intermediate", "command"),
    EvalQuestion("write a bash case statement for start/stop/restart/status", "scripting", "files", "intermediate", "command"),
]

# ═══════════════════════════════════════════════════════════════════════════
# DEVELOPER-LEVEL QUESTIONS (~50)
# Harder questions outside the fine-tuning data. Tests whether base
# knowledge survives fine-tuning AND whether the harness handles them.
# ═══════════════════════════════════════════════════════════════════════════

_DEVELOPER: list[EvalQuestion] = [
    # --- Git Advanced (8) ---
    EvalQuestion("bisect to find which commit broke the tests", "git", "files", "developer", "command"),
    EvalQuestion("cherry-pick commit abc123 from the release branch", "git", "files", "developer", "command"),
    EvalQuestion("recover a stash I accidentally dropped using reflog", "git", "files", "developer", "command"),
    EvalQuestion("set up a pre-commit hook that runs flake8", "git", "files", "developer", "command"),
    EvalQuestion("squash the last 3 commits into one", "git", "files", "developer", "command"),
    EvalQuestion("shallow clone only the main branch of this repo", "git", "files", "developer", "command"),
    EvalQuestion("set up a git worktree so I can work on the hotfix branch without switching", "git", "files", "developer", "command"),
    EvalQuestion("resolve this merge conflict from the command line", "git", "files", "developer", "command"),

    # --- Docker / Containers (8) ---
    EvalQuestion("build the docker image from the Dockerfile in this directory", "docker", "files", "developer", "command"),
    EvalQuestion("run the postgres container with /var/lib/postgres mounted from the host", "docker", "files", "developer", "command"),
    EvalQuestion("show running containers and their CPU/memory usage", "docker", "process", "developer", "command"),
    EvalQuestion("get a shell inside the running api container", "docker", "process", "developer", "command"),
    EvalQuestion("tail the logs from the worker container", "docker", "process", "developer", "command"),
    EvalQuestion("clean up all stopped containers and dangling images", "docker", "process", "developer", "command"),
    EvalQuestion("create a docker network for the app and db containers to talk", "docker", "network", "developer", "command"),
    EvalQuestion("what's the difference between CMD and ENTRYPOINT in a Dockerfile", "docker", "files", "developer", "conceptual"),

    # --- Debugging & Profiling (8) ---
    EvalQuestion("strace the curl command to see what syscalls it makes", "debugging", "process", "developer", "command"),
    EvalQuestion("attach gdb to the running myapp process PID 4521", "debugging", "process", "developer", "command"),
    EvalQuestion("run valgrind on my C program to check for memory leaks", "debugging", "process", "developer", "command"),
    EvalQuestion("profile CPU usage of the nginx master process with perf", "debugging", "process", "developer", "command"),
    EvalQuestion("generate a core dump from the crashed myapp binary and open it in gdb", "debugging", "process", "developer", "command"),
    EvalQuestion("find which process has /var/log/app.log open", "debugging", "process", "developer", "command"),
    EvalQuestion("show the memory map of PID 4521 from /proc", "debugging", "kernel", "developer", "command"),
    EvalQuestion("what's the difference between static and dynamic analysis", "debugging", "kernel", "developer", "conceptual"),

    # --- Build Systems & Compilation (6) ---
    EvalQuestion("compile main.c with gcc, enable all warnings and debug symbols", "build", "files", "developer", "command"),
    EvalQuestion("write a Makefile with build, clean, and install targets", "build", "files", "developer", "command"),
    EvalQuestion("compile libutils.c into a shared library .so", "build", "files", "developer", "command"),
    EvalQuestion("use pkg-config to get the compiler flags for libssl", "build", "files", "developer", "command"),
    EvalQuestion("cross-compile hello.c for ARM64", "build", "files", "developer", "command"),
    EvalQuestion("what's the difference between static and dynamic linking", "build", "kernel", "developer", "conceptual"),

    # --- Systemd Advanced (5) ---
    EvalQuestion("create a systemd timer that runs the cleanup script every hour", "systemd", "process", "developer", "command"),
    EvalQuestion("write a systemd service unit for our node app", "systemd", "process", "developer", "command"),
    EvalQuestion("show journald logs for the api service since yesterday", "systemd", "process", "developer", "command"),
    EvalQuestion("make the worker service start only after postgres is up", "systemd", "process", "developer", "command"),
    EvalQuestion("what is systemd socket activation and when would you use it", "systemd", "process", "developer", "conceptual"),

    # --- Security & Hardening (5) ---
    EvalQuestion("configure UFW to allow only SSH and HTTP", "security", "network", "developer", "command"),
    EvalQuestion("harden SSH: disable root login and password auth", "security", "network", "developer", "command"),
    EvalQuestion("set up fail2ban to block brute-force SSH attempts", "security", "network", "developer", "command"),
    EvalQuestion("check AppArmor profiles and which ones are enforcing", "security", "kernel", "developer", "command"),
    EvalQuestion("set up auditd to track access to /etc/shadow", "security", "kernel", "developer", "command"),

    # --- Database CLI (5) ---
    EvalQuestion("connect to the local postgres db and list all tables", "database", "files", "developer", "command"),
    EvalQuestion("dump the production mysql database, compressed, before the migration", "database", "files", "developer", "command"),
    EvalQuestion("create a sqlite db and run a quick query from the command line", "database", "files", "developer", "command"),
    EvalQuestion("restore the postgres backup from last night's pg_dump", "database", "files", "developer", "command"),
    EvalQuestion("show the size of each database in postgres", "database", "files", "developer", "command"),

    # --- CI/CD & DevOps (5) ---
    EvalQuestion("set up ssh-agent so I stop getting asked for my passphrase", "devops", "network", "developer", "command"),
    EvalQuestion("configure logrotate for /var/log/myapp/*.log, keep 7 days", "devops", "files", "developer", "command"),
    EvalQuestion("set up a cron job that logs output with timestamps to /var/log/cron.log", "devops", "process", "developer", "command"),
    EvalQuestion("use envsubst to fill in env vars in the nginx.conf.template", "devops", "files", "developer", "command"),
    EvalQuestion("write a health check that curls /healthz and alerts if it fails", "devops", "files", "developer", "command"),
]

# ═══════════════════════════════════════════════════════════════════════════
# HARNESS STRESS TESTS (~26)
# Designed to break routing, parser, or classification.
# ═══════════════════════════════════════════════════════════════════════════

_HARNESS_STRESS: list[EvalQuestion] = [
    # Cross-domain: keyword tie-breakers (process + network keywords)
    EvalQuestion("which process is hogging all the bandwidth", "harness", "process", "advanced", "routing"),
    EvalQuestion("show network connections grouped by process", "harness", "network", "advanced", "routing"),

    # Cross-domain: file + process overlap
    EvalQuestion("list all open files for PID 1234", "harness", "process", "advanced", "routing"),
    EvalQuestion("which process keeps writing to /var/log/syslog", "harness", "process", "advanced", "routing"),

    # Zero-keyword: force model fallback (no strong domain signals)
    EvalQuestion("make this system faster", "harness", "process", "basic", "adversarial"),
    EvalQuestion("something is wrong, things are slow", "harness", "process", "basic", "adversarial"),

    # Format test: should give explanation, NOT a command
    EvalQuestion("what are inodes", "harness", "kernel", "intermediate", "format"),
    EvalQuestion("explain how pipes work in Unix", "harness", "kernel", "intermediate", "format"),

    # Ambiguous: could be command or conceptual
    EvalQuestion("tell me about Linux permissions", "harness", "files", "basic", "format"),
    EvalQuestion("how does cron work", "harness", "process", "basic", "format"),

    # Out-of-domain: no specialist covers this well
    EvalQuestion("set up a python venv for this flask project", "harness", "packages", "intermediate", "adversarial"),
    EvalQuestion("configure nginx as a reverse proxy for port 3000", "harness", "network", "advanced", "adversarial"),

    # --- NEW: Multi-step / compound requests ---
    EvalQuestion("deploy the new config and then check if the service is healthy", "harness", "process", "advanced", "routing"),
    EvalQuestion("back up the database, then rotate the old backups", "harness", "files", "advanced", "routing"),

    # --- NEW: Typos, slang, shorthand ---
    EvalQuestion("whats eating all my ram", "harness", "process", "basic", "adversarial"),
    EvalQuestion("nuke all the old docker containers", "harness", "process", "intermediate", "adversarial"),
    EvalQuestion("yeet the tmp files", "harness", "files", "basic", "adversarial"),

    # --- NEW: Complaints / vague problems ---
    EvalQuestion("nginx is down again", "harness", "process", "basic", "adversarial"),
    EvalQuestion("the app is throwing 502s", "harness", "network", "intermediate", "adversarial"),
    EvalQuestion("disk is almost full", "harness", "files", "basic", "adversarial"),

    # --- NEW: Long compound queries ---
    EvalQuestion("find all core dumps older than a week, show their sizes, and delete them", "harness", "files", "advanced", "routing"),
    EvalQuestion("check which services failed to start, show their logs, and restart them", "harness", "process", "advanced", "routing"),

    # --- NEW: Mixed domain context ---
    EvalQuestion("is the postgres port exposed to the internet", "harness", "network", "intermediate", "routing"),
    EvalQuestion("the container can't resolve DNS, check the network config", "harness", "network", "advanced", "routing"),

    # --- NEW: Requests with irrelevant filler ---
    EvalQuestion("hey so like I was wondering if you could maybe show me what ports are open", "harness", "network", "basic", "adversarial"),
    EvalQuestion("urgent: production is down and we need to find the process that crashed", "harness", "process", "intermediate", "adversarial"),
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
