# Agent Harness — Tool Catalog

**Date**: 2026-03-30
**Purpose**: Inventory of every tool the agent harness provides, with capabilities, limits, and potential additions.

---

## Architecture Overview

The harness wraps a single GGUF model (Qwen 3.5 4B) with layers that compensate for what a small model can't reliably do alone:

```
Query
  │
  ▼
[1] Master Router          keyword match → domain (90.9%)
                           model fallback → domain (86.4%)
  │
  ▼
[2] Environment Context    CWD + system info injected into prompt
  │
  ▼
[3] FAISS Memory           prior solutions retrieved semantically
  │
  ▼
[4] RAG Injection          command-specific flag hints from help_db
  │
  ▼
[5] Inference Engine       single shared Qwen 3.5 4B (Q4_K_M, GPU)
  │
  ▼
[6] Validator              7 rules: invalid flags, type mismatches, deprecated syntax
  │
  ▼
[7] Risk Classifier        safe / moderate / dangerous
  │
  ▼
[8] Sandboxed Executor     bubblewrap isolation, 30s timeout
  │
  ▼
[9] Desktop Notifier       alerts for dangerous commands + low VRAM
  │
  ▼
[10] Session + Shared State  logs turn, stores solution, updates cross-agent context
```

---

## Implemented Tools

### 1. Master Router
**File**: `os_agent/agents/master.py` — `MasterAgent`

| Attribute | Value |
|---|---|
| Stage 1 | Keyword matching (instant, 0 inference) |
| Stage 2 | Model fallback (512 tokens, ~100ms) |
| Domains | files, network, process, packages, kernel |
| Keyword coverage | 90.9% of queries resolved without model |
| Combined routing accuracy | 93.2% (41/44 original, 81.1% on 164-question suite) |
| Fallback domain | files (broadest, most useful for unknowns) |

**What it does**: Classifies any user query to a domain specialist using word intersection scoring, falls back to LLM if ambiguous.

**Limitations**:
- Keyword matching has no semantic understanding — "what does cron do" scores same as "set up cron"
- 18.9% misroutes on the full suite (mostly scripting→files, users→process — both still get correct answers)
- No context carried into classifier (each query classified in isolation)

---

### 2. Environment Context
**File**: `os_agent/shell/context.py` — `EnvironmentContext`

| Attribute | Value |
|---|---|
| System info | OS, kernel, hostname, user, distro, Python version, RAM, GPU |
| CWD context | Current directory + first 10 entries |
| Cache | System info cached once per session; CWD refreshed per query |
| Token cost | ~40 tokens system info + ~20 tokens CWD |

**What it does**: Grounds the model's responses to the actual machine — prevents generic `/home/user/projects` paths, distro-wrong commands, version mismatches.

**Limitations**:
- CWD listing hard-capped at 10 entries (deep directories lose context)
- GPU query via nvidia-smi adds ~5ms overhead at session start
- No file content preview — model knows files exist but not what's in them

---

### 3. FAISS Semantic Memory (Tier 2)
**File**: `os_agent/memory/agent_memory.py` — `AgentMemory`

| Attribute | Value |
|---|---|
| Embedding model | all-MiniLM-L6-v2 (384-dim, CPU-only, 53 MB) |
| Index type | FAISS IndexFlatL2 (brute force, exact search) |
| Max vectors | 500 per domain (5 domains = 2,500 total) |
| Similarity threshold | L2² < 0.4 (≈ cosine similarity > 0.8) |
| Top-k results | 3 hits injected per query |
| Truncation | 200 chars per hit in prompt |
| Persistence | `.faiss` + `.json` in `~/.local/share/ai-daemon/memory/` |
| Prune strategy | Rebuild keeping most recent N on capacity hit |
| Lazy load | Embedding model loads on first call (~2s one-time) |

**What it does**: Stores successful problem→solution pairs per domain. Before calling the model, searches for semantically similar past queries and injects them into the prompt. Agents learn from use.

**Limitations**:
- Fresh indices (empty memory) give zero benefit — needs usage to warm up
- CPU-only embeddings: ~10ms per embed (fast enough but not zero-cost)
- 200-char truncation can cut off the most useful part of a long solution
- No effectiveness scoring — all stored solutions treated equally
- No decay — old solutions persist indefinitely unless pruned at 500 limit

---

### 4. Shared State (Tier 1)
**File**: `os_agent/memory/shared_state.py` — `SharedState`

| Attribute | Value |
|---|---|
| Storage | JSON at `~/.local/share/ai-daemon/state.json` |
| System snapshot | df, ip addr, ps aux — cached 30s TTL |
| Action log | Last 50 agent actions (FIFO) |
| Write safety | Atomic via tmp file + os.replace() |

**What it does**: Global state all agents can read/write. Lets agents know what other agents have recently done — prevents redundant actions and enables cross-domain awareness ("files agent freed 2GB, network agent is idle").

**Limitations**:
- Snapshots NOT injected into prompts by default (too expensive for 2048 ctx window)
- Only injected selectively when query keywords match (disk/net/proc)
- Action log is informational only — no automatic deduplication or summarisation

---

### 5. Session Context (Tier 3)
**File**: `os_agent/memory/session.py` — `SessionContext`

| Attribute | Value |
|---|---|
| Storage | In-memory only (lost on restart) |
| Max turns | 20 (FIFO eviction) |
| Context string | Last 5 turns, 120 chars/turn, ~62 tokens |
| Format | `[domain] Q: ... → A: ...` compact tags |

**What it does**: Tracks the current conversation. Last 5 turns injected into every specialist prompt so the model knows what was asked and answered recently. Prevents "already told you psql is installed" type repetition.

**Limitations**:
- Ephemeral — cleared on shell exit (no cross-session continuity)
- 120-char truncation can lose command context for long responses
- All domains see all turns (no per-domain filtering)

---

### 6. RAG Injection
**File**: `os_agent/inference/rag.py`

| Attribute | Value |
|---|---|
| Keyword patterns | 73 keyword → command mappings |
| Match strategy | Longest keyword wins (pre-sorted) |
| Help DB | 146 commands with flags, 8 hand-curated OVERRIDEs |
| Token cost | ~15–30 tokens per injected hint |
| Hit rate | ~16% of queries get RAG context (24/150 in eval) |

**What it does**: Detects which command a query is about, injects a one-line flag summary into the system prompt before inference. Prevents the model from inventing flags by showing it the real ones.

**Limitations**:
- Keyword-only detection (no semantic matching — "generate an auth key" doesn't hit ssh-keygen)
- Only one command detected per query (first match wins)
- help_db generated from `--help` output which is often incomplete (why curated OVERRIDE list exists)
- Cannot detect compound commands (`find ... | xargs ...` — only picks up `find`)

---

### 7. Rule-Based Validator
**File**: `os_agent/inference/validator.py`

| Rule | Scope | Action |
|---|---|---|
| 1. Explicitly invalid flag (e.g. `wc -n`) | All commands with `invalid_flags` list | Block |
| 2. Unknown flag | **Curated commands only** (8 total) | Block |
| 3. Deprecated syntax (`find -perm +4000`) | find | Block + suggest |
| 4. Arg type mismatch (`ssh-keygen -f email`) | Curated commands | Block + suggest |
| 5. Duplicate flag (`useradd -m -m`) | All commands | Block |
| 6. Distro mismatch (`dnf` on Ubuntu) | All | Warn only |
| Shell substitutions `$(...)` | Rule 4 | Skip type check (runtime-evaluated) |
| Unknown command | All | Warn only, never block |

**What it does**: Post-inference safety net. Catches the most common model errors before a command reaches the executor.

**Limitations**:
- Strict blocking only for 8 hand-curated commands (auto-parsed schemas are too incomplete)
- Rule 2 coverage could expand if more commands get curated OVERRIDE entries
- Cannot validate multi-command pipelines end-to-end (only validates first command)
- No semantic validation — `rm -rf /home/legit_dir` passes all rules

---

### 8. Sandboxed Executor
**File**: `os_agent/tools/executor.py` — `SandboxedExecutor`
**File**: `os_agent/tools/registry.py` — `DOMAIN_WHITELIST`, `DANGEROUS_PATTERNS`

| Attribute | Value |
|---|---|
| Sandbox | bubblewrap (bwrap) — Linux namespaces |
| Timeout | 30 seconds (configurable in daemon.yaml) |
| Risk levels | safe (auto-run) / moderate (y/n) / dangerous (y/n + warning) |
| Dangerous patterns | 17 regex rules |
| Safe commands | 45 read-only commands |
| Domain whitelists | 5 domains, 100+ commands each + 18 universal |
| Out-of-domain | y/n prompt (not hard block) |
| bwrap fallback | Direct execution with timeout if bwrap unavailable (AppArmor/container) |

**What it does**: Executes the model's suggested commands with isolation. Bubblewrap creates a new mount/user namespace per command. Dangerous commands require explicit user confirmation. Out-of-domain commands warn but don't block.

**Limitations**:
- Network isolation disabled (requires CAP_NET_ADMIN, breaks most network agent commands)
- bwrap blocked in some environments (VS Code terminal, certain AppArmor profiles)
- 30s timeout kills long-running commands (e.g. large apt install, make from source)
- No resource limits (CPU/memory cgroups not applied per-command)

---

### 9. Desktop Notifier
**File**: `os_agent/notify/desktop.py` — `DesktopNotifier`

| Attribute | Value |
|---|---|
| Backend | `notify-send` (D-Bus, fire-and-forget) |
| Triggers | Dangerous commands, VRAM < 500 MB |
| Action buttons | Not supported (GLib event loop incompatible with REPL) |
| Graceful degradation | Silently disabled if notify-send not found |

**What it does**: Alerts the user via desktop notification when the agent proposes a dangerous command or when VRAM is critically low. Runs non-blocking alongside the terminal y/n prompt.

**Limitations**:
- No interactive Accept/Reject buttons (deferred to C/C++ daemon rewrite)
- Cannot guarantee user sees notification if desktop is focused elsewhere
- VRAM check via nvidia-smi adds ~50ms overhead before every inference call

---

### 10. Command Parser
**File**: `os_agent/tools/parser.py`

| Attribute | Value |
|---|---|
| Pattern | ` ```bash`, ` ```sh`, ` ```shell`, or generic ` ``` ` |
| Mode | DOTALL (handles multi-line commands) |
| Functions | `extract_command()` (first block), `extract_all_commands()` (all blocks) |

**What it does**: Extracts the executable command from model responses. Required because the model always wraps commands in fenced code blocks.

**Limitations**:
- Returns `None` if model gives inline code or plain text (conceptual responses are fine, but command responses without a code block are silently dropped)
- `extract_all_commands()` exists but is unused — multi-step command chaining not yet implemented

---

## Harness Lift Summary

| Layer added | Accuracy | Delta |
|---|---|---|
| Raw model | 88% | — |
| + System prompts (Step 3) | ~91% | +3pp |
| + FAISS memory (Step 5) | TBD | — |
| + RAG + Validator (Step 8) | 96.0% | +8pp |
| + Full harness (Step 10) | **98.8%** | **+10.8pp** |

The harness contributes more to accuracy than post-training. A 4B model at 98.8% with harness outperforms what a raw 7B model would score on OS tasks.

---

## Potential Additions

Ordered by effort vs value for an OS-core use case:

### High Value, Low Effort

**Man Page Lookup**
- Trigger: validator blocks a command OR confidence is low
- Implementation: `man -P cat {cmd} | head -80` → inject relevant section into prompt
- Value: eliminates the remaining validator false negatives (obscure flags the model invents)
- Effort: ~50 lines — shell call + section parser + prompt injection hook

**Log Analyzer**
- Trigger: query contains "error", "failed", "why is X not working"
- Implementation: `journalctl -u {service} -n 50 --no-pager` or `tail -n 50 {logfile}`
- Value: gives the model real error messages to reason over instead of guessing
- Effort: ~60 lines — log file detector + truncation + injection

**Selective Shared-State Injection**
- Trigger: query keywords match "disk", "memory", "port", "process"
- Implementation: Already in shared_state.py — just needs a keyword-gated injection in base.py
- Value: model gets live system state (df output, ps output) for diagnosis queries
- Effort: ~10 lines — keyword gate in `augmented_prompt_with_context()`

### High Value, Medium Effort

**Flag Validator Coverage Expansion**
- Current: 8 hand-curated commands
- Addition: curate 20 more high-frequency commands (git, systemctl, docker, kubectl, apt, tar, ssh, rsync, grep, sed, awk, chmod, ps, kill, curl, wget, ip, ss, journalctl, crontab)
- Value: extends unknown-flag blocking to the most-used commands
- Effort: ~4 hours manual curation + testing per command batch

**Config File Parser**
- Trigger: query references a config file (nginx.conf, sshd_config, /etc/fstab, etc.)
- Implementation: detect config file path in query → read it → inject relevant section
- Value: model can give advice based on actual current config, not hypothetical defaults
- Effort: ~100 lines — path extractor + safe reader + section detection

**Cron Manager Tool**
- Trigger: query about scheduling, cron, timers
- Implementation: `crontab -l` output injected into prompt + syntax validator for generated cron expressions
- Value: prevents model from generating cron syntax for a time slot that already exists
- Effort: ~80 lines — crontab reader + cron expression validator

### Lower Priority

**Web Search** — Latency (external HTTP) incompatible with <2s target. Useful only for "how do I install X" where local knowledge is stale. Defer to Phase 2 with async execution.

**Package Database** — `apt-cache show {pkg}` already available via executor. Injection useful for version pinning queries. Low frequency, moderate effort.

**Semantic Router Upgrade** — Replace keyword matching with embedding similarity for routing. Eliminates the 18.9% misroute rate. Requires ~10ms embedding call per query (currently 0ms). Worthwhile when routing misses cause wrong domain context injection.

---

## Tool Gaps vs OS-Core Requirements

| OS-Core Need | Current Coverage | Gap |
|---|---|---|
| File operations | Full (FAISS + executor + whitelist) | None |
| Network configuration | Good (iptables, ip, ss, ssh) | No live traffic data injection |
| Process management | Good (ps, kill, systemctl, cron) | No cgroup resource limits |
| Package management | Good (apt, dpkg, pip, snap) | No dependency conflict detection |
| Kernel / hardware | Partial (lsmod, sysctl, dmesg) | No eBPF tooling, no perf integration |
| Log analysis | None (tool not yet built) | High-value gap — common OS diagnosis task |
| Config management | None (tool not yet built) | High-value gap — wrong config = broken service |
| Systemd unit files | None (model hallucination, no tool) | Critical gap — model has no knowledge of `.service`/`.timer` syntax |
| Security auditing | Partial (auditctl, openssl in whitelist) | No structured audit log analysis |
| Disk health | None | `smartctl` output injection for failing disks |
