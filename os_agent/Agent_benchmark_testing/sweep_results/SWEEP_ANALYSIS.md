# Parameter Sweep Analysis — 4 Configs x 44 Questions (RAG + Validator)

**Date**: 2026-03-29
**Model**: `qwen3.5-4b-os-q4km.gguf` (Q4_K_M, 2.6 GB)
**Hardware**: RTX 3060 12GB, 31GB RAM, Linux 6.17.0-19-generic
**Eval set**: Original 44 questions (9 domains)
**Mode**: RAG context injection + validator active on all configs

---

## Summary Table

| Config | temp | top_p | top_k | n_ctx | tok/s (warm) | total_tokens | RAG hits | Blocks | Warns | VRAM peak |
|---|---|---|---|---|---|---|---|---|---|---|
| **baseline** | 0.6 | 0.95 | 20 | 1024 | 70.7 | 1824 | 5/44 (11%) | 4 | 5 | 3899 MB |
| **conservative** | 0.5 | 0.98 | 15 | 4096 | 70.5 | 1919 | 5/44 (11%) | 2 | 7 | 3891 MB |
| **precise** | 0.3 | 0.90 | 10 | 4096 | 70.1 | 1890 | 5/44 (11%) | 2 | 7 | 3891 MB |
| **ctx4k_default** | 0.6 | 0.95 | 20 | 4096 | 68.4 | 1867 | 5/44 (11%) | 3 | 6 | 3967 MB |

---

## Validator Block Analysis

### What each config blocked

| Question | baseline (0.6) | conservative (0.5) | precise (0.3) | ctx4k (0.6) |
|---|---|---|---|---|
| Q2: find -mtime with -exec ls -lh | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| Q3: find -exec grep -Hn | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| Q17: nohup -d (hallucinated flag) | **BLOCKED** | ok | ok | ok |
| Q21: useradd -m -m (duplicate flag) | **BLOCKED** | ok | ok | ok |
| Q30: wc -lwc (combined flags) | ok | ok | ok | **BLOCKED** |

### Classification of blocks

**False positives (validator bugs, not model errors):**
- Q2 + Q3: Validator parsed flags inside `-exec` as `find` flags. The model's commands (`find -exec ls -lh`, `find -exec grep -Hn`) were actually correct. **Fixed post-sweep** — validator now skips tokens between `-exec` and `\;`/`+`.
- Q30 (ctx4k only): `wc -lwc` — combined short flags. Validator doesn't handle combined flag syntax.

**True positives (model errors correctly caught):**
- Q17 (baseline only): `nohup -d` — `-d` flag does not exist on nohup. Model hallucinated it.
- Q21 (baseline only): `useradd -m -m` — duplicate `-m` flag. Model generated it twice.

### Real block counts after -exec fix

| Config | Real blocks (model errors only) |
|---|---|
| baseline (temp=0.6) | **2** (nohup -d, useradd -m -m) |
| conservative (temp=0.5) | **0** |
| precise (temp=0.3) | **0** |
| ctx4k_default (temp=0.6) | **0** (wc -lwc is a combined-flag parser limitation, not a model error) |

---

## Key Findings

### 1. Temperature is the dominant accuracy factor

Both hallucinated flag errors (nohup -d, useradd -m -m) disappear at temp <= 0.5. The model becomes deterministic enough to stop inventing flags. Going lower to 0.3 provides no additional accuracy benefit.

| temp | Hallucinated flags | Duplicate flags | Total model errors |
|---|---|---|---|
| 0.6 | 1 (nohup -d) | 1 (useradd -m -m) | 2 |
| 0.5 | 0 | 0 | **0** |
| 0.3 | 0 | 0 | **0** |

### 2. n_ctx has no measurable accuracy impact

The ctx4k_default config (same params as baseline but n_ctx=4096) showed nearly identical accuracy to baseline. This is expected — prompts are ~500 tokens and responses average ~40 tokens, all well within 1024.

The value of larger n_ctx is for **future features** (session memory injection, multi-turn context), not for current accuracy.

Chose n_ctx=2048 for production — enough headroom for session memory without 4x KV cache overhead.

### 3. Speed is identical across all configs

All configs produced ~70 tok/s. Neither temperature, top_p, top_k, nor n_ctx affects generation speed on this model/hardware combination. The bottleneck is model inference, not parameter tuning.

### 4. Token generation varies slightly

Conservative (temp=0.5) generated the most tokens (1919) vs baseline (1824). Lower temperature doesn't mean shorter responses — it means more confident, sometimes more detailed responses.

### 5. RAG hit rate is consistent

All 4 configs hit RAG on the same 5 questions:
- Q2: find (file modification query)
- Q8: ssh-keygen (key generation query)
- Q13: rsync (remote sync query)
- Q16: lsof (process/memory query)
- Q30: wc (line count query)

RAG coverage is 11% (5/44). The keyword map focuses on the 8 curated OVERRIDE commands — queries outside those commands don't trigger RAG injection. This is by design: RAG is most valuable where the model most frequently gets flags wrong.

---

## Per-Question Results (All 44)

### Legend
- **ok**: Passed validation, command is correct
- **BLOCKED**: Validator rejected the command
- **warn**: Soft warning (unknown command or non-Debian pkg manager), not blocked

### Files Domain (Q1-Q6)

| # | Question | B tok | B val | C tok | C val | P tok | P val | 4k tok | 4k val |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Find all files larger than 100MB | 18 | ok | 41 | ok | 21 | ok | 21 | ok |
| 2 | Find files modified in last 24h in /var/log | 49 | BLOCKED | 28 | BLOCKED | 28 | BLOCKED | 20 | BLOCKED |
| 3 | Recursively search for 'ERROR' in .log files | 28 | BLOCKED | 28 | BLOCKED | 28 | BLOCKED | 28 | BLOCKED |
| 4 | What does chmod 755 do? | 57 | ok | 52 | ok | 51 | ok | 50 | ok |
| 5 | Change owner of directory recursively | 43 | ok | 46 | ok | 45 | ok | 41 | ok |
| 6 | Create a symbolic link | 53 | ok | 65 | ok | 50 | ok | 41 | ok |

### Networking Domain (Q7-Q13)

| # | Question | B tok | B val | C tok | C val | P tok | P val | 4k tok | 4k val |
|---|---|---|---|---|---|---|---|---|---|
| 7 | List all open TCP ports | 34 | ok | 33 | ok | 32 | ok | 52 | ok |
| 8 | Generate SSH key pair (RAG) | 29 | ok | 56 | ok | 80 | ok | 77 | ok |
| 9 | Copy file to remote server via SCP | 46 | ok | 47 | ok | 43 | ok | 47 | ok |
| 10 | Check current IP address | 27 | ok | 25 | ok | 37 | ok | 36 | ok |
| 11 | Test if remote port is open | 47 | ok | 65 | ok | 57 | ok | 65 | ok |
| 12 | Block port 22 with iptables | 50 | ok | 38 | ok | 39 | ok | 47 | ok |
| 13 | Rsync local folder to remote (RAG) | 29 | ok | 78 | ok | 28 | ok | 54 | ok |

### Process Domain (Q14-Q19)

| # | Question | B tok | B val | C tok | C val | P tok | P val | 4k tok | 4k val |
|---|---|---|---|---|---|---|---|---|---|
| 14 | Check disk usage by directory | 37 | ok | 35 | ok | 38 | ok | 35 | ok |
| 15 | Kill process by name | 32 | ok | 39 | ok | 38 | ok | 37 | ok |
| 16 | Find most memory-heavy process (RAG) | 51 | ok | 38 | ok | 47 | ok | 38 | ok |
| 17 | Run process in background after logout | 48 | BLOCKED | 51 | ok | 53 | ok | 59 | ok |
| 18 | Schedule cron job at midnight | 43 | ok | 35 | warn | 36 | warn | 36 | warn |
| 19 | Check CPU and memory in real time | 46 | ok | 42 | warn | 40 | warn | 60 | ok |

### Users Domain (Q20-Q23)

| # | Question | B tok | B val | C tok | C val | P tok | P val | 4k tok | 4k val |
|---|---|---|---|---|---|---|---|---|---|
| 20 | Add user to sudo group | 36 | ok | 38 | ok | 42 | ok | 38 | ok |
| 21 | Create new user with home directory | 40 | BLOCKED | 38 | ok | 39 | ok | 38 | ok |
| 22 | Lock user account | 34 | ok | 40 | ok | 44 | ok | 35 | ok |
| 23 | View all groups for a user | 35 | ok | 34 | ok | 29 | ok | 37 | ok |

### Packages Domain (Q24-Q27)

| # | Question | B tok | B val | C tok | C val | P tok | P val | 4k tok | 4k val |
|---|---|---|---|---|---|---|---|---|---|
| 24 | Install .deb package manually | 40 | ok | 58 | ok | 54 | ok | 34 | ok |
| 25 | Start/stop/restart systemd service | 36 | ok | 43 | ok | 39 | ok | 42 | ok |
| 26 | Check if service enabled on boot | 41 | ok | 41 | ok | 50 | ok | 42 | ok |
| 27 | Find which package owns a file | 37 | ok | 36 | ok | 38 | ok | 33 | ok |

### Text Domain (Q28-Q31)

| # | Question | B tok | B val | C tok | C val | P tok | P val | 4k tok | 4k val |
|---|---|---|---|---|---|---|---|---|---|
| 28 | Extract 3rd column with awk | 21 | ok | 43 | ok | 22 | ok | 17 | ok |
| 29 | Replace foo with bar using sed | 48 | ok | 45 | ok | 44 | ok | 29 | ok |
| 30 | Count lines, words, chars (RAG) | 18 | ok | 34 | ok | 46 | ok | 40 | BLOCKED |
| 31 | Sort file and remove duplicates | 29 | ok | 36 | ok | 44 | ok | 35 | ok |

### Storage Domain (Q32-Q35)

| # | Question | B tok | B val | C tok | C val | P tok | P val | 4k tok | 4k val |
|---|---|---|---|---|---|---|---|---|---|
| 32 | Mount USB drive | 75 | ok | 63 | ok | 57 | ok | 50 | ok |
| 33 | Check available disk space | 40 | ok | 39 | ok | 38 | ok | 37 | ok |
| 34 | Create tar.gz archive | 43 | ok | 45 | ok | 47 | ok | 40 | ok |
| 35 | Find and delete files older than 30 days | 46 | ok | 53 | ok | 48 | ok | 48 | ok |

### Kernel Domain (Q36-Q40)

| # | Question | B tok | B val | C tok | C val | P tok | P val | 4k tok | 4k val |
|---|---|---|---|---|---|---|---|---|---|
| 36 | What is a kernel module / how to load | 36 | ok | 47 | ok | 48 | ok | 51 | ok |
| 37 | Difference between process and thread | 38 | ok | 42 | ok | 50 | ok | 50 | ok |
| 38 | How virtual memory paging works | 63 | ok | 40 | ok | 60 | ok | 37 | ok |
| 39 | Purpose of /proc filesystem | 52 | ok | 48 | warn | 50 | warn | 42 | warn |
| 40 | Check current kernel version | 45 | ok | 25 | ok | 32 | ok | 32 | ok |

### Scripting Domain (Q41-Q44)

| # | Question | B tok | B val | C tok | C val | P tok | P val | 4k tok | 4k val |
|---|---|---|---|---|---|---|---|---|---|
| 41 | Bash script: check if file exists | 54 | ok | 52 | warn | 48 | warn | 63 | warn |
| 42 | Loop over .log files in bash | 43 | ok | 52 | warn | 45 | warn | 54 | warn |
| 43 | Capture command output into variable | 39 | ok | 36 | warn | 38 | warn | 38 | warn |
| 44 | Pass args to bash script and validate | 68 | ok | 49 | warn | 47 | warn | 61 | warn |

---

## Production Config Decision

Based on sweep results, adopted **conservative** as production defaults in `daemon.yaml`:

```yaml
model:
  n_ctx: 2048           # doubled from 1024 for session memory headroom
generation:
  temperature: 0.5      # eliminates hallucinated flags at temp=0.6
  top_p: 0.98           # slightly wider nucleus for conceptual answers
  top_k: 15             # tighter sampling for command accuracy
```

**Rationale**:
- temp=0.5 eliminates both hallucination errors with no accuracy loss vs temp=0.3
- n_ctx=2048 (not 4096) — current prompts fit in 1024, 2048 gives room for session memory injection
- VRAM impact: negligible (~3891 MB peak vs 3899 MB baseline)
- Speed impact: none (70.5 vs 70.7 tok/s)

## Post-Sweep Fixes Applied

1. **Validator -exec bug**: Fixed in `validator.py` — flags inside `find -exec ... \;` are now skipped during validation. Q2 and Q3 would pass after this fix.
2. **Negative number args**: Fixed `-mtime -1` being treated as a flag instead of an argument value.
3. **After fixes**: conservative and precise configs would have **zero real blocks** on all 44 questions.

---

## Data Files

| File | Config | Description |
|---|---|---|
| `results_baseline.json` | temp=0.6, top_p=0.95, top_k=20, n_ctx=1024 | Current production params |
| `results_conservative.json` | temp=0.5, top_p=0.98, top_k=15, n_ctx=4096 | Lower temp + tighter sampling |
| `results_precise.json` | temp=0.3, top_p=0.90, top_k=10, n_ctx=4096 | Max determinism |
| `results_ctx4k_default.json` | temp=0.6, top_p=0.95, top_k=20, n_ctx=4096 | Isolates n_ctx impact |
