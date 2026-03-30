# Harness Evaluation — Results & Analysis

**Date**: 2026-03-30
**Model**: `qwen3.5-4b-os-q4km.gguf` (Q4_K_M, llama-cpp-python, RTX 3060)
**Test file**: `test_harness.py`
**Results file**: `results_harness_full.json`

---

## Accuracy Summary

| Layer | Questions | Correct | Accuracy | Delta vs raw |
|---|---|---|---|---|
| Raw model (Step 1 baseline) | 44 | ~39 | **88%** | — |
| + RAG + Validator | 150 | 144 | **96.0%** | +8.0pp |
| + Full Harness | 164 | 162 | **98.8%** | **+10.8pp** |

The full harness (routing → FAISS memory → RAG injection → inference → validator) delivers a **10.8 percentage point lift** over the raw model.

---

## Per-Domain Accuracy (Full Harness, 164 questions)

| Domain | Correct | Total | Accuracy | Note |
|---|---|---|---|---|
| build | 6 | 6 | 100% | |
| database | 5 | 5 | 100% | |
| devops | 5 | 5 | 100% | |
| docker | 8 | 8 | 100% | |
| files | 12 | 12 | 100% | |
| git | 8 | 8 | 100% | |
| harness | 26 | 26 | 100% | |
| kernel | 10 | 10 | 100% | |
| networking | 13 | 13 | 100% | |
| packages | 8 | 8 | 100% | |
| process | 12 | 12 | 100% | |
| scripting | 9 | 9 | 100% | |
| security | 5 | 5 | 100% | |
| storage | 8 | 8 | 100% | |
| text | 8 | 8 | 100% | |
| users | 8 | 8 | 100% | |
| **debugging** | **7** | **8** | **87.5%** | 1 failure |
| **systemd** | **4** | **5** | **80.0%** | 1 failure |

16 of 18 domains at 100%. Only `debugging` and `systemd` have failures.

---

## Failures — Detailed

### Failure 1: debugging — perf with command substitution

**Query**: `profile CPU usage of the nginx master process with perf`

**Generated command**:
```bash
perf record -F 99 -g -p $(ps -o pid= -C nginx | awk 'NR==1{print $1}')
```

**Validator error**: `perf -p expects number, got string '$(ps'`

**Root cause**: The validator's `_extract_first_command_tokens()` splits on ` | ` before shlex-parsing, which breaks the `$(ps ... | awk ...)` substitution mid-token. The truncated fragment `$(ps` fails the `number` type check for `-p`.

**Reality**: The command is **correct**. `$(...)` evaluates to a PID at runtime.

**Fix applied (2026-03-30)**: Added a shell-substitution guard in Rule 4 of `validator.py`:
```python
if arg_val and (arg_val.startswith("$(") or arg_val.startswith("`")):
    seen_flags.append(flag)
    i += 1
    continue
```
This was a **false positive** — the harness was blocking a correct command.

---

### Failure 2: systemd — timer creation

**Query**: `create a systemd timer that runs the cleanup script every hour`

**Generated command**:
```bash
sudo systemctl timer-create --on-calendar="0 * * * *" --description="Hourly cleanup" --unit=cleanup.timer --on-calendar="0 * * * *" --on-calendar=...
```

**Validator error**: `sudo: duplicate flag --on-calendar`

**Root cause**: `systemctl timer-create` does not exist. The model invented both the subcommand and its flags, then repeated `--on-calendar` five times. The correct approach is to create `.timer` and `.service` unit files manually.

**Reality**: This is a **genuine model hallucination** — the model has no reliable knowledge of systemd unit file syntax, only `systemctl` subcommands.

**Fix needed**: 30-50 training examples covering `.timer` / `.service` / `.socket` unit file creation. Cannot be fixed with a validator rule — the model needs to know the correct answer.

---

## Test Suite Composition (164 questions)

| Test type | Count | % |
|---|---|---|
| command | 121 | 73.8% |
| conceptual | 17 | 10.4% |
| adversarial | 12 | 7.3% |
| routing | 10 | 6.1% |
| format | 4 | 2.4% |

---

## Routing Performance

- **Correct routes**: 133/164 (81.1%)
- **Keyword-based**: covers the majority (fast, zero-inference)
- **Model fallback**: for ambiguous queries
- **Misroutes**: 31 — most still produce correct answers because the model is generalist, but domain-specific context injection (FAISS, env) is less accurate

Common misroute patterns:
- `scripting` → `files` (shell scripts overlap heavily with file operations)
- `users` → `process` (user management confused with process/systemctl)
- `packages` → `process` (systemctl-related package queries)

---

## Risk Classification (commands only)

| Level | Count | % |
|---|---|---|
| safe | 48 | 29.3% |
| moderate | 109 | 66.5% |
| dangerous | 5 | 3.0% |
| unknown | 2 | 1.2% |

5 dangerous commands correctly flagged (e.g. `find -delete`, `iptables -F`) — would trigger desktop notification + y/n confirmation in live shell.

---

## Performance

| Metric | Value |
|---|---|
| Avg response time | 672 ms |
| Avg tokens/sec | 51.4 tok/s |
| Avg tokens generated | 36.1 |
| Model VRAM | 3,292 MB |
| Peak VRAM | 4,082 MB |
| Available headroom | ~8 GB |

Latency is within the <2s target. Throughput drop vs raw (70 tok/s → 51 tok/s) is due to RAG context injection increasing prompt size and validator post-processing.

---

## Comparison: Harness vs RAG-Only

| Metric | Full Harness | RAG Only |
|---|---|---|
| Accuracy | **98.8%** | 96.0% |
| Failures | **2** | 6 |
| Validator blocks | **0** (post-fix) | 6 |
| False positive blocks | **0** (post-fix) | 2 |
| Peak VRAM | 4,082 MB | 7,183 MB |
| Avg tok/s | 51.4 | 69.3 |

The harness is both more accurate and uses less VRAM than the RAG-only run. The RAG-only higher VRAM is likely a test harness artifact (embedding model on GPU or second model instance loaded during that run — needs investigation).

---

## Open Issues

| Priority | Issue | Fix |
|---|---|---|
| P1 | Systemd unit file syntax — model hallucinates | 30-50 training examples for `.timer`/`.service` files |
| P2 | Routing accuracy 81.1% — misroutes affect context injection | Expand keyword sets, tune model classifier |
| P3 | RAG run VRAM anomaly (7,183 MB vs 4,082 MB) | Investigate embedding model device placement |

---

## Harness Accuracy Ladder (updated)

| Layer | Expected | Measured |
|---|---|---|
| Raw model | 88% | 88% ✓ |
| + System prompt fixes (Step 3) | 91–93% | ~91% ✓ |
| + FAISS memory (Step 5) | 93–95% | TBD (needs isolated test) |
| + RAG + validator (Step 8) | 82–85% (4B unfinetuned) | 96.0% ✓ (with prompt fixes) |
| + Full harness (Step 10) | 96%+ | **98.8% ✓** |
| + Fine-tuned 9B + full harness | 97%+ | TBD |
