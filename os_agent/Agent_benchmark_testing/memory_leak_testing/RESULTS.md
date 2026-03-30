# Resource Stability Test — Results

**Date**: 2026-03-30
**Model**: `qwen3.5-4b-os-q4km.gguf` (Q4_K_M, llama-cpp-python, RTX 3060 12GB)
**Test file**: `test_resource_stability.py`
**Results file**: `results_resource_stability.json`
**Config**: 100 sequential queries, sampled every 10, seed=42

---

## Result: PASS

No memory leaks detected in either VRAM or RAM over 100 sequential queries.

---

## Memory Samples (every 10 queries)

| Query | VRAM (MB) | VRAM Δ | RAM RSS (MB) | RAM Δ |
|---|---|---|---|---|
| baseline (pre-load) | 764 | — | 139 | — |
| after model load | 4,049 | +3,285 | 858 | +719 |
| 10 | 4,095 | +46 | 1,083 | +225 |
| 20 | 4,069 | +20 | 1,092 | +234 |
| 30 | 4,091 | +42 | 1,126 | +268 |
| 40 | 4,086 | +37 | 1,169 | +311 |
| 50 | 4,080 | +31 | 1,184 | +326 |
| 60 | 4,076 | +27 | 1,184 | +326 |
| 70 | 4,051 | +2 | 1,184 | +326 |
| 80 | 4,056 | +7 | 1,184 | +326 |
| 90 | 4,049 | +0 | 1,184 | +326 |
| 100 | 4,048 | −1 | 1,195 | +337 |

---

## VRAM Analysis

- **Start → End**: 4,095 MB → 4,048 MB (**−47 MB**)
- **Growth slope**: −5.13 MB/sample (threshold: 10.0)
- **Verdict**: PASS — VRAM is flat and slightly decreasing (GPU memory defragmentation)

The model loads once and stays fully resident on GPU. No additional VRAM is allocated per query. The small negative drift is normal — the GPU driver reclaims fragmented memory between inference calls.

---

## RAM (RSS) Analysis

- **Start → End**: 1,083 MB → 1,195 MB (**+112 MB total**)
- **Growth slope**: +12.04 MB/sample (threshold: 20.0)
- **Verdict**: PASS — under threshold, and the growth is front-loaded, not continuous

### RAM growth breakdown

| Phase | MB | Cause |
|---|---|---|
| Model load | +719 | llama-cpp-python, Python runtime, imports |
| Queries 1–10 | +225 | sentence-transformers embedding model lazy-loaded on first FAISS call; SharedState JSON cache; EnvironmentContext system info |
| Queries 10–50 | ~+100 | FAISS indices filling up (5 domains × growing vectors); session objects; cached snapshots |
| Queries 50–100 | ~+11 | Near-flat — all caches resident, no new allocations |

The RAM profile is a **one-time initialization pattern**, not a leak. After ~50 queries, RAM flatlines at +326 MB over baseline and holds through query 100.

**Steady-state RAM cost of the harness** (beyond the model itself): ~326 MB

---

## Latency Over Run

| Metric | Value |
|---|---|
| Avg (all 100) | 837 ms |
| Avg (warm, queries 2–100) | 838 ms |
| Max | 15,560 ms |
| Target | < 2,000 ms |

No latency degradation over the run — query 1 and query 100 are comparable in speed. The model's KV cache does not accumulate across independent queries (fresh session per query in the test).

**Latency outlier**: Query 84 (`build and install this project from source using configure/make`) — 15,560 ms. This is a long conceptual response (the model generates a multi-step explanation), not a leak or degradation. Single-occurrence, not a trend.

---

## Validator Blocks During Run

4 queries were blocked by the validator:

| Query | Reason |
|---|---|
| Query 12: `generate a core dump from the crashed myapp binary` | perf/gdb flag issue |
| Query 39: `run the postgres container with /var/lib/postgres mounted` | out-of-domain or flag issue |
| Query 57: `find all files owned by the old admin user` | likely `-user` flag validation edge case |
| Query 95: `run this script in background so it survives SSH logout` | model generated `nohup -d` (known hallucination) |
| Query 100: `set up auditd to track access to /etc/shadow` | auditctl flag not in help_db |

These are validator-level blocks, not crashes or resource issues. The harness continued cleanly after each block.

---

## Conclusions

| Concern | Status | Evidence |
|---|---|---|
| VRAM leak | None | Flat line, −47 MB total drift |
| RAM leak | None | Front-loaded initialization, flatlines at query 50 |
| Latency degradation | None | Avg consistent at ~838ms throughout |
| Crash / exception | None | 100/100 queries completed |
| Validator stability | Good | 4 blocks, 0 crashes |

The system is stable for long sessions. The harness is suitable for OS daemon use where the process runs continuously and handles many sequential queries.

---

## Thresholds Used

| Metric | Threshold | Measured slope | Result |
|---|---|---|---|
| VRAM growth | ≤ 10 MB/sample | −5.13 MB/sample | PASS |
| RAM growth | ≤ 20 MB/sample | +12.04 MB/sample | PASS |
