"""
Resource stability test: run N sequential queries and track VRAM + RAM growth.

Detects memory leaks by measuring per-query memory delta and computing linear
regression over the full run. Fails if growth rate exceeds thresholds.

Usage:
    source os_agent_env/bin/activate
    python os_agent/Agent_benchmark_testing/test_resource_stability.py
    python os_agent/Agent_benchmark_testing/test_resource_stability.py --queries 50
    python os_agent/Agent_benchmark_testing/test_resource_stability.py --queries 100 --sample-every 5
"""

import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from eval_questions import ALL_QUESTIONS

from os_agent.agents.master import MasterAgent
from os_agent.inference.engine import InferenceEngine
from os_agent.memory.session import SessionContext
from os_agent.shell.context import EnvironmentContext
from os_agent.tools.executor import SandboxedExecutor
from os_agent.tools.parser import extract_command


# ── Memory samplers ──────────────────────────────────────────────────────────

def get_vram_mb() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return 0
    return int(result.stdout.strip())


def get_ram_mb() -> int:
    """Read process RSS from /proc/self/status (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024  # kB → MB
    except OSError:
        pass
    return 0


def sample_memory() -> dict:
    return {"vram_mb": get_vram_mb(), "ram_mb": get_ram_mb(), "ts": time.time()}


# ── Linear regression (no numpy) ─────────────────────────────────────────────

def linear_slope(ys: list[float]) -> float:
    """Slope (MB per sample) of least-squares line through ys."""
    n = len(ys)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    return num / den if den != 0 else 0.0


# ── Single query runner ───────────────────────────────────────────────────────

def run_query(q_text: str, master: MasterAgent, engine: InferenceEngine,
              executor: SandboxedExecutor, env_ctx: EnvironmentContext) -> dict:
    session = SessionContext()

    keyword_result = master._classify_by_keywords(q_text)
    domain = keyword_result if keyword_result else master._classify_by_model(q_text)
    agent = master.get_agent(domain)

    system_prompt = agent.augmented_prompt_with_context(
        q_text, env_ctx.full_context(), session.get_context_string(),
    )

    start = time.perf_counter()
    result = engine.infer_validated(system_prompt, q_text)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "domain": domain,
        "elapsed_ms": round(elapsed_ms, 1),
        "tokens": engine.last_completion_tokens,
        "blocked": result.get("blocked", False),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Resource stability test")
    parser.add_argument("--queries", type=int, default=100,
                        help="Number of sequential queries to run (default: 100)")
    parser.add_argument("--sample-every", type=int, default=10,
                        help="Sample memory every N queries (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for question selection")
    parser.add_argument("--output", default=None,
                        help="Save full results to JSON file")
    # Thresholds (MB per sample interval)
    parser.add_argument("--vram-leak-threshold", type=float, default=10.0,
                        help="Max acceptable VRAM growth slope in MB/sample (default: 10)")
    parser.add_argument("--ram-leak-threshold", type=float, default=20.0,
                        help="Max acceptable RAM growth slope in MB/sample (default: 20)")
    args = parser.parse_args()

    # Build question list — random sample with replacement from ALL_QUESTIONS
    rng = random.Random(args.seed)
    questions = [rng.choice(ALL_QUESTIONS).q for _ in range(args.queries)]

    print(f"\n{'=' * 60}")
    print(f"RESOURCE STABILITY TEST  ({args.queries} queries, sample every {args.sample_every})")
    print(f"{'=' * 60}")

    # Load harness
    print("\nLoading model and harness...")
    baseline = sample_memory()
    print(f"  Baseline  — VRAM: {baseline['vram_mb']} MB | RAM: {baseline['ram_mb']} MB")

    engine = InferenceEngine()
    tmpdir = tempfile.mkdtemp(prefix="neurosh_stability_")
    config = {
        "memory": {
            "state_dir": tmpdir,
            "faiss_dims": 384,
            "max_vectors_per_domain": 500,
        }
    }
    master = MasterAgent(engine, config=config)
    executor = SandboxedExecutor()
    env_ctx = EnvironmentContext()

    after_load = sample_memory()
    model_vram = after_load["vram_mb"] - baseline["vram_mb"]
    model_ram = after_load["ram_mb"] - baseline["ram_mb"]
    print(f"  After load — VRAM: {after_load['vram_mb']} MB (+{model_vram}) | RAM: {after_load['ram_mb']} MB (+{model_ram})")

    # Run queries, sampling memory every N
    memory_samples: list[dict] = []
    query_results: list[dict] = []
    latencies: list[float] = []

    print(f"\nRunning {args.queries} queries...\n")

    for i, q_text in enumerate(questions, 1):
        r = run_query(q_text, master, engine, executor, env_ctx)
        query_results.append(r)
        latencies.append(r["elapsed_ms"])

        status = "BLOCKED" if r["blocked"] else "ok"
        print(f"  [{i:>3}/{args.queries}] [{r['domain']:<8}] {r['elapsed_ms']:>6.0f}ms  {q_text[:55]:<55} {status}")

        if i % args.sample_every == 0 or i == args.queries:
            mem = sample_memory()
            mem["query_index"] = i
            mem["vram_delta"] = mem["vram_mb"] - after_load["vram_mb"]
            mem["ram_delta"] = mem["ram_mb"] - after_load["ram_mb"]
            memory_samples.append(mem)
            print(f"\n  ── Memory @ query {i:>3}: VRAM {mem['vram_mb']} MB (Δ{mem['vram_delta']:+d}) | RAM {mem['ram_mb']} MB (Δ{mem['ram_delta']:+d})\n")

    # ── Analysis ─────────────────────────────────────────────────────────────
    vram_series = [s["vram_mb"] for s in memory_samples]
    ram_series  = [s["ram_mb"]  for s in memory_samples]

    vram_slope = linear_slope(vram_series)
    ram_slope  = linear_slope(ram_series)

    vram_total_drift = vram_series[-1] - vram_series[0] if vram_series else 0
    ram_total_drift  = ram_series[-1]  - ram_series[0]  if ram_series  else 0

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    warm_latencies = latencies[1:] if len(latencies) > 1 else latencies
    avg_warm = sum(warm_latencies) / len(warm_latencies) if warm_latencies else 0
    max_latency = max(latencies) if latencies else 0

    vram_leak = vram_slope > args.vram_leak_threshold
    ram_leak  = ram_slope  > args.ram_leak_threshold

    print(f"\n{'═' * 60}")
    print("RESOURCE STABILITY RESULTS")
    print(f"{'═' * 60}")
    print(f"\nQueries run:       {args.queries}")
    print(f"Memory samples:    {len(memory_samples)}  (every {args.sample_every} queries)")

    print(f"\nVRAM:")
    print(f"  Start:           {vram_series[0] if vram_series else '?'} MB")
    print(f"  End:             {vram_series[-1] if vram_series else '?'} MB")
    print(f"  Total drift:     {vram_total_drift:+d} MB")
    print(f"  Growth slope:    {vram_slope:+.2f} MB/sample  (threshold: {args.vram_leak_threshold})")
    print(f"  Leak detected:   {'YES ⚠' if vram_leak else 'NO  ✓'}")

    print(f"\nRAM (RSS):")
    print(f"  Start:           {ram_series[0] if ram_series else '?'} MB")
    print(f"  End:             {ram_series[-1] if ram_series else '?'} MB")
    print(f"  Total drift:     {ram_total_drift:+d} MB")
    print(f"  Growth slope:    {ram_slope:+.2f} MB/sample  (threshold: {args.ram_leak_threshold})")
    print(f"  Leak detected:   {'YES ⚠' if ram_leak else 'NO  ✓'}")

    print(f"\nLatency:")
    print(f"  Avg (all):       {avg_latency:.0f} ms")
    print(f"  Avg (warm):      {avg_warm:.0f} ms")
    print(f"  Max:             {max_latency:.0f} ms")

    overall_pass = not vram_leak and not ram_leak
    print(f"\n{'─' * 60}")
    print(f"RESULT: {'PASS ✓' if overall_pass else 'FAIL ✗ — memory leak detected'}")
    print(f"{'═' * 60}\n")

    if args.output:
        output_data = {
            "config": {
                "queries": args.queries,
                "sample_every": args.sample_every,
                "seed": args.seed,
                "vram_leak_threshold": args.vram_leak_threshold,
                "ram_leak_threshold": args.ram_leak_threshold,
            },
            "baseline_vram_mb": baseline["vram_mb"],
            "baseline_ram_mb": baseline["ram_mb"],
            "model_load_vram_mb": model_vram,
            "model_load_ram_mb": model_ram,
            "memory_samples": memory_samples,
            "vram_slope_mb_per_sample": round(vram_slope, 3),
            "ram_slope_mb_per_sample": round(ram_slope, 3),
            "vram_total_drift_mb": vram_total_drift,
            "ram_total_drift_mb": ram_total_drift,
            "avg_latency_ms": round(avg_latency, 1),
            "avg_warm_latency_ms": round(avg_warm, 1),
            "max_latency_ms": round(max_latency, 1),
            "vram_leak": vram_leak,
            "ram_leak": ram_leak,
            "pass": overall_pass,
            "query_results": query_results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {args.output}")

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
