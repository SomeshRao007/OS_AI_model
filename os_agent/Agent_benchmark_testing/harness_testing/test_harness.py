"""
End-to-end harness evaluation for the OS AI agent.

Tests the FULL agent pipeline: routing → specialist prompt → FAISS memory →
RAG injection → model inference → validator → command extraction → risk
classification. No command execution — stops at risk classification.

Usage:
    source os_agent_env/bin/activate

    # Default: all questions
    python os_agent/Agent_benchmark_testing/test_harness.py

    # Original 44 only
    python os_agent/Agent_benchmark_testing/test_harness.py --benchmark

    # Quick smoke test (1 per domain)
    python os_agent/Agent_benchmark_testing/test_harness.py --quick

    # Filter + save
    python os_agent/Agent_benchmark_testing/test_harness.py --difficulty developer --output dev_results.json
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Project root: 3 levels up from os_agent/Agent_benchmark_testing/test_harness.py
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# eval_questions.py lives in the same directory
_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from eval_questions import (
    ALL_QUESTIONS,
    EvalQuestion,
    _ORIGINAL_44,
    filter_questions,
)

from os_agent.agents.master import MasterAgent
from os_agent.inference.engine import InferenceEngine
from os_agent.inference.rag import build_rag_context
from os_agent.inference.validator import validate
from os_agent.memory.session import SessionContext
from os_agent.shell.context import EnvironmentContext
from os_agent.tools.executor import SandboxedExecutor
from os_agent.tools.parser import extract_command


# ── VRAM tracking ─────────────────────────────────────────────────────────

def get_vram_mb() -> dict:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {"used": 0, "total": 0, "free": 0}
    parts = result.stdout.strip().split(", ")
    return {"used": int(parts[0]), "total": int(parts[1]), "free": int(parts[2])}


def print_vram(label: str) -> dict:
    v = get_vram_mb()
    print(f"  VRAM [{label}]: {v['used']} MB used / {v['total']} MB total ({v['free']} MB free)")
    return v


# ── Single question runner ───────────────────────────────────────────────

def run_single_question(
    eq: EvalQuestion,
    master: MasterAgent,
    engine: InferenceEngine,
    executor: SandboxedExecutor,
    env_ctx: EnvironmentContext,
) -> dict:
    """Run one question through the full harness. Returns result dict."""

    # Fresh session per question — no context bleed
    session = SessionContext()

    # 1. Classify (routing)
    keyword_result = master._classify_by_keywords(eq.q)
    actual_domain = keyword_result if keyword_result else master._classify_by_model(eq.q)
    routing_method = "keyword" if keyword_result else "model"

    # 2. Get specialist agent
    agent = master.get_agent(actual_domain)

    # 3. Check FAISS memory hits (will be 0 with fresh indices)
    memory_hits = 0
    if agent._memory:
        hits = agent._memory.search(eq.q, top_k=3)
        memory_hits = len(hits)

    # 4. Build augmented prompt (memory + env + session)
    system_prompt = agent.augmented_prompt_with_context(
        eq.q, env_ctx.full_context(), session.get_context_string(),
    )

    # 5. Track RAG context separately
    rag_ctx = build_rag_context(eq.q)

    # 6. Timed inference with validation
    start = time.perf_counter()
    result = engine.infer_validated(system_prompt, eq.q)
    elapsed_ms = (time.perf_counter() - start) * 1000

    tokens = engine.last_completion_tokens
    tok_per_sec = tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

    response = result["response"]
    command = result["command"]

    # 7. Get full validator result (including warns that infer_validated doesn't surface)
    validator_result = {"ok": True}
    if command:
        validator_result = validate(command)

    # 8. Risk classification (no execution)
    risk_level = None
    in_domain = None
    if command:
        risk_level = executor.classify_risk(command)
        in_domain = executor.check_domain_allowed(command, actual_domain)

    return {
        "question": eq.q,
        "eval_domain": eq.eval_domain,
        "expected_route": eq.route_domain,
        "actual_route": actual_domain,
        "routing_correct": actual_domain == eq.route_domain,
        "routing_method": routing_method,
        "memory_hits": memory_hits,
        "rag_context": rag_ctx,
        "response": response,
        "command": command,
        "validator": validator_result,
        "risk_level": risk_level,
        "in_domain": in_domain,
        "elapsed_ms": round(elapsed_ms, 1),
        "tokens": tokens,
        "tok_per_sec": round(tok_per_sec, 1),
        "difficulty": eq.difficulty,
        "test_type": eq.test_type,
    }


# ── Run all questions ────────────────────────────────────────────────────

def run_all_questions(
    master: MasterAgent,
    engine: InferenceEngine,
    executor: SandboxedExecutor,
    questions: list[EvalQuestion],
    env_ctx: EnvironmentContext,
) -> list[dict]:
    """Run all questions sequentially. Returns list of result dicts."""
    results = []
    vram_peak = 0
    total = len(questions)

    print(f"\n{'=' * 70}")
    print(f"E2E HARNESS EVALUATION — {total} questions")
    print(f"{'=' * 70}")

    for i, eq in enumerate(questions, 1):
        print(f"\n[{i}/{total}] [{eq.route_domain.upper():>8}] {eq.q}")
        print("-" * 50)

        result = run_single_question(eq, master, engine, executor, env_ctx)

        # Status line
        route_status = "OK" if result["routing_correct"] else f"MISS (got {result['actual_route']})"
        cmd_status = result["command"][:60] if result["command"] else "(no command)"
        risk_str = result["risk_level"] or "-"

        print(f"  Route: {route_status} ({result['routing_method']})")
        print(f"  Cmd:   {cmd_status}")
        if result["validator"] and not result["validator"].get("ok", True):
            print(f"  [BLOCKED] {result['validator'].get('error', '')}")
        elif result["validator"] and result["validator"].get("warn"):
            print(f"  [WARN] {result['validator']['warn']}")
        print(f"  Risk: {risk_str} | {result['tokens']} tok | {result['elapsed_ms']:.0f}ms | {result['tok_per_sec']:.1f} tok/s")

        vram = get_vram_mb()
        vram_peak = max(vram_peak, vram["used"])

        results.append(result)

    return results


# ── Summary report ───────────────────────────────────────────────────────

def print_summary(results: list[dict], vram_peak: int = 0):
    """Print formatted summary of evaluation results."""
    total = len(results)
    if total == 0:
        print("No results to summarize.")
        return

    # Routing
    routing_correct = sum(1 for r in results if r["routing_correct"])
    keyword_count = sum(1 for r in results if r["routing_method"] == "keyword")
    model_count = sum(1 for r in results if r["routing_method"] == "model")
    misrouted = total - routing_correct

    # Commands
    with_command = [r for r in results if r["command"]]
    no_command = total - len(with_command)

    # Validator (only for questions with commands)
    blocked = sum(1 for r in with_command if not r["validator"].get("ok", True))
    warned = sum(1 for r in with_command if r["validator"].get("ok", True) and r["validator"].get("warn"))
    clean = len(with_command) - blocked - warned

    # RAG
    rag_hits = sum(1 for r in results if r["rag_context"])

    # Risk (only for questions with commands)
    risk_safe = sum(1 for r in with_command if r["risk_level"] == "safe")
    risk_mod = sum(1 for r in with_command if r["risk_level"] == "moderate")
    risk_dang = sum(1 for r in with_command if r["risk_level"] == "dangerous")

    # Performance
    latencies = [r["elapsed_ms"] for r in results]
    avg_latency = sum(latencies) / len(latencies)
    tok_speeds = [r["tok_per_sec"] for r in results if r["tok_per_sec"] > 0]
    warm_speeds = tok_speeds[1:] if len(tok_speeds) > 1 else tok_speeds
    avg_tok_s = sum(warm_speeds) / len(warm_speeds) if warm_speeds else 0

    pct = lambda n, d: f"{100 * n / d:.1f}%" if d > 0 else "N/A"

    print(f"\n\n{'═' * 55}")
    print("E2E HARNESS EVALUATION RESULTS")
    print(f"{'═' * 55}")

    print(f"\nRouting:          {routing_correct}/{total} ({pct(routing_correct, total)})")
    print(f"  Keyword:        {keyword_count} ({pct(keyword_count, total)})")
    print(f"  Model fallback: {model_count} ({pct(model_count, total)})")
    print(f"  Misrouted:      {misrouted}")

    print(f"\nCommands:         {len(with_command)}/{total} extracted")
    print(f"  No code block:  {no_command} (conceptual/format)")

    print(f"\nValidator:        {clean}/{len(with_command)} clean")
    print(f"  Blocked:        {blocked}")
    print(f"  Warned:         {warned}")

    print(f"\nRAG hits:         {rag_hits}/{total} ({pct(rag_hits, total)})")

    print(f"\nRisk:")
    print(f"  Safe: {risk_safe} | Moderate: {risk_mod} | Dangerous: {risk_dang}")

    print(f"\nPerformance:")
    print(f"  Avg latency:    {avg_latency:.0f}ms")
    print(f"  Avg tok/s:      {avg_tok_s:.1f}")
    if vram_peak > 0:
        print(f"  VRAM peak:      {vram_peak / 1024:.1f} GB")

    # Per-domain breakdown
    domains = sorted(set(r["expected_route"] for r in results))
    print(f"\nPer-domain breakdown:")
    print(f"  {'Domain':<10} {'Routing':>12} {'Commands':>10} {'Avg ms':>8}")
    print(f"  {'-' * 42}")
    for d in domains:
        d_results = [r for r in results if r["expected_route"] == d]
        d_correct = sum(1 for r in d_results if r["routing_correct"])
        d_cmds = sum(1 for r in d_results if r["command"])
        d_latency = sum(r["elapsed_ms"] for r in d_results) / len(d_results)
        print(f"  {d:<10} {d_correct:>3}/{len(d_results):<3} ({pct(d_correct, len(d_results)):>6}) {d_cmds:>6} cmds {d_latency:>8.0f}")

    # Misrouted questions detail
    misrouted_qs = [r for r in results if not r["routing_correct"]]
    if misrouted_qs:
        print(f"\nMisrouted questions:")
        for r in misrouted_qs:
            print(f"  [{r['expected_route']}→{r['actual_route']}] {r['question'][:70]}")

    print(f"{'═' * 55}")


# ── CLI ───────────────────────────────────────────────────────────────────

def _build_question_list(args) -> list[EvalQuestion]:
    """Build question list from CLI args."""
    if args.quick:
        # One question per domain
        seen = set()
        quick_list = []
        for q in ALL_QUESTIONS:
            if q.route_domain not in seen:
                seen.add(q.route_domain)
                quick_list.append(q)
        return quick_list

    if args.benchmark:
        return list(_ORIGINAL_44)

    if args.difficulty or args.domain or args.test_type:
        return filter_questions(
            difficulty=args.difficulty,
            eval_domain=args.domain if not args.route_domain else None,
            test_type=args.test_type,
        )

    # Default: all questions
    return list(ALL_QUESTIONS)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end harness evaluation for the OS AI agent"
    )
    # Question selection
    parser.add_argument("--benchmark", action="store_true",
                        help="Original 44 questions only")
    parser.add_argument("--quick", action="store_true",
                        help="5 questions (one per domain)")
    parser.add_argument("--difficulty", default=None,
                        choices=["basic", "intermediate", "advanced", "developer"])
    parser.add_argument("--domain", default=None,
                        help="Filter by eval_domain")
    parser.add_argument("--route-domain", default=None,
                        choices=["files", "network", "process", "packages", "kernel"],
                        help="Filter by route_domain")
    parser.add_argument("--test-type", default=None,
                        choices=["command", "conceptual", "routing", "format", "adversarial"])

    # Output
    parser.add_argument("--output", default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    # Handle route-domain filter manually since filter_questions uses eval_domain
    questions = _build_question_list(args)
    if args.route_domain:
        questions = [q for q in questions if q.route_domain == args.route_domain]

    if not questions:
        print("No questions match the given filters.")
        sys.exit(1)

    print(f"Loading model and harness...")
    vram_before = print_vram("before load")

    # Load inference engine (from daemon.yaml)
    engine = InferenceEngine()

    # Fresh FAISS indices in temp dir for reproducibility
    tmpdir = tempfile.mkdtemp(prefix="neurosh_eval_")
    config = {
        "memory": {
            "state_dir": tmpdir,
            "faiss_dims": 384,
            "max_vectors_per_domain": 500,
        },
    }
    master = MasterAgent(engine, config=config)

    executor = SandboxedExecutor()
    env_ctx = EnvironmentContext()

    vram_after = print_vram("after load")
    model_vram = vram_after["used"] - vram_before["used"]
    print(f"  Model VRAM: ~{model_vram} MB")
    print(f"  Questions: {len(questions)}")
    print(f"  FAISS state: {tmpdir} (fresh/empty)")

    # Run
    results = run_all_questions(master, engine, executor, questions, env_ctx)

    # Track VRAM peak from results
    vram_final = get_vram_mb()
    vram_peak = max(vram_final["used"], vram_after["used"])

    # Summary
    print_summary(results, vram_peak)

    # Save results
    if args.output:
        output_data = {
            "total_questions": len(results),
            "model_vram_mb": model_vram,
            "vram_peak_mb": vram_peak,
            "results": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
