"""
Context Rot Test: does agent quality degrade over a long single session?

Unlike test_harness.py which uses a fresh SessionContext per question, this
test shares ONE SessionContext across all queries — simulating a real user
session. Measures whether session context accumulation degrades response quality.

What we're testing:
  1. Does answer quality drop as session grows (queries 1-10 vs 11-25 vs 26-50)?
  2. Does session context injection cause prompt bloat that crowds out reasoning?
  3. At what query count does quality start to degrade?
  4. What is the safe max session length?

Scoring:
  - command present:   +1  (model gave an actionable answer)
  - validator clean:   +1  (command passed all rules)
  - not blocked:       +1  (harness didn't catch a bad command)
  Max score per query: 3

Usage:
    source os_agent_env/bin/activate
    python os_agent/Agent_benchmark_testing/harness_testing/test_context_rot.py
    python os_agent/Agent_benchmark_testing/harness_testing/test_context_rot.py --queries 50
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_BENCH_DIR = str(Path(__file__).resolve().parent.parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from eval_questions import ALL_QUESTIONS

from os_agent.agents.master import MasterAgent
from os_agent.inference.engine import InferenceEngine
from os_agent.inference.rag import build_rag_context
from os_agent.inference.validator import validate
from os_agent.memory.session import SessionContext
from os_agent.shell.context import EnvironmentContext
from os_agent.tools.executor import SandboxedExecutor


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_result(result: dict) -> int:
    """0–3 score: command present + validator clean + not blocked."""
    s = 0
    if result.get("command"):
        s += 1
    v = result.get("validator", {})
    if v.get("ok", True) and not v.get("warn"):
        s += 1
    if not result.get("blocked"):
        s += 1
    return s


# ── Single query (shared session) ─────────────────────────────────────────────

def run_query(q_text: str, master: MasterAgent, engine: InferenceEngine,
              executor: SandboxedExecutor, env_ctx: EnvironmentContext,
              session: SessionContext) -> dict:
    """Run one query using the shared session — session state accumulates."""

    keyword_result = master._classify_by_keywords(q_text)
    domain = keyword_result if keyword_result else master._classify_by_model(q_text)
    agent = master.get_agent(domain)

    # Inject shared (growing) session context
    ctx_string = session.get_context_string(n=5)
    system_prompt = agent.augmented_prompt_with_context(
        q_text, env_ctx.full_context(), ctx_string,
    )
    ctx_tokens = len(ctx_string.split())

    start = time.perf_counter()
    result = engine.infer_validated(system_prompt, q_text)
    elapsed_ms = (time.perf_counter() - start) * 1000

    command = result["command"]
    blocked = result.get("blocked", False)

    validator_result = {"ok": True}
    if command and not blocked:
        validator_result = validate(command)

    # Record turn into shared session so next query sees this one
    response_text = result.get("response", "")
    session.add_turn(q_text, domain, response_text[:120])

    return {
        "query": q_text,
        "domain": domain,
        "command": command,
        "blocked": blocked,
        "validator": validator_result,
        "elapsed_ms": round(elapsed_ms, 1),
        "tokens": engine.last_completion_tokens,
        "session_turns": len(session._turns),
        "ctx_tokens_injected": ctx_tokens,
    }


# ── Window analysis ───────────────────────────────────────────────────────────

def window_stats(results: list[dict], lo: int, hi: int) -> dict:
    """Stats for queries lo..hi (1-indexed, inclusive)."""
    window = results[lo - 1: hi]
    if not window:
        return {}
    scores = [score_result(r) for r in window]
    avg_score = sum(scores) / len(scores)
    avg_latency = sum(r["elapsed_ms"] for r in window) / len(window)
    avg_ctx = sum(r["ctx_tokens_injected"] for r in window) / len(window)
    blocked = sum(1 for r in window if r["blocked"])
    no_cmd = sum(1 for r in window if not r["command"])
    return {
        "range": f"{lo}–{hi}",
        "count": len(window),
        "avg_score": round(avg_score, 2),
        "avg_latency_ms": round(avg_latency, 0),
        "avg_ctx_tokens": round(avg_ctx, 1),
        "blocked": blocked,
        "no_command": no_cmd,
        "scores": scores,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Context Rot test")
    parser.add_argument("--queries", type=int, default=50,
                        help="Number of queries in the shared session (default: 50)")
    parser.add_argument("--output", default=None,
                        help="Save full results to JSON")
    args = parser.parse_args()

    # Fixed question order: first N from ALL_QUESTIONS (diverse domains)
    questions = [q.q for q in list(ALL_QUESTIONS)[: args.queries]]
    total = len(questions)

    # Define analysis windows
    w1_end = min(10, total)
    w2_end = min(25, total)
    w3_end = total

    print(f"\n{'=' * 65}")
    print(f"CONTEXT ROT TEST  ({total} queries, single shared session)")
    print(f"{'=' * 65}")
    print(f"Windows:  1–{w1_end} (early) | {w1_end+1}–{w2_end} (mid) | {w2_end+1}–{w3_end} (late)")
    print(f"\nLoading model...\n")

    engine = InferenceEngine()
    tmpdir = tempfile.mkdtemp(prefix="neurosh_rot_")
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

    # ONE shared session for the entire run
    session = SessionContext(max_turns=20)

    results = []

    for i, q_text in enumerate(questions, 1):
        r = run_query(q_text, master, engine, executor, env_ctx, session)
        results.append(r)

        s = score_result(r)
        status = "BLOCKED" if r["blocked"] else ("no-cmd" if not r["command"] else "ok")
        print(
            f"  [{i:>2}/{total}] [{r['domain']:<8}] "
            f"turns={r['session_turns']:>2} ctx≈{r['ctx_tokens_injected']:>3}tok "
            f"{r['elapsed_ms']:>6.0f}ms  score={s}/3  {status}"
            f"\n         {q_text[:70]}"
        )

    # ── Analysis ─────────────────────────────────────────────────────────────
    w_early = window_stats(results, 1, w1_end)
    w_mid   = window_stats(results, w1_end + 1, w2_end) if total > w1_end else {}
    w_late  = window_stats(results, w2_end + 1, w3_end) if total > w2_end else {}

    def rot_verdict(early_score: float, late_score: float) -> str:
        delta = late_score - early_score
        if delta < -0.3:
            return f"ROT DETECTED  (Δ{delta:+.2f})"
        elif delta < -0.1:
            return f"Mild degradation  (Δ{delta:+.2f})"
        else:
            return f"Stable  (Δ{delta:+.2f})"

    print(f"\n\n{'═' * 65}")
    print("CONTEXT ROT ANALYSIS")
    print(f"{'═' * 65}")

    for w in [w_early, w_mid, w_late]:
        if not w:
            continue
        print(f"\n  Window {w['range']} ({w['count']} queries):")
        print(f"    Avg score:    {w['avg_score']:.2f} / 3.0")
        print(f"    Avg latency:  {w['avg_latency_ms']:.0f} ms")
        print(f"    Avg ctx injected: {w['avg_ctx_tokens']:.0f} tokens")
        print(f"    Blocked:      {w['blocked']}")
        print(f"    No command:   {w['no_command']}")

    if w_early and w_late:
        verdict = rot_verdict(w_early["avg_score"], w_late["avg_score"])
        print(f"\n  Early avg score: {w_early['avg_score']:.2f}")
        print(f"  Late  avg score: {w_late['avg_score']:.2f}")
        print(f"\n  Verdict: {verdict}")

    # Latency trend
    all_latencies = [r["elapsed_ms"] for r in results]
    if len(all_latencies) >= 10:
        first10_avg = sum(all_latencies[:10]) / 10
        last10_avg  = sum(all_latencies[-10:]) / 10
        lat_delta   = last10_avg - first10_avg
        print(f"\n  Latency trend:  first-10 avg={first10_avg:.0f}ms  last-10 avg={last10_avg:.0f}ms  (Δ{lat_delta:+.0f}ms)")
        if lat_delta > 200:
            print("  WARNING: Significant latency increase — prompt bloat likely")
        else:
            print("  Latency stable — no prompt bloat detected")

    # Context token trend
    ctx_values = [r["ctx_tokens_injected"] for r in results]
    if ctx_values:
        print(f"\n  Session context tokens: first={ctx_values[0]}  last={ctx_values[-1]}  max={max(ctx_values)}")

    print(f"\n{'═' * 65}\n")

    if args.output:
        output_data = {
            "total_queries": total,
            "windows": {
                "early": w_early,
                "mid": w_mid,
                "late": w_late,
            },
            "results": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
