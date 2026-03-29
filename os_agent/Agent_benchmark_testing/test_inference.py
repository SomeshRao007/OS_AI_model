"""
Inference benchmark with optional RAG + validator and parameter sweep.

Tests model accuracy and speed across different generation parameters.
Loads the model ONCE, then runs all configs sequentially.

Usage:
    source os_agent_env/bin/activate

    # Single config (defaults: temp=0.6, top_p=0.95, top_k=20, n_ctx=1024)
    python os_agent/Agent_benchmark_testing/test_inference.py --benchmark
    python os_agent/Agent_benchmark_testing/test_inference.py --full --rag --output results.json

    # Custom params
    python os_agent/Agent_benchmark_testing/test_inference.py --benchmark --temp 0.5 --top-k 15 --n-ctx 4096

    # Parameter sweep — runs all preset configs, prints comparison table
    python os_agent/Agent_benchmark_testing/test_inference.py --sweep --benchmark
    python os_agent/Agent_benchmark_testing/test_inference.py --sweep --benchmark --rag
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

# Project root: 3 levels up from os_agent/Agent_benchmark_testing/test_inference.py
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from llama_cpp import Llama

# eval_questions.py lives in the same directory
_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from eval_questions import (
    ALL_QUESTIONS,
    LEGACY_TUPLES,
    ORIGINAL_44_TUPLES,
    filter_questions,
)

MODEL_PATH = str(
    Path(_PROJECT_ROOT)
    / "finetuning"
    / "q4_k_m-deploy"
    / "qwen3.5-4b-os-q4km.gguf"
)

SYSTEM_PROMPT = (
    "You are an AI assistant built into a Linux-based operating system. "
    "Respond with one correct command in a bash code block followed by a one-line explanation. "
    "For conceptual questions, explain in 2-4 sentences. "
    "If the request is ambiguous, ask one clarifying question. "
    "Never list alternatives. Never restate the question. Never explain individual flags."
)

STOP_TOKENS = ["<|im_end|>", "<|endoftext|>"]
N_GPU_LAYERS = -1

TEST_QUESTIONS = ORIGINAL_44_TUPLES

# ── Parameter configs for sweep mode ─────────────────────────────────────
# Each config is tested on the same questions. Model loaded once with max n_ctx.

SWEEP_CONFIGS = [
    {
        "name": "baseline",
        "description": "Current production params (Modelfile.think)",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "n_ctx": 1024,
        "num_predict": 1024,
    },
    {
        "name": "conservative",
        "description": "Lower temp + narrower sampling for more deterministic output",
        "temperature": 0.5,
        "top_p": 0.98,
        "top_k": 15,
        "n_ctx": 4096,
        "num_predict": 1024,
    },
    {
        "name": "precise",
        "description": "Very low temp, tight sampling — max determinism",
        "temperature": 0.3,
        "top_p": 0.90,
        "top_k": 10,
        "n_ctx": 4096,
        "num_predict": 1024,
    },
    {
        "name": "ctx4k_default",
        "description": "Same as baseline but with 4k context (isolates n_ctx impact)",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "n_ctx": 4096,
        "num_predict": 1024,
    },
]


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


def print_vram(label: str):
    v = get_vram_mb()
    print(f"  VRAM [{label}]: {v['used']} MB used / {v['total']} MB total ({v['free']} MB free)")
    return v


# ── Core inference ────────────────────────────────────────────────────────

def strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def load_model(n_ctx: int = 1024) -> Llama:
    print(f"Loading model: {MODEL_PATH}")
    print(f"  n_ctx={n_ctx}, n_gpu_layers={N_GPU_LAYERS}")

    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    vram_before = print_vram("before load")
    start = time.time()
    model = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=n_ctx,
        verbose=False,
    )
    elapsed = time.time() - start
    vram_after = print_vram("after load")
    model_vram = vram_after["used"] - vram_before["used"]
    print(f"  Loaded in {elapsed:.1f}s (model VRAM: ~{model_vram} MB)")
    return model


def infer(model: Llama, question: str, params: dict,
          rag_context: str = "") -> tuple[str, int, float]:
    """Run inference with given params. Returns (response, tokens, tok/s)."""
    system = SYSTEM_PROMPT
    if rag_context:
        system = system + f"\n\nCOMMAND REFERENCE: {rag_context}"

    prompt = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    start = time.time()
    result = model.create_completion(
        prompt=prompt,
        temperature=params["temperature"],
        top_p=params["top_p"],
        top_k=params["top_k"],
        max_tokens=params["num_predict"],
        stop=STOP_TOKENS,
    )
    elapsed = time.time() - start

    content = strip_thinking(result["choices"][0]["text"])
    completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
    tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0

    return content, completion_tokens, tok_per_sec


# ── Benchmark runner ──────────────────────────────────────────────────────

def run_benchmark(model: Llama, questions: list, params: dict,
                  output_path: str | None = None, use_rag: bool = False) -> dict:
    """Run all questions with given params. Returns summary dict."""
    if use_rag:
        from os_agent.inference.rag import build_rag_context
        from os_agent.inference.validator import validate
        from os_agent.tools.parser import extract_command

    config_name = params.get("name", "custom")
    mode = f"{config_name} | {'RAG + validator' if use_rag else 'raw'}"

    print(f"\n{'=' * 70}")
    print(f"BENCHMARK — {mode}")
    print(f"  temp={params['temperature']}, top_p={params['top_p']}, "
          f"top_k={params['top_k']}, n_ctx={params['n_ctx']}, "
          f"num_predict={params['num_predict']}")
    print(f"{'=' * 70}")

    results = []
    tok_speeds = []
    total_tokens = 0
    vram_peak = 0
    rag_hits = 0
    validator_blocks = 0
    validator_warns = 0

    for i, (question, domain) in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] [{domain.upper()}] {question}")
        print("-" * 50)

        rag_ctx = ""
        if use_rag:
            rag_ctx = build_rag_context(question)
            if rag_ctx:
                rag_hits += 1
                print(f"  [RAG] {rag_ctx[:80]}{'...' if len(rag_ctx) > 80 else ''}")

        response, tokens, tok_s = infer(model, question, params, rag_context=rag_ctx)
        print(response)

        v_result = None
        cmd_extracted = None
        if use_rag:
            cmd_extracted = extract_command(response)
            if cmd_extracted:
                v_result = validate(cmd_extracted)
                if not v_result["ok"]:
                    validator_blocks += 1
                    print(f"  [BLOCKED] {v_result['error']}")
                    if v_result.get("suggestion"):
                        print(f"  [SUGGEST] {v_result['suggestion']}")
                elif v_result.get("warn"):
                    validator_warns += 1
                    print(f"  [WARN] {v_result['warn']}")

        vram = get_vram_mb()
        vram_peak = max(vram_peak, vram["used"])
        print(f"\n  ({tokens} tokens, {tok_s:.1f} tok/s | VRAM: {vram['used']} MB)")
        print("-" * 50)

        total_tokens += tokens
        tok_speeds.append(tok_s)
        entry = {
            "question": question,
            "domain": domain,
            "response": response,
            "tokens": tokens,
            "tok_per_sec": round(tok_s, 1),
            "vram_mb": vram["used"],
        }
        if use_rag:
            entry["rag_context"] = rag_ctx
            entry["command"] = cmd_extracted
            entry["validator"] = v_result
        results.append(entry)

    avg_tok_s = sum(tok_speeds) / len(tok_speeds) if tok_speeds else 0
    warm_speeds = tok_speeds[1:] if len(tok_speeds) > 1 else tok_speeds
    warm_avg = sum(warm_speeds) / len(warm_speeds) if warm_speeds else 0

    vram_final = print_vram("after benchmark")

    print(f"\n{'=' * 70}")
    print(f"SUMMARY — {config_name}")
    print(f"  Questions:          {len(questions)}")
    print(f"  Total tokens:       {total_tokens}")
    print(f"  Avg tok/s (all):    {avg_tok_s:.1f}")
    print(f"  Avg tok/s (warm):   {warm_avg:.1f}")
    print(f"  VRAM peak:          {vram_peak} MB")

    if use_rag:
        print(f"\n  RAG + Validator:")
        print(f"    RAG hits:         {rag_hits}/{len(questions)} "
              f"({100 * rag_hits / len(questions):.0f}%)")
        print(f"    Validator blocks: {validator_blocks}")
        print(f"    Validator warns:  {validator_warns}")
        print(f"    Clean passes:     {len(questions) - validator_blocks - validator_warns}")

    domain_speeds: dict[str, list[float]] = {}
    for r in results:
        domain_speeds.setdefault(r["domain"], []).append(r["tok_per_sec"])

    print(f"\n  Per-domain avg tok/s:")
    for domain, speeds in sorted(domain_speeds.items()):
        davg = sum(speeds) / len(speeds)
        print(f"    {domain:12s}: {davg:.1f} tok/s  ({len(speeds)} questions)")

    print(f"{'=' * 70}")

    summary = {
        "config": config_name,
        "params": {k: v for k, v in params.items() if k not in ("name", "description")},
        "questions": len(questions),
        "total_tokens": total_tokens,
        "avg_tok_s": round(avg_tok_s, 1),
        "warm_avg_tok_s": round(warm_avg, 1),
        "vram_peak_mb": vram_peak,
        "rag_hits": rag_hits if use_rag else None,
        "validator_blocks": validator_blocks if use_rag else None,
        "validator_warns": validator_warns if use_rag else None,
        "results": results,
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")

    return summary


# ── Sweep mode ────────────────────────────────────────────────────────────

def run_sweep(model: Llama, questions: list, configs: list[dict],
              use_rag: bool = False, output_dir: str | None = None):
    """Run all questions through each config, then print comparison table."""
    summaries = []

    for cfg in configs:
        out_path = None
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            out_path = str(Path(output_dir) / f"results_{cfg['name']}.json")

        summary = run_benchmark(model, questions, cfg, out_path, use_rag)
        summaries.append(summary)

    # ── Comparison table ──────────────────────────────────────────────────
    print(f"\n\n{'=' * 90}")
    print("PARAMETER SWEEP — COMPARISON")
    print(f"{'=' * 90}")
    print(f"{'Config':<18} {'temp':>5} {'top_p':>6} {'top_k':>6} {'n_ctx':>6} "
          f"{'tok/s':>7} {'tokens':>7}", end="")
    if use_rag:
        print(f" {'RAG%':>5} {'blocks':>7} {'warns':>6}", end="")
    print()
    print("-" * 90)

    for s in summaries:
        p = s["params"]
        print(f"{s['config']:<18} {p['temperature']:>5.1f} {p['top_p']:>6.2f} "
              f"{p['top_k']:>6} {p['n_ctx']:>6} "
              f"{s['warm_avg_tok_s']:>7.1f} {s['total_tokens']:>7}", end="")
        if use_rag:
            rag_pct = 100 * (s["rag_hits"] or 0) / s["questions"]
            print(f" {rag_pct:>4.0f}% {s['validator_blocks'] or 0:>7} "
                  f"{s['validator_warns'] or 0:>6}", end="")
        print()

    print(f"{'=' * 90}")
    print("\nReview the per-question responses in the JSON files to judge accuracy.")
    print("Lower temp + lower top_k = more deterministic (better for commands).")
    print("Higher n_ctx = more room for think block + response (reduces truncation).")


# ── CLI ───────────────────────────────────────────────────────────────────

def _build_question_list(args) -> list[tuple[str, str]]:
    if args.quick:
        return TEST_QUESTIONS[:3]
    if args.benchmark:
        return TEST_QUESTIONS
    if args.full:
        return LEGACY_TUPLES
    if args.difficulty or args.domain or args.test_type:
        filtered = filter_questions(
            difficulty=args.difficulty,
            eval_domain=args.domain,
            test_type=args.test_type,
        )
        return [(q.q, q.eval_domain) for q in filtered]
    return TEST_QUESTIONS


def main():
    parser = argparse.ArgumentParser(
        description="Inference benchmark with optional RAG + parameter sweep"
    )
    # Question selection
    parser.add_argument("--benchmark", action="store_true", help="Original 44 questions")
    parser.add_argument("--full", action="store_true", help="All 150 questions")
    parser.add_argument("--quick", action="store_true", help="3 questions only")
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--test-type", default=None)

    # Output
    parser.add_argument("--output", default=None, help="Save results JSON (single config)")
    parser.add_argument("--output-dir", default=None,
                        help="Save per-config results to this directory (sweep mode)")

    # Features
    parser.add_argument("--rag", action="store_true", help="Enable RAG + validator")
    parser.add_argument("--sweep", action="store_true",
                        help="Run all preset configs and compare")

    # Custom params (single-config mode)
    parser.add_argument("--temp", type=float, default=None, help="Temperature (default: 0.6)")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p (default: 0.95)")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k (default: 20)")
    parser.add_argument("--n-ctx", type=int, default=None, help="Context window (default: 1024)")

    args = parser.parse_args()
    questions = _build_question_list(args)

    if args.sweep:
        # Sweep: load model once with the max n_ctx across all configs
        max_ctx = max(c["n_ctx"] for c in SWEEP_CONFIGS)
        print(f"SWEEP MODE — {len(SWEEP_CONFIGS)} configs × {len(questions)} questions")
        print(f"Loading model with n_ctx={max_ctx} (max across configs)\n")
        model = load_model(n_ctx=max_ctx)
        run_sweep(model, questions, SWEEP_CONFIGS, use_rag=args.rag,
                  output_dir=args.output_dir)
    else:
        # Single config: use CLI overrides or defaults
        params = {
            "name": "custom" if any([args.temp, args.top_p, args.top_k, args.n_ctx]) else "baseline",
            "temperature": args.temp if args.temp is not None else 0.6,
            "top_p": args.top_p if args.top_p is not None else 0.95,
            "top_k": args.top_k if args.top_k is not None else 20,
            "n_ctx": args.n_ctx if args.n_ctx is not None else 1024,
            "num_predict": 1024,
        }
        mode = f"{'RAG + validator' if args.rag else 'raw'}"
        print(f"Running {len(questions)} questions ({mode})")
        model = load_model(n_ctx=params["n_ctx"])
        run_benchmark(model, questions, params, args.output, use_rag=args.rag)


if __name__ == "__main__":
    main()
