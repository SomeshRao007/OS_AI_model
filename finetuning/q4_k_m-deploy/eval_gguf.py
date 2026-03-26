"""
Evaluation script for GGUF models via Ollama API.
Tests the fine-tuned model on OS/Linux domain questions (44 original or 150 full suite).

Prerequisites:
    ollama serve  (must be running)
    ollama create os-ai -f Modelfile  (model must be imported)

Usage:
    python eval_gguf.py --no-score                   # original 44 questions
    python eval_gguf.py --full --no-score             # all 150 questions
    python eval_gguf.py --difficulty developer --no-score  # developer-level only
    python eval_gguf.py --domain docker --no-score    # single eval domain
    python eval_gguf.py --model os-ai
    python eval_gguf.py --no-score --output eval_results.json
    python eval_gguf.py --compare qwen3.5:4b          # side-by-side with base model
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

# Add project root to path for eval_questions import
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
from os_agent.eval_questions import (
    LEGACY_TUPLES,
    ORIGINAL_44_TUPLES,
    filter_questions,
)

DEFAULT_MODEL = "os-ai"
OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = (
    "You are an AI assistant built into a Linux-based operating system. "
    "Respond with one correct command in a bash code block followed by a one-line explanation. "
    "For conceptual questions, explain in 2-4 sentences. "
    "If the request is ambiguous, ask one clarifying question. "
    "Never list alternatives. Never restate the question. Never explain individual flags."
)

# Question bank imported from eval_questions.py
TEST_QUESTIONS = ORIGINAL_44_TUPLES


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GGUF model via Ollama")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama model name (default: os-ai)",
    )
    parser.add_argument(
        "--compare",
        default=None,
        help="Second Ollama model name for side-by-side comparison",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Skip interactive scoring, just print responses",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=250,
        help="Max tokens to generate per response (default: 250)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all 150 questions instead of original 44",
    )
    parser.add_argument(
        "--difficulty",
        default=None,
        help="Filter: basic|intermediate|advanced|developer",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="Filter by eval domain (e.g. docker, git, files)",
    )
    parser.add_argument(
        "--test-type",
        default=None,
        help="Filter: command|conceptual|routing|format|adversarial",
    )
    return parser.parse_args()


def build_question_list(args) -> list[tuple[str, str]]:
    """Build the question list based on CLI flags."""
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


def strip_thinking(text):
    """Remove <think>...</think> blocks from model output."""
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def ollama_chat(model_name, question, max_tokens):
    """Call Ollama chat API and return response content, token count, and speed."""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": max_tokens,
            "num_ctx": 512,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    start = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - start

    content = result.get("message", {}).get("content", "")
    content = strip_thinking(content)

    eval_count = result.get("eval_count", 0)
    eval_duration_ns = result.get("eval_duration", 1)
    tok_per_sec = (eval_count / eval_duration_ns * 1e9) if eval_duration_ns > 0 else 0

    return content, eval_count, tok_per_sec


def check_ollama(model_name):
    """Verify Ollama is running and the model exists."""
    # Check server
    health_req = urllib.request.Request("http://localhost:11434/api/tags")
    try:
        with urllib.request.urlopen(health_req, timeout=5) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, ConnectionRefusedError):
        print("ERROR: Ollama is not running. Start it with: ollama serve &")
        sys.exit(1)

    # Check model exists
    model_names = [m.get("name", "") for m in tags.get("models", [])]
    # Ollama model names can have :latest suffix
    found = any(model_name in name for name in model_names)
    if not found:
        print(f"ERROR: Model '{model_name}' not found in Ollama.")
        print(f"Available models: {', '.join(model_names)}")
        print(f"Import with: ollama create {model_name} -f Modelfile")
        sys.exit(1)

    print(f"Ollama model: {model_name}")


def run_evaluation(model_name, args, questions=None):
    questions = questions or TEST_QUESTIONS
    print(f"\n{'=' * 70}")
    print(f"EVALUATION — {model_name} ({len(questions)} questions)")
    print(f"{'=' * 70}")

    results = []
    scores = []
    domain_scores = {}
    total_tokens = 0
    total_time_tokens = []

    for i, (question, domain) in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] [{domain.upper()}] {question}")
        print("-" * 50)

        response, tokens, tok_s = ollama_chat(model_name, question, args.max_tokens)
        print(response)
        print(f"\n  ({tokens} tokens, {tok_s:.1f} tok/s)")
        print("-" * 50)

        total_tokens += tokens
        total_time_tokens.append(tok_s)

        result = {
            "question": question,
            "domain": domain,
            "response": response,
            "tokens": tokens,
            "tok_per_sec": round(tok_s, 1),
        }

        if not args.no_score:
            score_input = input("Rate 1-5 (or Enter to skip): ").strip()
            if score_input.isdigit() and 1 <= int(score_input) <= 5:
                score = int(score_input)
                scores.append(score)
                result["score"] = score
                domain_scores.setdefault(domain, []).append(score)

        results.append(result)

    avg_tok_s = sum(total_time_tokens) / len(total_time_tokens) if total_time_tokens else 0

    print(f"\n{'=' * 70}")
    print(f"SUMMARY — {model_name}")
    print(f"  Total tokens generated: {total_tokens}")
    print(f"  Avg speed: {avg_tok_s:.1f} tok/s")

    if scores:
        avg = sum(scores) / len(scores)
        print(f"  Rated: {len(scores)}/{len(questions)} | Avg score: {avg:.2f}/5")

        print(f"\n  Per-domain scores:")
        for domain, dscores in sorted(domain_scores.items()):
            davg = sum(dscores) / len(dscores)
            print(f"    {domain:12s}: {davg:.2f}/5  ({len(dscores)} rated)")

        if avg >= 4.0:
            print("\n  Model quality: EXCELLENT — ready for OS integration")
        elif avg >= 3.0:
            print("\n  Model quality: GOOD — minor gaps, consider targeted data")
        elif avg >= 2.0:
            print("\n  Model quality: FAIR — notable gaps, re-finetune with more data")
        else:
            print("\n  Model quality: POOR — significant issues, investigate training")

    print(f"{'=' * 70}")
    return results


def main():
    args = parse_args()
    questions = build_question_list(args)
    print(f"Running {len(questions)} questions")

    check_ollama(args.model)
    results_primary = run_evaluation(args.model, args, questions)

    results_compare = None
    if args.compare:
        check_ollama(args.compare)
        results_compare = run_evaluation(args.compare, args, questions)

    if args.output:
        output_data = {"model": args.model, "results": results_primary}
        if results_compare:
            output_data["compare_model"] = args.compare
            output_data["compare_results"] = results_compare
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
