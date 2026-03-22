"""
Evaluation script for GGUF models (Q4_K_M, Q5_K_M, etc.)
Tests the fine-tuned model on the same 44 OS/Linux domain questions as eval_adapter.py.

Usage (on Vast.ai or any machine with the GGUF file):
    pip install llama-cpp-python
    python eval_gguf.py
    python eval_gguf.py --no-score
    python eval_gguf.py --model /path/to/model.gguf
    python eval_gguf.py --compare /path/to/other.gguf   # side-by-side comparison
"""

import argparse
import json
import os
import re
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "qwen3.5-4b-os-q4km.gguf")

SYSTEM_PROMPT = (
    "You are an AI assistant built into a Linux-based operating system. "
    "Respond with one correct command in a bash code block followed by a one-line explanation. "
    "For conceptual questions, explain in 2-4 sentences. "
    "If the request is ambiguous, ask one clarifying question. "
    "Never list alternatives. Never restate the question. Never explain individual flags."
)

TEST_QUESTIONS = [
    # --- File operations ---
    ("Find all files larger than 100MB on Linux", "files"),
    ("Find files modified in the last 24 hours in /var/log", "files"),
    ("Recursively search for the string 'ERROR' in all .log files under /var", "files"),
    ("What does chmod 755 do and when would you use it?", "files"),
    ("How do I change the owner of a directory and all its contents?", "files"),
    ("How do I create a symbolic link?", "files"),

    # --- Networking & SSH ---
    ("List all open TCP ports on the system", "networking"),
    ("Generate an SSH key pair and add it to authorized_keys", "networking"),
    ("How do I copy a file to a remote server using SCP?", "networking"),
    ("How do I check my current IP address on Linux?", "networking"),
    ("How do I test if a remote port is open without telnet?", "networking"),
    ("How do I block port 22 with iptables?", "networking"),
    ("How do I use rsync to sync a local folder to a remote server?", "networking"),

    # --- Process & resource management ---
    ("How do I check disk usage broken down by directory?", "process"),
    ("How do I kill a process by name without knowing its PID?", "process"),
    ("Show me how to find which process is using the most memory", "process"),
    ("How do I run a process in the background and keep it after SSH logout?", "process"),
    ("How do I schedule a cron job to run a script every day at midnight?", "process"),
    ("How do I check CPU and memory usage in real time?", "process"),

    # --- User & permission management ---
    ("How do I add a user to the sudo group?", "users"),
    ("How do I create a new user with a home directory?", "users"),
    ("How do I lock a user account without deleting it?", "users"),
    ("How do I view all groups a user belongs to?", "users"),

    # --- Package & service management ---
    ("How do I install a .deb package manually?", "packages"),
    ("How do I start, stop, and restart a systemd service?", "packages"),
    ("How do I check if a service is enabled on boot with systemd?", "packages"),
    ("How do I find which package owns a specific file on Debian/Ubuntu?", "packages"),

    # --- Text processing ---
    ("How do I extract the 3rd column from a space-separated file using awk?", "text"),
    ("How do I replace all occurrences of 'foo' with 'bar' in a file using sed?", "text"),
    ("How do I count lines, words, and characters in a file?", "text"),
    ("How do I sort a file and remove duplicate lines?", "text"),

    # --- Storage & archiving ---
    ("How do I mount a USB drive on Linux?", "storage"),
    ("How do I check available disk space on all mounted filesystems?", "storage"),
    ("How do I create a compressed tar.gz archive of a directory?", "storage"),
    ("How do I find and delete files older than 30 days?", "storage"),

    # --- Kernel / OS concepts ---
    ("What is a Linux kernel module and how do you load one?", "kernel"),
    ("Explain the difference between a process and a thread in Linux", "kernel"),
    ("How does virtual memory paging work in Linux?", "kernel"),
    ("What is the purpose of the /proc filesystem?", "kernel"),
    ("How do I check the current kernel version and build info?", "kernel"),

    # --- Shell scripting ---
    ("Write a bash script that checks if a file exists and prints a message", "scripting"),
    ("How do I loop over all .log files in a directory in bash?", "scripting"),
    ("How do I capture the output of a command into a variable in bash?", "scripting"),
    ("How do I pass arguments to a bash script and validate them?", "scripting"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GGUF model on OS/Linux questions")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Path to GGUF model file",
    )
    parser.add_argument(
        "--compare",
        default=None,
        help="Path to second GGUF model for side-by-side comparison",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Skip interactive scoring, just print responses",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Max tokens to generate per response (default: 300)",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU (-1 = all, 0 = CPU only)",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=512,
        help="Context window size (default: 512)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save results to JSON file",
    )
    return parser.parse_args()


def load_model(model_path, n_gpu_layers, n_ctx):
    from llama_cpp import Llama

    if not os.path.isfile(model_path):
        print(f"ERROR: GGUF file not found: {model_path}")
        sys.exit(1)

    print(f"Loading GGUF model: {model_path}")
    print(f"  GPU layers: {n_gpu_layers}  |  Context: {n_ctx}")

    model = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=False,
    )

    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  Model size: {size_mb:.0f} MB")
    return model


def strip_thinking(text):
    """Remove <think>...</think> blocks from model output."""
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    # Also handle unclosed think blocks (model started thinking but got cut off)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def generate(model, question, max_tokens):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    start = time.time()
    response = model.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.1,
        top_p=0.9,
        stop=["<|im_end|>", "<|endoftext|>", "</think>"],
    )
    elapsed = time.time() - start

    content = response["choices"][0]["message"]["content"]
    content = strip_thinking(content)
    tokens_generated = response["usage"]["completion_tokens"]
    tok_per_sec = tokens_generated / elapsed if elapsed > 0 else 0

    return content, tokens_generated, tok_per_sec


def run_eval(model, model_name, args):
    print(f"\n{'=' * 70}")
    print(f"EVALUATION — {model_name}")
    print(f"{'=' * 70}")

    results = []
    scores = []
    domain_scores = {}
    total_tokens = 0
    total_time_tokens = []

    for i, (question, domain) in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] [{domain.upper()}] {question}")
        print("-" * 50)

        response, tokens, tok_s = generate(model, question, args.max_tokens)
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
        print(f"  Rated: {len(scores)}/{len(TEST_QUESTIONS)} | Avg score: {avg:.2f}/5")

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

    model = load_model(args.model, args.n_gpu_layers, args.n_ctx)
    model_name = os.path.basename(args.model)
    results_primary = run_eval(model, model_name, args)

    results_compare = None
    if args.compare:
        del model
        model2 = load_model(args.compare, args.n_gpu_layers, args.n_ctx)
        model2_name = os.path.basename(args.compare)
        results_compare = run_eval(model2, model2_name, args)

    if args.output:
        output_data = {"model": model_name, "results": results_primary}
        if results_compare:
            output_data["compare_model"] = os.path.basename(args.compare)
            output_data["compare_results"] = results_compare
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
