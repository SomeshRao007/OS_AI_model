"""
Evaluation script v2 — tests direct-answer behavior via prompt engineering.
Same questions as eval_adapter.py, but with a system prompt that demands
concise, single-answer responses and strips <think> blocks from output.

Usage:
    python finetuning/eval_adapter_v2.py
    python finetuning/eval_adapter_v2.py --no-score
    python finetuning/eval_adapter_v2.py --adapter finetuning/output/qlora-run
    python finetuning/eval_adapter_v2.py --compare   # run both prompts side-by-side
"""

import argparse
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3.5-4B"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ADAPTER = os.path.join(_SCRIPT_DIR, "output", "qlora-run")

# v1 prompt (original) — kept for --compare mode
SYSTEM_PROMPT_V1 = (
    "You are a Linux system expert and OS assistant. "
    "Provide accurate, concise answers about Linux commands, "
    "system administration, and kernel concepts."
)

# v2 prompt — demands direct, confident answers
SYSTEM_PROMPT_V2 = (
    "You are a Linux system expert built into an OS. "
    "Rules:\n"
    "1. When asked to DO something: give ONE correct command immediately, "
    "then a one-line explanation. No alternatives, no variations, no lists.\n"
    "2. When asked to EXPLAIN something: give a clear, focused explanation.\n"
    "3. If the question is ambiguous (e.g. which directory, which distro), "
    "ask the user to clarify — do not guess.\n"
    "4. Never repeat the question back. Never say \"The user wants to...\".\n"
    "5. Be direct. No preamble, no filler."
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
    parser = argparse.ArgumentParser(description="Eval v2 — direct-answer prompt")
    parser.add_argument(
        "--adapter", default=_DEFAULT_ADAPTER,
        help="Path to adapter directory",
    )
    parser.add_argument("--no-score", action="store_true", help="Skip interactive scoring")
    parser.add_argument(
        "--max-new-tokens", type=int, default=300,
        help="Max tokens to generate per response (default: 300)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run both v1 and v2 prompts side-by-side for each question",
    )
    return parser.parse_args()


def load_model(adapter_path: str):
    if not os.path.isdir(adapter_path):
        print(f"ERROR: Adapter path not found: {adapter_path}")
        sys.exit(1)

    print(f"Loading tokenizer from adapter: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    print(f"Loading base model: {BASE_MODEL}  (dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Applying adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    device = next(model.parameters()).device
    print(f"Model loaded on: {device}")
    return model, tokenizer


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output."""
    # Handle complete think blocks (greedy: outermost pair)
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    # Handle unclosed think block (model hit max_tokens mid-thought)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def generate(model, tokenizer, question: str, system_prompt: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=pad_id,
        )

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return strip_think_blocks(raw)


def run_eval(model, tokenizer, args, system_prompt: str, label: str):
    """Run evaluation with a given system prompt and return scores."""
    print(f"\n{'=' * 70}")
    print(f"EVALUATION [{label}] — Fine-tuned Qwen3.5-4B (QLoRA adapter)")
    print(f"{'=' * 70}")
    print(f"System prompt: {system_prompt[:80]}...")
    print(f"{'=' * 70}")

    scores = []
    for i, (question, domain) in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] [{domain.upper()}] {question}")
        print("-" * 50)
        response = generate(model, tokenizer, question, system_prompt, args.max_new_tokens)
        print(response)
        print("-" * 50)

        if not args.no_score:
            score_input = input("Rate 1-5 (or Enter to skip): ").strip()
            if score_input.isdigit() and 1 <= int(score_input) <= 5:
                scores.append(int(score_input))

    return scores


def print_results(scores: list[int], total: int, label: str):
    if not scores:
        return
    avg = sum(scores) / len(scores)
    print(f"\n{'=' * 70}")
    print(f"RESULTS [{label}]: {len(scores)}/{total} rated | avg score: {avg:.2f}/5")
    if avg >= 4.7:
        print("  EXCELLENT — direct, confident answers")
    elif avg >= 3.0:
        print("  GOOD — mostly direct, some verbosity")
    elif avg >= 2.0:
        print("  FAIR — still too verbose or hedging")
    else:
        print("  POOR — not following prompt instructions")
    print(f"{'=' * 70}")
    return avg


def main():
    args = parse_args()
    model, tokenizer = load_model(args.adapter)

    if args.compare:
        # Side-by-side comparison mode
        print(f"\n{'=' * 70}")
        print("COMPARE MODE — v1 (original) vs v2 (direct-answer)")
        print(f"{'=' * 70}")

        v1_scores = []
        v2_scores = []

        for i, (question, domain) in enumerate(TEST_QUESTIONS, 1):
            print(f"\n{'=' * 70}")
            print(f"[{i}/{len(TEST_QUESTIONS)}] [{domain.upper()}] {question}")

            print(f"\n--- v1 (original prompt) ---")
            r1 = generate(model, tokenizer, question, SYSTEM_PROMPT_V1, args.max_new_tokens)
            print(r1)

            print(f"\n--- v2 (direct-answer prompt) ---")
            r2 = generate(model, tokenizer, question, SYSTEM_PROMPT_V2, args.max_new_tokens)
            print(r2)
            print("-" * 50)

            if not args.no_score:
                s1 = input("Rate v1 (1-5 or Enter to skip): ").strip()
                if s1.isdigit() and 1 <= int(s1) <= 5:
                    v1_scores.append(int(s1))
                s2 = input("Rate v2 (1-5 or Enter to skip): ").strip()
                if s2.isdigit() and 1 <= int(s2) <= 5:
                    v2_scores.append(int(s2))

        print_results(v1_scores, len(TEST_QUESTIONS), "v1 original")
        print_results(v2_scores, len(TEST_QUESTIONS), "v2 direct-answer")

    else:
        # Standard v2 evaluation
        scores = run_eval(model, tokenizer, args, SYSTEM_PROMPT_V2, "v2 direct-answer")
        print_results(scores, len(TEST_QUESTIONS), "v2 direct-answer")


if __name__ == "__main__":
    main()
