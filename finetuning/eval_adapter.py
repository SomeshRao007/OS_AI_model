"""
Quick evaluation script for the QLoRA adapter.
Tests the fine-tuned model on OS/Linux domain questions.

Usage (from project root):
    python finetuning/eval_adapter.py
    python finetuning/eval_adapter.py --no-score   # skip manual scoring
    python finetuning/eval_adapter.py --adapter finetuning/output/qlora-run
"""

import argparse
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3.5-4B"

# Resolve default adapter path relative to this script — works regardless of cwd
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ADAPTER = os.path.join(_SCRIPT_DIR, "output", "qlora-run")

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default=_DEFAULT_ADAPTER,
        help="Path to adapter directory (default: finetuning/output/qlora-run)",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Skip interactive scoring, just print responses",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=300,
        help="Max tokens to generate per response (default: 300)",
    )
    return parser.parse_args()


def load_model(adapter_path: str):
    if not os.path.isdir(adapter_path):
        print(f"ERROR: Adapter path not found: {adapter_path}")
        print("Make sure you run this from the project root directory.")
        sys.exit(1)

    print(f"Loading tokenizer from adapter: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    # Use bfloat16 — better than float16 on Ampere/Blackwell (RTX 5090, A100)
    # Falls back gracefully on older GPUs
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


def strip_thinking(text):
    """Remove <think>...</think> blocks from model output."""
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def generate(model, tokenizer, question: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    # enable_thinking=False tells Qwen 3.5 to skip <think> reasoning
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
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
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return strip_thinking(response)


def main():
    args = parse_args()

    model, tokenizer = load_model(args.adapter)

    print("\n" + "=" * 70)
    print("EVALUATION — Fine-tuned Qwen3.5-4B (QLoRA adapter)")
    print("=" * 70)

    scores = []
    for i, (question, domain) in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] [{domain.upper()}] {question}")
        print("-" * 50)
        response = generate(model, tokenizer, question, args.max_new_tokens)
        print(response)
        print("-" * 50)

        if not args.no_score:
            score_input = input("Rate 1-5 (or Enter to skip): ").strip()
            if score_input.isdigit() and 1 <= int(score_input) <= 5:
                scores.append(int(score_input))

    if scores:
        avg = sum(scores) / len(scores)
        print(f"\n{'=' * 70}")
        print(f"RESULTS: {len(scores)}/{len(TEST_QUESTIONS)} rated | avg score: {avg:.2f}/5")
        if avg >= 4.7:
            print("Model quality: EXCELLENT — ready for deployment")
        elif avg >= 3.0:
            print("Model quality: GOOD — minor gaps, consider synthetic data for weak areas")
        elif avg >= 2.0:
            print("Model quality: FAIR — notable gaps, re-finetune with more data")
        else:
            print("Model quality: POOR — significant issues, investigate training")
        print("=" * 70)


if __name__ == "__main__":
    main()
