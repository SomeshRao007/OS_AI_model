"""
Compare GGUF Q4_K_M (via Ollama) vs adapter (via transformers, 4-bit NF4)
on 44 OS/Linux eval questions. Thinking enabled for both — stripped from output.

Outputs two separate txt files with full responses (no scoring).

Prerequisites:
    # GGUF model (thinking-enabled variant)
    cd finetuning/q4_k_m-deploy
    ollama create os-ai-think -f Modelfile.think

    # Python packages for adapter eval
    pip install torch transformers peft bitsandbytes accelerate

Usage:
    python finetuning/eval_compare.py                    # both models
    python finetuning/eval_compare.py --gguf-only        # GGUF only (no ML deps)
    python finetuning/eval_compare.py --adapter-only     # adapter only
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ADAPTER = os.path.join(SCRIPT_DIR, "output", "original", "output", "qlora-run")
DEFAULT_OLLAMA_MODEL = "os-ai-think"
OLLAMA_URL = "http://localhost:11434/api/chat"
BASE_MODEL = "Qwen/Qwen3.5-4B"

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


def strip_thinking(text):
    """Remove <think>...</think> blocks from model output, return (thinking, answer)."""
    # Match completed think blocks
    match = re.match(r"<think>(.*?)</think>\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    # Match unclosed think block (model hit token limit mid-think)
    match = re.match(r"<think>(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip(), "[TRUNCATED — model ran out of tokens during thinking]"
    return "", text.strip()


def format_result_block(idx, total, question, domain, answer, thinking, tokens, tok_s, elapsed):
    """Format a single Q&A result for the txt file."""
    lines = []
    lines.append(f"[{idx}/{total}] [{domain.upper()}]")
    lines.append(f"Q: {question}")
    lines.append("")
    lines.append(f"A: {answer}")
    lines.append("")
    lines.append(f"  tokens={tokens}  tok/s={tok_s:.1f}  elapsed={elapsed:.1f}s  thinking_len={len(thinking.split()) if thinking else 0}w")
    lines.append("=" * 70)
    return "\n".join(lines)


# ── Ollama (GGUF) inference ──


def check_ollama(model_name):
    """Verify Ollama is running and the model exists."""
    health_req = urllib.request.Request("http://localhost:11434/api/tags")
    with urllib.request.urlopen(health_req, timeout=5) as resp:
        tags = json.loads(resp.read().decode("utf-8"))

    model_names = [m.get("name", "") for m in tags.get("models", [])]
    found = any(model_name in name for name in model_names)
    if not found:
        print(f"ERROR: Model '{model_name}' not found in Ollama.")
        print(f"Available: {', '.join(model_names)}")
        print(f"\nTo create it:")
        print(f"  cd finetuning/q4_k_m-deploy")
        print(f"  ollama create {model_name} -f Modelfile.think")
        sys.exit(1)
    print(f"  Ollama model found: {model_name}")


def ollama_chat(model_name, question, max_tokens=1024):
    """Call Ollama with thinking enabled. Returns (answer, thinking, tokens, tok/s, elapsed)."""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "stream": False,
        # "think": True,
        "options": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "num_predict": max_tokens,
            "num_ctx": 1024,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
    )

    start = time.time()
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - start

    msg = result.get("message", {})
    content = msg.get("content", "")
    thinking = msg.get("thinking", "")

    # Fallback: if Ollama doesn't separate thinking (custom GGUF without renderer),
    # the thinking may be inline in content — strip it
    if not thinking and "<think>" in content:
        thinking, content = strip_thinking(content)

    eval_count = result.get("eval_count", 0)
    eval_duration_ns = result.get("eval_duration", 1)
    tok_per_sec = (eval_count / eval_duration_ns * 1e9) if eval_duration_ns > 0 else 0

    return content.strip(), thinking, eval_count, tok_per_sec, elapsed


def run_gguf(model_name, output_path, max_tokens=1024):
    """Run all questions through GGUF via Ollama, write results to txt."""
    print(f"\n{'=' * 70}")
    print(f"  GGUF Q4_K_M via Ollama — {model_name}")
    print(f"{'=' * 70}")

    lines = []
    lines.append(f"Model: {model_name} (GGUF Q4_K_M via Ollama)")
    lines.append(f"Thinking: enabled (stripped from answers below)")
    lines.append(f"Timestamp: {datetime.now().isoformat()}")
    lines.append(f"Questions: {len(TEST_QUESTIONS)}")
    lines.append("=" * 70)
    lines.append("")

    total_tokens = 0
    speeds = []

    for i, (question, domain) in enumerate(TEST_QUESTIONS, 1):
        print(f"  [{i:2d}/{len(TEST_QUESTIONS)}] {question[:55]}...", end=" ", flush=True)

        answer, thinking, tokens, tok_s, elapsed = ollama_chat(model_name, question, max_tokens)

        print(f"({tokens} tok, {tok_s:.0f} tok/s)")
        total_tokens += tokens
        speeds.append(tok_s)

        block = format_result_block(i, len(TEST_QUESTIONS), question, domain, answer, thinking, tokens, tok_s, elapsed)
        lines.append(block)
        lines.append("")

    avg_speed = sum(speeds) / len(speeds) if speeds else 0
    lines.append(f"\nSUMMARY")
    lines.append(f"  Total tokens: {total_tokens}")
    lines.append(f"  Avg speed: {avg_speed:.1f} tok/s")
    lines.append(f"  Total time: {sum(e for _, _, _, _, e in [ollama_chat(model_name, 'test', 1)] or speeds):.0f}s")

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Results saved to: {output_path}")
    return total_tokens, avg_speed


# ── Adapter (transformers) inference ──


def load_adapter_model(adapter_path):
    """Load adapter with 4-bit quantization for RTX 3060 (12GB)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    if not os.path.isdir(adapter_path):
        print(f"ERROR: Adapter path not found: {adapter_path}")
        sys.exit(1)

    print(f"  Loading tokenizer from: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"  Loading base model: {BASE_MODEL} (4-bit NF4 for 12GB VRAM)")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"  Applying adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    device = next(model.parameters()).device
    print(f"  Model loaded on: {device}")
    return model, tokenizer


def adapter_generate(model, tokenizer, question, max_new_tokens=1024):
    """Generate with thinking enabled, return (answer, thinking, tokens, tok/s, elapsed)."""
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    # enable_thinking=True (default) — model produces <think> blocks
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    start = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            do_sample=True,
            pad_token_id=pad_id,
        )
    elapsed = time.time() - start

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    num_tokens = len(new_tokens)
    tok_per_sec = num_tokens / elapsed if elapsed > 0 else 0
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    thinking, answer = strip_thinking(raw)
    return answer, thinking, num_tokens, tok_per_sec, elapsed


def run_adapter(adapter_path, output_path, max_new_tokens=1024):
    """Run all questions through adapter, write results to txt."""
    model, tokenizer = load_adapter_model(adapter_path)

    print(f"\n{'=' * 70}")
    print(f"  Adapter (4-bit NF4) — Run 3 v3-23k-synthetic")
    print(f"{'=' * 70}")

    lines = []
    lines.append(f"Model: Qwen3.5-4B + LoRA adapter (4-bit NF4 via BitsAndBytes)")
    lines.append(f"Adapter: {adapter_path}")
    lines.append(f"Thinking: enabled (stripped from answers below)")
    lines.append(f"Timestamp: {datetime.now().isoformat()}")
    lines.append(f"Questions: {len(TEST_QUESTIONS)}")
    lines.append("=" * 70)
    lines.append("")

    total_tokens = 0
    speeds = []

    for i, (question, domain) in enumerate(TEST_QUESTIONS, 1):
        print(f"  [{i:2d}/{len(TEST_QUESTIONS)}] {question[:55]}...", end=" ", flush=True)

        answer, thinking, tokens, tok_s, elapsed = adapter_generate(
            model, tokenizer, question, max_new_tokens
        )

        print(f"({tokens} tok, {tok_s:.0f} tok/s)")
        total_tokens += tokens
        speeds.append(tok_s)

        block = format_result_block(i, len(TEST_QUESTIONS), question, domain, answer, thinking, tokens, tok_s, elapsed)
        lines.append(block)
        lines.append("")

    avg_speed = sum(speeds) / len(speeds) if speeds else 0
    lines.append(f"\nSUMMARY")
    lines.append(f"  Total tokens: {total_tokens}")
    lines.append(f"  Avg speed: {avg_speed:.1f} tok/s")

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Results saved to: {output_path}")

    # Free VRAM
    import torch
    del model
    torch.cuda.empty_cache()

    return total_tokens, avg_speed


# ── Main ──


def main():
    parser = argparse.ArgumentParser(description="Compare GGUF vs Adapter (thinking enabled)")
    parser.add_argument("--gguf-only", action="store_true", help="Only run GGUF via Ollama")
    parser.add_argument("--adapter-only", action="store_true", help="Only run adapter via transformers")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="Adapter directory path")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model name")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens per response (default: 1024)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gguf_output = os.path.join(SCRIPT_DIR, f"results_gguf_{timestamp}.txt")
    adapter_output = os.path.join(SCRIPT_DIR, f"results_adapter_{timestamp}.txt")

    if not args.adapter_only:
        check_ollama(args.model)
        run_gguf(args.model, gguf_output, args.max_tokens)

    if not args.gguf_only:
        run_adapter(args.adapter, adapter_output, args.max_tokens)

    print(f"\n{'=' * 70}")
    print("  DONE — Compare the two files side by side:")
    if not args.adapter_only:
        print(f"    GGUF:    {gguf_output}")
    if not args.gguf_only:
        print(f"    Adapter: {adapter_output}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
