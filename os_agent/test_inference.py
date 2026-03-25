"""
Step 1 verification: llama-cpp-python inference with our fine-tuned GGUF.

Tests:
  1. Model loads on GPU (n_gpu_layers=-1)
  2. All 44 questions run successfully
  3. Measures tok/s per question and overall average
  4. Tracks VRAM utilization before/during/after inference
  5. Compares against Ollama baseline (67 tok/s)

Parameters match Modelfile.think exactly:
  num_ctx=1024, temperature=0.6, top_p=0.95, top_k=20, num_predict=1024

Usage:
    source os_agent_env/bin/activate
    python os_agent/test_inference.py --quick        # 3 questions only
    python os_agent/test_inference.py --benchmark    # full 44-question run
    python os_agent/test_inference.py --output results.json
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from llama_cpp import Llama

MODEL_PATH = str(
    Path(__file__).resolve().parent.parent
    / "finetuning"
    / "q4_k_m-deploy"
    / "qwen3.5-4b-os-q4km.gguf"
)

# ── Modelfile.think parameters (exact match) ─────────────────────────────
SYSTEM_PROMPT = (
    "You are an AI assistant built into a Linux-based operating system. "
    "Respond with one correct command in a bash code block followed by a one-line explanation. "
    "For conceptual questions, explain in 2-4 sentences. "
    "If the request is ambiguous, ask one clarifying question. "
    "Never list alternatives. Never restate the question. Never explain individual flags."
)

# PARAMETER num_ctx 1024
N_CTX = 1024
# PARAMETER num_predict 1024
NUM_PREDICT = 1024
# PARAMETER temperature 0.6
TEMPERATURE = 0.6
# PARAMETER top_p 0.95
TOP_P = 0.95
# PARAMETER top_k 20
TOP_K = 20
# PARAMETER stop "<|im_end|>" / PARAMETER stop "<|endoftext|>"
STOP_TOKENS = ["<|im_end|>", "<|endoftext|>"]
# n_gpu_layers=-1 → all layers on GPU
N_GPU_LAYERS = -1

# Same 44 questions from the existing Ollama-based evaluation script
TEST_QUESTIONS = [
    ("Find all files larger than 100MB on Linux", "files"),
    ("Find files modified in the last 24 hours in /var/log", "files"),
    ("Recursively search for the string 'ERROR' in all .log files under /var", "files"),
    ("What does chmod 755 do and when would you use it?", "files"),
    ("How do I change the owner of a directory and all its contents?", "files"),
    ("How do I create a symbolic link?", "files"),
    ("List all open TCP ports on the system", "networking"),
    ("Generate an SSH key pair and add it to authorized_keys", "networking"),
    ("How do I copy a file to a remote server using SCP?", "networking"),
    ("How do I check my current IP address on Linux?", "networking"),
    ("How do I test if a remote port is open without telnet?", "networking"),
    ("How do I block port 22 with iptables?", "networking"),
    ("How do I use rsync to sync a local folder to a remote server?", "networking"),
    ("How do I check disk usage broken down by directory?", "process"),
    ("How do I kill a process by name without knowing its PID?", "process"),
    ("Show me how to find which process is using the most memory", "process"),
    ("How do I run a process in the background and keep it after SSH logout?", "process"),
    ("How do I schedule a cron job to run a script every day at midnight?", "process"),
    ("How do I check CPU and memory usage in real time?", "process"),
    ("How do I add a user to the sudo group?", "users"),
    ("How do I create a new user with a home directory?", "users"),
    ("How do I lock a user account without deleting it?", "users"),
    ("How do I view all groups a user belongs to?", "users"),
    ("How do I install a .deb package manually?", "packages"),
    ("How do I start, stop, and restart a systemd service?", "packages"),
    ("How do I check if a service is enabled on boot with systemd?", "packages"),
    ("How do I find which package owns a specific file on Debian/Ubuntu?", "packages"),
    ("How do I extract the 3rd column from a space-separated file using awk?", "text"),
    ("How do I replace all occurrences of 'foo' with 'bar' in a file using sed?", "text"),
    ("How do I count lines, words, and characters in a file?", "text"),
    ("How do I sort a file and remove duplicate lines?", "text"),
    ("How do I mount a USB drive on Linux?", "storage"),
    ("How do I check available disk space on all mounted filesystems?", "storage"),
    ("How do I create a compressed tar.gz archive of a directory?", "storage"),
    ("How do I find and delete files older than 30 days?", "storage"),
    ("What is a Linux kernel module and how do you load one?", "kernel"),
    ("Explain the difference between a process and a thread in Linux", "kernel"),
    ("How does virtual memory paging work in Linux?", "kernel"),
    ("What is the purpose of the /proc filesystem?", "kernel"),
    ("How do I check the current kernel version and build info?", "kernel"),
    ("Write a bash script that checks if a file exists and prints a message", "scripting"),
    ("How do I loop over all .log files in a directory in bash?", "scripting"),
    ("How do I capture the output of a command into a variable in bash?", "scripting"),
    ("How do I pass arguments to a bash script and validate them?", "scripting"),
]


# ── VRAM tracking ─────────────────────────────────────────────────────────

def get_vram_mb() -> dict:
    """Query nvidia-smi for current VRAM usage. Returns dict with used/total/free in MB."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {"used": 0, "total": 0, "free": 0}
    parts = result.stdout.strip().split(", ")
    return {
        "used": int(parts[0]),
        "total": int(parts[1]),
        "free": int(parts[2]),
    }


def print_vram(label: str):
    """Print VRAM usage with a label."""
    v = get_vram_mb()
    print(f"  VRAM [{label}]: {v['used']} MB used / {v['total']} MB total ({v['free']} MB free)")
    return v


# ── Core inference ────────────────────────────────────────────────────────

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def load_model() -> Llama:
    """Load the GGUF model with full GPU offloading."""
    print(f"Loading model: {MODEL_PATH}")
    print(f"  n_ctx={N_CTX}, n_gpu_layers={N_GPU_LAYERS}")

    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    vram_before = print_vram("before load")

    start = time.time()
    model = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        verbose=False,
    )
    elapsed = time.time() - start

    vram_after = print_vram("after load")
    model_vram = vram_after["used"] - vram_before["used"]
    print(f"  Loaded in {elapsed:.1f}s (model VRAM: ~{model_vram} MB)")

    return model


def infer(model: Llama, question: str) -> tuple[str, int, float]:
    """Run inference with raw completion for full control over thinking tags.

    Uses manual ChatML formatting (matching Modelfile.think TEMPLATE) instead of
    create_chat_completion(), which mangles <think> tags during auto-detection.
    """
    # ChatML format matching Modelfile.think TEMPLATE exactly
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    start = time.time()
    result = model.create_completion(
        prompt=prompt,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=NUM_PREDICT,
        stop=STOP_TOKENS,
    )
    elapsed = time.time() - start

    content = result["choices"][0]["text"]
    content = strip_thinking(content)

    completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
    tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0

    return content, completion_tokens, tok_per_sec


# ── Benchmark runner ──────────────────────────────────────────────────────

def run_benchmark(model: Llama, questions: list, output_path: str | None = None):
    """Run all questions and print results with tok/s and VRAM tracking."""
    print(f"\n{'=' * 70}")
    print("BENCHMARK — llama-cpp-python (CUDA) | Modelfile.think params")
    print(f"  temperature={TEMPERATURE}, top_p={TOP_P}, top_k={TOP_K}, "
          f"n_ctx={N_CTX}, num_predict={NUM_PREDICT}")
    print(f"{'=' * 70}")

    results = []
    tok_speeds = []
    total_tokens = 0
    vram_peak = 0

    for i, (question, domain) in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] [{domain.upper()}] {question}")
        print("-" * 50)

        response, tokens, tok_s = infer(model, question)
        print(response)

        vram = get_vram_mb()
        vram_peak = max(vram_peak, vram["used"])
        print(f"\n  ({tokens} tokens, {tok_s:.1f} tok/s | VRAM: {vram['used']} MB)")
        print("-" * 50)

        total_tokens += tokens
        tok_speeds.append(tok_s)
        results.append({
            "question": question,
            "domain": domain,
            "response": response,
            "tokens": tokens,
            "tok_per_sec": round(tok_s, 1),
            "vram_mb": vram["used"],
        })

    avg_tok_s = sum(tok_speeds) / len(tok_speeds) if tok_speeds else 0
    warm_speeds = tok_speeds[1:] if len(tok_speeds) > 1 else tok_speeds
    warm_avg = sum(warm_speeds) / len(warm_speeds) if warm_speeds else 0

    vram_final = print_vram("after benchmark")

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"  Questions:          {len(questions)}")
    print(f"  Total tokens:       {total_tokens}")
    print(f"  Avg tok/s (all):    {avg_tok_s:.1f}")
    print(f"  Avg tok/s (warm):   {warm_avg:.1f}")
    print(f"  Ollama baseline:    67.0 tok/s")
    if warm_avg > 0:
        print(f"  Speedup vs Ollama:  {warm_avg / 67.0:.2f}x")
    print(f"\n  VRAM peak:          {vram_peak} MB")
    print(f"  VRAM final:         {vram_final['used']} MB / {vram_final['total']} MB")
    print(f"  VRAM headroom:      {vram_final['free']} MB free")

    # Per-domain breakdown
    domain_speeds: dict[str, list[float]] = {}
    for r in results:
        domain_speeds.setdefault(r["domain"], []).append(r["tok_per_sec"])

    print(f"\n  Per-domain avg tok/s:")
    for domain, speeds in sorted(domain_speeds.items()):
        davg = sum(speeds) / len(speeds)
        print(f"    {domain:12s}: {davg:.1f} tok/s  ({len(speeds)} questions)")

    print(f"{'=' * 70}")

    if output_path:
        output_data = {
            "backend": "llama-cpp-python",
            "model": MODEL_PATH,
            "params": {
                "n_ctx": N_CTX,
                "n_gpu_layers": N_GPU_LAYERS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "num_predict": NUM_PREDICT,
                "stop": STOP_TOKENS,
            },
            "avg_tok_s": round(avg_tok_s, 1),
            "warm_avg_tok_s": round(warm_avg, 1),
            "vram_peak_mb": vram_peak,
            "results": results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")

    return results, warm_avg


def main():
    parser = argparse.ArgumentParser(description="Test llama-cpp-python inference")
    parser.add_argument("--benchmark", action="store_true", help="Run all 44 questions")
    parser.add_argument("--quick", action="store_true", help="Run 3 questions only")
    parser.add_argument("--output", default=None, help="Save results to JSON")
    args = parser.parse_args()

    model = load_model()

    if args.quick:
        questions = TEST_QUESTIONS[:3]
    elif args.benchmark:
        questions = TEST_QUESTIONS
    else:
        questions = TEST_QUESTIONS

    run_benchmark(model, questions, args.output)


if __name__ == "__main__":
    main()
