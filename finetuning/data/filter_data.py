"""
Quality filtering pipeline for training data.

Filters:
  1. Remove too-short responses (< 50 chars)
  2. Remove too-long responses (> 4096 tokens approx)
  3. Deduplicate by instruction text
  4. Validate shell commands with shlex
  5. Flag destructive commands — keep but verify warnings are present
  6. Remove entries with hardcoded credentials/IPs

Usage:
  python filter_data.py --input raw/combined_raw.jsonl --output output/
"""

import argparse
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"

# Patterns that indicate destructive commands
DESTRUCTIVE_PATTERNS = [
    r"\brm\s+(-[rRf]+\s+|--recursive|--force)",
    r"\bdd\b\s+.*\bof=",
    r"\bmkfs\b",
    r"\bfdisk\b",
    r"\bparted\b.*\brm\b",
    r"\biptables\s+-F\b",
    r"\bnft\s+flush\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bsystemctl\s+(stop|disable|mask)\b",
    r"\bkill\s+-9\b",
    r"\bchmod\s+777\b",
    r"\b>\s*/dev/sd[a-z]",
    r"\bformat\b",
]

# Patterns that suggest leaked credentials
CREDENTIAL_PATTERNS = [
    r"password\s*=\s*['\"][^'\"]{4,}['\"]",
    r"api[_-]?key\s*=\s*['\"][^'\"]{8,}['\"]",
    r"secret\s*=\s*['\"][^'\"]{8,}['\"]",
    r"token\s*=\s*['\"][^'\"]{8,}['\"]",
    r"\b(?:sk|pk)[-_][a-zA-Z0-9]{20,}\b",  # API key patterns
    r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b(?!.*(?:0\.0\.0\.0|127\.0\.0\.1|localhost|192\.168|10\.|172\.(1[6-9]|2[0-9]|3[01])))",  # public IPs (allow private)
]

WARNING_KEYWORDS = ["warning", "caution", "careful", "dangerous", "destructive", "irreversible", "data loss"]


def get_assistant_content(example: dict) -> str:
    """Extract the assistant's response text."""
    for msg in example.get("messages", []):
        if msg["role"] == "assistant":
            return msg["content"]
    return ""


def get_user_content(example: dict) -> str:
    """Extract the user's instruction text."""
    for msg in example.get("messages", []):
        if msg["role"] == "user":
            return msg["content"]
    return ""


def extract_code_blocks(text: str) -> list[str]:
    """Extract code from ```bash ... ``` blocks."""
    blocks = re.findall(r"```(?:bash|sh)?\s*\n(.*?)```", text, re.DOTALL)
    return [b.strip() for b in blocks if b.strip()]


def is_valid_shell(cmd: str) -> bool:
    """Check if a command is syntactically parseable."""
    try:
        # Handle pipes, redirections
        # shlex can parse simple commands
        shlex.split(cmd)
        return True
    except ValueError:
        # Unmatched quotes, etc.
        return False


def has_destructive_cmd(text: str) -> bool:
    """Check if text contains potentially destructive commands."""
    for pattern in DESTRUCTIVE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def has_warning(text: str) -> bool:
    """Check if the response includes safety warnings."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in WARNING_KEYWORDS)


def has_credentials(text: str) -> bool:
    """Check if text contains potential credentials."""
    for pattern in CREDENTIAL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def instruction_hash(text: str) -> str:
    """Create a normalized hash for deduplication."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.md5(normalized.encode()).hexdigest()


def filter_dataset(input_path: Path, output_dir: Path):
    """Run the full filtering pipeline."""
    print(f"Loading data from {input_path}...")
    examples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples")
    stats = {
        "total_input": len(examples),
        "too_short": 0,
        "too_long": 0,
        "duplicates": 0,
        "credentials": 0,
        "invalid_syntax": 0,
        "destructive_no_warning": 0,
        "destructive_with_warning": 0,
        "passed": 0,
    }

    seen_hashes = set()
    filtered = []
    flagged_destructive = []

    for ex in examples:
        response = get_assistant_content(ex)
        instruction = get_user_content(ex)

        # 1. Too short — only filter truly empty/trivial responses
        # Many valid bash commands are short (e.g. "chmod +x *.sh" = 28 chars)
        if len(response) < 10:
            stats["too_short"] += 1
            continue

        # 2. Too long (rough estimate: 1 token ≈ 4 chars)
        if len(response) > 16384:
            stats["too_long"] += 1
            continue

        # 3. Deduplicate
        h = instruction_hash(instruction)
        if h in seen_hashes:
            stats["duplicates"] += 1
            continue
        seen_hashes.add(h)

        # 4. Credentials check
        if has_credentials(response):
            stats["credentials"] += 1
            continue

        # 5. Shell syntax validation (for code blocks)
        code_blocks = extract_code_blocks(response)
        if code_blocks:
            valid_count = sum(1 for cmd in code_blocks if is_valid_shell(cmd))
            if valid_count == 0 and len(code_blocks) > 0:
                # All code blocks invalid — skip, but only if they look like shell
                # (kernel C code won't parse as shell, and that's fine)
                if not any(kw in response for kw in ["#include", "struct", "void", "int main"]):
                    stats["invalid_syntax"] += 1
                    continue

        # 6. Destructive command check
        if has_destructive_cmd(response):
            if has_warning(response):
                stats["destructive_with_warning"] += 1
                filtered.append(ex)
            else:
                stats["destructive_no_warning"] += 1
                flagged_destructive.append(ex)
                # Still include, but flag for review
                filtered.append(ex)
            continue

        stats["passed"] += 1
        filtered.append(ex)

    # Save filtered output
    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(filtered, output_dir / "filtered.jsonl")

    if flagged_destructive:
        save_jsonl(flagged_destructive, output_dir / "flagged_destructive.jsonl")
        print(f"\n  REVIEW NEEDED: {len(flagged_destructive)} destructive examples lack warnings")
        print(f"  See: {output_dir / 'flagged_destructive.jsonl'}")

    # Print stats
    print(f"\n{'='*60}")
    print("Filtering Results:")
    print(f"{'='*60}")
    for key, val in stats.items():
        print(f"  {key:30s}: {val}")
    print(f"  {'output_total':30s}: {len(filtered)}")
    print(f"{'='*60}")

    return filtered


def split_dataset(data: list[dict], output_dir: Path, val_ratio: float = 0.1):
    """Split into train/validation sets."""
    import random
    random.seed(42)
    random.shuffle(data)

    split_idx = int(len(data) * (1 - val_ratio))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    save_jsonl(train_data, output_dir / "train.jsonl")
    save_jsonl(val_data, output_dir / "val.jsonl")

    print(f"\nSplit: {len(train_data)} train / {len(val_data)} validation")


def save_jsonl(data: list[dict], filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} examples to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Filter and split training data")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--split", action="store_true", help="Also split into train/val")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio (default: 0.1)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    filtered = filter_dataset(Path(args.input), output_dir)

    if args.split:
        split_dataset(filtered, output_dir, args.val_ratio)


if __name__ == "__main__":
    main()
