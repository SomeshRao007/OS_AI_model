"""
Local validation for training setup — no GPU required.

Checks syntax, config, and data before spending money on Vast.ai.

Usage:
  python validate_setup.py
  python validate_setup.py --verbose
"""

import argparse
import json
import math
import py_compile
import sys
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).parent
REQUIRED_SCRIPTS = ["train_trl.py", "train_qlora.py"]
REQUIRED_CONFIG_KEYS = {
    "model": ["name", "max_seq_length"],
    "lora": ["r", "alpha", "dropout", "target_modules", "bias", "task_type"],
    "quantization": ["load_in_4bit", "bnb_4bit_quant_type"],
    "training": ["num_epochs", "per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate"],
    "data": ["train_file", "val_file"],
    "output": ["dir"],
}


def check_syntax(verbose: bool) -> bool:
    """Compile all training scripts to check for syntax errors."""
    print("\n[1/4] Checking Python syntax...")
    all_ok = True

    for script in REQUIRED_SCRIPTS:
        path = BASE_DIR / script
        if not path.exists():
            print(f"  MISSING: {script}")
            all_ok = False
            continue

        compiled = py_compile.compile(str(path), doraise=False)
        if compiled:
            print(f"  OK: {script}")
        else:
            print(f"  SYNTAX ERROR: {script}")
            all_ok = False

    # Also check this file and any other .py in finetuning/
    for py_file in BASE_DIR.glob("*.py"):
        if py_file.name in REQUIRED_SCRIPTS or py_file.name == "validate_setup.py":
            continue
        compiled = py_compile.compile(str(py_file), doraise=False)
        if verbose:
            status = "OK" if compiled else "SYNTAX ERROR"
            print(f"  {status}: {py_file.name}")

    return all_ok


def check_config(verbose: bool) -> tuple[bool, dict | None]:
    """Load and validate the YAML config."""
    print("\n[2/4] Validating config...")
    config_path = BASE_DIR / "configs" / "qlora_config.yaml"

    if not config_path.exists():
        print(f"  MISSING: {config_path}")
        return False, None

    with open(config_path) as f:
        config = yaml.safe_load(f)

    all_ok = True
    for section, keys in REQUIRED_CONFIG_KEYS.items():
        if section not in config:
            print(f"  MISSING SECTION: {section}")
            all_ok = False
            continue
        for key in keys:
            if key not in config[section]:
                print(f"  MISSING KEY: {section}.{key}")
                all_ok = False

    if not all_ok:
        return False, None

    # Value validation
    lora = config["lora"]
    if lora["r"] <= 0:
        print(f"  INVALID: lora.r must be > 0 (got {lora['r']})")
        all_ok = False
    if lora["alpha"] <= 0:
        print(f"  INVALID: lora.alpha must be > 0 (got {lora['alpha']})")
        all_ok = False
    if not (0 <= lora["dropout"] < 1):
        print(f"  INVALID: lora.dropout must be in [0, 1) (got {lora['dropout']})")
        all_ok = False

    t = config["training"]
    if t["learning_rate"] <= 0:
        print(f"  INVALID: training.learning_rate must be > 0 (got {t['learning_rate']})")
        all_ok = False

    if all_ok:
        print(f"  OK: {config_path.name}")
        if verbose:
            print(f"    Model: {config['model']['name']}")
            print(f"    LoRA: r={lora['r']}, alpha={lora['alpha']}")
            print(f"    Training: {t['num_epochs']} epochs, lr={t['learning_rate']}, batch={t['per_device_train_batch_size']}x{t['gradient_accumulation_steps']}")

    return all_ok, config


def check_data(config: dict, verbose: bool) -> bool:
    """Validate training data files exist and have correct format."""
    print("\n[3/4] Validating data...")
    all_ok = True

    for split, key in [("train", "train_file"), ("val", "val_file")]:
        path = BASE_DIR / config["data"][key]
        if not path.exists():
            print(f"  MISSING: {path}")
            all_ok = False
            continue

        # Count total lines first
        with open(path) as f:
            line_count = sum(1 for _ in f)

        # Validate format (first 100 lines in non-verbose, all in verbose)
        check_limit = line_count if verbose else min(100, line_count)
        errors = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= check_limit:
                    break

                row = json.loads(line)
                if "messages" not in row:
                    if errors == 0:
                        print(f"  FORMAT ERROR in {split} line {i+1}: missing 'messages' key")
                    errors += 1
                    continue

                messages = row["messages"]
                if not isinstance(messages, list) or len(messages) < 2:
                    if errors == 0:
                        print(f"  FORMAT ERROR in {split} line {i+1}: messages must be list with >= 2 entries")
                    errors += 1
                    continue

                valid_roles = {"system", "user", "assistant"}
                for msg in messages:
                    if msg.get("role") not in valid_roles:
                        if errors == 0:
                            print(f"  FORMAT ERROR in {split} line {i+1}: invalid role '{msg.get('role')}'")
                        errors += 1
                        break

        if errors > 0:
            print(f"  {split}: {line_count} examples, {errors} format errors")
            all_ok = False
        else:
            print(f"  OK: {split} — {line_count} examples")

    return all_ok


def check_requirements(verbose: bool) -> bool:
    """Check requirements file exists."""
    print("\n[4/4] Checking requirements...")
    req_path = BASE_DIR / "requirements-train.txt"
    if not req_path.exists():
        print(f"  MISSING: {req_path}")
        return False

    with open(req_path) as f:
        deps = [line.split("#")[0].strip() for line in f if line.strip() and not line.startswith("#")]
        deps = [d for d in deps if d]

    print(f"  OK: {len(deps)} dependencies listed")
    if verbose:
        for dep in deps:
            print(f"    {dep}")
    return True


def print_summary(config: dict):
    """Print what will happen on Vast.ai."""
    t = config["training"]
    train_path = BASE_DIR / config["data"]["train_file"]
    with open(train_path) as f:
        train_count = sum(1 for _ in f)

    effective_batch = t["per_device_train_batch_size"] * t["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(train_count / effective_batch)
    total_steps = steps_per_epoch * t["num_epochs"]

    # TRL estimate: ~1.5 sec/step on 5090 (batch=8), ~3.5 sec/step on 3090 (batch=4)
    est_hours_5090 = (total_steps * 1.5) / 3600
    est_cost_5090 = est_hours_5090 * 0.272

    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"  Model:           {config['model']['name']}")
    print(f"  LoRA:            r={config['lora']['r']}, alpha={config['lora']['alpha']}")
    print(f"  Train examples:  {train_count}")
    print(f"  Effective batch: {effective_batch} ({t['per_device_train_batch_size']} x {t['gradient_accumulation_steps']})")
    print(f"  Steps/epoch:     {steps_per_epoch}")
    print(f"  Total steps:     {total_steps} ({t['num_epochs']} epochs)")
    print(f"  Checkpoints:     every {t['save_steps']} steps (keep {t['save_total_limit']})")
    print(f"  Max seq length:  {config['model']['max_seq_length']}")
    print(f"")
    print(f"  Estimated time (TRL on RTX 5090):  ~{est_hours_5090:.1f} hours")
    print(f"  Estimated cost ($0.272/hr):          ~${est_cost_5090:.2f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Validate training setup locally")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("Validating training setup...")

    syntax_ok = check_syntax(args.verbose)
    config_ok, config = check_config(args.verbose)
    data_ok = check_data(config, args.verbose) if config else False
    req_ok = check_requirements(args.verbose)

    all_ok = syntax_ok and config_ok and data_ok and req_ok

    if config:
        print_summary(config)

    if all_ok:
        print("\nAll checks passed! Ready for Vast.ai.")
        print("  1. Upload this directory to Vast.ai")
        print("  2. pip install -r requirements-train.txt")
        print("  3. python train_trl.py --epochs 1 --max-steps 1  # dry run")
        print("  4. python train_trl.py                           # full training")
    else:
        print("\nSome checks FAILED. Fix issues above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
