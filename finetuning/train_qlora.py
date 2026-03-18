"""
QLoRA fine-tuning for Qwen3.5-4B using Unsloth (OPTIONAL try-first).

Unsloth provides ~2x speedup and ~60% less VRAM, but may not support
Qwen3.5's hybrid architecture (Gated DeltaNet + Sparse MoE) yet.
If this script fails, use train_trl.py instead — same config, same CLI.

Usage:
  python train_qlora.py                                # use config defaults
  python train_qlora.py --epochs 1 --max-steps 1       # 1-step dry run
  python train_qlora.py --resume output/qlora-run/checkpoint-400
  python train_qlora.py --export-gguf                  # train + export GGUF
"""

import argparse
import sys
from pathlib import Path

import yaml
from datasets import load_dataset

BASE_DIR = Path(__file__).parent


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply CLI args on top of YAML config. Only non-None values override."""
    overrides = {
        ("model", "name"): args.model,
        ("model", "max_seq_length"): args.max_seq_len,
        ("training", "num_epochs"): args.epochs,
        ("training", "learning_rate"): args.lr,
        ("training", "per_device_train_batch_size"): args.batch_size,
        ("training", "gradient_accumulation_steps"): args.grad_accum,
        ("training", "save_steps"): args.save_steps,
        ("training", "report_to"): "wandb" if args.wandb else None,
        ("output", "dir"): args.output_dir,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            config[section][key] = value

    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps

    if args.resume:
        config["resume_from_checkpoint"] = args.resume

    return config


def resolve_path(relative: str) -> Path:
    """Resolve a path relative to the finetuning/ directory."""
    return BASE_DIR / relative


def build_model_and_tokenizer(config: dict):
    """Load model via Unsloth's FastLanguageModel."""
    from unsloth import FastLanguageModel

    model_name = config["model"]["name"]
    print(f"Loading model via Unsloth: {model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config["model"]["max_seq_length"],
        load_in_4bit=config["quantization"]["load_in_4bit"],
        dtype=None,  # auto-detect
    )

    # Apply LoRA via Unsloth
    lora = config["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias=lora["bias"],
        use_gradient_checkpointing="unsloth",  # 60% less VRAM than HF
    )

    model.print_trainable_parameters()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def load_data(config: dict):
    """Load train and validation JSONL datasets."""
    train_path = resolve_path(config["data"]["train_file"])
    val_path = resolve_path(config["data"]["val_file"])

    if not train_path.exists():
        print(f"ERROR: Training data not found: {train_path}")
        sys.exit(1)
    if not val_path.exists():
        print(f"ERROR: Validation data not found: {val_path}")
        sys.exit(1)

    data = load_dataset("json", data_files={
        "train": str(train_path),
        "validation": str(val_path),
    })

    print(f"Train: {len(data['train'])} examples, Val: {len(data['validation'])} examples")
    return data["train"], data["validation"]


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning with Unsloth")
    parser.add_argument("--config", type=str, default="configs/qlora_config.yaml")
    parser.add_argument("--model", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--max-steps", type=int, help="Override epochs with fixed step count")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--resume", type=str, help="Resume from checkpoint path")
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--export-gguf", action="store_true", help="Export GGUF after training")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    config = apply_cli_overrides(config, args)

    # Load model — this is where it fails if Unsloth doesn't support qwen3_5
    model, tokenizer = build_model_and_tokenizer(config)

    # Load data
    train_data, val_data = load_data(config)

    # Build training args — import here since trl is a dependency of unsloth
    from trl import SFTConfig, SFTTrainer

    t = config["training"]
    output_dir = str(resolve_path(config["output"]["dir"]))

    sft_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=t["num_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        optim=t["optimizer"],
        gradient_checkpointing=t["gradient_checkpointing"],
        bf16=t["bf16"],
        fp16=t["fp16"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        seed=t["seed"],
        report_to=t["report_to"],
        max_seq_length=config["model"]["max_seq_length"],
        packing=False,
        remove_unused_columns=False,
    )

    if "max_steps" in t:
        sft_kwargs["max_steps"] = t["max_steps"]

    sft_config = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
    )

    # Train
    resume = config.get("resume_from_checkpoint")
    print(f"\nStarting training{' (resuming from ' + resume + ')' if resume else ''}...")
    result = trainer.train(resume_from_checkpoint=resume)

    # Save adapter + tokenizer
    out = resolve_path(config["output"]["dir"])
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    # GGUF export (Unsloth native)
    if args.export_gguf:
        gguf_dir = resolve_path(config["output"]["gguf_dir"])
        gguf_quant = config["output"]["gguf_quant"]
        print(f"\nExporting GGUF ({gguf_quant}) to {gguf_dir}...")
        model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=gguf_quant)
        print(f"GGUF saved to: {gguf_dir}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Train loss: {result.training_loss:.4f}")
    print(f"  Steps: {result.global_step}")
    print(f"  Runtime: {result.metrics['train_runtime']:.0f}s")
    print(f"  Adapter saved to: {out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
