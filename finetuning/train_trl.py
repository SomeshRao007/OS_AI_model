"""
QLoRA fine-tuning for Qwen3.5-4B using TRL + PEFT.

PRIMARY training script — uses HF Transformers directly, guaranteed Qwen3.5 support.

Usage:
  python train_trl.py                                # use config defaults
  python train_trl.py --epochs 1 --max-steps 1       # 1-step dry run
  python train_trl.py --lr 1e-5 --epochs 5           # override hyperparams
  python train_trl.py --resume output/qlora-run/checkpoint-400
  python train_trl.py --wandb                        # enable W&B logging
  python train_trl.py --push-to-hub SomeshRao007/qwen3.5-4b-os-qlora  # upload adapter to HF Hub
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

# Reduce memory fragmentation — recommended by PyTorch for OOM scenarios
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

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
        ("training", "run_name"): args.run_name,
        ("output", "dir"): args.output_dir,
    }

    if args.push_to_hub:
        config.setdefault("hub", {})
        config["hub"]["push_to_hub"] = True
        config["hub"]["repo_id"] = args.push_to_hub
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
    """Load 4-bit quantized model and tokenizer."""
    model_name = config["model"]["name"]
    quant = config["quantization"]

    compute_dtype = torch.bfloat16 if quant["bnb_4bit_compute_dtype"] == "bfloat16" else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant["load_in_4bit"],
        bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Use device_map for single-GPU, let accelerate handle multi-GPU (DDP)
    num_gpus = torch.cuda.device_count()
    device_map = {"": int(os.environ.get("LOCAL_RANK", 0))} if num_gpus > 1 else "auto"
    print(f"  GPUs detected: {num_gpus}, device_map: {device_map}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",   # memory-efficient attention (works on all GPUs)
    )

    # Prepare for QLoRA training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config["training"]["gradient_checkpointing"])

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def apply_lora(model, config: dict):
    """Apply LoRA adapters and print trainable param summary."""
    lora = config["lora"]

    lora_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias=lora["bias"],
        task_type=lora["task_type"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


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


def build_training_args(config: dict) -> SFTConfig:
    """Build SFTConfig from the config dict."""
    t = config["training"]
    output_dir = str(resolve_path(config["output"]["dir"]))

    kwargs = dict(
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
        logging_first_step=t.get("logging_first_step", True),
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        seed=t["seed"],
        report_to=t["report_to"],
        run_name=t.get("run_name"),
        max_length=config["model"]["max_seq_length"],
        packing=True,           # pack short examples to fill 1024-token windows (~10-15x speedup; 99% of examples are <756 tokens so no truncation)
        remove_unused_columns=False,
        dataloader_num_workers=8,  # prefetch batches in parallel to keep GPU fed (96 cores available)
        dataloader_pin_memory=True,  # faster CPU→GPU transfers
    )

    # max_steps overrides num_epochs when set (useful for dry runs)
    if "max_steps" in t:
        kwargs["max_steps"] = t["max_steps"]

    return SFTConfig(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning with TRL+PEFT")
    parser.add_argument("--config", type=str, default="configs/qlora_config.yaml")
    parser.add_argument("--model", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--max-steps", type=int, help="Override epochs with fixed step count (for dry runs)")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--resume", type=str, help="Resume from checkpoint path")
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--run-name", type=str, help="W&B run name")
    parser.add_argument("--push-to-hub", type=str, metavar="REPO_ID",
                        help="Push adapter to HF Hub (e.g. SomeshRao007/qwen3.5-4b-os-qlora)")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    config = apply_cli_overrides(config, args)

    # Load model + tokenizer
    model, tokenizer = build_model_and_tokenizer(config)

    # Apply LoRA
    model = apply_lora(model, config)

    # Load data
    train_data, val_data = load_data(config)

    # Build trainer
    sft_config = build_training_args(config)

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
    output_dir = resolve_path(config["output"]["dir"])
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Push to HF Hub if configured
    hub_config = config.get("hub", {})
    if hub_config.get("push_to_hub") and hub_config.get("repo_id"):
        repo_id = hub_config["repo_id"]
        private = hub_config.get("private", True)
        print(f"\nPushing adapter to HF Hub: {repo_id} (private={private})...")
        model.push_to_hub(repo_id, private=private)
        tokenizer.push_to_hub(repo_id, private=private)
        print(f"Adapter uploaded to: https://huggingface.co/{repo_id}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Train loss: {result.training_loss:.4f}")
    print(f"  Steps: {result.global_step}")
    print(f"  Runtime: {result.metrics['train_runtime']:.0f}s")
    print(f"  Adapter saved to: {output_dir}")
    if hub_config.get("push_to_hub") and hub_config.get("repo_id"):
        print(f"  HF Hub: https://huggingface.co/{hub_config['repo_id']}")
    print(f"\nTo export GGUF, merge the adapter and convert with llama.cpp:")
    print(f"  1. Merge: python -c \"from peft import AutoPeftModelForCausalLM; m = AutoPeftModelForCausalLM.from_pretrained('{output_dir}'); m = m.merge_and_unload(); m.save_pretrained('{output_dir}/merged')\"")
    print(f"  2. Convert: python llama.cpp/convert_hf_to_gguf.py {output_dir}/merged --outtype q5_k_m")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
