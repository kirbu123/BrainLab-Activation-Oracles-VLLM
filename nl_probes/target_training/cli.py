"""Command-line entry point for visual target-organism training."""

from __future__ import annotations

import argparse
import os

from nl_probes.configs.target_sft_config import (
    TARGET_MODEL_NAME,
    TARGET_MODEL_REVISION,
    TARGET_ORGANISMS,
    TargetSFTConfig,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a bf16 Qwen3-VL PEFT target-organism adapter"
    )
    parser.add_argument("--organism", required=True, choices=TARGET_ORGANISMS)
    parser.add_argument("--organism-id", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl")
    parser.add_argument("--image-root")
    parser.add_argument("--model-name", default=TARGET_MODEL_NAME)
    parser.add_argument("--model-revision", default=TARGET_MODEL_REVISION)

    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Positive value overrides epochs; use 1 for a smoke run",
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=10_000)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)

    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--resume-adapter")
    parser.add_argument("--resume-from-checkpoint")

    parser.add_argument("--logs-root", default="logs/target_training")
    parser.add_argument("--targets-root", default="targets")
    parser.add_argument(
        "--adapter-registry",
        default="data/val/target_organisms/adapter_registry.json",
    )
    parser.add_argument("--run-id", default=os.environ.get("TARGET_RUN_ID", ""))
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--target-dir", default="")
    parser.add_argument("--report-to", default="none", choices=("none", "tensorboard", "wandb"))
    return parser


def parse_config(argv: list[str] | None = None) -> TargetSFTConfig:
    args = build_parser().parse_args(argv)
    config = TargetSFTConfig(
        organism=args.organism,
        organism_id=args.organism_id,
        train_jsonl=args.train_jsonl,
        eval_jsonl=args.eval_jsonl,
        model_name=args.model_name,
        model_revision=args.model_revision,
        image_root=args.image_root,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        max_pixels=args.max_pixels,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        resume_adapter=args.resume_adapter,
        resume_from_checkpoint=args.resume_from_checkpoint,
        logs_root=args.logs_root,
        targets_root=args.targets_root,
        adapter_registry=args.adapter_registry,
        run_id=args.run_id,
        run_dir=args.run_dir,
        target_dir=args.target_dir,
        report_to=args.report_to,
    )
    return config.finalize()


def main(argv: list[str] | None = None) -> None:
    from nl_probes.target_training.trainer import train_target

    train_target(parse_config(argv))
