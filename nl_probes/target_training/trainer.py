"""Deterministic distributed PEFT training for visual target organisms."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments
from transformers.trainer_utils import set_seed

from nl_probes.configs.target_sft_config import (
    LANGUAGE_LORA_MODULES,
    MERGER_LORA_MODULES,
    TARGET_LORA_PATTERN,
    TargetSFTConfig,
)
from nl_probes.target_training.collator import Qwen3VLAssistantOnlyCollator
from nl_probes.target_training.data import TargetConversationDataset, load_target_jsonl


def _is_world_process_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_run_directories(config: TargetSFTConfig) -> tuple[Path, Path]:
    """Create the timestamped log and final-adapter directories."""

    config.finalize()
    run_dir = Path(config.run_dir)
    target_dir = Path(config.target_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, target_dir


def write_run_artifacts(
    config: TargetSFTConfig,
    train_dataset: TargetConversationDataset,
    eval_dataset: TargetConversationDataset | None,
) -> None:
    """Persist resolved settings and source hashes before model loading."""

    run_dir = Path(config.run_dir)
    target_dir = Path(config.target_dir)
    config_json = json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n"
    (run_dir / "config.json").write_text(config_json, encoding="utf-8")
    (target_dir / "config.json").write_text(config_json, encoding="utf-8")

    manifest: dict[str, Any] = {
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "organism": config.organism,
        "organism_id": config.organism_id,
        "train": {
            "path": str(train_dataset.source_path.resolve()),
            "examples": len(train_dataset),
            "sha256": _sha256(str(train_dataset.source_path)),
        },
        "eval": None,
    }
    if eval_dataset is not None:
        manifest["eval"] = {
            "path": str(eval_dataset.source_path.resolve()),
            "examples": len(eval_dataset),
            "sha256": _sha256(str(eval_dataset.source_path)),
        }
    manifest_json = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (run_dir / "data_manifest.json").write_text(manifest_json, encoding="utf-8")
    (target_dir / "data_manifest.json").write_text(manifest_json, encoding="utf-8")


def update_adapter_registry(config: TargetSFTConfig) -> Path:
    registry_path = Path(config.adapter_registry)
    if registry_path.is_file():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if set(registry) != {"schema_version", "base_model", "base_revision", "adapters"}:
            raise ValueError(f"Unexpected adapter registry keys: {sorted(registry)}")
        if registry["base_model"] != config.model_name:
            raise ValueError("Adapter registry base_model differs from target config")
        if registry["base_revision"] != config.model_revision:
            raise ValueError("Adapter registry base_revision differs from target config")
    else:
        registry = {
            "schema_version": 1,
            "base_model": config.model_name,
            "base_revision": config.model_revision,
            "adapters": [],
        }
    if not isinstance(registry["adapters"], list):
        raise TypeError("Adapter registry adapters must be a list")
    registry["adapters"] = [
        entry
        for entry in registry["adapters"]
        if (entry["family"], entry["organism_id"])
        != (config.organism, config.organism_id)
    ]
    registry["adapters"].append(
        {
            "family": config.organism,
            "organism_id": config.organism_id,
            "adapter_path": str(Path(config.target_dir).resolve()),
            "enabled": True,
        }
    )
    registry["adapters"].sort(key=lambda entry: (entry["family"], entry["organism_id"]))
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = registry_path.with_suffix(registry_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(registry_path)
    return registry_path


def _assert_bf16_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Target training requires CUDA; CPU training is intentionally unsupported")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Target training requires a CUDA device with native bfloat16 support")


def _new_lora_config(config: TargetSFTConfig) -> LoraConfig:
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_LORA_PATTERN,
    )


def validate_trainable_lora(model: torch.nn.Module) -> None:
    """Require language and merger adapters, with no trainable vision encoder."""

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("PEFT produced no trainable parameters")
    non_lora = [name for name in trainable if "lora_" not in name]
    if non_lora:
        raise RuntimeError(f"Only LoRA parameters may be trainable; found {non_lora[:5]}")

    language = [
        name
        for name in trainable
        if "language_model" in name and any(module in name for module in LANGUAGE_LORA_MODULES)
    ]
    merger = [
        name
        for name in trainable
        if ".visual.merger." in name and any(module in name for module in MERGER_LORA_MODULES)
    ]
    vision_encoder = [name for name in trainable if ".visual.blocks." in name]
    if not language:
        raise RuntimeError("No language-model LoRA parameters were created")
    if not merger:
        raise RuntimeError("No multimodal-merger LoRA parameters were created")
    if vision_encoder:
        raise RuntimeError(f"Vision encoder must remain frozen; found {vision_encoder[:5]}")


def _load_model(config: TargetSFTConfig) -> torch.nn.Module:
    base_model = AutoModelForImageTextToText.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    base_model.config.use_cache = False

    adapter_path = config.resume_adapter or config.resume_from_checkpoint
    if adapter_path:
        peft_config = PeftConfig.from_pretrained(adapter_path)
        if peft_config.base_model_name_or_path != config.model_name:
            raise ValueError(
                f"Adapter base model is {peft_config.base_model_name_or_path!r}, "
                f"expected {config.model_name!r}"
            )
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
    else:
        model = get_peft_model(base_model, _new_lora_config(config))

    if config.gradient_checkpointing:
        model.enable_input_require_grads()
    validate_trainable_lora(model)
    return model


def build_training_arguments(config: TargetSFTConfig, has_eval: bool) -> TrainingArguments:
    """Translate the stable public config into Transformers arguments."""

    return TrainingArguments(
        output_dir=str(Path(config.run_dir) / "checkpoints"),
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        tf32=True,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        full_determinism=True,
        seed=config.seed,
        data_seed=config.seed,
        logging_dir=str(Path(config.run_dir) / "tensorboard"),
        logging_steps=config.logging_steps,
        logging_first_step=True,
        report_to=config.report_to,
        run_name=f"{config.organism}_{config.run_id}",
        eval_strategy="steps" if has_eval else "no",
        eval_steps=config.eval_steps if has_eval else None,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_drop_last=False,
    )


def train_target(config: TargetSFTConfig) -> dict[str, float]:
    """Run DDP-safe target training and save a resumable final PEFT adapter."""

    config.finalize()
    config.validate(check_paths=True)
    _assert_bf16_cuda()
    set_seed(config.seed, deterministic=True)
    run_dir, _ = create_run_directories(config)
    run_logger = logging.getLogger(f"target_training.{config.run_id}.{config.organism_id}")
    if _is_world_process_zero():
        run_logger.setLevel(logging.INFO)
        run_logger.propagate = False
        handler = logging.FileHandler(run_dir / "training.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        run_logger.addHandler(handler)
        run_logger.info("Starting target training: %s", json.dumps(config.as_dict(), sort_keys=True))

    train_dataset = load_target_jsonl(
        config.train_jsonl,
        config.image_root,
        organism_id=config.organism_id,
    )
    eval_dataset = (
        load_target_jsonl(
            config.eval_jsonl,
            config.image_root,
            organism_id=config.organism_id,
        )
        if config.eval_jsonl
        else None
    )
    if _is_world_process_zero():
        write_run_artifacts(config, train_dataset, eval_dataset)

    processor = AutoProcessor.from_pretrained(
        config.model_name,
        revision=config.model_revision,
    )
    collator = Qwen3VLAssistantOnlyCollator(
        processor=processor,
        max_length=config.max_length,
        max_pixels=config.max_pixels,
    )
    model = _load_model(config)
    trainer = Trainer(
        model=model,
        args=build_training_arguments(config, has_eval=eval_dataset is not None),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
    )
    result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    metrics = {name: float(value) for name, value in result.metrics.items()}
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    if trainer.is_world_process_zero():
        trainer.save_model(config.target_dir)
        processor.save_pretrained(config.target_dir)
        registry_path = update_adapter_registry(config)
        (run_dir / "train_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        run_logger.info("Saved final adapter: %s", config.target_dir)
        run_logger.info("Updated adapter registry: %s", registry_path)
        for handler in run_logger.handlers:
            handler.close()
        run_logger.handlers.clear()
    return metrics
