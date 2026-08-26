"""Configuration for Visual Target Organism adapter training."""

from __future__ import annotations

import datetime
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TARGET_MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
TARGET_MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
MOVING_MODEL_REVISIONS = frozenset({"main", "master", "latest"})
TARGET_ORGANISMS = (
    "visual_taboo",
    "visual_user_attribute",
    "visual_ssc",
    "visual_personaqa",
)
LANGUAGE_LORA_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
MERGER_LORA_MODULES = ("linear_fc1", "linear_fc2")
TARGET_LORA_MODULES = LANGUAGE_LORA_MODULES + MERGER_LORA_MODULES
TARGET_LORA_PATTERN = (
    r"model\.(?:"
    r"language_model\.layers\.\d+\.(?:"
    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj)"
    r")|"
    r"visual\.merger\.(?:linear_fc1|linear_fc2)"
    r")"
)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not slug:
        raise ValueError(f"Cannot create an artifact name from {value!r}")
    return slug


@dataclass
class TargetSFTConfig:
    """Strict, serializable settings for Qwen3-VL target LoRA training."""

    organism: str
    organism_id: str
    train_jsonl: str
    eval_jsonl: str | None = None
    model_name: str = TARGET_MODEL_NAME
    model_revision: str = TARGET_MODEL_REVISION
    image_root: str | None = None

    num_train_epochs: float = 3.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_length: int = 2048
    max_pixels: int = 1280 * 28 * 28
    gradient_checkpointing: bool = True
    seed: int = 42
    logging_steps: int = 1
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 2
    dataloader_num_workers: int = 0

    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = TARGET_LORA_MODULES

    resume_adapter: str | None = None
    resume_from_checkpoint: str | None = None
    logs_root: str = "logs/target_training"
    targets_root: str = "targets"
    adapter_registry: str = "data/val/target_organisms/adapter_registry.json"
    run_id: str = ""
    run_dir: str = ""
    target_dir: str = ""
    report_to: str = "none"

    def validate(self, check_paths: bool = False) -> None:
        if self.model_name != TARGET_MODEL_NAME:
            raise ValueError(
                f"Target training is pinned to {TARGET_MODEL_NAME}; got {self.model_name!r}"
            )
        if not self.model_revision:
            raise ValueError("model_revision must be non-empty")
        if self.model_revision.casefold() in MOVING_MODEL_REVISIONS:
            raise ValueError(
                "model_revision must be an immutable snapshot; "
                f"got {self.model_revision!r}"
            )
        if self.organism not in TARGET_ORGANISMS:
            raise ValueError(
                f"Unknown organism {self.organism!r}; expected one of {TARGET_ORGANISMS}"
            )
        if not self.organism_id:
            raise ValueError("organism_id must be non-empty")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be -1 or a positive integer")
        if self.max_length <= 0 or self.max_pixels <= 0:
            raise ValueError("max_length and max_pixels must be positive")
        for name in (
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "logging_steps",
            "save_steps",
            "eval_steps",
            "save_total_limit",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.resume_adapter and self.resume_from_checkpoint:
            raise ValueError(
                "Use either resume_adapter (adapter weights only) or "
                "resume_from_checkpoint (adapter plus Trainer state), not both"
            )
        if tuple(self.lora_target_modules) != TARGET_LORA_MODULES:
            raise ValueError(
                "lora_target_modules is fixed to language projections plus the "
                "Qwen3-VL multimodal merger"
            )
        if check_paths:
            train_path = Path(self.train_jsonl)
            if not train_path.is_file():
                raise FileNotFoundError(f"Training JSONL not found: {train_path}")
            if self.eval_jsonl and not Path(self.eval_jsonl).is_file():
                raise FileNotFoundError(f"Evaluation JSONL not found: {self.eval_jsonl}")
            resume_path = self.resume_adapter or self.resume_from_checkpoint
            if resume_path and not Path(resume_path).is_dir():
                raise FileNotFoundError(f"Resume directory not found: {resume_path}")

    def finalize(self, timestamp: str | None = None) -> "TargetSFTConfig":
        self.validate()
        if not self.run_id:
            self.run_id = timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = (
            f"{_slug(self.run_id)}_{_slug(self.organism)}_{_slug(self.organism_id)}"
        )
        if not self.run_dir:
            self.run_dir = str(Path(self.logs_root) / run_name)
        if not self.target_dir:
            self.target_dir = str(Path(self.targets_root) / self.organism / run_name)
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
