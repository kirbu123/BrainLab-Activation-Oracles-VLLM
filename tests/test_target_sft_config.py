import json
from pathlib import Path
import re

import pytest

from nl_probes.configs.target_sft_config import (
    TARGET_LORA_PATTERN,
    TARGET_LORA_MODULES,
    TARGET_MODEL_NAME,
    TARGET_MODEL_REVISION,
    TargetSFTConfig,
)
from nl_probes.target_training.cli import parse_config
from nl_probes.target_training.trainer import (
    create_run_directories,
    update_adapter_registry,
)


def test_target_config_is_pinned_and_timestamped(tmp_path):
    config = TargetSFTConfig(
        organism="visual_taboo",
        organism_id="taboo-cat",
        train_jsonl="train.jsonl",
        logs_root=str(tmp_path / "logs"),
        targets_root=str(tmp_path / "targets"),
    ).finalize(timestamp="20260824_040000")

    assert config.model_name == TARGET_MODEL_NAME
    assert config.model_revision == TARGET_MODEL_REVISION
    assert config.lora_target_modules == TARGET_LORA_MODULES
    assert Path(config.run_dir) == tmp_path / "logs/20260824_040000_visual_taboo_taboo-cat"
    assert Path(config.target_dir) == (
        tmp_path / "targets/visual_taboo/20260824_040000_visual_taboo_taboo-cat"
    )


def test_target_config_rejects_model_and_resume_ambiguity():
    with pytest.raises(ValueError, match="pinned"):
        TargetSFTConfig(
            organism="visual_ssc",
            organism_id="visual-ssc-shared-codebook",
            train_jsonl="train.jsonl",
            model_name="another/model",
        ).validate()

    with pytest.raises(ValueError, match="immutable snapshot"):
        TargetSFTConfig(
            organism="visual_ssc",
            organism_id="visual-ssc-shared-codebook",
            train_jsonl="train.jsonl",
            model_revision="main",
        ).validate()

    with pytest.raises(ValueError, match="either resume_adapter"):
        TargetSFTConfig(
            organism="visual_ssc",
            organism_id="visual-ssc-shared-codebook",
            train_jsonl="train.jsonl",
            resume_adapter="adapter",
            resume_from_checkpoint="checkpoint",
        ).validate()


def test_lora_pattern_includes_merger_but_excludes_vision_encoder():
    assert re.fullmatch(
        TARGET_LORA_PATTERN,
        "model.language_model.layers.3.self_attn.q_proj",
    )
    assert re.fullmatch(TARGET_LORA_PATTERN, "model.visual.merger.linear_fc1")
    assert not re.fullmatch(TARGET_LORA_PATTERN, "model.visual.blocks.3.mlp.linear_fc1")


def test_cli_supports_smoke_steps_and_explicit_run_directories(tmp_path):
    config = parse_config(
        [
            "--organism",
            "visual_personaqa",
            "--organism-id",
            "visual-personaqa-shuffled",
            "--train-jsonl",
            "train.jsonl",
            "--max-steps",
            "1",
            "--run-id",
            "smoke_001",
            "--logs-root",
            str(tmp_path / "logs"),
            "--targets-root",
            str(tmp_path / "targets"),
        ]
    )
    run_dir, target_dir = create_run_directories(config)

    assert config.max_steps == 1
    assert run_dir.is_dir()
    assert target_dir.is_dir()
    assert run_dir.name == "smoke_001_visual_personaqa_visual-personaqa-shuffled"
    assert target_dir.name == "smoke_001_visual_personaqa_visual-personaqa-shuffled"


def test_completed_target_updates_registry_idempotently(tmp_path):
    config = TargetSFTConfig(
        organism="visual_taboo",
        organism_id="taboo-cat",
        train_jsonl="train.jsonl",
        logs_root=str(tmp_path / "logs"),
        targets_root=str(tmp_path / "targets"),
        adapter_registry=str(tmp_path / "adapter_registry.json"),
    ).finalize(timestamp="20260824_040000")
    Path(config.target_dir).mkdir(parents=True)

    update_adapter_registry(config)
    update_adapter_registry(config)

    registry = json.loads(Path(config.adapter_registry).read_text())
    assert registry["base_model"] == TARGET_MODEL_NAME
    assert registry["base_revision"] == TARGET_MODEL_REVISION
    assert registry["adapters"] == [
        {
            "adapter_path": str(Path(config.target_dir).resolve()),
            "enabled": True,
            "family": "visual_taboo",
            "organism_id": "taboo-cat",
        }
    ]
