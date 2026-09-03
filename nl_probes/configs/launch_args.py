from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass


def _add_target_data_root_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--target-adapter-registry",
        default="data/val/target_organisms/adapter_registry.json",
        help="Path to the target-organism adapter registry",
    )
    parser.add_argument(
        "--target-val-root",
        default="data/val",
        help="Directory containing per-family validation_manifest.json files",
    )
    parser.add_argument(
        "--target-cache-dir",
        default="data/val/cache",
        help="Directory for adapter-on target validation caches",
    )


@dataclass(frozen=True)
class DatasetFamilyFlags:
    visual_spqa: bool = True
    classification: bool = True
    context_prediction: bool = True
    snli_ve: bool = True
    visual_taboo_val: bool = False
    visual_user_attribute_val: bool = False
    visual_ssc_val: bool = False
    visual_personaqa_val: bool = False
    target_adapter_registry: str = "data/val/target_organisms/adapter_registry.json"
    target_val_root: str = "data/val"
    target_cache_dir: str = "data/val/cache"

    def as_dict(self) -> dict[str, bool]:
        return {
            key: value
            for key, value in asdict(self).items()
            if isinstance(value, bool)
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the vision-language Activation Oracle")
    parser.add_argument(
        "--visual-spqa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Visual SPQA training data (default: enabled)",
    )
    parser.add_argument(
        "--classification",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable VSR, GQA yes/no, and COCO object-presence train/validation data (default: enabled)",
    )
    parser.add_argument(
        "--context-prediction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable COCO-caption context-prediction train/validation data (default: enabled)",
    )
    parser.add_argument(
        "--snli-ve",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable SNLI-VE validation data (default: enabled)",
    )
    parser.add_argument(
        "--visual-taboo-val",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable adapter-on Visual Taboo validation (default: disabled)",
    )
    parser.add_argument(
        "--visual-user-attribute-val",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable randomized visual user-attribute validation (default: disabled)",
    )
    parser.add_argument(
        "--visual-ssc-val",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable visual secret-side-constraint validation (default: disabled)",
    )
    parser.add_argument(
        "--visual-personaqa-val",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Visual PersonaQA validation (default: disabled)",
    )
    _add_target_data_root_args(parser)
    return parser


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained vision-language Activation Oracle"
    )
    parser.add_argument("--lora-path", required=True, help="Path to the trained oracle LoRA")
    parser.add_argument(
        "--source-tokens",
        nargs="+",
        choices=["mixed", "text", "visual"],
        default=["mixed", "text", "visual"],
        help="Source-token modes to evaluate (default: mixed text visual)",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Base VLM name matching the oracle LoRA",
    )
    parser.add_argument(
        "--classification",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate VSR, GQA yes/no, and COCO object-presence (default: enabled)",
    )
    parser.add_argument(
        "--context-prediction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate COCO-caption context-prediction (default: enabled)",
    )
    parser.add_argument(
        "--snli-ve",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate SNLI-VE (default: enabled)",
    )
    parser.add_argument(
        "--visual-taboo-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate adapter-on Visual Taboo (default: enabled)",
    )
    parser.add_argument(
        "--visual-user-attribute-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate visual user-attribute (default: enabled)",
    )
    parser.add_argument(
        "--visual-ssc-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate visual secret-side-constraint (default: enabled)",
    )
    parser.add_argument(
        "--visual-personaqa-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate Visual PersonaQA (default: enabled)",
    )
    _add_target_data_root_args(parser)
    return parser


@dataclass(frozen=True)
class OracleModalityEvalArgs:
    lora_path: str
    source_tokens: tuple[str, ...]
    model_name: str
    dataset_flags: DatasetFamilyFlags


def parse_eval_launch_args(argv: list[str] | None = None) -> OracleModalityEvalArgs:
    namespace = build_eval_parser().parse_args(argv)
    flags = DatasetFamilyFlags(
        visual_spqa=False,
        classification=namespace.classification,
        context_prediction=namespace.context_prediction,
        snli_ve=namespace.snli_ve,
        visual_taboo_val=namespace.visual_taboo_val,
        visual_user_attribute_val=namespace.visual_user_attribute_val,
        visual_ssc_val=namespace.visual_ssc_val,
        visual_personaqa_val=namespace.visual_personaqa_val,
        target_adapter_registry=namespace.target_adapter_registry,
        target_val_root=namespace.target_val_root,
        target_cache_dir=namespace.target_cache_dir,
    )
    validate_eval_family_flags(flags)
    modes = tuple(dict.fromkeys(namespace.source_tokens))
    if not modes:
        raise ValueError("At least one --source-tokens mode is required")
    return OracleModalityEvalArgs(
        lora_path=namespace.lora_path,
        source_tokens=modes,
        model_name=namespace.model_name,
        dataset_flags=flags,
    )


def validate_eval_family_flags(flags: DatasetFamilyFlags) -> None:
    if not validation_enabled(flags):
        raise ValueError(
            "No validation datasets selected. Enable at least one of "
            "--classification, --context-prediction, --snli-ve, or a target-organism val flag."
        )


def parse_launch_args(argv: list[str] | None = None) -> DatasetFamilyFlags:
    namespace = build_parser().parse_args(argv)
    flags = DatasetFamilyFlags(
        visual_spqa=namespace.visual_spqa,
        classification=namespace.classification,
        context_prediction=namespace.context_prediction,
        snli_ve=namespace.snli_ve,
        visual_taboo_val=namespace.visual_taboo_val,
        visual_user_attribute_val=namespace.visual_user_attribute_val,
        visual_ssc_val=namespace.visual_ssc_val,
        visual_personaqa_val=namespace.visual_personaqa_val,
        target_adapter_registry=namespace.target_adapter_registry,
        target_val_root=namespace.target_val_root,
        target_cache_dir=namespace.target_cache_dir,
    )
    validate_family_flags(flags)
    return flags


def validate_family_flags(flags: DatasetFamilyFlags) -> None:
    if not (flags.visual_spqa or flags.classification or flags.context_prediction):
        raise ValueError(
            "No training datasets selected. Enable at least one of "
            "--visual-spqa, --classification, or --context-prediction."
        )


def enabled_family_tokens(flags: DatasetFamilyFlags) -> list[str]:
    tokens = []
    if flags.visual_spqa:
        tokens.append("visual_spqa")
    if flags.classification:
        tokens.append("cls")
    if flags.context_prediction:
        tokens.append("cococtx")
    if flags.snli_ve:
        tokens.append("snlive")
    if flags.visual_taboo_val:
        tokens.append("vtaboo")
    if flags.visual_user_attribute_val:
        tokens.append("vuser")
    if flags.visual_ssc_val:
        tokens.append("vssc")
    if flags.visual_personaqa_val:
        tokens.append("vpqa")
    return tokens


def validation_enabled(flags: DatasetFamilyFlags) -> bool:
    return any(
        (
            flags.classification,
            flags.context_prediction,
            flags.snli_ve,
            flags.visual_taboo_val,
            flags.visual_user_attribute_val,
            flags.visual_ssc_val,
            flags.visual_personaqa_val,
        )
    )


def target_validation_enabled(flags: DatasetFamilyFlags) -> bool:
    return any(
        (
            flags.visual_taboo_val,
            flags.visual_user_attribute_val,
            flags.visual_ssc_val,
            flags.visual_personaqa_val,
        )
    )


def compose_wandb_suffix(flags: DatasetFamilyFlags, model_name: str) -> str:
    model_name_str = model_name.split("/")[-1].replace(".", "_").replace(" ", "_")
    return f"_{'_'.join(enabled_family_tokens(flags))}_{model_name_str}"
