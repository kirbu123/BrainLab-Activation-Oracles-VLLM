from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass


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
    parser.add_argument(
        "--target-adapter-registry",
        default="data/val/target_organisms/adapter_registry.json",
        help="Path to the target-organism adapter registry",
    )
    return parser


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
