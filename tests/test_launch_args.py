import pytest

from nl_probes.configs.launch_args import (
    DatasetFamilyFlags,
    compose_wandb_suffix,
    enabled_family_tokens,
    parse_launch_args,
    target_validation_enabled,
    validation_enabled,
)


def test_dataset_families_are_enabled_by_default():
    flags = parse_launch_args([])

    assert flags == DatasetFamilyFlags()
    assert not flags.visual_taboo_val
    assert not flags.visual_user_attribute_val
    assert not flags.visual_ssc_val
    assert not flags.visual_personaqa_val


def test_dataset_families_can_be_disabled_independently():
    flags = parse_launch_args(["--no-visual-spqa", "--no-snli-ve"])

    assert flags == DatasetFamilyFlags(
        visual_spqa=False,
        classification=True,
        context_prediction=True,
        snli_ve=False,
    )


def test_at_least_one_training_family_is_required():
    with pytest.raises(ValueError, match="No training datasets selected"):
        parse_launch_args(
            [
                "--no-visual-spqa",
                "--no-classification",
                "--no-context-prediction",
            ]
        )


def test_wandb_suffix_uses_stable_enabled_family_order():
    flags = DatasetFamilyFlags(
        visual_spqa=True,
        classification=False,
        context_prediction=True,
        snli_ve=False,
    )

    assert (
        compose_wandb_suffix(flags, "Qwen/Qwen3-VL-4B-Instruct")
        == "_visual_spqa_cococtx_Qwen3-VL-4B-Instruct"
    )


def test_visual_spqa_only_launch_has_no_validation_family():
    flags = parse_launch_args(
        [
            "--no-classification",
            "--no-context-prediction",
            "--no-snli-ve",
        ]
    )

    assert flags.visual_spqa
    assert not validation_enabled(flags)


def test_target_validation_flags_and_registry_are_parsed():
    flags = parse_launch_args(
        [
            "--visual-taboo-val",
            "--visual-ssc-val",
            "--target-adapter-registry",
            "fixtures/registry.json",
        ]
    )

    assert flags.visual_taboo_val
    assert flags.visual_ssc_val
    assert validation_enabled(flags)
    assert target_validation_enabled(flags)
    assert flags.target_adapter_registry == "fixtures/registry.json"
    assert "vtaboo" in enabled_family_tokens(flags)
    assert "vssc" in enabled_family_tokens(flags)
