import pytest

from nl_probes.configs.launch_args import (
    DatasetFamilyFlags,
    compose_wandb_suffix,
    enabled_family_tokens,
    parse_eval_launch_args,
    parse_launch_args,
    target_activation_source,
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


def test_eval_steps_default_and_override():
    assert parse_launch_args([]).eval_steps == 2000
    flags = parse_launch_args(["--eval-steps", "5000"])
    assert flags.eval_steps == 5000
    assert "eval_steps" not in flags.as_dict()


@pytest.mark.parametrize("interval", ["0", "-1"])
def test_eval_steps_must_be_positive(interval):
    with pytest.raises(ValueError, match="--eval-steps must be a positive integer"):
        parse_launch_args(["--eval-steps", interval])


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
    assert not flags.target_activation_diff
    assert target_activation_source(flags) == "target_lora"


def test_deepstack_injection_is_off_by_default():
    flags = parse_launch_args([])
    assert flags.deepstack_coefficient_init == 1.0
    assert not flags.deepstack_injection
    assert not flags.train_deepstack_coefficients
    assert "deepstack" not in enabled_family_tokens(flags)
    assert "dscoef" not in enabled_family_tokens(flags)


@pytest.mark.parametrize("init_value", [0.0, 0.5, 1.0])
def test_deepstack_coefficient_init_override(init_value):
    flags = parse_launch_args([
        "--deepstack-injection",
        "--deepstack-trainable-coefficients",
        "--deepstack-coefficient-init", str(init_value),
    ])
    assert flags.deepstack_coefficient_init == init_value
    assert "deepstack_coefficient_init" not in flags.as_dict()


def test_deepstack_injection_flag_and_wandb_token():
    flags = parse_launch_args(["--deepstack-injection"])
    assert flags.deepstack_injection
    assert not flags.train_deepstack_coefficients
    assert enabled_family_tokens(flags)[-1] == "deepstack"
    assert compose_wandb_suffix(flags, "Qwen/Qwen3-VL-4B-Instruct").endswith(
        "_deepstack_Qwen3-VL-4B-Instruct"
    )

    eval_args = parse_eval_launch_args(
        ["--lora-path", "logs/run/checkpoints/final", "--deepstack-injection"]
    )
    assert eval_args.dataset_flags.deepstack_injection
    assert "deepstack" in enabled_family_tokens(eval_args.dataset_flags)


def test_deepstack_trainable_coefficients_flag_and_wandb_token():
    flags = parse_launch_args(["--deepstack-injection", "--deepstack-trainable-coefficients"])
    assert flags.train_deepstack_coefficients
    assert enabled_family_tokens(flags)[-2:] == ["deepstack", "dscoef"]
    assert compose_wandb_suffix(flags, "Qwen/Qwen3-VL-4B-Instruct").endswith(
        "_deepstack_dscoef_Qwen3-VL-4B-Instruct"
    )

    eval_args = parse_eval_launch_args(
        [
            "--lora-path",
            "logs/run/checkpoints/final",
            "--deepstack-injection",
            "--deepstack-trainable-coefficients",
        ]
    )
    assert eval_args.dataset_flags.train_deepstack_coefficients
    assert enabled_family_tokens(eval_args.dataset_flags)[-2:] == ["deepstack", "dscoef"]


def test_deepstack_trainable_coefficients_requires_injection():
    with pytest.raises(
        ValueError, match="--deepstack-trainable-coefficients requires --deepstack-injection"
    ):
        parse_launch_args(["--deepstack-trainable-coefficients"])

    with pytest.raises(
        ValueError, match="--deepstack-trainable-coefficients requires --deepstack-injection"
    ):
        parse_eval_launch_args(
            ["--lora-path", "logs/run/checkpoints/final", "--deepstack-trainable-coefficients"]
        )


def test_target_activation_diff_is_parsed_for_train_and_eval():
    train_flags = parse_launch_args(["--target-activation-diff"])
    assert train_flags.target_activation_diff
    assert target_activation_source(train_flags) == "adapter_base_diff"
    assert "adiff" in enabled_family_tokens(train_flags)

    eval_args = parse_eval_launch_args(
        ["--lora-path", "logs/run/checkpoints/final", "--target-activation-diff"]
    )
    assert eval_args.dataset_flags.target_activation_diff
    assert target_activation_source(eval_args.dataset_flags) == "adapter_base_diff"
    assert "adiff" in enabled_family_tokens(eval_args.dataset_flags)
