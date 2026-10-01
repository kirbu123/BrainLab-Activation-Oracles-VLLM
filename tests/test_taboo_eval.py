import os
import subprocess
from pathlib import Path

import pytest

from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.trl_training.taboo_eval import (
    ONE_TO_ONE_EXP,
    build_taboo_jobs,
    dataset_flags_from_training_config,
    default_exps,
    discover_local_taboo_adapters,
    fill_steering_coefficients,
    one_to_one_lora_path,
    parse_taboo_eval_args,
    render_taboo_eval_html,
    resolve_taboo_lora_path,
)
from nl_probes.trl_training.taboo_train import taboo_secret_train_eval_sizes


ROOT = Path(__file__).resolve().parents[1]


def test_one_to_one_hf_path():
    assert one_to_one_lora_path("smile") == "adamkarvonen/Qwen3-8B-taboo-smile_50_mix"


def test_discover_mixes_and_default_exps_include_1to1(tmp_path: Path):
    (tmp_path / "Qwen3-8B-taboo-smile_1to2").mkdir()
    (tmp_path / "Qwen3-8B-taboo-ship_1to2").mkdir()
    (tmp_path / "Qwen3-8B-taboo-smile_1to5").mkdir()
    (tmp_path / "unrelated").mkdir()
    found = discover_local_taboo_adapters(tmp_path)
    assert set(found) == {"1to2", "1to5"}
    assert set(found["1to2"]) == {"smile", "ship"}
    assert default_exps(found)[0] == ONE_TO_ONE_EXP
    assert "1to2" in default_exps(found)
    assert "1to5" in default_exps(found)


def test_parse_val_exps_and_exps():
    args = parse_taboo_eval_args(
        [
            "--val-exps",
            "logs/20260927_173048_attn10_Qwen3-VL-4B-Instruct",
            "logs/other_run",
            "--exps",
            "1to1",
            "1to2",
        ]
    )
    assert args.val_exps == [
        "logs/20260927_173048_attn10_Qwen3-VL-4B-Instruct",
        "logs/other_run",
    ]
    assert args.exps == ["1to1", "1to2"]


def test_resolve_1to1_and_missing_local(tmp_path: Path):
    discovered = discover_local_taboo_adapters(tmp_path)
    assert resolve_taboo_lora_path("1to1", "smile", discovered) == one_to_one_lora_path("smile")
    with pytest.raises(FileNotFoundError, match="No local adapters for mix 1to2"):
        resolve_taboo_lora_path("1to2", "smile", discovered)


def test_build_jobs_smoke_one_word(tmp_path: Path):
    (tmp_path / "Qwen3-8B-taboo-smile_1to2").mkdir()
    jobs = build_taboo_jobs(["1to2"], ["smile", "ship"], tmp_path, smoke=True)
    assert len(jobs) == 1
    assert jobs[0]["word"] == "smile"
    assert jobs[0]["exp"] == "1to2"


def test_dataset_flags_use_cfg_optimize_steering_when_missing_from_families():
    cfg = SelfInterpTrainingConfig(
        dataset_families={
            "visual_spqa": True,
            "classification": True,
            "context_prediction": True,
            "snli_ve": True,
            "visual_taboo_val": True,
            "visual_user_attribute_val": True,
            "visual_ssc_val": True,
            "visual_personaqa_val": True,
            "target_activation_diff": False,
        },
        use_deepstack_injection=False,
        train_deepstack_coefficients=False,
        optimize_steering_coefs=True,
        num_injection_layers=3,
    )
    flags = dataset_flags_from_training_config(cfg)
    assert flags.optimize_steering_coefs is True
    assert flags.deepstack_injection is False
    assert flags.num_injection_layers == 3


def test_fill_steering_coefficients_from_scalar():
    cfg = SelfInterpTrainingConfig(
        steering_coefficient=1.0,
        steering_coefficients=[],
        num_injection_layers=3,
    )
    filled = fill_steering_coefficients(cfg)
    assert filled.steering_coefficients == [1.0, 1.0, 1.0]


def test_eval_split_sizes_match_train_holdout():
    train_size, eval_size = taboo_secret_train_eval_sizes(100, smoke=False)
    assert train_size == 90
    assert eval_size == 10
    train_size, eval_size = taboo_secret_train_eval_sizes(8, smoke=True)
    assert train_size == 7
    assert eval_size == 1


def test_html_renderer_includes_val_exp_and_eval_loss():
    html = render_taboo_eval_html(
        {
            "val_exps": [
                {
                    "run_name": "20260927_173048_attn10_Qwen3-VL-4B-Instruct",
                    "checkpoint": "/tmp/checkpoints/final",
                    "step": 15000,
                    "metrics": {
                        "eval_ans_correct/vsr": 0.61,
                        "eval_format_correct/vsr": 0.99,
                    },
                    "n_by_dataset": {"vsr": 250},
                }
            ],
            "taboo": {
                "sft": [
                    {
                        "exp": "1to1",
                        "word": "smile",
                        "eval_loss": 1.234567,
                        "lora_path": "adamkarvonen/Qwen3-8B-taboo-smile_50_mix",
                    }
                ],
                "open_ended": [
                    {
                        "exp": "1to1",
                        "word": "smile",
                        "verbalizer": "checkpoints_latentqa_only_addition_Qwen3-8B",
                        "accuracy": 0.4,
                    }
                ],
            },
        }
    )
    assert "20260927_173048_attn10_Qwen3-VL-4B-Instruct" in html
    assert "15000" in html
    assert "1to1" in html
    assert "1.234567" in html
    assert "eval_ans_correct/vsr" in html


def test_run_taboo_eval_launcher(tmp_path: Path):
    arguments_path = tmp_path / "arguments.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$FAKE_PYTHON_ARGUMENTS"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/trl/run_taboo_eval.sh"),
            "--exps",
            "1to1",
            "--val-exps",
            "logs/run_a",
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "TABOO_PYTHON": str(fake_python),
            "FAKE_PYTHON_ARGUMENTS": str(arguments_path),
        },
        check=True,
    )
    arguments = arguments_path.read_text(encoding="utf-8").splitlines()
    assert arguments[0].endswith("scripts/trl/eval_taboo_loras.py")
    assert "--exps" in arguments
    assert "--val-exps" in arguments
    assert "logs/run_a" in arguments
