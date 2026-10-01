from pathlib import Path

import pytest

from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.utils.sft_resume import (
    latest_step_checkpoint,
    resume_epoch_batch_start,
    run_id_from_run_dir,
)


def test_run_id_from_run_dir():
    path = Path(
        "logs/20260927_173048_attn10_visual_spqa_cls_cococtx_snlive_"
        "vtaboo_vuser_vssc_vpqa_deepstack_optsteer_inj3_ep3_Qwen3-VL-4B-Instruct"
    )
    assert run_id_from_run_dir(path) == "20260927_173048"


def test_run_id_from_run_dir_rejects_bad_name():
    with pytest.raises(ValueError, match="YYYYMMDD_HHMMSS"):
        run_id_from_run_dir(Path("logs/manual_run"))


def test_latest_step_checkpoint_picks_max(tmp_path):
    save_dir = tmp_path / "checkpoints"
    (save_dir / "step_5000").mkdir(parents=True)
    (save_dir / "step_2000").mkdir()
    (save_dir / "final").mkdir()
    step, path = latest_step_checkpoint(save_dir)
    assert step == 5000
    assert path == save_dir / "step_5000"


def test_latest_step_checkpoint_rejects_empty_and_final_only(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No checkpoints/step_"):
        latest_step_checkpoint(empty)

    final_only = tmp_path / "done"
    (final_only / "final").mkdir(parents=True)
    with pytest.raises(ValueError, match="already has checkpoints/final"):
        latest_step_checkpoint(final_only)

    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="Checkpoint directory not found"):
        latest_step_checkpoint(missing)


def test_resume_epoch_batch_start():
    start_epoch, start_batch = resume_epoch_batch_start(
        completed_updates=5000,
        steps_per_epoch=10000,
        num_epochs=3,
        gradient_accumulation_steps=1,
    )
    assert start_epoch == 0
    assert start_batch == 5000

    start_epoch, start_batch = resume_epoch_batch_start(
        completed_updates=12,
        steps_per_epoch=10,
        num_epochs=3,
        gradient_accumulation_steps=2,
    )
    assert start_epoch == 1
    assert start_batch == 4


def test_resume_epoch_batch_start_rejects_completed_run():
    with pytest.raises(ValueError, match="training only has"):
        resume_epoch_batch_start(30, steps_per_epoch=10, num_epochs=3, gradient_accumulation_steps=1)


def test_finalize_forces_resume_run_dir_paths(tmp_path):
    run_dir = tmp_path / "20260927_173048_attn10_Qwen3-VL-4B-Instruct"
    run_dir.mkdir()
    cfg = SelfInterpTrainingConfig(
        model_name="Qwen/Qwen3-VL-4B-Instruct",
        act_layers=[9, 18, 27],
        run_id="ignored_timestamp",
        wandb_suffix="_other_suffix",
        resume_run_dir=str(run_dir),
    )
    cfg.finalize(dataset_loaders=[])
    assert Path(cfg.run_dir) == run_dir.resolve()
    assert cfg.run_id == "20260927_173048"
    assert Path(cfg.save_dir) == run_dir.resolve() / "checkpoints"
    assert Path(cfg.result_log_path) == run_dir.resolve() / "training.log"
    assert Path(cfg.tensorboard_dir) == run_dir.resolve() / "tensorboard"
