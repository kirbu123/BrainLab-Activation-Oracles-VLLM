from pathlib import Path

from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.dataset_classes.snli_ve import SNLIVEDatasetConfig
from nl_probes.dataset_classes.visual_spqa_dataset import VisualSPQADatasetConfig


def test_vlm_source_data_is_split_between_train_and_val():
    train = VisualSPQADatasetConfig()
    val = SNLIVEDatasetConfig()

    assert Path(train.llava_json_path).parts[:2] == ("data", "train")
    assert Path(train.coco_image_dir).parts[:2] == ("data", "train")
    assert Path(train.latentqa_dir).parts[:2] == ("data", "train")
    assert Path(val.annotations_path).parts[:2] == ("data", "val")
    assert Path(val.flickr_image_dir).parts[:2] == ("data", "val")


def test_run_artifacts_share_one_timestamped_directory():
    cfg = SelfInterpTrainingConfig(
        model_name="Qwen/Qwen3-VL-4B-Instruct",
        act_layers=[9, 18, 27],
        run_id="20260824_021700",
        wandb_suffix="_visual_spqa_snlive_Qwen3-VL-4B-Instruct",
    )

    cfg.finalize(dataset_loaders=[])

    expected = Path("logs/20260824_021700_visual_spqa_snlive_Qwen3-VL-4B-Instruct")
    assert Path(cfg.run_dir) == expected
    assert Path(cfg.save_dir) == expected / "checkpoints"
    assert Path(cfg.results_html_path) == expected / "results.html"
    assert Path(cfg.result_log_path) == expected / "training.log"
    assert Path(cfg.tensorboard_dir) == expected / "tensorboard"
