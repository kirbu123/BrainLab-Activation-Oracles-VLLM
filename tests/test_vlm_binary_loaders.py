import json
import random
from collections import Counter
from pathlib import Path

import pytest
import torch

import nl_probes.dataset_classes.vlm_binary as vlm_binary
from nl_probes.dataset_classes.coco_presence import (
    COCOObjectPresenceDatasetConfig,
    COCOObjectPresenceDatasetLoader,
    load_coco_presence_records,
)
from nl_probes.dataset_classes.gqa_yesno import (
    GQAYesNoDatasetConfig,
    GQAYesNoDatasetLoader,
    load_gqa_yesno_records,
)
from nl_probes.dataset_classes.snli_ve import SNLIVEDatasetConfig
from nl_probes.dataset_classes.vlm_binary import (
    NO_TOKEN,
    YES_TOKEN,
    VLMBinaryRecord,
    VLMBinaryDatasetConfig,
    create_vlm_binary_vector_dataset,
    normalize_binary_label,
    subsample_binary_records,
)
from nl_probes.dataset_classes.vsr import VSRDatasetConfig, VSRDatasetLoader, load_vsr_records


def _touch_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _record(index: int, answer: str) -> VLMBinaryRecord:
    return VLMBinaryRecord(
        source_id=str(index),
        image_path=f"/images/{index}.jpg",
        context_text=f"context {index}",
        question=f"question {index}",
        answer=answer,
    )


def test_binary_record_and_label_normalization():
    assert normalize_binary_label(" YES ") == YES_TOKEN
    assert normalize_binary_label("False") == NO_TOKEN
    assert normalize_binary_label(1) == YES_TOKEN
    with pytest.raises(ValueError, match="Unsupported binary label"):
        normalize_binary_label("maybe")
    with pytest.raises(ValueError, match="Binary answer"):
        _record(0, "maybe")


def test_subsample_is_deterministic_and_can_balance():
    records = [_record(i, YES_TOKEN if i < 6 else NO_TOKEN) for i in range(10)]

    first = subsample_binary_records(records, 6, seed=17)
    second = subsample_binary_records(records, 6, seed=17)
    balanced = subsample_binary_records(records, 8, seed=17, balanced=True)

    assert [record.source_id for record in first] == [record.source_id for record in second]
    assert Counter(record.answer for record in balanced) == {YES_TOKEN: 4, NO_TOKEN: 4}
    assert [record.source_id for record in records] == [str(i) for i in range(10)]


def test_vector_builder_selects_one_lazy_layer_and_all_saved_layers(monkeypatch):
    records = [_record(0, YES_TOKEN), _record(1, NO_TOKEN)]
    monkeypatch.setattr(
        vlm_binary,
        "vlm_tokenize_target",
        lambda *args, **kwargs: (list(range(8)), {"input_ids": torch.arange(8).unsqueeze(0)}),
    )
    monkeypatch.setattr(vlm_binary, "extract_image_paths", lambda messages: ["/images/0.jpg"])
    monkeypatch.setattr(
        vlm_binary,
        "create_training_datapoint",
        lambda **kwargs: kwargs,
    )

    common = {
        "records": records,
        "processor": object(),
        "tokenizer": object(),
        "model_name": "test-model",
        "act_layers": [2, 4, 6],
        "dataset_params": VLMBinaryDatasetConfig(),
        "datapoint_type": "classification_test",
        "batch_size": 2,
        "model_kwargs": {},
    }
    lazy = create_vlm_binary_vector_dataset(
        **common,
        save_acts=False,
        rng=random.Random(3),
    )

    class FakeModel:
        def eval(self):
            return self

        def parameters(self):
            return iter([torch.zeros(1)])

    monkeypatch.setattr(vlm_binary, "load_model", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(vlm_binary, "get_hf_submodule", lambda model, layer: layer)
    monkeypatch.setattr(
        vlm_binary,
        "collect_activations_multiple_layers",
        lambda model, submodules, inputs, min_offset, max_offset: {
            layer: torch.zeros(1, 8, 3) for layer in submodules
        },
    )
    saved = create_vlm_binary_vector_dataset(
        **common,
        save_acts=True,
        rng=random.Random(3),
    )

    assert len(lazy) == len(records)
    assert len(saved) == len(records) * 3
    assert all(datapoint["acts_BD"] is None for datapoint in lazy)
    assert all(datapoint["acts_BD"] is not None for datapoint in saved)


def test_vsr_parser_normalizes_labels(tmp_path: Path):
    images = tmp_path / "images"
    _touch_image(images / "one.jpg")
    _touch_image(images / "two.jpg")
    annotations = tmp_path / "vsr.jsonl"
    annotations.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "image": "one.jpg", "caption": "Left of.", "label": True}),
                json.dumps({"id": "b", "image": "two.jpg", "caption": "Right of.", "label": "0"}),
            ]
        ),
        encoding="utf-8",
    )

    records = load_vsr_records(str(annotations), str(images))

    assert [record.answer for record in records] == [YES_TOKEN, NO_TOKEN]
    assert [record.source_id for record in records] == ["a", "b"]


def test_vsr_parser_can_enforce_coco_image_split(tmp_path: Path):
    images = tmp_path / "images"
    _touch_image(images / "train.jpg")
    _touch_image(images / "val.jpg")
    annotations = tmp_path / "vsr.jsonl"
    annotations.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "image": "train.jpg",
                        "image_link": "http://images.cocodataset.org/train2017/train.jpg",
                        "caption": "Train image.",
                        "label": 1,
                    }
                ),
                json.dumps(
                    {
                        "image": "val.jpg",
                        "image_link": "http://images.cocodataset.org/val2017/val.jpg",
                        "caption": "Validation image.",
                        "label": 0,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    records = load_vsr_records(str(annotations), str(images), required_coco_split="val2017")

    assert [record.context_text for record in records] == ["Validation image."]


def test_gqa_parser_keeps_only_yes_no_questions(tmp_path: Path):
    images = tmp_path / "images"
    _touch_image(images / "10.jpg")
    _touch_image(images / "11.jpg")
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps(
            {
                "q1": {"imageId": "10", "question": "Is it red?", "answer": "YES"},
                "q2": {"imageId": "11", "question": "Is it blue?", "answer": "false"},
                "q3": {"imageId": "missing", "question": "What color?", "answer": "green"},
            }
        ),
        encoding="utf-8",
    )

    records = load_gqa_yesno_records(str(questions), str(images))

    assert [record.source_id for record in records] == ["q1", "q2"]
    assert [record.answer for record in records] == [YES_TOKEN, NO_TOKEN]


def test_coco_parser_emits_balanced_deterministic_pairs(tmp_path: Path):
    images = tmp_path / "images"
    _touch_image(images / "1.jpg")
    _touch_image(images / "2.jpg")
    annotations = tmp_path / "instances.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "1.jpg"}, {"id": 2, "file_name": "2.jpg"}],
                "categories": [
                    {"id": 1, "name": "cat"},
                    {"id": 2, "name": "dog"},
                    {"id": 3, "name": "chair"},
                ],
                "annotations": [
                    {"image_id": 1, "category_id": 1},
                    {"image_id": 2, "category_id": 2},
                ],
            }
        ),
        encoding="utf-8",
    )

    first = load_coco_presence_records(str(annotations), str(images), seed=9)
    second = load_coco_presence_records(str(annotations), str(images), seed=9)

    assert Counter(record.answer for record in first) == {YES_TOKEN: 2, NO_TOKEN: 2}
    assert [record.source_id for record in first] == [record.source_id for record in second]
    assert len({record.source_id for record in first}) == len(first)


@pytest.mark.parametrize(
    ("config", "loader", "dataset_name", "annotations_field"),
    [
        (VSRDatasetConfig(), VSRDatasetLoader, "classification_vsr", "annotations_path"),
        (
            GQAYesNoDatasetConfig(),
            GQAYesNoDatasetLoader,
            "classification_gqa_yesno",
            "questions_path",
        ),
        (
            COCOObjectPresenceDatasetConfig(),
            COCOObjectPresenceDatasetLoader,
            "classification_coco_presence",
            "annotations_path",
        ),
    ],
)
def test_official_split_paths_and_dataset_names(config, loader, dataset_name, annotations_field):
    assert Path(getattr(config, f"train_{annotations_field}")).parts[:2] == ("data", "train")
    assert Path(getattr(config, f"test_{annotations_field}")).parts[:2] == ("data", "val")
    assert loader.dataset_name == dataset_name


def test_snli_ve_exposes_official_train_and_test_paths():
    config = SNLIVEDatasetConfig()

    assert Path(config.train_annotations_path).parts[:2] == ("data", "train")
    assert Path(config.annotations_path).parts[:2] == ("data", "val")
