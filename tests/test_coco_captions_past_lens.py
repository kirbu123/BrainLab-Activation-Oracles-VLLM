import json
from pathlib import Path

import pytest
import torch

import nl_probes.dataset_classes.coco_captions_past_lens_dataset as coco_past_lens
from nl_probes.dataset_classes.coco_captions_past_lens_dataset import (
    CocoCaptionsPastLensDatasetConfig,
    create_coco_captions_past_lens_dataset,
    load_official_coco_caption_records,
)
from nl_probes.utils.vlm_utils import vlm_tokenize_target


class FakeTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        tokens = []
        index = 0
        while index < len(text):
            if text[index : index + 2] == " ?":
                tokens.append(999)
                index += 2
            else:
                tokens.append(1000 + ord(text[index]))
                index += 1
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token_id - 1000) for token_id in token_ids if token_id >= 1000)

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    ):
        del return_tensors, padding, enable_thinking
        assert tokenize
        ids = [10]
        for message in messages:
            if message["role"] == "user":
                ids.extend([20, *self.encode(message["content"]), 21])
            elif message["role"] == "assistant":
                ids.extend([30, *self.encode(message["content"]), 31])
            else:
                raise ValueError(message["role"])
        if add_generation_prompt:
            ids.append(30)
        return ids


class FakeProcessor:
    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
        return_tensors=None,
        enable_thinking=False,
    ):
        del return_tensors, enable_thinking
        assert tokenize
        assert not add_generation_prompt
        content = messages[0]["content"]
        assert content[0]["type"] == "image"
        caption = content[1]["text"]
        ids = [40, 50, 51, *[1000 + ord(character) for character in caption], 60]
        if return_dict:
            return {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                "pixel_values": torch.ones(1, 3, 2, 2),
            }
        return ids


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _official_coco(images, annotations):
    return {"info": {}, "licenses": [], "images": images, "annotations": annotations}


def test_official_splits_and_llava_exclusion(tmp_path):
    train_images = tmp_path / "data/train/coco/train2017"
    val_images = tmp_path / "data/val/coco/val2017"
    train_images.mkdir(parents=True)
    val_images.mkdir(parents=True)
    for root, names in ((train_images, ["a.jpg", "b.jpg"]), (val_images, ["v.jpg"])):
        for name in names:
            (root / name).touch()

    train_json = tmp_path / "data/train/coco/annotations/captions_train2017.json"
    val_json = tmp_path / "data/val/coco/annotations/captions_val2017.json"
    llava_json = tmp_path / "data/train/llava/llava_instruct_150k.json"
    _write_json(
        train_json,
        _official_coco(
            [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
            [
                {"id": 12, "image_id": 2, "caption": "kept train caption"},
                {"id": 11, "image_id": 1, "caption": "excluded caption"},
            ],
        ),
    )
    _write_json(
        val_json,
        _official_coco(
            [{"id": 3, "file_name": "v.jpg"}],
            [{"id": 21, "image_id": 3, "caption": "official validation caption"}],
        ),
    )
    _write_json(llava_json, [{"id": "x", "image": "nested/a.jpg"}])

    config = CocoCaptionsPastLensDatasetConfig(
        train_annotations_path=str(train_json),
        train_image_dir=str(train_images),
        val_annotations_path=str(val_json),
        val_image_dir=str(val_images),
        llava_json_path=str(llava_json),
    )
    train = load_official_coco_caption_records(config, "train")
    val = load_official_coco_caption_records(config, "val")

    assert [record["file_name"] for record in train] == ["b.jpg"]
    assert [record["annotation_id"] for record in val] == [21]


def test_malformed_official_caption_reference_fails_clearly(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    annotations = tmp_path / "captions.json"
    llava = tmp_path / "llava.json"
    _write_json(
        annotations,
        _official_coco(
            [{"id": 1, "file_name": "a.jpg"}],
            [{"id": 2, "image_id": 999, "caption": "bad reference"}],
        ),
    )
    _write_json(llava, [])
    config = CocoCaptionsPastLensDatasetConfig(
        train_annotations_path=str(annotations),
        train_image_dir=str(image_dir),
        llava_json_path=str(llava),
    )

    with pytest.raises(ValueError, match="unknown image_id 999"):
        load_official_coco_caption_records(config, "train")


def _records(tmp_path, count=3):
    records = []
    for index in range(count):
        image = tmp_path / f"{index}.jpg"
        image.touch()
        records.append(
            {
                "annotation_id": index,
                "image_id": index,
                "file_name": image.name,
                "image_path": str(image),
                "caption": f"caption number {index}",
            }
        )
    return records


def test_on_the_fly_samples_caption_only_and_reproduces_context(tmp_path):
    processor = FakeProcessor()
    tokenizer = FakeTokenizer()
    params = CocoCaptionsPastLensDatasetConfig(
        min_k_tokens=2,
        max_k_tokens=2,
        min_k_activations=3,
        max_k_activations=3,
        directions=["past", "future"],
    )
    kwargs = dict(
        records=_records(tmp_path),
        processor=processor,
        tokenizer=tokenizer,
        model_name="fake",
        act_layers=[4, 8, 12],
        dataset_params=params,
        save_acts=False,
        num_examples=2,
        seed=7,
    )

    first = create_coco_captions_past_lens_dataset(**kwargs)
    second = create_coco_captions_past_lens_dataset(**kwargs)

    assert [(dp.layer, dp.meta_info) for dp in first] == [(dp.layer, dp.meta_info) for dp in second]
    assert len(first) == 2
    for datapoint in first:
        assert datapoint.steering_vectors is None
        assert datapoint.layer in {4, 8, 12}
        assert len(datapoint.context_positions) == 3
        metadata = datapoint.meta_info
        assert set(metadata["activation_positions"]).isdisjoint(metadata["target_positions"])
        assert set(metadata["activation_positions"]) <= set(metadata["caption_positions"])
        assert set(metadata["target_positions"]) <= set(metadata["caption_positions"])
        reproduced_ids, _ = vlm_tokenize_target(
            processor,
            metadata["target_messages"],
            add_generation_prompt=metadata["add_generation_prompt"],
        )
        assert reproduced_ids == datapoint.context_input_ids


def test_saved_activations_emit_every_layer(monkeypatch, tmp_path):
    class FakeModel:
        def __init__(self):
            self._parameter = torch.nn.Parameter(torch.zeros(1))

        def eval(self):
            return self

        def parameters(self):
            yield self._parameter

    def fake_collect(model, submodules, inputs, min_offset, max_offset):
        del model, min_offset, max_offset
        length = inputs["input_ids"].shape[1]
        return {
            layer: torch.full((1, length, 2), float(layer))
            for layer in submodules
        }

    monkeypatch.setattr(coco_past_lens, "load_model", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(coco_past_lens, "get_hf_submodule", lambda model, layer: layer)
    monkeypatch.setattr(coco_past_lens, "collect_activations_multiple_layers", fake_collect)

    data = create_coco_captions_past_lens_dataset(
        records=_records(tmp_path, count=1),
        processor=FakeProcessor(),
        tokenizer=FakeTokenizer(),
        model_name="fake",
        act_layers=[2, 6],
        dataset_params=CocoCaptionsPastLensDatasetConfig(
            min_k_tokens=1,
            max_k_tokens=1,
            min_k_activations=2,
            max_k_activations=2,
            directions=["future"],
        ),
        save_acts=True,
        num_examples=1,
        seed=3,
    )

    assert [datapoint.layer for datapoint in data] == [2, 6]
    for datapoint in data:
        assert datapoint.steering_vectors.shape == (2, 2)
        assert torch.all(datapoint.steering_vectors == datapoint.layer)
        assert datapoint.context_input_ids is not None
        assert datapoint.meta_info["target_messages"][0]["content"][1]["text"] == "caption number 0"
