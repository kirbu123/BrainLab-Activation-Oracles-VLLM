import pytest

from nl_probes.utils.vlm_utils import (
    messages_with_resolved_images,
    resolve_vlm_image_path,
    vlm_image_path_candidates,
)


def test_legacy_coco_train_path_maps_to_split_layout():
    assert vlm_image_path_candidates("data/coco/train2017/000000086075.jpg") == (
        "data/coco/train2017/000000086075.jpg",
        "data/train/coco/train2017/000000086075.jpg",
    )


def test_resolve_legacy_coco_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image = tmp_path / "data/train/coco/train2017/000000086075.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"jpeg")
    assert resolve_vlm_image_path("data/coco/train2017/000000086075.jpg") == (
        "data/train/coco/train2017/000000086075.jpg"
    )


def test_resolve_missing_image_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="Tried:"):
        resolve_vlm_image_path("data/coco/train2017/missing.jpg")


def test_messages_rewrite_legacy_image_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image = tmp_path / "data/train/coco/train2017/000000086075.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"jpeg")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "data/coco/train2017/000000086075.jpg"},
                {"type": "text", "text": "Describe the image."},
            ],
        }
    ]
    rewritten = messages_with_resolved_images(messages)
    assert rewritten[0]["content"][0]["image"] == "data/train/coco/train2017/000000086075.jpg"
    assert messages[0]["content"][0]["image"] == "data/coco/train2017/000000086075.jpg"
