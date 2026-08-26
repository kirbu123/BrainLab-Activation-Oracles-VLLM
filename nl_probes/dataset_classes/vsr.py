"""Visual Spatial Reasoning binary classification loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from nl_probes.dataset_classes.vlm_binary import (
    VLMBinaryDatasetConfig,
    VLMBinaryDatasetLoader,
    VLMBinaryRecord,
    normalize_binary_label,
)


@dataclass
class VSRDatasetConfig(VLMBinaryDatasetConfig):
    train_annotations_path: str = "data/train/vsr/train.jsonl"
    test_annotations_path: str = "data/val/vsr/test.jsonl"
    train_image_dir: str = "data/train/vsr/images"
    test_image_dir: str = "data/val/vsr/images"
    train_coco_split: str = "train2017"
    test_coco_split: str = "val2017"


class VSRDatasetLoader(VLMBinaryDatasetLoader):
    dataset_name = "classification_vsr"

    def records_for_split(self, split: str) -> list[VLMBinaryRecord]:
        params: VSRDatasetConfig = self.dataset_params
        if split == "train":
            return load_vsr_records(
                params.train_annotations_path,
                params.train_image_dir,
                required_coco_split=params.train_coco_split,
            )
        return load_vsr_records(
            params.test_annotations_path,
            params.test_image_dir,
            required_coco_split=params.test_coco_split,
        )


def _vsr_image_name(item: dict) -> str:
    for key in ("image", "image_path", "img", "image_id"):
        if key in item:
            return str(item[key])
    raise KeyError("VSR record has no image field")


def _resolve_vsr_image(image_name: str, image_dir: Path) -> Path:
    supplied = Path(image_name)
    candidates = [image_dir / supplied, image_dir / supplied.name]
    if supplied.suffix == "":
        candidates.extend([image_dir / f"{image_name}.jpg", image_dir / f"{image_name}.png"])
        if image_name.isdigit():
            candidates.append(image_dir / f"COCO_train2014_{int(image_name):012d}.jpg")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"VSR image {image_name!r} not found under {image_dir}")


def load_vsr_records(
    annotations_path: str,
    image_dir: str,
    required_coco_split: str | None = None,
) -> list[VLMBinaryRecord]:
    """Parse the official VSR JSONL format."""

    path = Path(annotations_path)
    if not path.is_file():
        raise FileNotFoundError(f"VSR annotations not found: {path}")

    records: list[VLMBinaryRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if required_coco_split is not None and required_coco_split not in str(item["image_link"]):
                continue
            caption = str(item["caption"]).strip()
            image_path = _resolve_vsr_image(_vsr_image_name(item), Path(image_dir))
            label = normalize_binary_label(item["label"])
            source_id = str(item["id"]) if "id" in item else f"{path.stem}:{line_number}"
            records.append(
                VLMBinaryRecord(
                    source_id=source_id,
                    image_path=str(image_path),
                    context_text=caption,
                    question=(
                        "Answer with 'Yes' or 'No' only. "
                        f"Does the image correctly depict this statement? {caption}"
                    ),
                    answer=label,
                    metadata={"caption": caption},
                )
            )
    return records
