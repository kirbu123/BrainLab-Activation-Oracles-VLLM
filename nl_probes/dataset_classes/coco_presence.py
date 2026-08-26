"""Balanced COCO object-presence binary classification loader."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from nl_probes.dataset_classes.vlm_binary import (
    NO_TOKEN,
    YES_TOKEN,
    VLMBinaryDatasetConfig,
    VLMBinaryDatasetLoader,
    VLMBinaryRecord,
)


@dataclass
class COCOObjectPresenceDatasetConfig(VLMBinaryDatasetConfig):
    train_annotations_path: str = "data/train/coco/annotations/instances_train2017.json"
    test_annotations_path: str = "data/val/coco/annotations/instances_val2017.json"
    train_image_dir: str = "data/train/coco/train2017"
    test_image_dir: str = "data/val/coco/val2017"


class COCOObjectPresenceDatasetLoader(VLMBinaryDatasetLoader):
    dataset_name = "classification_coco_presence"

    def records_for_split(self, split: str) -> list[VLMBinaryRecord]:
        params: COCOObjectPresenceDatasetConfig = self.dataset_params
        if split == "train":
            return load_coco_presence_records(
                params.train_annotations_path,
                params.train_image_dir,
                self.dataset_config.seed,
            )
        return load_coco_presence_records(
            params.test_annotations_path,
            params.test_image_dir,
            self.dataset_config.seed + 1,
        )

    def should_balance_split(self, split: str) -> bool:
        return True


# Short aliases for callers that use the dataset name rather than the task description.
COCOPresenceDatasetConfig = COCOObjectPresenceDatasetConfig
COCOPresenceDatasetLoader = COCOObjectPresenceDatasetLoader


def load_coco_presence_records(
    annotations_path: str,
    image_dir: str,
    seed: int = 42,
) -> list[VLMBinaryRecord]:
    """Build one positive and one negative object-presence QA per COCO image."""

    path = Path(annotations_path)
    if not path.is_file():
        raise FileNotFoundError(f"COCO annotations not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    categories = {int(category["id"]): str(category["name"]) for category in raw["categories"]}
    category_ids = sorted(categories)
    present_by_image: dict[int, set[int]] = defaultdict(set)
    for annotation in raw["annotations"]:
        present_by_image[int(annotation["image_id"])].add(int(annotation["category_id"]))

    rng = random.Random(seed)
    records: list[VLMBinaryRecord] = []
    image_root = Path(image_dir)
    for image in sorted(raw["images"], key=lambda item: int(item["id"])):
        image_id = int(image["id"])
        present_ids = sorted(present_by_image[image_id])
        if not present_ids:
            continue
        absent_ids = [category_id for category_id in category_ids if category_id not in present_by_image[image_id]]
        if not absent_ids:
            continue

        image_path = image_root / str(image["file_name"])
        if not image_path.is_file():
            raise FileNotFoundError(f"COCO image not found: {image_path}")
        positive_id = rng.choice(present_ids)
        negative_id = rng.choice(absent_ids)
        for category_id, answer in ((positive_id, YES_TOKEN), (negative_id, NO_TOKEN)):
            category_name = categories[category_id]
            records.append(
                VLMBinaryRecord(
                    source_id=f"{image_id}:{category_id}",
                    image_path=str(image_path),
                    context_text=f"Is there a {category_name} in this image?",
                    question=(
                        "Answer with 'Yes' or 'No' only. "
                        f"Is a {category_name} present in the image?"
                    ),
                    answer=answer,
                    metadata={
                        "coco_image_id": image_id,
                        "coco_category_id": category_id,
                        "coco_category": category_name,
                    },
                )
            )
    return records
