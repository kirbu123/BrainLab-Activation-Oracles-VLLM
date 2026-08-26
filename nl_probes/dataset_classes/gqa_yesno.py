"""GQA yes/no binary classification loader."""

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

GQA_BINARY_LABELS = {"yes", "no", "true", "false", "1", "0"}


@dataclass
class GQAYesNoDatasetConfig(VLMBinaryDatasetConfig):
    train_questions_path: str = "data/train/gqa/train_balanced_questions.json"
    test_questions_path: str = "data/val/gqa/val_balanced_questions.json"
    train_image_dir: str = "data/train/gqa/images"
    test_image_dir: str = "data/val/gqa/images"


class GQAYesNoDatasetLoader(VLMBinaryDatasetLoader):
    dataset_name = "classification_gqa_yesno"

    def records_for_split(self, split: str) -> list[VLMBinaryRecord]:
        params: GQAYesNoDatasetConfig = self.dataset_params
        if split == "train":
            return load_gqa_yesno_records(params.train_questions_path, params.train_image_dir)
        return load_gqa_yesno_records(params.test_questions_path, params.test_image_dir)


def _gqa_items(raw: object) -> list[tuple[str, dict]]:
    if isinstance(raw, dict):
        return [(str(question_id), item) for question_id, item in raw.items()]
    if isinstance(raw, list):
        items = []
        for index, item in enumerate(raw):
            question_id = str(item["questionId"]) if "questionId" in item else str(index)
            items.append((question_id, item))
        return items
    raise TypeError(f"GQA questions must be a JSON object or list, got {type(raw).__name__}")


def _resolve_gqa_image(image_id: str, image_dir: Path) -> Path:
    supplied = Path(image_id)
    candidates = [image_dir / supplied, image_dir / supplied.name]
    if supplied.suffix == "":
        candidates.extend([image_dir / f"{image_id}.jpg", image_dir / f"{image_id}.png"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"GQA image {image_id!r} not found under {image_dir}")


def load_gqa_yesno_records(questions_path: str, image_dir: str) -> list[VLMBinaryRecord]:
    """Parse the yes/no subset of an official GQA question split."""

    path = Path(questions_path)
    if not path.is_file():
        raise FileNotFoundError(f"GQA questions not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    records: list[VLMBinaryRecord] = []
    for question_id, item in _gqa_items(raw):
        raw_answer = str(item["answer"]).strip().lower()
        if raw_answer not in GQA_BINARY_LABELS:
            continue
        image_id = str(item["imageId"])
        image_path = _resolve_gqa_image(image_id, Path(image_dir))
        question = str(item["question"]).strip()
        records.append(
            VLMBinaryRecord(
                source_id=question_id,
                image_path=str(image_path),
                context_text=question,
                question=(
                    "Answer with 'Yes' or 'No' only. "
                    f"What is the correct binary answer to this visual question? {question}"
                ),
                answer=normalize_binary_label(raw_answer),
                metadata={"gqa_question_id": question_id, "gqa_image_id": image_id},
            )
        )
    return records
