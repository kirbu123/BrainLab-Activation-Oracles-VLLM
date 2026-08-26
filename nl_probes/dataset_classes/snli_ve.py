"""SNLI-VE binary visual entailment loader for Activation Oracle validation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig
from nl_probes.dataset_classes.vlm_binary import (
    NO_TOKEN,
    YES_TOKEN,
    VLMBinaryDatasetConfig,
    VLMBinaryDatasetLoader,
    VLMBinaryRecord,
    create_vlm_binary_vector_dataset,
)
from nl_probes.utils.dataset_utils import TrainingDataPoint

SNLI_VE_PARAPHRASES = [
    "Does the image entail this statement?",
    "Is this hypothesis true given the image?",
    "Can we conclude the following from the image?",
    "Does the photograph support this claim?",
    "Is the following sentence entailed by the image?",
    "Given only the image, is this statement true?",
    "Does the visual evidence entail this hypothesis?",
]


@dataclass
class SNLIVEDatasetConfig(VLMBinaryDatasetConfig):
    train_annotations_path: str = "data/train/snli_ve/snli_ve_train.jsonl"
    train_flickr_image_dir: str = "data/train/flickr30k/flickr30k-images"
    annotations_path: str = "data/val/snli_ve/snli_ve_dev.jsonl"
    flickr_image_dir: str = "data/val/flickr30k/flickr30k-images"
    num_qa_per_sample: int = 2
    drop_neutral: bool = True


class SNLIVEDatasetLoader(VLMBinaryDatasetLoader):
    dataset_name = "classification_snli_ve"

    def records_for_split(self, split: str) -> list[VLMBinaryRecord]:
        params: SNLIVEDatasetConfig = self.dataset_params
        if split == "train":
            raw_records = load_snli_ve_records(
                params.train_annotations_path,
                params.train_flickr_image_dir,
                drop_neutral=params.drop_neutral,
                hf_split="train",
            )
        else:
            raw_records = load_snli_ve_records(
                params.annotations_path,
                params.flickr_image_dir,
                drop_neutral=params.drop_neutral,
                hf_split="validation",
            )
        return [
            VLMBinaryRecord(
                source_id=record["flickr_id"],
                image_path=record["image_path"],
                context_text=record["hypothesis"],
                question=record["hypothesis"],
                answer=record["answer"],
                metadata={"flickr_id": record["flickr_id"], "hypothesis": record["hypothesis"]},
            )
            for record in raw_records
        ]

    def default_num_records(self, split: str, available: int) -> int:
        return min(500, available)

    def expand_records(
        self,
        records: list[VLMBinaryRecord],
        rng: random.Random,
    ) -> list[VLMBinaryRecord]:
        params: SNLIVEDatasetConfig = self.dataset_params
        n_paraphrases = min(params.num_qa_per_sample, len(SNLI_VE_PARAPHRASES))
        expanded = []
        for record in records:
            for paraphrase in rng.sample(SNLI_VE_PARAPHRASES, n_paraphrases):
                expanded.append(
                    VLMBinaryRecord(
                        source_id=record.source_id,
                        image_path=record.image_path,
                        context_text=record.context_text,
                        question=(
                            "Answer with 'Yes' or 'No' only. "
                            f"# {paraphrase} {record.context_text}"
                        ),
                        answer=record.answer,
                        metadata=record.metadata,
                    )
                )
        return expanded


def _label_to_yes_no(label: str | int) -> str | None:
    if label in (0, "0", "entailment", "entail"):
        return YES_TOKEN
    if label in (2, "2", "contradiction", "contradict"):
        return NO_TOKEN
    return None


def _flickr_id(record: dict) -> str | None:
    for key in (
        "Flickr30K_ID",
        "Flickr30kID",
        "Flikr30kID",
        "flickr30k_id",
        "image_id",
        "image",
    ):
        if key in record and record[key] not in (None, ""):
            value = str(record[key])
            value = value.split("#")[0]
            value = value.replace(".jpg", "").replace(".png", "")
            return Path(value).name
    return None


def _hypothesis(record: dict) -> str | None:
    for key in ("sentence2", "hypothesis", "gold_hypothesis", "H"):
        if key in record and record[key]:
            return str(record[key])
    return None


def _gold_label(record: dict) -> str | int | None:
    for key in ("gold_label", "label", "goldLabel"):
        if key in record:
            return record[key]
    return None


def _resolve_image_path(flickr_id: str, flickr_dir: Path) -> Path | None:
    candidates = [
        flickr_dir / f"{flickr_id}.jpg",
        flickr_dir / flickr_id,
        flickr_dir / f"{flickr_id}.png",
    ]
    # Some dumps nest images one more level
    if flickr_dir.exists():
        nested = flickr_dir / "flickr30k-images" / f"{flickr_id}.jpg"
        candidates.append(nested)
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_json_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        for key in ("data", "records", "examples"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
        raise ValueError(f"Unrecognized JSON object in {path}")
    return raw


def load_snli_ve_records(
    annotations_path: str,
    flickr_dir: str,
    drop_neutral: bool = True,
    hf_split: str = "validation",
) -> list[dict]:
    path = Path(annotations_path)
    records_raw: list[dict] = []
    if path.exists():
        records_raw = _load_json_records(path)
    else:
        records_raw = _try_huggingface_snli_ve(hf_split)

    flickr_root = Path(flickr_dir)
    kept = []
    missing_images = 0
    for record in records_raw:
        answer = _label_to_yes_no(_gold_label(record))
        if answer is None:
            if drop_neutral:
                continue
            continue
        flickr_id = _flickr_id(record)
        hypothesis = _hypothesis(record)
        if not flickr_id or not hypothesis:
            continue
        image_path = _resolve_image_path(flickr_id, flickr_root)
        if image_path is None:
            missing_images += 1
            continue
        kept.append(
            {
                "flickr_id": flickr_id,
                "image_path": str(image_path),
                "hypothesis": hypothesis,
                "answer": answer,
            }
        )
    print(f"SNLI-VE: {len(kept)} labeled image pairs, {missing_images} missing images")
    return kept


def _try_huggingface_snli_ve(split: str = "validation") -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    for repo in ("HuggingFaceM4/SNLI-VE", "nlphuji/snli_ve"):
        try:
            ds = load_dataset(repo, split=split)
            return [dict(row) for row in ds]
        except Exception as exc:
            print(f"Could not load {repo}: {exc}")
    return []


def records_to_datapoints(
    records: list[dict],
    dataset_params: SNLIVEDatasetConfig,
    rng: random.Random,
) -> list[dict]:
    datapoints = []
    n_para = min(dataset_params.num_qa_per_sample, len(SNLI_VE_PARAPHRASES))
    for record in records:
        paraphrases = rng.sample(SNLI_VE_PARAPHRASES, n_para)
        for paraphrase in paraphrases:
            question = f"Answer with 'Yes' or 'No' only. # {paraphrase} {record['hypothesis']}"
            datapoints.append(
                {
                    "image_path": record["image_path"],
                    "hypothesis": record["hypothesis"],
                    "question": question,
                    "answer": record["answer"],
                    "flickr_id": record["flickr_id"],
                }
            )
    return datapoints


def create_snli_ve_vector_dataset(
    datapoints: list[dict],
    processor,
    tokenizer,
    model_name: str,
    act_layers: list[int],
    dataset_params: SNLIVEDatasetConfig,
    save_acts: bool,
    batch_size: int,
    model_kwargs: dict[str, Any],
    rng: random.Random,
) -> list[TrainingDataPoint]:
    records = [
        VLMBinaryRecord(
            source_id=datapoint["flickr_id"],
            image_path=datapoint["image_path"],
            context_text=datapoint["hypothesis"],
            question=datapoint["question"],
            answer=datapoint["answer"],
            metadata={"flickr_id": datapoint["flickr_id"]},
        )
        for datapoint in datapoints
    ]
    return create_vlm_binary_vector_dataset(
        records=records,
        processor=processor,
        tokenizer=tokenizer,
        model_name=model_name,
        act_layers=act_layers,
        dataset_params=dataset_params,
        datapoint_type="classification_snli_ve",
        save_acts=save_acts,
        batch_size=batch_size,
        model_kwargs=model_kwargs,
        rng=rng,
    )
