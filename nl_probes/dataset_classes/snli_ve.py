"""SNLI-VE binary visual entailment loader for Activation Oracle validation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from nl_probes.dataset_classes.act_dataset_manager import (
    ActDatasetLoader,
    BaseDatasetConfig,
    DatasetLoaderConfig,
)
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_processor, load_tokenizer, set_seed
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint
from nl_probes.utils.vlm_utils import DEFAULT_MAX_PIXELS, extract_image_paths, vision_inputs_to_device, vlm_tokenize_target

YES_TOKEN = "Yes"
NO_TOKEN = "No"

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
class SNLIVEDatasetConfig(BaseDatasetConfig):
    annotations_path: str = "data/val/snli_ve/snli_ve_dev.jsonl"
    flickr_image_dir: str = "data/val/flickr30k/flickr30k-images"
    num_qa_per_sample: int = 2
    min_end_offset: int = -1
    max_end_offset: int = -5
    max_window_size: int = 5
    min_window_size: int = 1
    max_pixels: int = DEFAULT_MAX_PIXELS
    drop_neutral: bool = True


class SNLIVEDatasetLoader(ActDatasetLoader):
    def __init__(self, dataset_config: DatasetLoaderConfig, model_kwargs: dict[str, Any] | None = None):
        super().__init__(dataset_config)
        assert self.dataset_config.dataset_name == "", "SNLI-VE dataset name gets overridden here"
        self.dataset_config.dataset_name = "classification_snli_ve"
        self.dataset_params: SNLIVEDatasetConfig = dataset_config.custom_dataset_params
        self.model_kwargs = model_kwargs or {}

    def create_dataset(self) -> None:
        set_seed(self.dataset_config.seed)
        tokenizer = load_tokenizer(self.dataset_config.model_name)
        processor = load_processor(self.dataset_config.model_name)
        layers = [
            layer_percent_to_layer(self.dataset_config.model_name, layer_percent)
            for layer_percent in self.dataset_config.layer_percents
        ]

        records = load_snli_ve_records(
            self.dataset_params.annotations_path,
            self.dataset_params.flickr_image_dir,
            drop_neutral=self.dataset_params.drop_neutral,
        )
        if not records:
            raise FileNotFoundError(
                "No SNLI-VE records with images. Run scripts/download_vlm_ao_data.sh"
            )

        rng = random.Random(self.dataset_config.seed)
        rng.shuffle(records)

        n_test = self.dataset_config.num_test
        if n_test <= 0 or n_test > len(records):
            n_test = min(500, len(records))
        test_records = records[:n_test]

        datapoints = records_to_datapoints(test_records, self.dataset_params, rng)
        data = create_snli_ve_vector_dataset(
            datapoints=datapoints,
            processor=processor,
            tokenizer=tokenizer,
            model_name=self.dataset_config.model_name,
            act_layers=layers,
            dataset_params=self.dataset_params,
            save_acts=self.dataset_config.save_acts,
            batch_size=max(1, self.dataset_config.batch_size),
            model_kwargs=self.model_kwargs,
            rng=rng,
        )
        if "test" in self.dataset_config.splits:
            self.save_dataset(data, "test")


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


def load_snli_ve_records(annotations_path: str, flickr_dir: str, drop_neutral: bool = True) -> list[dict]:
    path = Path(annotations_path)
    records_raw: list[dict] = []
    if path.exists():
        records_raw = _load_json_records(path)
    else:
        records_raw = _try_huggingface_snli_ve()

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


def _try_huggingface_snli_ve() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    for repo, split in (
        ("HuggingFaceM4/SNLI-VE", "validation"),
        ("nlphuji/snli_ve", "validation"),
    ):
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
    model = None
    submodules = None
    device = torch.device("cpu")
    if save_acts:
        model = load_model(model_name, torch.bfloat16, **model_kwargs)
        model.eval()
        submodules = {layer: get_hf_submodule(model, layer) for layer in act_layers}
        device = next(model.parameters()).device

    training_data: list[TrainingDataPoint] = []
    for i in tqdm(range(0, len(datapoints), batch_size), desc="SNLI-VE vector dataset"):
        batch = datapoints[i : i + batch_size]
        for dp in batch:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": dp["image_path"]},
                        {"type": "text", "text": dp["hypothesis"]},
                    ],
                }
            ]
            context_input_ids, proc_inputs = vlm_tokenize_target(
                processor,
                messages,
                add_generation_prompt=True,
                max_pixels=dataset_params.max_pixels,
            )
            L = len(context_input_ids)
            end_offset = rng.randint(dataset_params.max_end_offset, dataset_params.min_end_offset)
            end_pos = L + end_offset
            end_pos = max(1, min(end_pos, L - 1 if L > 1 else 0))
            k = rng.randint(dataset_params.min_window_size, dataset_params.max_window_size)
            k = min(k, end_pos + 1)
            k = max(k, 1)
            begin_pos = end_pos - k + 1
            positions_K = list(range(begin_pos, end_pos + 1))

            acts_by_layer = None
            if save_acts:
                inputs_BL = vision_inputs_to_device(proc_inputs, device)
                with torch.no_grad():
                    acts_by_layer = collect_activations_multiple_layers(
                        model,
                        submodules,
                        inputs_BL,
                        None,
                        None,
                    )

            for layer in act_layers:
                acts_KD = None
                if save_acts:
                    acts_KD = acts_by_layer[layer][0, positions_K].detach().contiguous()
                training_data.append(
                    create_training_datapoint(
                        datapoint_type="classification_snli_ve",
                        prompt=dp["question"],
                        target_response=dp["answer"],
                        layer=layer,
                        num_positions=len(positions_K),
                        tokenizer=tokenizer,
                        acts_BD=acts_KD,
                        feature_idx=-1,
                        context_input_ids=context_input_ids,
                        context_positions=positions_K,
                        context_image_paths=extract_image_paths(messages),
                        ds_label=dp["answer"],
                        meta_info={
                            "target_messages": messages,
                            "add_generation_prompt": True,
                            "flickr_id": dp["flickr_id"],
                        },
                    )
                )
    return training_data
