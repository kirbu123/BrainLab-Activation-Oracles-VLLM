"""COCO-caption VLM adjacent-token context prediction dataset."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from tqdm import tqdm

from nl_probes.dataset_classes.act_dataset_manager import (
    ActDatasetLoader,
    BaseDatasetConfig,
    DatasetLoaderConfig,
)
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_processor, load_tokenizer
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint
from nl_probes.utils.vlm_utils import (
    DEFAULT_MAX_PIXELS,
    extract_image_paths,
    vision_inputs_to_device,
    vlm_tokenize_target,
)

Direction = Literal["past", "future"]


@dataclass
class CocoCaptionsPastLensDatasetConfig(BaseDatasetConfig):
    """Paths and sampling ranges for COCO-caption context prediction."""

    train_annotations_path: str = "data/train/coco/annotations/captions_train2017.json"
    train_image_dir: str = "data/train/coco/train2017"
    val_annotations_path: str = "data/val/coco/annotations/captions_val2017.json"
    val_image_dir: str = "data/val/coco/val2017"
    llava_json_path: str = "data/train/llava/llava_instruct_150k.json"
    min_k_tokens: int = 1
    max_k_tokens: int = 20
    min_k_activations: int = 1
    max_k_activations: int = 20
    directions: list[str] = field(default_factory=lambda: ["past", "future"])


class CocoCaptionsPastLensDatasetLoader(ActDatasetLoader):
    """Map official COCO train/val data to the framework's train/test splits."""

    def __init__(
        self,
        dataset_config: DatasetLoaderConfig,
        model_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(dataset_config)
        if self.dataset_config.dataset_name:
            raise ValueError("dataset_name is set by CocoCaptionsPastLensDatasetLoader")
        self.dataset_config.dataset_name = "coco_captions_past_lens"
        if not isinstance(dataset_config.custom_dataset_params, CocoCaptionsPastLensDatasetConfig):
            raise TypeError("custom_dataset_params must be CocoCaptionsPastLensDatasetConfig")
        self.dataset_params: CocoCaptionsPastLensDatasetConfig = dataset_config.custom_dataset_params
        self.model_kwargs = model_kwargs or {}
        validate_coco_captions_config(self.dataset_params)

    def create_dataset(self) -> None:
        tokenizer = load_tokenizer(self.dataset_config.model_name)
        processor = load_processor(self.dataset_config.model_name)
        layers = [
            layer_percent_to_layer(self.dataset_config.model_name, layer_percent)
            for layer_percent in self.dataset_config.layer_percents
        ]
        if not layers:
            raise ValueError("layer_percents must contain at least one layer")

        for split in self.dataset_config.splits:
            official_split: Literal["train", "val"] = "train" if split == "train" else "val"
            records = load_official_coco_caption_records(
                dataset_params=self.dataset_params,
                split=official_split,
            )
            num_examples = self.dataset_config.num_train if split == "train" else self.dataset_config.num_test
            data = create_coco_captions_past_lens_dataset(
                records=records,
                processor=processor,
                tokenizer=tokenizer,
                model_name=self.dataset_config.model_name,
                act_layers=layers,
                dataset_params=self.dataset_params,
                save_acts=self.dataset_config.save_acts,
                num_examples=num_examples,
                seed=self.dataset_config.seed + (0 if split == "train" else 1),
                model_kwargs=self.model_kwargs,
            )
            self.save_dataset(data, split)


def validate_coco_captions_config(config: CocoCaptionsPastLensDatasetConfig) -> None:
    for name in ("min_k_tokens", "max_k_tokens", "min_k_activations", "max_k_activations"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be at least 1")
    if config.min_k_tokens > config.max_k_tokens:
        raise ValueError("min_k_tokens must not exceed max_k_tokens")
    if config.min_k_activations > config.max_k_activations:
        raise ValueError("min_k_activations must not exceed max_k_activations")
    if not config.directions:
        raise ValueError("directions must not be empty")
    invalid = set(config.directions) - {"past", "future"}
    if invalid:
        raise ValueError(f"Unsupported directions: {sorted(invalid)}")


def _read_json_object(path: str, description: str) -> dict:
    json_path = Path(path)
    if not json_path.is_file():
        raise FileNotFoundError(f"{description} not found: {json_path}")
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {description} {json_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {json_path}")
    return value


def load_llava_referenced_image_names(llava_json_path: str) -> set[str]:
    path = Path(llava_json_path)
    if not path.is_file():
        raise FileNotFoundError(f"LLaVA annotations not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in LLaVA annotations {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError(f"LLaVA annotations must contain a JSON list: {path}")

    names: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"LLaVA item {index} is not an object")
        image = item.get("image")
        if image is None:
            image = item.get("image_id")
        if image in (None, ""):
            raise ValueError(f"LLaVA item {index} has no image or image_id")
        name = Path(str(image)).name
        if Path(name).suffix == "":
            name = f"{name}.jpg"
        names.add(name)
    return names


def load_official_coco_caption_records(
    dataset_params: CocoCaptionsPastLensDatasetConfig,
    split: Literal["train", "val"],
) -> list[dict[str, Any]]:
    """Parse official COCO captions, preserving the official split boundary."""

    if split == "train":
        annotations_path = dataset_params.train_annotations_path
        image_dir = dataset_params.train_image_dir
        excluded_names = load_llava_referenced_image_names(dataset_params.llava_json_path)
    elif split == "val":
        annotations_path = dataset_params.val_annotations_path
        image_dir = dataset_params.val_image_dir
        excluded_names = set()
    else:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    root = _read_json_object(annotations_path, f"COCO {split} captions")
    images = root.get("images")
    annotations = root.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError(
            f"COCO {split} captions must contain list-valued 'images' and 'annotations': {annotations_path}"
        )
    image_root = Path(image_dir)
    if not image_root.is_dir():
        raise FileNotFoundError(f"COCO {split} image directory not found: {image_root}")

    image_names: dict[int, str] = {}
    for index, image in enumerate(images):
        if not isinstance(image, dict) or not isinstance(image.get("id"), int):
            raise ValueError(f"COCO {split} image {index} must have an integer id")
        file_name = image.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            raise ValueError(f"COCO {split} image {index} must have a non-empty file_name")
        if image["id"] in image_names:
            raise ValueError(f"Duplicate COCO {split} image id: {image['id']}")
        image_names[image["id"]] = Path(file_name).name

    records: list[dict[str, Any]] = []
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            raise ValueError(f"COCO {split} annotation {index} is not an object")
        annotation_id = annotation.get("id")
        image_id = annotation.get("image_id")
        caption = annotation.get("caption")
        if not isinstance(annotation_id, int) or not isinstance(image_id, int):
            raise ValueError(f"COCO {split} annotation {index} must have integer id and image_id")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"COCO {split} annotation {annotation_id} has an empty or invalid caption")
        if image_id not in image_names:
            raise ValueError(f"COCO {split} annotation {annotation_id} references unknown image_id {image_id}")
        file_name = image_names[image_id]
        if file_name in excluded_names:
            continue
        image_path = image_root / file_name
        if not image_path.is_file():
            raise FileNotFoundError(
                f"COCO {split} image referenced by annotation {annotation_id} not found: {image_path}"
            )
        records.append(
            {
                "annotation_id": annotation_id,
                "image_id": image_id,
                "file_name": file_name,
                "image_path": str(image_path),
                "caption": caption,
            }
        )

    records.sort(key=lambda record: (record["annotation_id"], record["image_id"], record["file_name"]))
    if not records:
        exclusion_note = " after excluding LLaVA-referenced images" if split == "train" else ""
        raise ValueError(f"COCO {split} captions produced no records{exclusion_note}")
    return records


def coco_caption_messages(image_path: str, caption: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": caption},
            ],
        }
    ]


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def _common_suffix_length(left: list[int], right: list[int], max_length: int) -> int:
    length = 0
    while length < max_length and left[-1 - length] == right[-1 - length]:
        length += 1
    return length


def tokenize_coco_caption_context(
    processor,
    image_path: str,
    caption: str,
) -> tuple[list[dict[str, Any]], list[int], dict[str, torch.Tensor], list[int]]:
    """Tokenize the exact VLM context and locate its caption-dependent tokens."""

    messages = coco_caption_messages(image_path, caption)
    context_input_ids, proc_inputs = vlm_tokenize_target(
        processor,
        messages,
        add_generation_prompt=False,
        max_pixels=DEFAULT_MAX_PIXELS,
    )
    empty_ids, _ = vlm_tokenize_target(
        processor,
        coco_caption_messages(image_path, ""),
        add_generation_prompt=False,
        max_pixels=DEFAULT_MAX_PIXELS,
    )
    prefix_length = _common_prefix_length(context_input_ids, empty_ids)
    max_suffix = min(len(context_input_ids) - prefix_length, len(empty_ids) - prefix_length)
    suffix_length = _common_suffix_length(context_input_ids, empty_ids, max_suffix)
    caption_end = len(context_input_ids) - suffix_length
    caption_positions = list(range(prefix_length, caption_end))
    if not caption_positions:
        raise ValueError(f"Caption produced no caption-specific tokens: {caption!r}")
    return messages, context_input_ids, proc_inputs, caption_positions


def sample_adjacent_caption_spans(
    caption_positions: list[int],
    dataset_params: CocoCaptionsPastLensDatasetConfig,
    rng: random.Random,
) -> tuple[Direction, list[int], list[int]]:
    """Sample adjacent, non-overlapping activation and target spans."""

    direction = rng.choice(dataset_params.directions)
    valid_sizes = [
        (k_tokens, k_activations)
        for k_tokens in range(dataset_params.min_k_tokens, dataset_params.max_k_tokens + 1)
        for k_activations in range(
            dataset_params.min_k_activations,
            dataset_params.max_k_activations + 1,
        )
        if k_tokens + k_activations <= len(caption_positions)
    ]
    if not valid_sizes:
        minimum_required = dataset_params.min_k_tokens + dataset_params.min_k_activations
        raise ValueError(
            f"Caption has {len(caption_positions)} token positions, "
            f"but configured spans require at least {minimum_required}"
        )
    k_tokens, k_activations = rng.choice(valid_sizes)
    required = k_tokens + k_activations
    first = rng.randint(0, len(caption_positions) - required)
    if direction == "past":
        target_positions = caption_positions[first : first + k_tokens]
        activation_positions = caption_positions[first + k_tokens : first + required]
    else:
        activation_positions = caption_positions[first : first + k_activations]
        target_positions = caption_positions[first + k_activations : first + required]
    if set(activation_positions) & set(target_positions):
        raise AssertionError("Activation and target spans overlap")
    return direction, activation_positions, target_positions


def _prediction_prompt(direction: Direction, k_tokens: int) -> str:
    if direction == "past":
        return f"Can you predict the previous {k_tokens} tokens that came before this?"
    return f"Can you predict the next {k_tokens} tokens that come after this?"


def create_coco_captions_past_lens_dataset(
    records: list[dict[str, Any]],
    processor,
    tokenizer,
    model_name: str,
    act_layers: list[int],
    dataset_params: CocoCaptionsPastLensDatasetConfig,
    save_acts: bool,
    num_examples: int,
    seed: int,
    model_kwargs: dict[str, Any] | None = None,
) -> list[TrainingDataPoint]:
    """Create one sampled context task per selected caption record."""

    validate_coco_captions_config(dataset_params)
    if not act_layers:
        raise ValueError("act_layers must contain at least one layer")
    if not records:
        raise ValueError("records must not be empty")
    if num_examples < 0:
        raise ValueError("num_examples must be non-negative")
    requested = len(records) if num_examples == 0 else num_examples
    if requested > len(records):
        raise ValueError(f"Requested {requested} examples, but only {len(records)} COCO captions are available")

    rng = random.Random(seed)
    shuffled_records = list(records)
    rng.shuffle(shuffled_records)

    model = None
    submodules = None
    device = torch.device("cpu")
    if save_acts:
        model = load_model(model_name, torch.bfloat16, **(model_kwargs or {}))
        model.eval()
        submodules = {layer: get_hf_submodule(model, layer) for layer in act_layers}
        device = next(model.parameters()).device

    data: list[TrainingDataPoint] = []
    accepted = 0
    for record in tqdm(shuffled_records, desc="Creating COCO caption context dataset"):
        messages, context_ids, proc_inputs, caption_positions = tokenize_coco_caption_context(
            processor,
            record["image_path"],
            record["caption"],
        )
        try:
            direction, activation_positions, target_positions = sample_adjacent_caption_spans(
                caption_positions,
                dataset_params,
                rng,
            )
        except ValueError:
            continue

        target_ids = [context_ids[position] for position in target_positions]
        target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
        if not target_text:
            continue

        acts_by_layer = None
        selected_layers = act_layers
        if save_acts:
            inputs = vision_inputs_to_device(proc_inputs, device)
            with torch.no_grad():
                acts_by_layer = collect_activations_multiple_layers(
                    model,
                    submodules,
                    inputs,
                    None,
                    None,
                )
        else:
            selected_layers = [rng.choice(act_layers)]

        for layer in selected_layers:
            acts = None
            if save_acts:
                acts = acts_by_layer[layer][0, activation_positions].detach().contiguous()
            data.append(
                create_training_datapoint(
                    datapoint_type="coco_captions_past_lens",
                    prompt=_prediction_prompt(direction, len(target_positions)),
                    target_response=target_text,
                    layer=layer,
                    num_positions=len(activation_positions),
                    tokenizer=tokenizer,
                    acts_BD=acts,
                    feature_idx=-1,
                    context_input_ids=context_ids,
                    context_positions=activation_positions,
                    context_image_paths=extract_image_paths(messages),
                    meta_info={
                        "target_messages": messages,
                        "add_generation_prompt": False,
                        "annotation_id": record["annotation_id"],
                        "image_id": record["image_id"],
                        "file_name": record["file_name"],
                        "caption": record["caption"],
                        "caption_positions": caption_positions,
                        "activation_positions": activation_positions,
                        "target_positions": target_positions,
                        "direction": direction,
                    },
                )
            )
        accepted += 1
        if accepted == requested:
            break

    if accepted != requested:
        raise RuntimeError(
            f"Could create only {accepted} of {requested} requested examples; "
            "captions were too short for the configured token and activation ranges"
        )
    return data
