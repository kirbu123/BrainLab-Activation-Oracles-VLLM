"""Shared infrastructure for binary visual classification datasets."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class VLMBinaryRecord:
    """One source image/question pair with a normalized binary answer."""

    source_id: str
    image_path: str
    context_text: str
    question: str
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.answer not in {YES_TOKEN, NO_TOKEN}:
            raise ValueError(f"Binary answer must be {YES_TOKEN!r} or {NO_TOKEN!r}, got {self.answer!r}")


@dataclass
class VLMBinaryDatasetConfig(BaseDatasetConfig):
    min_end_offset: int = -1
    max_end_offset: int = -5
    max_window_size: int = 5
    min_window_size: int = 1
    max_pixels: int = DEFAULT_MAX_PIXELS


def normalize_binary_label(
    label: object,
    *,
    positive_labels: tuple[object, ...] = (True, 1, "1", "yes", "true"),
    negative_labels: tuple[object, ...] = (False, 0, "0", "no", "false"),
) -> str:
    """Normalize a known binary label, rejecting ambiguous values."""

    normalized = label.strip().lower() if isinstance(label, str) else label
    normalized_positive = tuple(
        value.strip().lower() if isinstance(value, str) else value for value in positive_labels
    )
    normalized_negative = tuple(
        value.strip().lower() if isinstance(value, str) else value for value in negative_labels
    )
    if normalized in normalized_positive:
        return YES_TOKEN
    if normalized in normalized_negative:
        return NO_TOKEN
    raise ValueError(f"Unsupported binary label: {label!r}")


def subsample_binary_records(
    records: list[VLMBinaryRecord],
    num_records: int,
    seed: int,
    *,
    balanced: bool = False,
) -> list[VLMBinaryRecord]:
    """Return a deterministic shuffled subset without mutating the input."""

    rng = random.Random(seed)
    if not balanced:
        shuffled = list(records)
        rng.shuffle(shuffled)
        return shuffled if num_records <= 0 or num_records >= len(shuffled) else shuffled[:num_records]

    positives = [record for record in records if record.answer == YES_TOKEN]
    negatives = [record for record in records if record.answer == NO_TOKEN]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    available = min(len(positives) + len(negatives), 2 * min(len(positives), len(negatives)) + 1)
    requested = available if num_records <= 0 else min(num_records, available)
    positive_count = (requested + 1) // 2
    negative_count = requested // 2
    selected = positives[:positive_count] + negatives[:negative_count]
    rng.shuffle(selected)
    return selected


class VLMBinaryDatasetLoader(ActDatasetLoader):
    """Base loader that materializes official train/test binary VLM splits."""

    dataset_name: str

    def __init__(
        self,
        dataset_config: DatasetLoaderConfig,
        model_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(dataset_config)
        assert self.dataset_config.dataset_name == "", (
            f"{self.dataset_config.dataset_name}, dataset name gets overridden here"
        )
        self.dataset_config.dataset_name = self.dataset_name
        self.dataset_params: VLMBinaryDatasetConfig = dataset_config.custom_dataset_params
        self.model_kwargs = {} if model_kwargs is None else model_kwargs

    def records_for_split(self, split: str) -> list[VLMBinaryRecord]:
        raise NotImplementedError

    def should_balance_split(self, split: str) -> bool:
        return False

    def default_num_records(self, split: str, available: int) -> int:
        return available

    def expand_records(
        self,
        records: list[VLMBinaryRecord],
        rng: random.Random,
    ) -> list[VLMBinaryRecord]:
        return records

    def create_dataset(self) -> None:
        set_seed(self.dataset_config.seed)
        tokenizer = load_tokenizer(self.dataset_config.model_name)
        processor = load_processor(self.dataset_config.model_name)
        layers = [
            layer_percent_to_layer(self.dataset_config.model_name, layer_percent)
            for layer_percent in self.dataset_config.layer_percents
        ]

        for split in self.dataset_config.splits:
            records = self.records_for_split(split)
            requested = (
                self.dataset_config.num_train if split == "train" else self.dataset_config.num_test
            )
            if requested <= 0:
                requested = self.default_num_records(split, len(records))
            split_seed = self.dataset_config.seed + (0 if split == "train" else 1)
            selected = subsample_binary_records(
                records,
                requested,
                split_seed,
                balanced=self.should_balance_split(split),
            )
            if not selected:
                raise RuntimeError(f"{self.dataset_name} {split} split produced zero records")
            rng = random.Random(split_seed)
            expanded = self.expand_records(selected, rng)
            data = create_vlm_binary_vector_dataset(
                records=expanded,
                processor=processor,
                tokenizer=tokenizer,
                model_name=self.dataset_config.model_name,
                act_layers=layers,
                dataset_params=self.dataset_params,
                datapoint_type=self.dataset_name,
                save_acts=self.dataset_config.save_acts,
                batch_size=max(1, self.dataset_config.batch_size),
                model_kwargs=self.model_kwargs,
                rng=rng,
            )
            self.save_dataset(data, split)


def sample_context_positions(
    context_length: int,
    dataset_params: VLMBinaryDatasetConfig,
    rng: random.Random,
) -> list[int]:
    if context_length < 2:
        raise ValueError(f"Binary VLM context must contain at least two tokens, got {context_length}")
    end_offset = rng.randint(dataset_params.max_end_offset, dataset_params.min_end_offset)
    end_pos = max(1, min(context_length + end_offset, context_length - 1))
    window_size = rng.randint(dataset_params.min_window_size, dataset_params.max_window_size)
    window_size = max(1, min(window_size, end_pos + 1))
    return list(range(end_pos - window_size + 1, end_pos + 1))


def create_vlm_binary_vector_dataset(
    records: list[VLMBinaryRecord],
    processor,
    tokenizer,
    model_name: str,
    act_layers: list[int],
    dataset_params: VLMBinaryDatasetConfig,
    datapoint_type: str,
    save_acts: bool,
    batch_size: int,
    model_kwargs: dict[str, Any],
    rng: random.Random,
) -> list[TrainingDataPoint]:
    """Convert normalized records into activation-oracle datapoints."""

    if not act_layers:
        raise ValueError("At least one activation layer is required")

    model = None
    submodules = None
    device = torch.device("cpu")
    if save_acts:
        model = load_model(model_name, torch.bfloat16, **model_kwargs)
        model.eval()
        submodules = {layer: get_hf_submodule(model, layer) for layer in act_layers}
        device = next(model.parameters()).device

    training_data: list[TrainingDataPoint] = []
    for batch_start in tqdm(
        range(0, len(records), batch_size),
        desc=f"{datapoint_type} vector dataset",
    ):
        batch = records[batch_start : batch_start + batch_size]
        for record in batch:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": record.image_path},
                        {"type": "text", "text": record.context_text},
                    ],
                }
            ]
            context_input_ids, proc_inputs = vlm_tokenize_target(
                processor,
                messages,
                add_generation_prompt=True,
                max_pixels=dataset_params.max_pixels,
            )
            positions = sample_context_positions(len(context_input_ids), dataset_params, rng)

            acts_by_layer = None
            layers_for_record = act_layers if save_acts else [rng.choice(act_layers)]
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

            for layer in layers_for_record:
                acts = None
                if save_acts:
                    acts = acts_by_layer[layer][0, positions].detach().contiguous()
                metadata = {
                    "target_messages": messages,
                    "add_generation_prompt": True,
                    "source_id": record.source_id,
                    **record.metadata,
                }
                training_data.append(
                    create_training_datapoint(
                        datapoint_type=datapoint_type,
                        prompt=record.question,
                        target_response=record.answer,
                        layer=layer,
                        num_positions=len(positions),
                        tokenizer=tokenizer,
                        acts_BD=acts,
                        feature_idx=-1,
                        context_input_ids=context_input_ids,
                        context_positions=positions,
                        context_image_paths=extract_image_paths(messages),
                        ds_label=record.answer,
                        meta_info=metadata,
                    )
                )
    return training_data
