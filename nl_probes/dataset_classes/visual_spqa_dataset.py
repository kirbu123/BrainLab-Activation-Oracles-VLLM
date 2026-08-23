"""Visual SPQA: LLaVA-Instruct image chats + hidden LatentQA system prompts."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

import nl_probes.dataset_classes.misc.latentqa_loader as latentqa_loader
from nl_probes.dataset_classes.act_dataset_manager import (
    ActDatasetLoader,
    BaseDatasetConfig,
    DatasetLoaderConfig,
)
from nl_probes.utils.common import layer_percent_to_layer, load_processor, load_tokenizer
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint
from nl_probes.utils.vlm_utils import DEFAULT_MAX_PIXELS, extract_image_paths, vlm_tokenize_target

IMAGE_TOKEN_RE = re.compile(r"\s*<image>\s*", re.IGNORECASE)


@dataclass
class VisualSPQADatasetConfig(BaseDatasetConfig):
    llava_json_path: str = "data/train/llava/llava_instruct_150k.json"
    coco_image_dir: str = "data/train/coco/train2017"
    latentqa_dir: str = "data/train/latentqa"
    max_window_size: int = 3
    min_window_size: int = 1
    min_end_offset: int = -1
    max_end_offset: int = -10
    position_types: list[str] = field(default_factory=lambda: ["all", "window"])
    max_all_positions: int = 64
    include_assistant_prob: float = 0.5
    max_pixels: int = DEFAULT_MAX_PIXELS


class VisualSPQADatasetLoader(ActDatasetLoader):
    def __init__(self, dataset_config: DatasetLoaderConfig):
        super().__init__(dataset_config)
        assert self.dataset_config.dataset_name == "", (
            f"{self.dataset_config.dataset_name}, Dataset name gets overridden here"
        )
        self.dataset_config.dataset_name = "visual_spqa"
        self.dataset_params: VisualSPQADatasetConfig = dataset_config.custom_dataset_params

        if "train" in self.dataset_config.splits and self.dataset_config.num_train < self.dataset_config.batch_size:
            raise ValueError(
                f"num_train {self.dataset_config.num_train} must be >= batch_size {self.dataset_config.batch_size}"
            )

    def create_dataset(self) -> None:
        tokenizer = load_tokenizer(self.dataset_config.model_name)
        processor = load_processor(self.dataset_config.model_name)
        layers = [
            layer_percent_to_layer(self.dataset_config.model_name, layer_percent)
            for layer_percent in self.dataset_config.layer_percents
        ]

        llava_examples = load_llava_examples(
            self.dataset_params.llava_json_path,
            self.dataset_params.coco_image_dir,
        )
        latentqa_items = load_latentqa_overlays(
            latentqa_dir=self.dataset_params.latentqa_dir,
            seed=self.dataset_config.seed,
        )
        if not llava_examples:
            raise FileNotFoundError(
                f"No LLaVA examples with images under {self.dataset_params.coco_image_dir}. "
                "Run scripts/download_vlm_ao_data.sh"
            )
        if not latentqa_items:
            raise RuntimeError("Failed to load LatentQA overlays")

        rng = random.Random(self.dataset_config.seed)
        rng.shuffle(llava_examples)

        n_train = self.dataset_config.num_train
        if n_train <= 0 or n_train > len(llava_examples):
            n_train = len(llava_examples)

        training_data: list[TrainingDataPoint] = []
        skipped = 0
        for i in tqdm(range(n_train), desc="Creating visual SPQA dataset"):
            llava = llava_examples[i]
            overlay = latentqa_items[i % len(latentqa_items)]
            try:
                dp = create_visual_spqa_datapoint(
                    llava=llava,
                    overlay=overlay,
                    processor=processor,
                    tokenizer=tokenizer,
                    act_layers=layers,
                    dataset_params=self.dataset_params,
                    rng=rng,
                )
            except Exception as exc:
                skipped += 1
                if skipped <= 10:
                    print(f"Skipping LLaVA example {llava.get('id')}: {exc}")
                continue
            training_data.append(dp)

        print(
            f"Visual SPQA: kept {len(training_data)} / {n_train} "
            f"(skipped {skipped}, images available {len(llava_examples)})"
        )
        if not training_data:
            raise RuntimeError("Visual SPQA produced zero training examples")

        if "train" in self.dataset_config.splits:
            self.save_dataset(training_data, "train")


def load_llava_examples(json_path: str, coco_dir: str) -> list[dict]:
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"LLaVA JSON not found: {json_path}. Run scripts/download_vlm_ao_data.sh")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    coco_root = Path(coco_dir)
    examples = []
    missing_images = 0
    for item in raw:
        image_name = item.get("image") or item.get("image_id")
        if not image_name:
            continue
        image_name = str(image_name)
        if not image_name.endswith(".jpg"):
            image_name = f"{image_name}.jpg"
        image_path = coco_root / image_name
        if not image_path.exists():
            # Some dumps store COCO ids without zero-padding
            missing_images += 1
            continue
        human, gpt = _first_human_gpt_turn(item.get("conversations") or [])
        if not human:
            continue
        examples.append(
            {
                "id": item.get("id"),
                "image_path": str(image_path),
                "human": strip_image_token(human),
                "gpt": gpt or "",
            }
        )
    print(f"LLaVA: {len(examples)} examples with images, {missing_images} missing image files")
    return examples


def _first_human_gpt_turn(conversations: list[dict]) -> tuple[str, str]:
    human = ""
    gpt = ""
    for turn in conversations:
        speaker = (turn.get("from") or turn.get("role") or "").lower()
        value = turn.get("value") or turn.get("content") or ""
        if speaker in {"human", "user"} and not human:
            human = value
        elif speaker in {"gpt", "assistant"} and human and not gpt:
            gpt = value
            break
    return human, gpt


def strip_image_token(text: str) -> str:
    return IMAGE_TOKEN_RE.sub(" ", text).strip()


def load_latentqa_overlays(latentqa_dir: str, seed: int = 42) -> list[dict]:
    root = Path(latentqa_dir)
    paths = latentqa_loader.DataPaths(
        system=None,
        stimulus_completion=str(root / "stimulus_completion.json"),
        stimulus=str(root / "stimulus.json"),
        control=str(root / "control.json"),
        qa=str(root / "qa.json"),
    )
    ds = latentqa_loader.load_latentqa_dataset(
        paths,
        filter_prefixes=[],
        train_percent=1.0,
        add_thought_tokens=False,
        seed=seed,
    )
    overlays = []
    for item in ds:
        instruction = hidden_instruction_from_latentqa(item)
        dialog = item["dialog"]
        if len(dialog) < 2 or not instruction:
            continue
        overlays.append(
            {
                "instruction": instruction,
                "question": dialog[0]["content"],
                "answer": dialog[1]["content"],
                "source": item.get("source", "stimulus"),
                "label": item.get("label", ""),
            }
        )
    print(f"LatentQA overlays: {len(overlays)}")
    return overlays


def hidden_instruction_from_latentqa(item: dict) -> str:
    for msg in item.get("read_prompt") or []:
        if msg.get("role") == "system" and msg.get("content"):
            return msg["content"]
    read_prompt = item.get("read_prompt") or []
    if read_prompt and read_prompt[0].get("role") == "user":
        return read_prompt[0].get("content") or ""
    return ""


def _system_prefix_len(processor, instruction: str) -> int:
    tokenizer = getattr(processor, "tokenizer", processor)
    system_messages = [{"role": "system", "content": instruction}]
    try:
        ids = tokenizer.apply_chat_template(
            system_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
            padding=False,
            enable_thinking=False,
        )
    except TypeError:
        ids = tokenizer.apply_chat_template(
            system_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
            padding=False,
        )
    if not isinstance(ids, list):
        ids = list(ids)
    return len(ids)


def sample_unmasked_positions(
    context_positions: list[int],
    dataset_params: VisualSPQADatasetConfig,
    rng: random.Random,
) -> list[int]:
    if not context_positions:
        raise ValueError("No unmasked context positions")
    position_type = rng.choice(dataset_params.position_types)
    if position_type == "all":
        if len(context_positions) > dataset_params.max_all_positions:
            return context_positions[-dataset_params.max_all_positions :]
        return list(context_positions)

    window_size = rng.randint(dataset_params.min_window_size, dataset_params.max_window_size)
    end_offset = rng.randint(dataset_params.max_end_offset, dataset_params.min_end_offset)
    if abs(end_offset) > len(context_positions):
        end_offset = -len(context_positions) + 1
    window_size = min(window_size, len(context_positions) + end_offset)
    window_size = max(window_size, 1)
    window_start = end_offset - window_size
    sliced = context_positions[window_start:end_offset]
    if not sliced:
        return context_positions[-1:]
    return sliced


def create_visual_spqa_datapoint(
    llava: dict,
    overlay: dict,
    processor,
    tokenizer,
    act_layers: list[int],
    dataset_params: VisualSPQADatasetConfig,
    rng: random.Random,
) -> TrainingDataPoint:
    include_assistant = bool(llava.get("gpt")) and rng.random() < dataset_params.include_assistant_prob
    messages: list[dict] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": overlay["instruction"]}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": llava["image_path"]},
                {"type": "text", "text": llava["human"]},
            ],
        },
    ]
    if include_assistant:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": llava["gpt"]}]})

    add_generation_prompt = not include_assistant
    context_input_ids, _ = vlm_tokenize_target(
        processor,
        messages,
        add_generation_prompt=add_generation_prompt,
        max_pixels=dataset_params.max_pixels,
    )

    system_len = min(_system_prefix_len(processor, overlay["instruction"]), len(context_input_ids) - 1)
    unmasked = list(range(system_len, len(context_input_ids)))
    context_positions = sample_unmasked_positions(unmasked, dataset_params, rng)
    layer = rng.choice(act_layers)

    return create_training_datapoint(
        datapoint_type="visual_spqa",
        prompt=overlay["question"],
        target_response=overlay["answer"],
        layer=layer,
        num_positions=len(context_positions),
        tokenizer=tokenizer,
        acts_BD=None,
        feature_idx=-1,
        context_input_ids=context_input_ids,
        context_positions=context_positions,
        context_image_paths=extract_image_paths(messages),
        ds_label=overlay.get("label"),
        meta_info={
            "target_messages": messages,
            "add_generation_prompt": add_generation_prompt,
            "llava_id": llava.get("id"),
            "source": overlay.get("source"),
        },
    )
