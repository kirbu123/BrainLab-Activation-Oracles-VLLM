"""Multimodal, assistant-only loss collation for Qwen3-VL."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from qwen_vl_utils import process_vision_info

from nl_probes.utils.vlm_utils import apply_chat_template_safe


VisionLoader = Callable[[list[dict[str, Any]]], tuple[Any, Any]]


class Qwen3VLAssistantOnlyCollator:
    """Encode image chats and mask every non-assistant response token."""

    def __init__(
        self,
        processor: Any,
        max_length: int,
        max_pixels: int,
        vision_loader: VisionLoader = process_vision_info,
    ):
        if max_length <= 0 or max_pixels <= 0:
            raise ValueError("max_length and max_pixels must be positive")
        self.processor = processor
        self.max_length = max_length
        self.vision_loader = vision_loader

        image_processor = processor.image_processor
        image_processor.max_pixels = max_pixels
        if isinstance(image_processor.size, dict):
            image_processor.size["longest_edge"] = max_pixels

        tokenizer = processor.tokenizer
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Processor tokenizer has neither pad_token_id nor eos_token_id")
            tokenizer.pad_token = tokenizer.eos_token
        self.pad_token_id = tokenizer.pad_token_id

    def _encode(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> dict[str, torch.Tensor]:
        rendered = apply_chat_template_safe(
            self.processor,
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        image_inputs, video_inputs = self.vision_loader(messages)
        if video_inputs:
            raise ValueError("Target training accepts images, not videos")
        encoded = self.processor(
            text=[rendered],
            images=image_inputs,
            videos=None,
            padding=False,
            return_tensors="pt",
        )
        tensors = {name: value for name, value in dict(encoded).items() if torch.is_tensor(value)}
        if "input_ids" not in tensors or tensors["input_ids"].shape[0] != 1:
            raise ValueError("Processor must return one input_ids row per encoded example")
        return tensors

    @staticmethod
    def _require_prefix(prefix: list[int], sequence: list[int], description: str) -> None:
        if sequence[: len(prefix)] != prefix:
            raise ValueError(
                f"Qwen chat tokenization changed before {description}; "
                "cannot construct an exact assistant-only loss mask"
            )

    def _encode_example(self, feature: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if set(feature) != {"organism_id", "messages"}:
            raise ValueError(
                "Collator features must contain exactly 'organism_id' and 'messages'"
            )
        messages = feature["messages"]
        full = self._encode(messages, add_generation_prompt=False)
        full_ids = full["input_ids"][0].tolist()
        if len(full_ids) > self.max_length:
            raise ValueError(
                f"Encoded example has {len(full_ids)} tokens, exceeding max_length="
                f"{self.max_length}; regenerate or raise --max-length"
            )

        assistant_mask = torch.zeros(len(full_ids), dtype=torch.bool)
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            prompt = self._encode(messages[:index], add_generation_prompt=True)
            through_response = self._encode(messages[: index + 1], add_generation_prompt=False)
            prompt_ids = prompt["input_ids"][0].tolist()
            response_ids = through_response["input_ids"][0].tolist()
            self._require_prefix(prompt_ids, response_ids, f"assistant turn {index}")
            self._require_prefix(response_ids, full_ids, f"assistant turn {index}")
            if len(response_ids) == len(prompt_ids):
                raise ValueError(f"Assistant turn {index} contains no trainable tokens")
            assistant_mask[len(prompt_ids) : len(response_ids)] = True

        if not assistant_mask.any():
            raise ValueError("Example contains no assistant response tokens")
        return full, assistant_mask

    @staticmethod
    def _pad_rows(
        rows: list[torch.Tensor],
        max_length: int,
        pad_value: int | float,
    ) -> torch.Tensor:
        padded = []
        for row in rows:
            pad_width = max_length - row.shape[-1]
            padding = torch.full(
                (*row.shape[:-1], pad_width),
                pad_value,
                dtype=row.dtype,
                device=row.device,
            )
            padded.append(torch.cat((row, padding), dim=-1))
        return torch.cat(padded, dim=0)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty batch")
        encoded_and_masks = [self._encode_example(feature) for feature in features]
        encoded = [item[0] for item in encoded_and_masks]
        masks = [item[1] for item in encoded_and_masks]
        lengths = [item["input_ids"].shape[-1] for item in encoded]
        batch_length = max(lengths)

        key_sets = [set(item) for item in encoded]
        if any(keys != key_sets[0] for keys in key_sets[1:]):
            raise ValueError("Processor returned inconsistent tensor fields across examples")

        batch: dict[str, torch.Tensor] = {}
        sequence_keys = {
            key
            for key, value in encoded[0].items()
            if value.ndim == 2 and value.shape[-1] == lengths[0]
        }
        for key in key_sets[0]:
            values = [item[key] for item in encoded]
            if key in sequence_keys:
                pad_value = self.pad_token_id if key == "input_ids" else 0
                batch[key] = self._pad_rows(values, batch_length, pad_value)
            else:
                batch[key] = torch.cat(values, dim=0)

        labels = batch["input_ids"].clone()
        labels.fill_(-100)
        for row_index, (mask, length) in enumerate(zip(masks, lengths, strict=True)):
            labels[row_index, :length][mask] = batch["input_ids"][row_index, :length][mask]
        batch["labels"] = labels
        return batch
