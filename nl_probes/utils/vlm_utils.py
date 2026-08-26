"""Helpers for Qwen3-VL / multimodal target-side tokenization and forwards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

LEGACY_COCO_IMAGE_PREFIXES = (
    ("data/coco/train2017/", "data/train/coco/train2017/"),
    ("data/coco/val2017/", "data/val/coco/val2017/"),
)


def vlm_image_path_candidates(image: str) -> tuple[str, ...]:
    path = image[2:] if image.startswith("./") else image
    candidates = [path]
    for old_prefix, new_prefix in LEGACY_COCO_IMAGE_PREFIXES:
        if path.startswith(old_prefix):
            candidates.append(new_prefix + path[len(old_prefix) :])
    return tuple(dict.fromkeys(candidates))


def resolve_vlm_image_path(image: str) -> str:
    tried = vlm_image_path_candidates(image)
    for candidate in tried:
        if Path(candidate).is_file():
            return candidate
    raise FileNotFoundError(f"VLM image not found. Tried: {list(tried)}")


def messages_with_resolved_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    for message in messages:
        msg = dict(message)
        content = msg.get("content")
        if isinstance(content, list):
            new_content = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    part = dict(part)
                    key = "image" if "image" in part else "path"
                    if isinstance(part.get(key), str):
                        part[key] = resolve_vlm_image_path(part[key])
                new_content.append(part)
            msg["content"] = new_content
        rewritten.append(msg)
    return rewritten


DEFAULT_MAX_PIXELS = 1280 * 28 * 28

VLM_FORWARD_KEYS = {
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
    "mm_token_type_ids",
}


def apply_chat_template_safe(tokenizer_or_processor, messages, **kwargs):
    """apply_chat_template with optional enable_thinking for Qwen3 vs Qwen3-VL."""
    try:
        return tokenizer_or_processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer_or_processor.apply_chat_template(messages, **kwargs)


def extract_image_paths(messages: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "image":
                continue
            image = part.get("image") or part.get("path")
            if isinstance(image, str):
                paths.append(resolve_vlm_image_path(image))
            elif isinstance(image, Path):
                paths.append(resolve_vlm_image_path(str(image)))
    return paths


def _load_pil_images(image_paths: list[str]) -> list[Image.Image]:
    images = []
    for path in image_paths:
        with Image.open(resolve_vlm_image_path(path)) as img:
            images.append(img.convert("RGB"))
    return images


def _maybe_set_max_pixels(processor, max_pixels: int | None) -> None:
    if max_pixels is None:
        return
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = max_pixels
    if hasattr(image_processor, "size") and isinstance(image_processor.size, dict):
        image_processor.size["longest_edge"] = max_pixels


def vlm_tokenize_target(
    processor,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool = True,
    max_pixels: int | None = DEFAULT_MAX_PIXELS,
) -> tuple[list[int], dict[str, torch.Tensor]]:
    """Tokenize a multimodal target prompt after image-token expansion.

    Returns (unpadded input_ids, processor batch dict on CPU).
    """
    _maybe_set_max_pixels(processor, max_pixels)
    messages = messages_with_resolved_images(messages)
    image_paths = extract_image_paths(messages)

    try:
        inputs = apply_chat_template_safe(
            processor,
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )
        if hasattr(inputs, "pop"):
            inputs.pop("token_type_ids", None)
    except Exception:
        text = apply_chat_template_safe(
            processor,
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        images = _load_pil_images(image_paths) if image_paths else None
        inputs = processor(text=[text], images=images, return_tensors="pt", padding=False)
        if hasattr(inputs, "pop"):
            inputs.pop("token_type_ids", None)

    input_ids = inputs["input_ids"]
    if input_ids.ndim == 2:
        ids_list = input_ids[0].tolist()
    else:
        ids_list = input_ids.tolist()
    filtered = {k: v for k, v in dict(inputs).items() if k in VLM_FORWARD_KEYS}
    return ids_list, filtered


def vision_inputs_to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved
