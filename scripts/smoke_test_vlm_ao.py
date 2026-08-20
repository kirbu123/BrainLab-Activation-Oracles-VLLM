"""Smoke test for Qwen3-VL Visual Activation Oracle plumbing (no DDP)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image

from nl_probes.utils.activation_utils import (
    freeze_vision_parameters,
    get_hf_submodule,
    get_text_only_lora_targets,
)
from nl_probes.utils.common import get_layer_count, layer_percent_to_layer, load_model, load_processor, load_tokenizer
from nl_probes.utils.dataset_utils import create_training_datapoint, materialize_missing_steering_vectors
from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook
from nl_probes.utils.vlm_utils import extract_image_paths, vlm_tokenize_target


def _tiny_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(120, 80, 40)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--image", default="")
    args = parser.parse_args()

    model_name = args.model_name
    n_layers = get_layer_count(model_name)
    layer = layer_percent_to_layer(model_name, 50)
    print(f"layer_count={n_layers} layer_50={layer}")
    print(f"LoRA targets={get_text_only_lora_targets(model_name)}")

    image_path = Path(args.image) if args.image else Path("data/smoke/tiny.jpg")
    if not image_path.exists():
        _tiny_image(image_path)
        print(f"Wrote dummy image {image_path}")

    dtype = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_name, dtype, device_map={"": str(device)} if device.type == "cuda" else {})
    freeze_vision_parameters(model)
    processor = load_processor(model_name)
    tokenizer = load_tokenizer(model_name)

    submodule = get_hf_submodule(model, 1)
    print(f"hook submodule: {type(submodule).__name__}")

    targets = get_text_only_lora_targets(model_name)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=targets or "all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config, autocast_adapter_dtype=True)
    model.print_trainable_parameters()

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "Answer like a pirate."}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": "What is in this picture?"},
            ],
        },
    ]
    context_input_ids, _ = vlm_tokenize_target(processor, messages, add_generation_prompt=True)
    context_positions = list(range(max(0, len(context_input_ids) - 4), len(context_input_ids)))
    print(f"target seq_len={len(context_input_ids)} act_positions={context_positions}")

    dp = create_training_datapoint(
        datapoint_type="visual_spqa_smoke",
        prompt="What is the assistant's tone?",
        target_response="Playful and pirate-like.",
        layer=layer,
        num_positions=len(context_positions),
        tokenizer=tokenizer,
        acts_BD=None,
        feature_idx=-1,
        context_input_ids=context_input_ids,
        context_positions=context_positions,
        context_image_paths=extract_image_paths(messages),
        meta_info={"target_messages": messages, "add_generation_prompt": True},
    )

    filled = materialize_missing_steering_vectors([dp], tokenizer, model, processor=processor)
    vec = filled[0].steering_vectors
    assert vec is not None and vec.ndim == 2, vec
    print(f"steering_vectors shape={tuple(vec.shape)}")

    from nl_probes.utils.dataset_utils import construct_batch

    batch = construct_batch(filled, tokenizer, device)
    hook_fn = get_hf_activation_steering_hook(
        vectors=batch.steering_vectors,
        positions=batch.positions,
        steering_coefficient=1.0,
        device=device,
        dtype=dtype,
    )
    hook_sub = get_hf_submodule(model, 1, use_lora=True)
    with add_hook(hook_sub, hook_fn):
        out = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
        )
    loss = float(out.loss.detach().cpu())
    print(f"oracle CE loss={loss}")
    if not torch.isfinite(out.loss):
        raise RuntimeError("Non-finite smoke-test loss")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
