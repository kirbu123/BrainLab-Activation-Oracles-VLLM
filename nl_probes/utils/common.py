import random
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig


def is_qwen3_vl(model_name: str) -> bool:
    return "qwen3-vl" in model_name.lower()


def is_vlm_model(model_name: str) -> bool:
    name = model_name.lower()
    return "qwen3-vl" in name or "gemma-3" in name


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch for reproducible runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(
    model_name: str,
    dtype: torch.dtype,
    **model_kwargs,
) -> AutoModelForCausalLM:
    print("🧠 Loading model...")

    requested_attn = model_kwargs.pop("attn_implementation", None)
    if requested_attn is None:
        requested_attn = "eager" if "gemma" in model_name.lower() else "flash_attention_2"

    kwargs: dict = {
        "device_map": "auto",
        "torch_dtype": dtype,
        **model_kwargs,
    }
    # transformers>=4.57 prefers `dtype`; keep torch_dtype for older callers.
    kwargs.setdefault("dtype", dtype)

    def _from_pretrained(attn: str):
        local_kwargs = {**kwargs, "attn_implementation": attn}
        if is_qwen3_vl(model_name):
            from transformers import Qwen3VLForConditionalGeneration

            return Qwen3VLForConditionalGeneration.from_pretrained(model_name, **local_kwargs)
        return AutoModelForCausalLM.from_pretrained(model_name, **local_kwargs)

    try:
        return _from_pretrained(requested_attn)
    except (ImportError, ValueError) as exc:
        if requested_attn != "sdpa":
            print(f"Falling back to sdpa attention ({exc})")
            return _from_pretrained("sdpa")
        raise


def load_processor(model_name: str) -> Any:
    print("📦 Loading processor...")
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if not tokenizer.bos_token_id:
            tokenizer.bos_token_id = tokenizer.eos_token_id
    return processor


def load_tokenizer(
    model_name: str,
) -> AutoTokenizer:
    # Load tokenizer. For VLMs, share the processor tokenizer so chat templates stay aligned.
    print("📦 Loading tokenizer...")
    if is_vlm_model(model_name):
        processor = load_processor(model_name)
        tokenizer = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if not tokenizer.bos_token_id:
        tokenizer.bos_token_id = tokenizer.eos_token_id
    return tokenizer


def list_decode(x: torch.Tensor, tokenizer: AutoTokenizer) -> list[list[str]]:
    """
    Input: torch.Tensor of shape [batch_size, seq_length]
    Output: list of list of strings of len [batch_size, seq_length] Each inner list corresponds to a single token
    """
    assert len(x.shape) == 1 or len(x.shape) == 2
    # Convert to list of lists, even if x is 1D
    if len(x.shape) == 1:
        x = x.unsqueeze(0)  # Make it 2D for consistent handling

    # Convert tensor to list of list of ints
    token_ids = x.tolist()

    # Convert token ids to token strings
    return [tokenizer.batch_decode(seq, skip_special_tokens=False) for seq in token_ids]


def get_bos_eos_pad_mask(tokenizer: AutoTokenizer, token_ids: torch.Tensor) -> torch.Tensor:
    """Create mask for BOS, EOS, and PAD tokens"""
    mask = torch.zeros_like(token_ids, dtype=torch.bool)

    if tokenizer.bos_token_id is not None:
        mask |= token_ids == tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        mask |= token_ids == tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        mask |= token_ids == tokenizer.pad_token_id

    return mask


def assert_no_peft_present(model, check_for_active_adapter_only=False):
    """
    Asserts that no PEFT adapters are present or active on the model.

    Args:
        model: The model to check.
        check_for_active_adapter_only (bool):
            - If False (default), asserts that NO adapters are loaded on the model at all.
            - If True, asserts only that no adapter is currently *active*.
              This allows inactive adapters to still be loaded in memory.
    """
    is_peft_model = isinstance(model, PeftModel)

    if not is_peft_model and not hasattr(model, "peft_config"):
        # If it's not a PeftModel and has no peft_config, we're 100% sure no adapters are loaded.
        return

    # At this point, the model has had PEFT adapters at some point.

    # getattr is used to safely access peft_config, which might be an empty dict.
    loaded_adapters = list(getattr(model, "peft_config", {}).keys())

    if not check_for_active_adapter_only:
        assert not loaded_adapters, (
            f"PEFT check failed! Found loaded adapters: {loaded_adapters}. "
            "Model should have no adapters loaded in memory."
        )

    # PeftModel has an `active_adapters` property which is a list of active adapter names.
    # It's an empty list when the base model is active.
    active_adapters = getattr(model, "active_adapters", [])
    assert not active_adapters, (
        f"PEFT check failed! Found active adapters: {active_adapters}. Model should be running in base mode."
    )


def get_layer_count(model_name: str) -> int:
    """Get the number of layers from a HuggingFace model config."""
    config = AutoConfig.from_pretrained(model_name)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        n = getattr(text_config, "num_hidden_layers", None)
        if n:
            return int(n)
    n = getattr(config, "num_hidden_layers", None)
    if n:
        return int(n)
    raise AttributeError(f"Could not find layer count for {model_name}")


def layer_percent_to_layer(model_name: str, layer_percent: int) -> int:
    """Convert a layer percent to a layer number."""
    max_layers = get_layer_count(model_name)
    return int(max_layers * (layer_percent / 100))
