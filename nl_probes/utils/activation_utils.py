import contextlib

import torch
from transformers import AutoModelForCausalLM


class EarlyStopException(Exception):
    """Custom exception for stopping model forward pass early."""

    pass


def collect_activations(
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    inputs_BL: dict[str, torch.Tensor],
    use_no_grad: bool = True,
) -> torch.Tensor:
    """
    Registers a forward hook on the submodule to capture the residual (or hidden)
    activations. We then raise an EarlyStopException to skip unneeded computations.

    Args:
        model: The model to run.
        submodule: The submodule to hook into.
        inputs_BL: The inputs to the model.
        use_no_grad: Whether to run the forward pass within a `torch.no_grad()` context. Defaults to True.
    """
    activations_BLD = None

    def gather_target_act_hook(module, inputs, outputs):
        nonlocal activations_BLD
        # For many models, the submodule outputs are a tuple or a single tensor:
        # If "outputs" is a tuple, pick the relevant item:
        #   e.g. if your layer returns (hidden, something_else), you'd do outputs[0]
        # Otherwise just do outputs
        if isinstance(outputs, tuple):
            activations_BLD = outputs[0]
        else:
            activations_BLD = outputs

        raise EarlyStopException("Early stopping after capturing activations")

    handle = submodule.register_forward_hook(gather_target_act_hook)

    # Determine the context manager based on the flag
    context_manager = torch.no_grad() if use_no_grad else contextlib.nullcontext()

    try:
        # Use the selected context manager
        with context_manager:
            _ = model(**inputs_BL)  # type: ignore
    except EarlyStopException:
        pass
    except Exception as e:
        print(f"Unexpected error during forward pass: {str(e)}")
        raise
    finally:
        handle.remove()

    return activations_BLD  # type: ignore


def collect_activations_multiple_layers(
    model: AutoModelForCausalLM,
    submodules: dict[int, torch.nn.Module],
    inputs_BL: dict[str, torch.Tensor],
    min_offset: int | None,
    max_offset: int | None,
) -> dict[int, torch.Tensor]:
    if min_offset is not None:
        assert max_offset is not None, "max_offset must be provided if min_offset is provided"
        assert max_offset < min_offset, "max_offset must be less than min_offset"
        assert min_offset < 0, "min_offset must be less than 0"
        assert max_offset < 0, "max_offset must be less than 0"
    else:
        assert max_offset is None, "max_offset must be provided if min_offset is not provided"

    activations_BLD_by_layer = {}

    module_to_layer = {submodule: layer for layer, submodule in submodules.items()}

    max_layer = max(submodules.keys())

    def gather_target_act_hook(module, inputs, outputs):
        layer = module_to_layer[module]

        if isinstance(outputs, tuple):
            activations_BLD_by_layer[layer] = outputs[0]
        else:
            activations_BLD_by_layer[layer] = outputs

        if min_offset is not None:
            activations_BLD_by_layer[layer] = activations_BLD_by_layer[layer][:, max_offset:min_offset, :]

        if layer == max_layer:
            raise EarlyStopException("Early stopping after capturing activations")

    handles = []

    for layer, submodule in submodules.items():
        handles.append(submodule.register_forward_hook(gather_target_act_hook))

    try:
        # Use the selected context manager
        with torch.no_grad():
            _ = model(**inputs_BL)
    except EarlyStopException:
        pass
    except Exception as e:
        print(f"Unexpected error during forward pass: {str(e)}")
        raise
    finally:
        for handle in handles:
            handle.remove()

    return activations_BLD_by_layer


# LoRA target patterns for text-only training on VLMs (vision-language models).
# When using target_modules='all-linear', LoRA adapters get added to the vision
# tower which won't receive gradients during text-only training, causing DDP errors.
# These patterns target only the language model layers.
VLM_TEXT_ONLY_LORA_TARGETS = {
    "gemma-3": r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    "qwen3-vl": r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
}


def get_text_only_lora_targets(model_name: str) -> str | None:
    """Returns LoRA target pattern for text-only training on VLMs, or None if not a VLM."""
    name = model_name.lower()
    # Longer / more specific keys first so "qwen3-vl" wins over a future generic "qwen" entry.
    for pattern, targets in sorted(VLM_TEXT_ONLY_LORA_TARGETS.items(), key=lambda kv: -len(kv[0])):
        if pattern in name:
            return targets
    return None


def freeze_vision_parameters(model: torch.nn.Module) -> int:
    """Freeze vision-tower params so oracle text-only steps do not DDP-error."""
    frozen = 0
    candidates = []
    for attr in ("visual", "vision_model", "vision_tower"):
        if hasattr(model, attr):
            candidates.append(getattr(model, attr))
    inner = getattr(model, "model", None)
    if inner is not None:
        for attr in ("visual", "vision_model", "vision_tower"):
            if hasattr(inner, attr):
                candidates.append(getattr(inner, attr))
    for module in candidates:
        if module is None:
            continue
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen += 1
    print(f"Froze {frozen} vision-tower parameters")
    return frozen


def _unwrap_peft_model(model: torch.nn.Module) -> torch.nn.Module:
    inner = model
    if hasattr(inner, "base_model") and hasattr(inner, "peft_config"):
        inner = inner.base_model
        if hasattr(inner, "model"):
            inner = inner.model
    return inner


def _get_language_layers(model: torch.nn.Module):
    root = _unwrap_peft_model(model)
    if hasattr(root, "model") and hasattr(root.model, "language_model"):
        return root.model.language_model.layers
    if hasattr(root, "language_model"):
        return root.language_model.layers
    if hasattr(root, "model") and hasattr(root.model, "layers"):
        return root.model.layers
    raise ValueError(f"Could not find language layers on {type(root)}")


def _get_vision_module(model: torch.nn.Module) -> torch.nn.Module:
    root = _unwrap_peft_model(model)
    candidates = [root]
    inner = getattr(root, "model", None)
    if inner is not None:
        candidates.append(inner)
    for obj in candidates:
        for attr in ("visual", "vision_model", "vision_tower"):
            module = getattr(obj, attr, None)
            if module is not None:
                return module
    raise ValueError(f"Could not find vision module on {type(root)}")


def _extract_deepstack_from_visual_output(output) -> list[torch.Tensor]:
    if hasattr(output, "deepstack_features") and output.deepstack_features is not None:
        features = list(output.deepstack_features)
        if not features:
            raise ValueError("visual output.deepstack_features is empty")
        return features
    if isinstance(output, tuple) and len(output) >= 2:
        maybe = output[-1]
        if isinstance(maybe, (list, tuple)) and maybe and torch.is_tensor(maybe[0]):
            return list(maybe)
    raise ValueError(f"Could not extract DeepStack features from visual output type {type(output)}")


def collect_deepstack_features(model: torch.nn.Module, inputs_BL: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    """Run the vision tower and return DeepStack merger outputs, one tensor per decoder layer."""
    if "pixel_values" not in inputs_BL:
        raise ValueError("DeepStack collection requires pixel_values")
    if "image_grid_thw" not in inputs_BL:
        raise ValueError("DeepStack collection requires image_grid_thw")
    visual = _get_vision_module(model)
    pixel_values = inputs_BL["pixel_values"]
    grid_thw = inputs_BL["image_grid_thw"]
    with torch.no_grad():
        output = visual(pixel_values, grid_thw=grid_thw)
    features = _extract_deepstack_from_visual_output(output)
    return [feat.detach() for feat in features]


def align_deepstack_to_oracle_slots(
    context_input_ids: list[int],
    context_positions: list[int],
    oracle_positions: list[int],
    visual_token_ids: frozenset[int],
    deepstack_features: list[torch.Tensor],
) -> tuple[list[int], list[torch.Tensor]]:
    """Map DeepStack visual rows onto the oracle `?` slots whose source tokens are visual."""
    if len(context_positions) != len(oracle_positions):
        raise ValueError(
            f"context_positions length {len(context_positions)} != oracle positions {len(oracle_positions)}"
        )
    if not deepstack_features:
        raise ValueError("deepstack_features must not be empty")
    visual_pos = [i for i, tok in enumerate(context_input_ids) if tok in visual_token_ids]
    n_visual = len(visual_pos)
    for layer_idx, feat in enumerate(deepstack_features):
        if feat.ndim != 2:
            raise ValueError(f"DeepStack layer {layer_idx} must be [V, D], got {tuple(feat.shape)}")
        if feat.shape[0] != n_visual:
            raise ValueError(
                f"DeepStack layer {layer_idx} has {feat.shape[0]} visual rows, "
                f"but context has {n_visual} visual tokens"
            )
    pos_to_row = {src_pos: row for row, src_pos in enumerate(visual_pos)}
    slot_oracle_positions: list[int] = []
    row_indices: list[int] = []
    for slot, src_pos in enumerate(context_positions):
        row = pos_to_row.get(src_pos)
        if row is None:
            continue
        slot_oracle_positions.append(oracle_positions[slot])
        row_indices.append(row)
    if not slot_oracle_positions:
        d_model = deepstack_features[0].shape[-1]
        empty = [feat.new_zeros((0, d_model)) for feat in deepstack_features]
        return [], empty
    aligned = [feat[row_indices].contiguous() for feat in deepstack_features]
    return slot_oracle_positions, aligned


def get_hf_submodule(model: AutoModelForCausalLM, layer: int, use_lora: bool = False):
    """Gets the residual stream submodule for HF transformers"""
    model_name = model.config._name_or_path

    if "qwen3-vl" in model_name.lower():
        return _get_language_layers(model)[layer]

    if use_lora:
        if "pythia" in model_name:
            raise ValueError("Need to determine how to get submodule for LoRA")
        elif "gemma-3" in model_name:
            return model.base_model.language_model.layers[layer]
        elif "gemma-2" in model_name or "mistral" in model_name or "Llama" in model_name or "Qwen" in model_name:
            return model.base_model.model.model.layers[layer]
        else:
            raise ValueError(f"Please add submodule for model {model_name}")

    if "pythia" in model_name:
        return model.gpt_neox.layers[layer]
    elif "gemma-3" in model_name:
        return model.language_model.layers[layer]
    elif "gemma-2" in model_name or "mistral" in model_name or "Llama" in model_name or "Qwen" in model_name:
        return model.model.layers[layer]
    else:
        raise ValueError(f"Please add submodule for model {model_name}")
