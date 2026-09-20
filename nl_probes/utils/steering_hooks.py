import contextlib
from collections.abc import Sequence
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

DEEPSTACK_COEFFICIENTS_FILENAME = "deepstack_steering_coefficients.pt"


class DeepStackSteeringCoefficients(nn.Module):
    def __init__(self, n_layers: int, init_value: float):
        super().__init__()
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        self.weight = nn.Parameter(
            torch.full((n_layers,), float(init_value), dtype=torch.float32)
        )


def deepstack_layer_count(model: nn.Module) -> int:
    inner = _unwrap_forward_model(model)
    vision_config = inner.config.vision_config
    indexes = vision_config.deepstack_visual_indexes
    if not indexes:
        raise ValueError("model.config.vision_config.deepstack_visual_indexes is empty")
    return len(indexes)


def attached_deepstack_coefficients(model: nn.Module) -> torch.Tensor | None:
    inner = _unwrap_forward_model(model)
    module = getattr(inner, "deepstack_steering_coefficients", None)
    if module is None:
        return None
    return module.weight


def save_deepstack_steering_coefficients(directory: str | Path, coefficients: torch.Tensor) -> Path:
    path = Path(directory) / DEEPSTACK_COEFFICIENTS_FILENAME
    torch.save(coefficients.detach().float().cpu(), path)
    return path


def load_deepstack_steering_coefficients(
    directory: str | Path,
    n_layers: int,
    device: torch.device | None = None,
) -> DeepStackSteeringCoefficients:
    path = Path(directory) / DEEPSTACK_COEFFICIENTS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"DeepStack coefficients file not found: {path}")
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if tensor.shape != (n_layers,):
        raise ValueError(
            f"DeepStack coefficients shape {tuple(tensor.shape)} != ({n_layers},)"
        )
    module = DeepStackSteeringCoefficients(n_layers, 0.0)
    module.weight.data.copy_(tensor)
    if device is not None:
        module = module.to(device)
    return module


def get_vllm_steering_hook(
    vectors: list[torch.Tensor],
    positions: list[int],
    prompt_lengths: list[int],
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    Debug version of your steering hook with detailed logging
    """
    vec_BD = torch.stack(vectors)  # (B, d_model)
    pos_B = torch.tensor(positions, dtype=torch.long)  # (B,)
    B, d_model = vec_BD.shape
    vec_BD = vec_BD.to(device, dtype)
    pos_B = pos_B.to(device)

    def hook_fn(module, _input, output):
        # passed prompt lengths should line up hopefully
        tokens_L = _input[0]

        if tokens_L.shape[0] == B:
            # means we are in decoding, not prefill. So no need to steer.
            return output

        # if there aren't any 0s in tokens_L, then we are NOT in prefill. So skip
        if not torch.any(tokens_L == 0):
            return output

        number_of_zeroes = torch.sum(tokens_L == 0).item()
        # should be equal to number of prompts
        if number_of_zeroes != len(prompt_lengths):
            breakpoint()
            raise ValueError(
                f"Number of zeroes {number_of_zeroes} is not equal to number of prompt lengths {len(prompt_lengths)}"
            )

        count = 0
        for prompt_length in prompt_lengths:
            expected_position_indices_L = torch.arange(prompt_length, device=device)
            try:
                assert tokens_L[count : count + prompt_length].equal(expected_position_indices_L), (
                    f"Position indices mismatch at index {count}, expected {expected_position_indices_L}, got {tokens_L[count : count + prompt_length]}"
                )
            except AssertionError as e:
                raise e

            count += prompt_length

        before_resid_flat, resid_flat, *rest = output

        assert count == tokens_L.shape[0]
        assert resid_flat.shape[0] == tokens_L.shape[0]
        assert resid_flat.shape[1] == d_model

        intervention_indices_L = []
        idx = 0

        for i in range(len(prompt_lengths)):
            intervention_idx = torch.tensor(idx + positions[i], device=device)
            intervention_indices_L.append(intervention_idx)
            idx += prompt_lengths[i]

        assert idx >= tokens_L.shape[0]

        intervention_indices_L = torch.stack(intervention_indices_L)

        assert intervention_indices_L.shape[0] == B

        orig_BD = resid_flat[intervention_indices_L]

        assert orig_BD.shape == (B, d_model)

        # Compute norms and steering
        norms_B1 = orig_BD.norm(dim=-1, keepdim=True).detach()
        normalized_features = torch.nn.functional.normalize(vec_BD, dim=-1)
        steered_BD = normalized_features * norms_B1 * steering_coefficient

        # print(f"  Normalized feature norms: {normalized_features.norm(dim=-1).tolist()}")
        # print(f"  Original norms: {norms_B1.squeeze().tolist()}")
        # print(f"  Steered activation norms: {steered_BD.norm(dim=-1).tolist()}")

        # Calculate the change magnitude BEFORE applying
        change_magnitude = (steered_BD - orig_BD).norm(dim=-1)
        print(f"  Change magnitudes: {change_magnitude.tolist()}")

        if change_magnitude.max() < 1e-4:
            print("  ⚠️  WARNING: Very small change magnitude!")

        # Apply the steering
        # print(f"  Applying steering at positions: {pos_B.tolist()}")
        resid_flat[intervention_indices_L] = steered_BD

        return (before_resid_flat, resid_flat, *rest)

    return hook_fn


@contextlib.contextmanager
def add_hook(
    module: torch.nn.Module,
    hook: Callable,
):
    """Temporarily adds a forward hook to a model module.

    Args:
        module: The PyTorch module to hook
        hook: The hook function to apply

    Yields:
        None: Used as a context manager

    Example:
        with add_hook(model.layer, hook_fn):
            output = model(input)
    """
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_hf_activation_steering_hook(
    vectors: list[torch.Tensor],  # len B, each tensor is (K_b, d_model)
    positions: list[list[int]],  # len B, each list has length K_b
    steering_coefficient: float | torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    detach_write: bool = True,
) -> Callable:
    """
    HF hook with debug prints to compare against vLLM.
    Supports a variable number of target positions per batch element.

    Semantics:
      For each batch item b and slot k, replace the residual at token index positions[b][k]
      with normalize(vectors[b][k]) * ||resid[b, positions[b][k], :]|| * steering_coefficient.

    We use a for loop instead of vectorized operations as it's simpler and we are just doing indexing in
    a single layer, so the simplicity won out for now.
    """

    # ---- move inputs to device and prepare ragged tensors ----
    assert len(vectors) == len(positions), "vectors and positions must have same batch length"
    B = len(vectors)
    if B == 0:
        raise ValueError("Empty batch")

    # Pre-normalize once; we never backprop through these
    normed_list = [_normalize_steering_vectors(v_b) for v_b in vectors]

    def hook_fn(module, _input, output):
        resid_BLD, output_is_tuple, rest = _unpack_residual_output(output)

        B_actual, L, d_model_actual = resid_BLD.shape
        if B_actual != B:
            raise ValueError(f"Batch mismatch: module B={B_actual}, provided vectors B={B}")

        # Only touch the prompt forward pass
        if L <= 1:
            return (resid_BLD, *rest) if output_is_tuple else resid_BLD

        _apply_steering_writes(
            resid_BLD,
            orig_BLD=resid_BLD,
            vectors=normed_list,
            positions=positions,
            steering_coefficient=steering_coefficient,
            device=device,
            dtype=dtype,
            detach_write=detach_write,
        )

        return (resid_BLD, *rest) if output_is_tuple else resid_BLD

    return hook_fn


def _normalize_steering_vectors(vectors: torch.Tensor) -> torch.Tensor:
    if vectors.numel() == 0:
        return vectors.detach()
    return torch.nn.functional.normalize(vectors, dim=-1).detach()


def _unpack_residual_output(output):
    if isinstance(output, tuple):
        resid_BLD, *rest = output
        return resid_BLD, True, rest
    return output, False, ()


def _apply_steering_writes(
    resid_BLD: torch.Tensor,
    orig_BLD: torch.Tensor,
    vectors: list[torch.Tensor],
    positions: list[list[int]],
    steering_coefficient: float | torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    detach_write: bool = True,
) -> None:
    B = resid_BLD.shape[0]
    if len(vectors) != B or len(positions) != B:
        raise ValueError(
            f"vectors/positions batch mismatch: B={B}, vectors={len(vectors)}, positions={len(positions)}"
        )
    coeff = steering_coefficient
    if torch.is_tensor(coeff):
        coeff = coeff.to(device=device, dtype=dtype)
    for b in range(B):
        pos_b = positions[b]
        if len(pos_b) == 0:
            continue
        vec_b = vectors[b]
        if vec_b.shape[0] == 0:
            continue
        if vec_b.shape[0] != len(pos_b):
            raise ValueError(
                f"batch {b}: {vec_b.shape[0]} steering vectors vs {len(pos_b)} positions"
            )
        pos_t = torch.tensor(pos_b, dtype=torch.long, device=device)
        if pos_t.min() < 0:
            raise ValueError(f"batch {b}: negative steering position {pos_t.min().item()}")
        if pos_t.max() >= resid_BLD.shape[1]:
            raise ValueError(
                f"batch {b}: steering position {pos_t.max().item()} >= sequence length {resid_BLD.shape[1]}"
            )
        orig_KD = orig_BLD[b, pos_t, :]
        norms_K1 = orig_KD.norm(dim=-1, keepdim=True)
        if b == 0 and norms_K1.max() > 300:
            print(f"\n\n\n\n\nWARNING: Large norm detected in batch! {norms_K1}\n\n\n\n\n")
        vec_KD = vec_b.to(device=device, dtype=dtype)
        if detach_write:
            steered_KD = vec_KD * norms_K1 * coeff
            resid_BLD[b, pos_t, :] = resid_BLD[b, pos_t, :] + steered_KD.detach()
        else:
            steered_KD = vec_KD.detach() * norms_K1.detach() * coeff
            resid_BLD[b, pos_t, :] = resid_BLD[b, pos_t, :] + steered_KD


def get_hf_multi_source_steering_hook(
    write_groups: list[tuple[list[torch.Tensor], list[list[int]]]],
    steering_coefficients: Sequence[float | torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    detach_writes: Sequence[bool] | None = None,
) -> Callable:
    """Apply several +h writes from the unmodified residual.

    Each group is (vectors_per_batch, positions_per_batch). Overlapping positions
    receive the sum of steered vectors, with norms taken from the original residual.
    """
    if not write_groups:
        raise ValueError("write_groups must not be empty")
    if len(steering_coefficients) != len(write_groups):
        raise ValueError(
            f"steering_coefficients length {len(steering_coefficients)} != write_groups {len(write_groups)}"
        )
    B = len(write_groups[0][0])
    if B == 0:
        raise ValueError("Empty batch")
    for vectors, positions in write_groups:
        if len(vectors) != B or len(positions) != B:
            raise ValueError("All write groups must share the same batch length")
    if detach_writes is None:
        group_detach = [True] * len(write_groups)
    else:
        if len(detach_writes) != len(write_groups):
            raise ValueError(
                f"detach_writes length {len(detach_writes)} != write_groups {len(write_groups)}"
            )
        group_detach = list(detach_writes)

    normed_groups = [
        ([_normalize_steering_vectors(v_b) for v_b in vectors], positions)
        for vectors, positions in write_groups
    ]

    def hook_fn(module, _input, output):
        resid_BLD, output_is_tuple, rest = _unpack_residual_output(output)
        B_actual, L, _d_model_actual = resid_BLD.shape
        if B_actual != B:
            raise ValueError(f"Batch mismatch: module B={B_actual}, provided vectors B={B}")
        if L <= 1:
            return (resid_BLD, *rest) if output_is_tuple else resid_BLD

        orig_BLD = resid_BLD.clone()
        for (vectors, positions), coefficient, detach_write in zip(
            normed_groups, steering_coefficients, group_detach, strict=True
        ):
            _apply_steering_writes(
                resid_BLD,
                orig_BLD=orig_BLD,
                vectors=vectors,
                positions=positions,
                steering_coefficient=coefficient,
                device=device,
                dtype=dtype,
                detach_write=detach_write,
            )
        return (resid_BLD, *rest) if output_is_tuple else resid_BLD

    return hook_fn


def _unwrap_forward_model(model: torch.nn.Module) -> torch.nn.Module:
    inner = model
    if hasattr(inner, "module"):
        inner = inner.module
    return inner


def _per_dest_decoder_vectors(
    dest_layers: list[int],
    batch_steering_vectors: list[torch.Tensor],
    dest_steering_vectors: list[list[torch.Tensor]] | None,
) -> list[list[torch.Tensor]]:
    if dest_steering_vectors:
        n_dest = len(dest_layers)
        if all(not item_vectors for item_vectors in dest_steering_vectors):
            if n_dest != 1:
                raise ValueError("Multiple dest layers require dest_steering_vectors")
            return [batch_steering_vectors]
        per_dest: list[list[torch.Tensor]] = []
        for dest_idx in range(n_dest):
            dest_vecs = []
            for item_idx, item_vectors in enumerate(dest_steering_vectors):
                if not item_vectors:
                    raise ValueError(f"dest_steering_vectors[{item_idx}] is empty")
                if dest_idx >= len(item_vectors):
                    raise ValueError(
                        f"dest_steering_vectors[{item_idx}] has {len(item_vectors)} dests, expected {n_dest}"
                    )
                dest_vecs.append(item_vectors[dest_idx])
            per_dest.append(dest_vecs)
        return per_dest
    if len(dest_layers) != 1:
        raise ValueError("Multiple dest layers require dest_steering_vectors")
    return [batch_steering_vectors]


def _dest_module(
    model: torch.nn.Module,
    decoder_submodule: torch.nn.Module,
    dest_idx: int,
    dest_layer: int,
) -> torch.nn.Module:
    if dest_idx == 0:
        return decoder_submodule
    from nl_probes.utils.activation_utils import get_hf_submodule

    return get_hf_submodule(_unwrap_forward_model(model), dest_layer)


@contextlib.contextmanager
def oracle_steering_hooks(
    model: torch.nn.Module,
    decoder_submodule: torch.nn.Module,
    batch_steering_vectors: list[torch.Tensor],
    batch_positions: list[list[int]],
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
    use_deepstack_injection: bool = False,
    hook_onto_layer: int = 1,
    hook_onto_layers: list[int] | None = None,
    dest_steering_vectors: list[list[torch.Tensor]] | None = None,
    deepstack_steering_vectors: list[list[torch.Tensor]] | None = None,
    deepstack_positions: list[list[int]] | None = None,
    deepstack_coefficients: torch.Tensor | None = None,
):
    """Register decoder-act and optional DeepStack encoder steering hooks."""
    from nl_probes.utils.activation_utils import get_hf_submodule

    dest_layers = list(hook_onto_layers) if hook_onto_layers is not None else [hook_onto_layer]
    if not dest_layers:
        raise ValueError("hook_onto_layers must be non-empty")
    per_dest_vectors = _per_dest_decoder_vectors(
        dest_layers, batch_steering_vectors, dest_steering_vectors
    )

    def _decoder_hook(vectors: list[torch.Tensor]):
        return get_hf_activation_steering_hook(
            vectors=vectors,
            positions=batch_positions,
            steering_coefficient=steering_coefficient,
            device=device,
            dtype=dtype,
            detach_write=True,
        )

    decoder_hook = _decoder_hook(per_dest_vectors[0])
    if not use_deepstack_injection:
        if len(dest_layers) == 1:
            with add_hook(decoder_submodule, decoder_hook):
                yield
            return
        with contextlib.ExitStack() as stack:
            for dest_idx, dest_layer in enumerate(dest_layers):
                stack.enter_context(
                    add_hook(
                        _dest_module(model, decoder_submodule, dest_idx, dest_layer),
                        _decoder_hook(per_dest_vectors[dest_idx]),
                    )
                )
            yield
        return

    if deepstack_steering_vectors is None or deepstack_positions is None:
        raise ValueError("DeepStack injection requires deepstack_steering_vectors and deepstack_positions")

    n_layers = 0
    for item_vectors in deepstack_steering_vectors:
        if item_vectors:
            n_layers = len(item_vectors)
            break

    if n_layers == 0:
        if len(dest_layers) == 1:
            with add_hook(decoder_submodule, decoder_hook):
                yield
            return
        with contextlib.ExitStack() as stack:
            for dest_idx, dest_layer in enumerate(dest_layers):
                stack.enter_context(
                    add_hook(
                        _dest_module(model, decoder_submodule, dest_idx, dest_layer),
                        _decoder_hook(per_dest_vectors[dest_idx]),
                    )
                )
            yield
        return

    if deepstack_coefficients is not None and deepstack_coefficients.numel() != n_layers:
        raise ValueError(
            f"DeepStack coefficients length {deepstack_coefficients.numel()} != n_layers {n_layers}"
        )
    detach_deepstack = deepstack_coefficients is None

    inner = _unwrap_forward_model(model)
    d_model = None
    for item_vectors in deepstack_steering_vectors:
        for tensor in item_vectors:
            d_model = tensor.shape[-1]
            break
        if d_model is not None:
            break
    if d_model is None:
        raise ValueError("DeepStack tensors are missing a hidden size")

    per_layer_vectors: list[list[torch.Tensor]] = []
    for layer_idx in range(n_layers):
        layer_vecs = []
        for item_vectors in deepstack_steering_vectors:
            if not item_vectors:
                layer_vecs.append(torch.zeros((0, d_model), device=device, dtype=dtype))
            else:
                layer_vecs.append(item_vectors[layer_idx])
        per_layer_vectors.append(layer_vecs)

    def _layer_coeff(layer_idx: int) -> float | torch.Tensor:
        if deepstack_coefficients is None:
            return steering_coefficient
        return deepstack_coefficients[layer_idx]

    dest_index = {layer: idx for idx, layer in enumerate(dest_layers)}
    with contextlib.ExitStack() as stack:
        for layer_idx in range(n_layers):
            if layer_idx in dest_index:
                continue
            hook = get_hf_activation_steering_hook(
                vectors=per_layer_vectors[layer_idx],
                positions=deepstack_positions,
                steering_coefficient=_layer_coeff(layer_idx),
                device=device,
                dtype=dtype,
                detach_write=detach_deepstack,
            )
            stack.enter_context(add_hook(get_hf_submodule(inner, layer_idx), hook))

        for dest_idx, dest_layer in enumerate(dest_layers):
            dest_module = _dest_module(model, decoder_submodule, dest_idx, dest_layer)
            dest_vecs = per_dest_vectors[dest_idx]
            if 0 <= dest_layer < n_layers:
                combined = get_hf_multi_source_steering_hook(
                    write_groups=[
                        (dest_vecs, batch_positions),
                        (per_layer_vectors[dest_layer], deepstack_positions),
                    ],
                    steering_coefficients=[steering_coefficient, _layer_coeff(dest_layer)],
                    device=device,
                    dtype=dtype,
                    detach_writes=[True, detach_deepstack],
                )
                stack.enter_context(add_hook(dest_module, combined))
            else:
                stack.enter_context(add_hook(dest_module, _decoder_hook(dest_vecs)))
        yield