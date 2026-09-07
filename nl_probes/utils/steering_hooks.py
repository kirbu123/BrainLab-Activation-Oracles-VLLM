import contextlib
from typing import Callable

import torch

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
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
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
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    B = resid_BLD.shape[0]
    if len(vectors) != B or len(positions) != B:
        raise ValueError(
            f"vectors/positions batch mismatch: B={B}, vectors={len(vectors)}, positions={len(positions)}"
        )
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
        steered_KD = (vec_b.to(device=device, dtype=dtype) * norms_K1 * steering_coefficient)
        resid_BLD[b, pos_t, :] = resid_BLD[b, pos_t, :] + steered_KD.detach()


def get_hf_multi_source_steering_hook(
    write_groups: list[tuple[list[torch.Tensor], list[list[int]]]],
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """Apply several +h writes from the unmodified residual.

    Each group is (vectors_per_batch, positions_per_batch). Overlapping positions
    receive the sum of steered vectors, with norms taken from the original residual.
    """
    if not write_groups:
        raise ValueError("write_groups must not be empty")
    B = len(write_groups[0][0])
    if B == 0:
        raise ValueError("Empty batch")
    for vectors, positions in write_groups:
        if len(vectors) != B or len(positions) != B:
            raise ValueError("All write groups must share the same batch length")

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
        for vectors, positions in normed_groups:
            _apply_steering_writes(
                resid_BLD,
                orig_BLD=orig_BLD,
                vectors=vectors,
                positions=positions,
                steering_coefficient=steering_coefficient,
                device=device,
                dtype=dtype,
            )
        return (resid_BLD, *rest) if output_is_tuple else resid_BLD

    return hook_fn


def _unwrap_forward_model(model: torch.nn.Module) -> torch.nn.Module:
    inner = model
    if hasattr(inner, "module"):
        inner = inner.module
    return inner


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
    deepstack_steering_vectors: list[list[torch.Tensor]] | None = None,
    deepstack_positions: list[list[int]] | None = None,
):
    """Register decoder-act and optional DeepStack encoder steering hooks."""
    from nl_probes.utils.activation_utils import get_hf_submodule

    decoder_hook = get_hf_activation_steering_hook(
        vectors=batch_steering_vectors,
        positions=batch_positions,
        steering_coefficient=steering_coefficient,
        device=device,
        dtype=dtype,
    )
    if not use_deepstack_injection:
        with add_hook(decoder_submodule, decoder_hook):
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
        with add_hook(decoder_submodule, decoder_hook):
            yield
        return

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

    with contextlib.ExitStack() as stack:
        for layer_idx in range(n_layers):
            if layer_idx == hook_onto_layer:
                continue
            hook = get_hf_activation_steering_hook(
                vectors=per_layer_vectors[layer_idx],
                positions=deepstack_positions,
                steering_coefficient=steering_coefficient,
                device=device,
                dtype=dtype,
            )
            stack.enter_context(add_hook(get_hf_submodule(inner, layer_idx), hook))

        if 0 <= hook_onto_layer < n_layers:
            combined = get_hf_multi_source_steering_hook(
                write_groups=[
                    (batch_steering_vectors, batch_positions),
                    (per_layer_vectors[hook_onto_layer], deepstack_positions),
                ],
                steering_coefficient=steering_coefficient,
                device=device,
                dtype=dtype,
            )
            stack.enter_context(add_hook(decoder_submodule, combined))
        else:
            stack.enter_context(add_hook(decoder_submodule, decoder_hook))
        yield