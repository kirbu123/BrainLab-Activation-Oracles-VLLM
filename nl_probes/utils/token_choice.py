"""Shared source-token ranking, attention collection, and validation accounting."""

import contextlib
import copy
import html
import math

import torch

TOKEN_CHOICE_VERSION = "received-attention-safe-spans-v1"


def validate_token_choice(mode, percent):
    if mode not in ("default", "attn_choice"):
        raise ValueError(f"Unknown token choice mode: {mode}")
    if mode == "default" and percent is not None:
        raise ValueError("--token-choice-percent requires --token-choice-mode attn_choice")
    if mode == "attn_choice" and (percent is None or not 0 < percent <= 100):
        raise ValueError("attn_choice requires --token-choice-percent in (0, 100]")


def received_attention(weights, attention_mask):
    """Mean over heads and valid query rows; causal zeros remain in the mean."""
    if weights.ndim != 4 or weights.shape[-2:] != (attention_mask.shape[1],) * 2:
        raise ValueError("Expected square [B, H, N, N] attention for source prefill")
    valid = attention_mask.to(device=weights.device, dtype=torch.bool)
    if not valid.any(dim=1).all():
        raise ValueError("Attention batch contains an empty source sequence")
    # Reduce before conversion to float32 to avoid a second full attention map.
    scores = weights.mean(dim=1).masked_fill(~valid[:, :, None], 0).sum(dim=1, dtype=torch.float32)
    scores = scores / valid.sum(dim=1, keepdim=True)
    return scores.masked_fill(~valid, -torch.inf).detach()


def select_attention_positions(scores, input_ids, visual_ids, percent, *, excluded=(), mode="mixed", example=""):
    validate_token_choice("attn_choice", percent)
    if mode not in ("mixed", "text", "visual"):
        raise ValueError(f"Invalid modality: {mode}")
    if scores.ndim != 1 or scores.numel() != len(input_ids):
        raise ValueError(f"{example}: scores do not match source length")
    excluded = set(excluded)
    candidates = [i for i, token in enumerate(input_ids) if i not in excluded
                  and (mode == "mixed" or (token in visual_ids) == (mode == "visual"))]
    if not candidates:
        raise ValueError(f"{example}: empty eligible {mode} token pool")
    values = scores[candidates].float().cpu()
    if not torch.isfinite(values).all():
        raise ValueError(f"{example}: nonfinite candidate attention scores")
    k = max(1, math.floor(len(candidates) * percent / 100))
    order = torch.argsort(values, descending=True, stable=True)[:k].tolist()
    return sorted(candidates[i] for i in order)


@contextlib.contextmanager
def capture_attention_scores(model, layers, attention_mask):
    """Temporarily collect eager decoder attention, without retaining full maps."""
    from nl_probes.utils.activation_utils import _get_language_layers

    blocks = _get_language_layers(model)
    configs = {id(block.self_attn.config): block.self_attn.config for block in blocks}
    saved = [(config, config._attn_implementation) for config in configs.values()]
    original_configs = [(block.self_attn, block.self_attn.config) for block in blocks]
    handles = []
    scores = {}
    was_training = model.training

    def capture(layer):
        def hook(module, inputs, output):
            if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                raise ValueError("Attention backend did not expose eager attention weights")
            scores[layer] = received_attention(output[1], attention_mask).cpu()
            return (output[0], None, *output[2:])
        return hook

    try:
        model.eval()
        for config, _ in saved:
            config._attn_implementation = "eager"
        # The parent needs an explicit causal mask; unrequested layers need no maps.
        for layer, (attention, config) in enumerate(original_configs):
            attention.config = copy.copy(config)
            attention.config._attn_implementation = "eager" if layer in layers else "sdpa"
        for layer in layers:
            handles.append(blocks[layer].self_attn.register_forward_hook(capture(layer)))
        yield scores
        if set(scores) != set(layers):
            raise ValueError(f"Missing source attention layers: {set(layers) - set(scores)}")
    finally:
        for handle in handles:
            handle.remove()
        for attention, config in original_configs:
            attention.config = config
        for config, implementation in saved:
            config._attn_implementation = implementation
        model.train(was_training)


def selection_identity(mode, percent, source_mode="mixed"):
    return {"mode": mode, "percent": percent, "source_mode": source_mode,
            "version": TOKEN_CHOICE_VERSION}


def token_count_record(point, visual_ids, deepstack_layers=0):
    from nl_probes.utils.dataset_utils import source_token_ids

    ids = source_token_ids(point)
    positions = point.context_positions
    if positions is None:
        positions = point.meta_info["source_positions"]
    if len(positions) != len(point.positions):
        raise ValueError("Selected source positions do not match the injected oracle slots")
    visual = sum(ids[i] in visual_ids for i in positions)
    ds_counts = [0] * deepstack_layers
    if point.deepstack_steering_vectors is not None:
        ds_counts = [len(vectors) for vectors in point.deepstack_steering_vectors]
    return {"layer": point.layer, "text": len(positions) - visual, "visual": visual,
            "deepstack": ds_counts}


def materialize_attention_choices(points, tokenizer, model, processor, percent, use_deepstack, source_mode):
    from nl_probes.utils.activation_utils import (
        _get_vision_module, _extract_deepstack_from_visual_output,
        align_deepstack_to_oracle_slots, collect_activations_multiple_layers, get_hf_submodule,
    )
    from nl_probes.utils.dataset_utils import (
        create_training_datapoint, recover_oracle_question, source_token_ids,
    )
    from nl_probes.utils.vlm_utils import vlm_tokenize_target, vision_inputs_to_device, visual_token_ids_from_tokenizer

    identity = selection_identity("attn_choice", percent, source_mode)
    device = next(model.parameters()).device
    visual_ids = visual_token_ids_from_tokenizer(tokenizer)
    result = []
    for point in points:
        if point.meta_info.get("token_choice") == identity and point.steering_vectors is not None:
            result.append(point)
            continue
        if "cache_identity" in point.meta_info:
            raise ValueError("Target attention selection must be rebuilt with its target adapter")
        ids = source_token_ids(point)
        meta = dict(point.meta_info)
        if "target_messages" in meta:
            actual_ids, inputs = vlm_tokenize_target(
                processor, meta["target_messages"],
                add_generation_prompt=meta.get("add_generation_prompt", True),
            )
            if list(actual_ids) != list(ids):
                raise ValueError(f"{point.datapoint_type}: source retokenization differs from cached IDs")
            inputs = vision_inputs_to_device(inputs, device)
        else:
            if point.context_image_paths:
                raise ValueError(f"{point.datapoint_type}: attention selection requires target_messages")
            inputs = {"input_ids": torch.tensor([ids], device=device),
                      "attention_mask": torch.ones((1, len(ids)), device=device, dtype=torch.long)}
        excluded = list(meta.get("target_positions", []))
        if point.datapoint_type == "visual_spqa":
            from nl_probes.dataset_classes.visual_spqa_dataset import _system_prefix_len
            instruction = meta["target_messages"][0]["content"][0]["text"]
            excluded.extend(range(min(_system_prefix_len(processor, instruction), len(ids) - 1)))
        ds_features = []
        with contextlib.ExitStack() as stack:
            stack.enter_context(model.disable_adapter())
            if use_deepstack and point.context_image_paths:
                def capture_visual(module, args, output):
                    ds_features.extend(_extract_deepstack_from_visual_output(output))
                handle = _get_vision_module(model).register_forward_hook(capture_visual)
                stack.callback(handle.remove)
            scores = stack.enter_context(capture_attention_scores(model, [point.layer], inputs["attention_mask"]))
            acts = collect_activations_multiple_layers(
                model, {point.layer: get_hf_submodule(model, point.layer, use_lora=True)}, inputs, None, None,
            )
        positions = select_attention_positions(
            scores[point.layer][0], ids, visual_ids, percent, excluded=excluded,
            mode=source_mode, example=point.datapoint_type,
        )
        meta.update(token_choice=identity, source_positions=positions, source_token_mode=source_mode)
        new = create_training_datapoint(
            datapoint_type=point.datapoint_type, prompt=recover_oracle_question(point, tokenizer),
            target_response=point.target_output, layer=point.layer, num_positions=len(positions),
            tokenizer=tokenizer, acts_BD=acts[point.layer][0, positions].detach(), feature_idx=point.feature_idx,
            context_input_ids=list(ids), context_positions=positions, context_image_paths=point.context_image_paths,
            ds_label=point.ds_label, meta_info=meta,
        )
        if use_deepstack and point.context_image_paths:
            if not ds_features:
                raise ValueError("Vision tower did not return DeepStack features")
            new.deepstack_positions, new.deepstack_steering_vectors = align_deepstack_to_oracle_slots(
                list(ids), positions, new.positions, visual_ids,
                [v[0].detach().cpu() if v.ndim == 3 else v.detach().cpu() for v in ds_features],
            )
        result.append(new)
    return result


def token_count_metrics(records, dataset):
    result = {}
    for layer in sorted({r["layer"] for r in records}):
        group = [r for r in records if r["layer"] == layer]
        prefix = f"eval_token_count/{dataset}/layer_{layer}"
        result[f"{prefix}/examples"] = len(group)
        for modality in ("text", "visual"):
            result[f"{prefix}/{modality}"] = sum(r[modality] for r in group) / len(group)
    n_ds = max((len(r["deepstack"]) for r in records), default=0)
    for layer in range(n_ds):
        prefix = f"eval_deepstack_token_count/{dataset}/layer_{layer}"
        result[f"{prefix}/examples"] = len(records)
        result[f"{prefix}/text"] = 0.0
        result[f"{prefix}/visual"] = sum(
            r["deepstack"][layer] if layer < len(r["deepstack"]) else 0 for r in records
        ) / len(records)
    return result


def render_token_counts(metrics):
    rows = []
    for key, value in sorted(metrics.items()):
        if key.startswith(("eval_token_count/", "eval_deepstack_token_count/")):
            rows.append(f"<tr><td>{html.escape(key)}</td><td>{value:.3f}</td></tr>")
    if not rows:
        return ""
    return ('<h2>Validation token counts</h2><div style="overflow-x:auto"><table>'
            '<thead><tr><th>Dataset / layer / count</th><th>Mean or examples</th></tr></thead>'
            '<tbody>' + "".join(rows) + '</tbody></table></div>')
