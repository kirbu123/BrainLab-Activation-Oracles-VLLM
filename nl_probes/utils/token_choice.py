"""Shared source-token ranking, attention collection, and validation accounting."""

import contextlib
import copy
import html
import math

import torch

TOKEN_CHOICE_VERSION = "last-query-per-layer-modality-sinks-v3"
ATTN_SINK_VISUAL_PREFIX = 4


def validate_token_choice(mode, percent):
    if mode not in ("default", "attn_choice"):
        raise ValueError(f"Unknown token choice mode: {mode}")
    if mode == "default" and percent is not None:
        raise ValueError("--token-choice-percent requires --token-choice-mode attn_choice")
    if mode == "attn_choice" and (percent is None or not 0 < percent <= 100):
        raise ValueError("attn_choice requires --token-choice-percent in (0, 100]")


def special_token_ids(tokenizer, visual_ids=()) -> frozenset[int]:
    ids = {int(value) for value in (getattr(tokenizer, "all_special_ids", None) or ())}
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            ids.add(int(value))
    return frozenset(ids) - {int(token) for token in visual_ids}


def sink_excluded_positions(
    input_ids,
    visual_ids,
    special_ids,
    visual_prefix=ATTN_SINK_VISUAL_PREFIX,
):
    excluded = []
    visual_seen = 0
    for i, token in enumerate(input_ids):
        if token in visual_ids:
            if visual_seen < visual_prefix:
                excluded.append(i)
            visual_seen += 1
        elif token in special_ids:
            excluded.append(i)
    return excluded


def last_user_text_query_index(input_ids, visual_ids, special_ids, assistant_indices=()):
    banned = set(assistant_indices)
    for i in range(len(input_ids) - 1, -1, -1):
        if i in banned:
            continue
        token = input_ids[i]
        if token in visual_ids or token in special_ids:
            continue
        return i
    raise ValueError("no user text token for attention query")


def received_attention(weights, attention_mask, query_index=None):
    """Mean over heads at one query row. Default query is the last valid token."""
    if weights.ndim != 4 or weights.shape[-2:] != (attention_mask.shape[1],) * 2:
        raise ValueError("Expected square [B, H, N, N] attention for source prefill")
    valid = attention_mask.to(device=weights.device, dtype=torch.bool)
    if not valid.any(dim=1).all():
        raise ValueError("Attention batch contains an empty source sequence")
    if query_index is None:
        query = valid.sum(dim=1) - 1
    elif isinstance(query_index, int):
        query = torch.full((valid.shape[0],), int(query_index), device=weights.device, dtype=torch.long)
    else:
        query = torch.as_tensor(query_index, device=weights.device, dtype=torch.long)
        if query.ndim != 1 or query.shape[0] != valid.shape[0]:
            raise ValueError(
                f"query_index shape {tuple(query.shape)} != batch {valid.shape[0]}"
            )
    if (query < 0).any() or (query >= valid.shape[1]).any():
        raise ValueError("Attention query index is out of range")
    batch = torch.arange(valid.shape[0], device=valid.device)
    if not valid[batch, query].all():
        raise ValueError("Attention query index is padded")
    scores = weights.mean(dim=1)[batch, query].to(torch.float32)
    return scores.masked_fill(~valid, -torch.inf).detach()


def _topk_positions(scores, candidates, percent, example, pool):
    if not candidates:
        raise ValueError(f"{example}: empty eligible {pool} token pool")
    values = scores[candidates].float().cpu()
    if not torch.isfinite(values).all():
        raise ValueError(f"{example}: nonfinite candidate attention scores")
    k = max(1, math.floor(len(candidates) * percent / 100))
    order = torch.argsort(values, descending=True, stable=True)[:k].tolist()
    return [candidates[i] for i in order]


def select_attention_positions(scores, input_ids, visual_ids, percent, *, excluded=(), mode="mixed", example=""):
    validate_token_choice("attn_choice", percent)
    if mode not in ("mixed", "text", "visual"):
        raise ValueError(f"Invalid modality: {mode}")
    if scores.ndim != 1 or scores.numel() != len(input_ids):
        raise ValueError(f"{example}: scores do not match source length")
    excluded = set(excluded)
    if mode == "mixed":
        text = [i for i, token in enumerate(input_ids) if i not in excluded and token not in visual_ids]
        visual = [i for i, token in enumerate(input_ids) if i not in excluded and token in visual_ids]
        return sorted(
            _topk_positions(scores, text, percent, example, "text")
            + _topk_positions(scores, visual, percent, example, "visual")
        )
    candidates = [
        i
        for i, token in enumerate(input_ids)
        if i not in excluded and (token in visual_ids) == (mode == "visual")
    ]
    return sorted(_topk_positions(scores, candidates, percent, example, mode))


@contextlib.contextmanager
def capture_attention_scores(model, layers, attention_mask, query_index=None):
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
            scores[layer] = received_attention(
                output[1], attention_mask, query_index=query_index
            ).cpu()
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
    from nl_probes.utils.dataset_utils import resolved_source_layers
    return {"layer": resolved_source_layers(point)[0], "text": len(positions) - visual, "visual": visual,
            "deepstack": ds_counts}


def _message_role(message):
    return message["role"] if isinstance(message, dict) else message.role


def assistant_indices_from_prefix(full_ids, prefix_ids, example=""):
    if list(full_ids[: len(prefix_ids)]) != list(prefix_ids):
        raise ValueError(f"{example}: prompt ids are not a prefix of the full sequence")
    return list(range(len(prefix_ids), len(full_ids)))


def system_indices_for_attn(processor, messages, ids, datapoint_type, example=""):
    if not messages or _message_role(messages[0]) != "system":
        return []
    if datapoint_type == "visual_spqa":
        from nl_probes.dataset_classes.visual_spqa_dataset import _system_prefix_len

        content = messages[0]["content"]
        instruction = content[0]["text"] if isinstance(content, list) else content
        return list(range(min(_system_prefix_len(processor, instruction), len(ids) - 1)))
    from nl_probes.utils.vlm_utils import vlm_tokenize_target

    prefix_ids, _ = vlm_tokenize_target(processor, messages[:1], add_generation_prompt=False)
    if list(ids[: len(prefix_ids)]) != list(prefix_ids):
        raise ValueError(f"{example}: system prefix is not a prefix of source ids")
    return list(range(len(prefix_ids)))


def assistant_indices_from_messages(processor, messages, ids, example=""):
    if not messages or _message_role(messages[-1]) != "assistant":
        return []
    if processor is None:
        raise ValueError(f"{example}: assistant exclusion requires a processor")
    from nl_probes.utils.vlm_utils import vlm_tokenize_target

    prefix_ids, _ = vlm_tokenize_target(processor, messages[:-1], add_generation_prompt=True)
    return assistant_indices_from_prefix(ids, prefix_ids, example)


def attn_selection_exclusions(
    input_ids,
    visual_ids,
    tokenizer,
    *,
    assistant_indices=(),
    system_indices=(),
    extra_excluded=(),
):
    return (
        list(extra_excluded)
        + list(assistant_indices)
        + list(system_indices)
        + sink_excluded_positions(input_ids, visual_ids, special_token_ids(tokenizer, visual_ids))
    )


def select_attention_positions_per_layer(
    scores_by_layer,
    source_layers,
    input_ids,
    visual_ids,
    percent,
    *,
    excluded=(),
    mode="mixed",
    example="",
):
    positions_per_layer = {
        layer: select_attention_positions(
            scores_by_layer[layer][0],
            input_ids,
            visual_ids,
            percent,
            excluded=excluded,
            mode=mode,
            example=example,
        )
        for layer in source_layers
    }
    k = len(positions_per_layer[source_layers[0]])
    if any(len(positions) != k for positions in positions_per_layer.values()):
        raise ValueError(f"{example}: per-layer attention selection produced unequal slot counts")
    return positions_per_layer


def materialize_attention_choices(points, tokenizer, model, processor, percent, use_deepstack, source_mode):
    from nl_probes.utils.activation_utils import (
        collect_deepstack_features,
        align_deepstack_to_oracle_slots, collect_activations_multiple_layers, get_hf_submodule,
    )
    from nl_probes.utils.dataset_utils import (
        create_training_datapoint, recover_oracle_question, resolved_source_layers, source_token_ids,
    )
    from nl_probes.utils.vlm_utils import vlm_tokenize_target, vision_inputs_to_device, visual_token_ids_from_tokenizer

    identity = selection_identity("attn_choice", percent, source_mode)
    device = next(model.parameters()).device
    visual_ids = visual_token_ids_from_tokenizer(tokenizer)
    special_ids = special_token_ids(tokenizer, visual_ids)
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
        messages = meta.get("target_messages") or []
        assistant_indices = assistant_indices_from_messages(
            processor, messages, ids, example=point.datapoint_type,
        )
        excluded = attn_selection_exclusions(
            ids,
            visual_ids,
            tokenizer,
            assistant_indices=assistant_indices,
            system_indices=system_indices_for_attn(
                processor, messages, ids, point.datapoint_type, point.datapoint_type,
            ),
            extra_excluded=meta.get("target_positions", []),
        )
        query_index = last_user_text_query_index(ids, visual_ids, special_ids, assistant_indices)
        source_layers = resolved_source_layers(point)
        scores = {}
        acts = {}
        ds_features = []
        with model.disable_adapter():
            for i, layer in enumerate(source_layers):
                with capture_attention_scores(
                    model, [layer], inputs["attention_mask"], query_index=query_index,
                ) as layer_scores:
                    layer_acts = collect_activations_multiple_layers(
                        model,
                        {layer: get_hf_submodule(model, layer, use_lora=True)},
                        inputs,
                        None,
                        None,
                    )
                    if use_deepstack and point.context_image_paths and i == 0:
                        ds_features = collect_deepstack_features(model, inputs)
                scores[layer] = layer_scores[layer]
                acts[layer] = layer_acts[layer]
        positions_per_layer = select_attention_positions_per_layer(
            scores, source_layers, ids, visual_ids, percent,
            excluded=excluded, mode=source_mode, example=point.datapoint_type,
        )
        positions = positions_per_layer[source_layers[0]]
        meta.update(
            token_choice=identity,
            source_positions=positions,
            source_positions_per_layer={str(layer): pos for layer, pos in positions_per_layer.items()},
            source_token_mode=source_mode,
        )
        dest_acts = [acts[layer][0, positions_per_layer[layer]].detach() for layer in source_layers]
        new = create_training_datapoint(
            datapoint_type=point.datapoint_type, prompt=recover_oracle_question(point, tokenizer),
            target_response=point.target_output, layer=source_layers[0], num_positions=len(positions),
            tokenizer=tokenizer, acts_BD=dest_acts[0], feature_idx=point.feature_idx,
            context_input_ids=list(ids), context_positions=positions, context_image_paths=point.context_image_paths,
            ds_label=point.ds_label, meta_info=meta,
            source_layers=source_layers,
            dest_acts=dest_acts if len(source_layers) > 1 else None,
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
