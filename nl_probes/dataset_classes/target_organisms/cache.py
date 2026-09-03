from __future__ import annotations

import gc
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from nl_probes.dataset_classes.target_organisms.families import (
    load_target_validation_manifest,
)
from nl_probes.dataset_classes.target_organisms.registry import (
    enabled_adapters,
    load_adapter_registry,
)
from nl_probes.dataset_classes.target_organisms.schema import (
    AdapterEntry,
    AdapterRegistry,
    CacheIdentity,
    FrozenMetadata,
    ProbeSettings,
    TargetMessage,
    TargetValidationRecord,
    VisualPersonaQARecord,
    VisualSSCRecord,
    VisualTabooRecord,
    VisualUserAttributeRecord,
    checksum_json,
    checksum_path,
)
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint
from nl_probes.utils.vlm_utils import sample_modality_positions, visual_token_ids_from_tokenizer


class TargetValidationDataPoint(TrainingDataPoint):
    """TrainingDataPoint with immutable validation provenance."""

    meta_info: FrozenMetadata


@dataclass(frozen=True)
class TokenizedTarget:
    input_ids: tuple[int, ...]
    model_inputs: Mapping[str, Any]


@dataclass(frozen=True)
class TargetModelOperations:
    """Injectable model boundary used by network-free cache tests."""

    load_base: Callable[[AdapterRegistry], Any]
    tokenizer: Callable[[Any], Any]
    enable_adapter: Callable[[Any, AdapterEntry], None]
    tokenize: Callable[[Any, Sequence[TargetMessage], bool], TokenizedTarget]
    generate: Callable[[Any, TokenizedTarget, int], str]
    collect_activations: Callable[
        [Any, TokenizedTarget, tuple[int, ...]], Mapping[int, torch.Tensor]
    ]
    disable_adapter: Callable[[Any, AdapterEntry], None]
    close: Callable[[Any], None]


def build_cache_identity(
    registry: AdapterRegistry,
    manifest_path: str | Path,
    settings: ProbeSettings,
    family: str,
) -> CacheIdentity:
    entries = enabled_adapters(registry, family)  # type: ignore[arg-type]
    adapter_checksums = {
        entry.organism_id: checksum_path(entry.adapter_path) for entry in entries
    }
    model_path = Path(registry.base_model)
    model_source_checksum = (
        checksum_path(model_path)
        if model_path.exists()
        else checksum_json(
            {"base_model": registry.base_model, "base_revision": registry.base_revision}
        )
    )
    model_checksum = checksum_json(
        {
            "source_checksum": model_source_checksum,
            "revision": registry.base_revision,
        }
    )
    return CacheIdentity(
        family=family,
        manifest_checksum=checksum_path(manifest_path),
        adapter_checksums=adapter_checksums,
        model_checksum=model_checksum,
        probe_checksum=checksum_json(settings.model_dump(mode="json")),
    )


def target_validation_cache_path(
    cache_dir: str | Path,
    identity: CacheIdentity,
) -> Path:
    return Path(cache_dir) / f"{identity.family}_{identity.digest}.pt"


def precompute_target_validation_cache(
    *,
    registry_path: str | Path,
    manifest_path: str | Path,
    settings: ProbeSettings,
    cache_dir: str | Path = "data/val/cache",
    operations: TargetModelOperations | None = None,
    force: bool = False,
) -> Path:
    """Build one adapter-on family cache, loading adapters sequentially."""
    registry = load_adapter_registry(registry_path, require_local_adapters=True)
    manifest = load_target_validation_manifest(
        manifest_path,
        registry=registry,
        require_images=True,
    )
    identity = build_cache_identity(registry, manifest_path, settings, manifest.family)
    output_path = target_validation_cache_path(cache_dir, identity)
    if output_path.exists() and not force:
        load_target_validation_cache(output_path, expected_identity=identity)
        return output_path

    ops = operations or default_target_model_operations()
    runtime = ops.load_base(registry)
    datapoints: list[TrainingDataPoint] = []
    try:
        records_by_organism: dict[str, list[TargetValidationRecord]] = {}
        for record in manifest.records:
            records_by_organism.setdefault(record.organism_id, []).append(record)

        for entry in enabled_adapters(registry, manifest.family):
            organism_records = records_by_organism.pop(entry.organism_id, [])
            if not organism_records:
                continue
            ops.enable_adapter(runtime, entry)
            try:
                for record in organism_records:
                    datapoints.extend(
                        build_record_datapoints(
                            record=record,
                            registry=registry,
                            adapter=entry,
                            settings=settings,
                            runtime=runtime,
                            operations=ops,
                            cache_identity=identity,
                        )
                    )
            finally:
                ops.disable_adapter(runtime, entry)
        if records_by_organism:
            raise KeyError(
                f"No enabled adapters processed manifest organisms: {sorted(records_by_organism)}"
            )
    finally:
        ops.close(runtime)

    if not datapoints:
        raise RuntimeError(f"Target validation cache produced zero rows for {manifest.family}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "identity": identity.model_dump(mode="json"),
            "data": [_serializable_datapoint(point) for point in datapoints],
        },
        output_path,
    )
    return output_path


def precompute_target_validation_caches(
    *,
    registry_path: str | Path,
    manifest_paths: Sequence[str | Path],
    settings: ProbeSettings,
    cache_dir: str | Path = "data/val/cache",
    operations_factory: Callable[[], TargetModelOperations] | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Build independent family caches; each call releases the base model."""
    outputs: dict[str, Path] = {}
    for manifest_path in manifest_paths:
        manifest = load_target_validation_manifest(
            manifest_path, require_images=False
        )
        if manifest.family in outputs:
            raise ValueError(f"Duplicate manifest family: {manifest.family}")
        operations = (
            operations_factory() if operations_factory is not None else None
        )
        outputs[manifest.family] = precompute_target_validation_cache(
            registry_path=registry_path,
            manifest_path=manifest_path,
            settings=settings,
            cache_dir=cache_dir,
            operations=operations,
            force=force,
        )
    return outputs


def build_record_datapoints(
    *,
    record: TargetValidationRecord,
    registry: AdapterRegistry,
    adapter: AdapterEntry,
    settings: ProbeSettings,
    runtime: Any,
    operations: TargetModelOperations,
    cache_identity: CacheIdentity,
) -> list[TrainingDataPoint]:
    prompt = operations.tokenize(runtime, record.target_messages, True)
    response = (
        operations.generate(runtime, prompt, settings.max_response_tokens)
        if settings.generate_target_response
        else record.target_response
    )
    if response is None:
        raise ValueError(
            f"Record {record.record_id} has no target_response while generation is disabled"
        )
    if not response.strip():
        raise ValueError(f"Record {record.record_id} produced an empty target response")
    _enforce_non_disclosure(record, response)

    full_messages = tuple(record.target_messages) + (
        TargetMessage(role="assistant", content=response),
    )
    full = operations.tokenize(runtime, full_messages, False)
    prompt_acts = operations.collect_activations(runtime, prompt, settings.layers)
    full_acts = operations.collect_activations(runtime, full, settings.layers)
    _validate_activation_map(prompt_acts, settings.layers, len(prompt.input_ids), "prompt")
    _validate_activation_map(full_acts, settings.layers, len(full.input_ids), "prompt_response")

    prompt_positions = tuple(
        range(
            max(0, len(prompt.input_ids) - settings.prompt_tail_tokens),
            len(prompt.input_ids),
        )
    )
    full_positions = tuple(
        range(
            max(
                0,
                min(
                    len(prompt.input_ids) - settings.prompt_tail_tokens,
                    len(full.input_ids) - 1,
                ),
            ),
            len(full.input_ids),
        )
    )
    if not prompt_positions or not full_positions:
        raise ValueError(f"Record {record.record_id} produced empty probe positions")

    tokenizer = operations.tokenizer(runtime)
    visual_token_ids = None
    if settings.source_token_mode != "mixed":
        visual_token_ids = visual_token_ids_from_tokenizer(tokenizer)
    datapoints = []
    for layer in settings.layers:
        for variant in settings.variants:
            if variant == "prompt_tail":
                source_ids = prompt.input_ids
                default_positions = prompt_positions
                vectors_source = prompt_acts
            elif variant == "prompt_response":
                source_ids = full.input_ids
                default_positions = full_positions
                vectors_source = full_acts
            else:
                raise ValueError(f"Unsupported probe variant: {variant}")
            positions = tuple(
                sample_modality_positions(
                    list(source_ids),
                    visual_token_ids if visual_token_ids is not None else frozenset(),
                    settings.source_token_mode,
                    len(default_positions),
                    original_positions=list(default_positions),
                )
            )
            vectors = vectors_source[layer][0, list(positions), :]
            metadata = FrozenMetadata(
                _record_metadata(
                    record=record,
                    registry=registry,
                    adapter=adapter,
                    cache_identity=cache_identity,
                    probe_variant=variant,
                    target_response=response,
                    source_input_ids=source_ids,
                    source_positions=positions,
                    source_token_mode=settings.source_token_mode,
                )
            )
            point = create_training_datapoint(
                    datapoint_type=record.family,
                    prompt=record.oracle_prompt,
                    target_response=record.oracle_target,
                    layer=layer,
                    num_positions=len(positions),
                    tokenizer=tokenizer,
                    acts_BD=vectors.detach().to(device="cpu").contiguous(),
                    feature_idx=-1,
                    ds_label=record.oracle_target,
                    meta_info=metadata,
                )
            datapoints.append(
                TargetValidationDataPoint.model_validate(
                    {**point.model_dump(), "meta_info": metadata}
                )
            )
    return datapoints


def load_target_validation_cache(
    cache_path: str | Path,
    *,
    expected_identity: CacheIdentity | None = None,
) -> list[TrainingDataPoint]:
    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"Target validation cache not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Target validation cache must contain a dictionary: {path}")
    if payload.keys() != {"schema_version", "identity", "data"}:
        raise ValueError(f"Unexpected target validation cache keys: {sorted(payload.keys())}")
    if payload["schema_version"] != 1:
        raise ValueError(f"Unsupported target validation cache schema: {payload['schema_version']}")
    identity = CacheIdentity.model_validate(payload["identity"], strict=True)
    if expected_identity is not None and identity != expected_identity:
        raise ValueError(
            f"Target validation cache identity mismatch: expected {expected_identity.digest}, "
            f"found {identity.digest}"
        )
    if not isinstance(payload["data"], list) or not payload["data"]:
        raise ValueError(f"Target validation cache contains no rows: {path}")
    datapoints = []
    for raw in payload["data"]:
        raw["meta_info"] = FrozenMetadata(raw["meta_info"])
        point = TargetValidationDataPoint.model_validate(raw)
        if point.steering_vectors is None:
            raise ValueError("Target validation cache rows must materialize steering_vectors")
        datapoints.append(point)
    return datapoints


def load_cached_target_validation_family(
    cache_path: str | Path,
    family: str,
    *,
    expected_identity: CacheIdentity | None = None,
) -> list[TrainingDataPoint]:
    datapoints = load_target_validation_cache(
        cache_path, expected_identity=expected_identity
    )
    mismatches = [
        point.meta_info["record_id"]
        for point in datapoints
        if point.datapoint_type != family or point.meta_info["family"] != family
    ]
    if mismatches:
        raise ValueError(
            f"Cache contains rows outside requested family {family}: {mismatches}"
        )
    return datapoints


def _validate_activation_map(
    activations: Mapping[int, torch.Tensor],
    layers: tuple[int, ...],
    sequence_length: int,
    variant: str,
) -> None:
    if set(activations) != set(layers):
        raise ValueError(
            f"{variant} activation layers mismatch: expected {layers}, found {sorted(activations)}"
        )
    for layer, tensor in activations.items():
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[1] != sequence_length:
            raise ValueError(
                f"Layer {layer} {variant} activations must have shape [1, {sequence_length}, D], "
                f"found {tuple(tensor.shape)}"
            )


def _enforce_non_disclosure(record: TargetValidationRecord, response: str) -> None:
    forbidden = list(record.forbidden_strings)
    if not forbidden:
        raise ValueError(f"Record {record.record_id} has empty forbidden_strings")
    response_normalized = _lexical_normalize(response)
    leaked = sorted(
        value
        for value in forbidden
        if value
        and re.search(
            rf"(?:^|\s){re.escape(_lexical_normalize(value))}(?:$|\s)",
            response_normalized,
        )
    )
    if leaked:
        raise ValueError(
            f"Target response for {record.record_id} disclosed forbidden strings: {leaked}"
        )


def _lexical_normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(
        "".join(
            character if character.isalnum() else " " for character in normalized
        ).split()
    )


def _record_metadata(
    *,
    record: TargetValidationRecord,
    registry: AdapterRegistry,
    adapter: AdapterEntry,
    cache_identity: CacheIdentity,
    probe_variant: str,
    target_response: str,
    source_input_ids: tuple[int, ...],
    source_positions: tuple[int, ...],
    source_token_mode: str = "mixed",
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "record_id": record.record_id,
        "family": record.family,
        "organism_id": record.organism_id,
        "adapter_path": adapter.adapter_path,
        "activation_source": "target_lora",
        "base_model": registry.base_model,
        "base_revision": registry.base_revision,
        "probe_variant": probe_variant,
        "render_split": record.render_split,
        "ood_slices": record.ood_slices,
        "scoring_mode": record.scoring_mode,
        "oracle_target": record.oracle_target,
        "target_response": target_response,
        "source_input_ids": source_input_ids,
        "source_positions": source_positions,
        "source_token_mode": source_token_mode,
        "cache_identity": cache_identity.model_dump(mode="json"),
    }
    if isinstance(record, VisualTabooRecord):
        metadata.update(secret=record.secret, allowed_values=record.allowed_values)
    elif isinstance(record, VisualUserAttributeRecord):
        metadata.update(
            attribute_name=record.attribute_name,
            attribute_value=record.attribute_value,
            allowed_values=record.allowed_values,
        )
    elif isinstance(record, VisualSSCRecord):
        metadata.update(
            constraint_id=record.constraint_id,
            constraints=tuple(
                {"name": item.name, "value": item.value} for item in record.constraints
            ),
            allowed_values=record.allowed_values,
            aliases=record.aliases,
        )
    elif isinstance(record, VisualPersonaQARecord):
        metadata.update(
            identity_id=record.identity_id,
            attribute_name=record.attribute_name,
            attribute_value=record.attribute_value,
            allowed_values=record.allowed_values,
        )
    else:
        raise TypeError(f"Unsupported target validation record: {type(record)}")
    return metadata


def _serializable_datapoint(point: TrainingDataPoint) -> dict[str, Any]:
    raw = point.model_dump()
    raw["meta_info"] = _thaw(raw["meta_info"])
    return raw


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def default_target_model_operations() -> TargetModelOperations:
    """Construct the real Qwen3-VL operations lazily; tests inject this boundary."""
    from nl_probes.utils.activation_utils import (
        collect_activations_multiple_layers,
        get_hf_submodule,
    )
    from transformers import AutoProcessor

    from nl_probes.utils.common import load_model
    from nl_probes.utils.vlm_utils import vision_inputs_to_device, vlm_tokenize_target

    def load_base(registry: AdapterRegistry):
        model = load_model(
            registry.base_model,
            dtype=torch.bfloat16,
            revision=registry.base_revision,
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(
            registry.base_model,
            revision=registry.base_revision,
        )
        processor_tokenizer = getattr(processor, "tokenizer", None)
        if processor_tokenizer is None:
            raise TypeError("Qwen3-VL processor must expose a tokenizer")
        processor_tokenizer.padding_side = "left"
        if processor_tokenizer.pad_token_id is None:
            processor_tokenizer.pad_token_id = processor_tokenizer.eos_token_id
        return {"model": model, "processor": processor}

    def tokenizer(runtime):
        processor = runtime["processor"]
        return getattr(processor, "tokenizer", processor)

    def enable_adapter(runtime, entry: AdapterEntry) -> None:
        model = runtime["model"]
        model.load_adapter(
            entry.adapter_path,
            adapter_name=entry.organism_id,
            is_trainable=False,
            low_cpu_mem_usage=True,
        )
        model.set_adapter(entry.organism_id)

    def tokenize(runtime, messages: Sequence[TargetMessage], add_generation_prompt: bool):
        message_dicts = [message.model_dump(mode="python") for message in messages]
        input_ids, inputs = vlm_tokenize_target(
            runtime["processor"],
            message_dicts,
            add_generation_prompt=add_generation_prompt,
        )
        device = next(runtime["model"].parameters()).device
        return TokenizedTarget(
            input_ids=tuple(input_ids),
            model_inputs=vision_inputs_to_device(inputs, device),
        )

    def generate(runtime, tokenized: TokenizedTarget, max_new_tokens: int) -> str:
        model = runtime["model"]
        output_ids = model.generate(
            **tokenized.model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        generated = output_ids[:, len(tokenized.input_ids) :]
        return tokenizer(runtime).batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()

    def collect(runtime, tokenized: TokenizedTarget, layers: tuple[int, ...]):
        model = runtime["model"]
        submodules = {
            layer: get_hf_submodule(model, layer, use_lora=True) for layer in layers
        }
        acts = collect_activations_multiple_layers(
            model=model,
            submodules=submodules,
            inputs_BL=dict(tokenized.model_inputs),
            min_offset=None,
            max_offset=None,
        )
        return {
            layer: tensor.detach().to(device="cpu").contiguous()
            for layer, tensor in acts.items()
        }

    def disable_adapter(runtime, entry: AdapterEntry) -> None:
        runtime["model"].delete_adapter(entry.organism_id)

    def close(runtime) -> None:
        del runtime["model"]
        del runtime["processor"]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return TargetModelOperations(
        load_base=load_base,
        tokenizer=tokenizer,
        enable_adapter=enable_adapter,
        tokenize=tokenize,
        generate=generate,
        collect_activations=collect,
        disable_adapter=disable_adapter,
        close=close,
    )
