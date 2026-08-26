from __future__ import annotations

import json
from pathlib import Path

from nl_probes.dataset_classes.target_organisms.schema import (
    AdapterEntry,
    AdapterRegistry,
    TargetFamily,
)


def load_adapter_registry(
    registry_path: str | Path,
    *,
    require_local_adapters: bool = True,
) -> AdapterRegistry:
    """Load and strictly validate a version-1 target-adapter registry."""
    path = Path(registry_path)
    if not path.is_file():
        raise FileNotFoundError(f"Target adapter registry not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    registry = AdapterRegistry.model_validate(raw, strict=True)

    resolved_entries = []
    for entry in registry.adapters:
        adapter_path = Path(entry.adapter_path)
        if not adapter_path.is_absolute():
            adapter_path = (path.parent / adapter_path).resolve()
        if require_local_adapters and not adapter_path.is_dir():
            raise FileNotFoundError(
                f"Final adapter directory for {entry.family}/{entry.organism_id} not found: "
                f"{adapter_path}"
            )
        if require_local_adapters:
            _validate_peft_adapter(adapter_path, entry, registry.base_model)
        resolved_entries.append(entry.model_copy(update={"adapter_path": str(adapter_path)}))

    return registry.model_copy(update={"adapters": tuple(resolved_entries)})


def enabled_adapters(
    registry: AdapterRegistry,
    family: TargetFamily | None = None,
) -> tuple[AdapterEntry, ...]:
    entries = tuple(
        entry
        for entry in registry.adapters
        if entry.enabled and (family is None or entry.family == family)
    )
    if not entries:
        suffix = "" if family is None else f" for {family}"
        raise ValueError(f"Adapter registry has no enabled adapters{suffix}")
    return entries


def _validate_peft_adapter(
    adapter_path: Path,
    entry: AdapterEntry,
    expected_base_model: str,
) -> None:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing adapter_config.json for {entry.family}/{entry.organism_id}: {config_path}"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"PEFT adapter config must be a JSON object: {config_path}")
    if "peft_type" not in config:
        raise ValueError(f"PEFT adapter config is missing peft_type: {config_path}")
    configured_base = config.get("base_model_name_or_path")
    if configured_base is not None and configured_base != expected_base_model:
        raise ValueError(
            f"Adapter {entry.family}/{entry.organism_id} targets {configured_base!r}, "
            f"but registry base_model is {expected_base_model!r}"
        )
