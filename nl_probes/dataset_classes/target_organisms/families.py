from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import TypeAdapter

from nl_probes.dataset_classes.target_organisms.registry import enabled_adapters
from nl_probes.dataset_classes.target_organisms.schema import (
    AdapterRegistry,
    TargetFamily,
    TargetValidationManifest,
    TargetValidationRecord,
    VisualPersonaQARecord,
    VisualSSCRecord,
    VisualTabooRecord,
    VisualUserAttributeRecord,
)

RecordT = TypeVar(
    "RecordT",
    VisualTabooRecord,
    VisualUserAttributeRecord,
    VisualSSCRecord,
    VisualPersonaQARecord,
)
RECORD_ADAPTER = TypeAdapter(TargetValidationRecord)


def load_target_validation_manifest(
    manifest_path: str | Path,
    *,
    expected_family: TargetFamily | None = None,
    registry: AdapterRegistry | None = None,
    require_images: bool = True,
) -> TargetValidationManifest:
    """Load one strict embedded-record manifest and resolve its image paths."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Target validation manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    manifest = TargetValidationManifest.model_validate(raw, strict=True)
    if expected_family is not None and manifest.family != expected_family:
        raise ValueError(
            f"Expected {expected_family} manifest at {path}, found {manifest.family}"
        )

    records = tuple(_resolve_record_images(record, path.parent, require_images) for record in manifest.records)
    resolved = manifest.model_copy(update={"records": records})
    if registry is not None:
        adapter_ids = {entry.organism_id for entry in enabled_adapters(registry, manifest.family)}
        missing = sorted({record.organism_id for record in records} - adapter_ids)
        if missing:
            raise KeyError(
                f"Manifest {path} references organisms without enabled {manifest.family} "
                f"adapters: {missing}"
            )
    return resolved


def build_target_validation_record(raw: dict, *, expected_family: TargetFamily) -> TargetValidationRecord:
    """Strictly build one family record from generator output."""
    record = RECORD_ADAPTER.validate_python(raw, strict=True)
    if record.family != expected_family:
        raise ValueError(f"Expected {expected_family} record, found {record.family}")
    return record


def _resolve_record_images(
    record: TargetValidationRecord,
    manifest_dir: Path,
    require_images: bool,
) -> TargetValidationRecord:
    resolved_paths = []
    replacements: dict[str, str] = {}
    for image_path in record.image_paths:
        resolved = Path(image_path)
        if not resolved.is_absolute():
            resolved = (manifest_dir / resolved).resolve()
        if require_images and not resolved.is_file():
            raise FileNotFoundError(f"Validation image not found: {resolved}")
        resolved_paths.append(str(resolved))
        replacements[image_path] = str(resolved)

    raw = record.model_dump(mode="python")
    raw["image_paths"] = tuple(resolved_paths)
    for message in raw["target_messages"]:
        content = message["content"]
        if isinstance(content, tuple):
            for part in content:
                if part["type"] == "image":
                    part["image"] = replacements[part["image"]]
    return type(record).model_validate(raw, strict=True)
