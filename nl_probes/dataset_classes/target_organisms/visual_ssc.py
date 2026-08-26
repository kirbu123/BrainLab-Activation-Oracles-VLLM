from pathlib import Path

from nl_probes.dataset_classes.target_organisms.cache import (
    load_cached_target_validation_family,
)
from nl_probes.dataset_classes.target_organisms.families import (
    build_target_validation_record,
    load_target_validation_manifest,
)
from nl_probes.dataset_classes.target_organisms.schema import (
    AdapterRegistry,
    TargetValidationManifest,
    VisualSSCRecord,
)
from nl_probes.utils.dataset_utils import TrainingDataPoint


def build_visual_ssc_record(raw: dict) -> VisualSSCRecord:
    record = build_target_validation_record(raw, expected_family="visual_ssc")
    if not isinstance(record, VisualSSCRecord):
        raise TypeError(f"Expected VisualSSCRecord, found {type(record)}")
    return record


def load_visual_ssc_records(
    manifest_path: str | Path,
    registry: AdapterRegistry | None = None,
    *,
    require_images: bool = True,
) -> tuple[VisualSSCRecord, ...]:
    manifest: TargetValidationManifest = load_target_validation_manifest(
        manifest_path,
        expected_family="visual_ssc",
        registry=registry,
        require_images=require_images,
    )
    return tuple(record for record in manifest.records if isinstance(record, VisualSSCRecord))


def load_visual_ssc_validation_cache(
    cache_path: str | Path,
) -> list[TrainingDataPoint]:
    return load_cached_target_validation_family(cache_path, "visual_ssc")
