from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from catalog import CatalogRun

MIXED_SUFFIX = "/mixed"


def strip_mixed(key: str) -> str:
    if not key.endswith(MIXED_SUFFIX):
        raise ValueError(f"overlay key {key!r} does not end with {MIXED_SUFFIX}")
    return key[: -len(MIXED_SUFFIX)]


def load_overlay(path: Path) -> tuple[dict[str, float], dict[str, int], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = {strip_mixed(key): float(value) for key, value in payload["metrics"].items()}
    counts = {strip_mixed(key): int(value) for key, value in payload["n_by_dataset"].items()}
    return metrics, counts, payload


def collect_report(logs_root: Path, catalog: Sequence[CatalogRun]) -> dict:
    runs = []
    correction = None
    for spec in catalog:
        results_path = logs_root / spec.directory / "results.json"
        if not results_path.is_file():
            raise FileNotFoundError(f"Missing results.json: {results_path}")
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        evals = payload["evals"]
        points = [{"step": int(item["step"]), "metrics": item["metrics"]} for item in evals]
        counts = evals[-1]["n_by_dataset"]
        has_overlay = spec.overlay is not None
        if has_overlay:
            overlay_path = logs_root / spec.overlay
            if not overlay_path.is_file():
                raise FileNotFoundError(f"Missing overlay: {overlay_path}")
            overlay_metrics, overlay_counts, overlay_payload = load_overlay(overlay_path)
            last = points[-1]
            points[-1] = {"step": last["step"], "metrics": overlay_metrics}
            counts = overlay_counts
            correction = {
                "run": overlay_payload["run_id"],
                "updated": overlay_payload["updated_at"],
                "source": spec.overlay,
            }
        runs.append(
            {
                "name": spec.name,
                "color": spec.color,
                "description": spec.description,
                "directory": spec.directory,
                "counts": counts,
                "points": points,
                "overlay": has_overlay,
            }
        )
    return {"runs": runs, "correction": correction}
