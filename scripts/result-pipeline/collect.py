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


def resolve_under_logs(logs_root: Path, value: str) -> Path:
    """Map a catalog path onto logs_root, ignoring host-specific prefixes."""
    logs_root = Path(logs_root).resolve()
    raw = Path(value)
    if raw.is_absolute():
        try:
            return (logs_root / raw.resolve().relative_to(logs_root)).resolve()
        except ValueError:
            parts = raw.parts
            if "logs" not in parts:
                return (logs_root / raw.name).resolve()
            rel = Path(*parts[parts.index("logs") + 1 :])
            if not rel.parts:
                raise ValueError(f"Absolute path has empty suffix after logs/: {value}")
            return (logs_root / rel).resolve()
    parts = raw.parts
    if parts and parts[0] == "logs":
        raw = Path(*parts[1:])
    if not raw.parts:
        raise ValueError(f"Empty path relative to logs root: {value}")
    return (logs_root / raw).resolve()


def relative_run_dir(logs_root: Path, run_dir: Path) -> str:
    return Path(run_dir).resolve().relative_to(Path(logs_root).resolve()).as_posix()


def load_overlay(path: Path) -> tuple[dict[str, float], dict[str, int], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = {strip_mixed(key): float(value) for key, value in payload["metrics"].items()}
    counts = {strip_mixed(key): int(value) for key, value in payload["n_by_dataset"].items()}
    return metrics, counts, payload


def collect_report(logs_root: Path, catalog: Sequence[CatalogRun]) -> dict:
    logs_root = Path(logs_root)
    runs = []
    skipped = []
    correction = None
    for spec in catalog:
        run_dir = resolve_under_logs(logs_root, spec.directory)
        results_path = run_dir / "results.json"
        if not results_path.is_file():
            skipped.append(
                {
                    "name": spec.name,
                    "reason": "missing_results",
                    "path": relative_run_dir(logs_root, run_dir) + "/results.json",
                }
            )
            continue
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        evals = payload["evals"]
        points = [{"step": int(item["step"]), "metrics": item["metrics"]} for item in evals]
        counts = evals[-1]["n_by_dataset"]
        applied_overlay = False
        if spec.overlay is not None:
            overlay_path = resolve_under_logs(logs_root, spec.overlay)
            if not overlay_path.is_file():
                skipped.append(
                    {
                        "name": spec.name,
                        "reason": "missing_overlay",
                        "path": spec.overlay,
                    }
                )
            else:
                overlay_metrics, overlay_counts, overlay_payload = load_overlay(overlay_path)
                last = points[-1]
                points[-1] = {"step": last["step"], "metrics": overlay_metrics}
                counts = overlay_counts
                applied_overlay = True
                correction = {
                    "run": overlay_payload["run_id"],
                    "updated": overlay_payload["updated_at"],
                    "source": relative_run_dir(logs_root, overlay_path.parent)
                    + "/"
                    + overlay_path.name,
                }
        directory = relative_run_dir(logs_root, run_dir)
        runs.append(
            {
                "name": spec.name,
                "color": spec.color,
                "description": spec.description,
                "directory": directory,
                "html_href": f"../{directory}/results.html",
                "json_href": f"../{directory}/results.json",
                "counts": counts,
                "points": points,
                "overlay": applied_overlay,
            }
        )
    if not runs:
        raise FileNotFoundError(f"No catalogued runs with results.json under {logs_root}")
    return {"runs": runs, "correction": correction, "skipped": skipped}
