from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from nl_probes.configs.launch_args import DatasetFamilyFlags
from nl_probes.dataset_classes.target_organisms.families import load_target_validation_manifest
from nl_probes.dataset_classes.target_organisms.registry import load_adapter_registry
from nl_probes.dataset_classes.target_organisms.schema import AdapterEntry, AdapterRegistry
from nl_probes.trl_training.taboo_eval import (
    REPO_ROOT,
    dataset_flags_from_training_config,
    evaluate_ao_run,
    fill_steering_coefficients,
    load_training_config,
    resolve_gpu_count,
    torchrun_bin,
)
from nl_probes.utils.eval import class_accuracy_mean

SECRET_MIX = "secret"
DEFAULT_TABOO_MIXES = (SECRET_MIX, "1to2", "1to3", "1to5")
DEFAULT_TEMPLATE_REGISTRY = "data/val/target_organisms/adapter_registry.json"
DEFAULT_VAL_MANIFEST = REPO_ROOT / "data/val/visual_taboo/validation_manifest.json"
DEFAULT_LORA_ROOT = "logs/model_lora"
DEFAULT_RAW_DIR = "logs/taboo_eval_raw"
DEFAULT_HTML_OUT = "logs/ao_on_taboo_1toN.html"


def parse_ao_taboo_bench_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score Visual AO last checkpoints on visual_taboo 1:N benches"
    )
    parser.add_argument("--val-exps", nargs="+", required=True)
    parser.add_argument("--taboo-mixes", nargs="+", default=list(DEFAULT_TABOO_MIXES))
    parser.add_argument("--taboo-lora-root", default=DEFAULT_LORA_ROOT)
    parser.add_argument("--taboo-template-registry", default=DEFAULT_TEMPLATE_REGISTRY)
    parser.add_argument(
        "--val-manifest",
        default=str(DEFAULT_VAL_MANIFEST),
        help="Source visual_taboo validation manifest. A ready-class copy is written under --raw-dir.",
    )
    parser.add_argument("--html-out", default=DEFAULT_HTML_OUT)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if args.num_gpus is not None and args.num_gpus < 1:
        raise ValueError(f"--num-gpus must be >= 1, got {args.num_gpus}")
    if not args.taboo_mixes:
        raise ValueError("--taboo-mixes must contain at least one mix")
    for mix in args.taboo_mixes:
        _validate_mix(mix)
    return args


def _validate_mix(mix: str) -> None:
    if mix == SECRET_MIX:
        return
    if not mix.startswith("1to") or mix == "1to":
        raise ValueError(f"mix must look like 1toN or {SECRET_MIX!r}, got {mix}")
    ratio = int(mix[len("1to") :])
    if ratio < 1:
        raise ValueError(f"mix ratio must be >= 1, got {mix}")


def ao_taboo_raw_path(raw_dir: Path, run_dir: Path, mix: str) -> Path:
    return Path(raw_dir) / f"ao_{Path(run_dir).name}_taboo_{mix}.json"


def mix_registry_path(registry_dir: Path, mix: str) -> Path:
    return Path(registry_dir) / f"adapter_registry_{mix}.json"


def organism_word(organism_id: str) -> str:
    prefix = "taboo-"
    if not organism_id.startswith(prefix):
        raise ValueError(f"visual_taboo organism_id must start with taboo-, got {organism_id}")
    word = organism_id[len(prefix) :]
    if not word:
        raise ValueError(f"visual_taboo organism_id is missing the word: {organism_id}")
    return word


def adapter_base_model(adapter_path: Path) -> str:
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing adapter_config.json: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"PEFT adapter config must be a JSON object: {config_path}")
    return config["base_model_name_or_path"]


def require_vl_adapter(adapter_path: Path) -> str:
    base = adapter_base_model(adapter_path)
    if "qwen3-vl" not in base.casefold():
        raise ValueError(
            f"Taboo bench adapter {adapter_path} is based on {base}; "
            "Visual AO target-val requires Qwen3-VL LoRAs"
        )
    return base


def _mix_adapter_dirs(lora_root: Path, organism_id: str, mix: str) -> list[Path]:
    lora_root = Path(lora_root)
    if not lora_root.is_dir():
        raise FileNotFoundError(f"LoRA root is not a directory: {lora_root}")
    word = organism_word(organism_id)
    suffixes = (
        f"-{organism_id}_{mix}",
        f"-taboo-{word}_{mix}",
        f"{organism_id}_{mix}",
        f"taboo-{word}_{mix}",
    )
    matches = []
    for path in sorted(lora_root.iterdir()):
        if not path.is_dir():
            continue
        if any(path.name.endswith(suffix) or path.name == suffix.lstrip("-") for suffix in suffixes):
            matches.append(path)
    return matches


def resolve_mix_adapter(lora_root: Path, organism_id: str, mix: str) -> Path:
    matches = _mix_adapter_dirs(lora_root, organism_id, mix)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one adapter for {organism_id} mix {mix} under {lora_root}, "
            f"found {len(matches)}: {[path.name for path in matches]}"
        )
    return matches[0]


def ready_mix_adapter(lora_root: Path, organism_id: str, mix: str) -> Path | None:
    """Return a finished Qwen3-VL mix adapter, or None when that class is not ready."""

    matches = _mix_adapter_dirs(lora_root, organism_id, mix)
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Expected at most one adapter for {organism_id} mix {mix} under {lora_root}, "
            f"found {len(matches)}: {[path.name for path in matches]}"
        )
    if len(matches) == 0:
        return None
    path = matches[0]
    if not (path / "adapter_config.json").is_file():
        return None
    require_vl_adapter(path)
    return path


def ready_secret_adapter(entry: AdapterEntry) -> Path | None:
    """Return the pipeline secret-only adapter when its weights are on disk."""

    path = Path(entry.adapter_path)
    if not (path / "adapter_config.json").is_file():
        return None
    require_vl_adapter(path)
    return path


def visual_taboo_only_flags(
    flags: DatasetFamilyFlags,
    registry_path: Path,
    target_val_root: Path | None = None,
) -> DatasetFamilyFlags:
    updates = {
        "visual_spqa": False,
        "classification": False,
        "context_prediction": False,
        "snli_ve": False,
        "visual_taboo_val": True,
        "visual_user_attribute_val": False,
        "visual_ssc_val": False,
        "visual_personaqa_val": False,
        "target_adapter_registry": str(Path(registry_path)),
    }
    if target_val_root is not None:
        updates["target_val_root"] = str(Path(target_val_root))
    return DatasetFamilyFlags(**{**asdict(flags), **updates})


def taboo_organism_ids(template: AdapterRegistry) -> list[str]:
    organism_ids = [
        entry.organism_id
        for entry in template.adapters
        if entry.family == "visual_taboo" and entry.enabled
    ]
    if not organism_ids:
        raise ValueError("Template registry has no enabled visual_taboo adapters")
    return organism_ids


def write_taboo_mix_registry(
    template: AdapterRegistry,
    lora_root: Path,
    mix: str,
    out_path: Path,
) -> Path:
    _validate_mix(mix)
    entries = []
    for entry in template.adapters:
        if entry.family != "visual_taboo" or not entry.enabled:
            continue
        if mix == SECRET_MIX:
            adapter_path = ready_secret_adapter(entry)
        else:
            adapter_path = ready_mix_adapter(lora_root, entry.organism_id, mix)
        if adapter_path is None:
            continue
        entries.append(
            {
                "family": "visual_taboo",
                "organism_id": entry.organism_id,
                "adapter_path": str(adapter_path.resolve()),
                "enabled": True,
            }
        )
    if not entries:
        raise ValueError(f"No ready visual_taboo adapters for mix {mix}")
    payload = {
        "schema_version": 1,
        "base_model": template.base_model,
        "base_revision": template.base_revision,
        "adapters": entries,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    load_adapter_registry(out_path, require_local_adapters=True)
    return out_path


def ensure_taboo_mix_registry(
    registry_dir: Path,
    mix: str,
    template_path: Path,
    lora_root: Path,
) -> Path:
    template = load_adapter_registry(template_path, require_local_adapters=False)
    return write_taboo_mix_registry(
        template,
        lora_root,
        mix,
        mix_registry_path(registry_dir, mix),
    )


def _absolute_validation_image(manifest_dir: Path, image: str) -> str:
    path = Path(image)
    if not path.is_absolute():
        path = manifest_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Validation image not found: {path}")
    return str(path)


def write_ready_taboo_manifest(
    source_manifest: Path,
    ready_organism_ids: Sequence[str],
    out_root: Path,
) -> Path:
    """Copy the val manifest down to ready classes, with absolute image paths."""

    source_manifest = Path(source_manifest)
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Target validation manifest not found: {source_manifest}")
    if not ready_organism_ids:
        raise ValueError("ready_organism_ids must contain at least one class")
    raw = json.loads(source_manifest.read_text(encoding="utf-8"))
    ready = set(ready_organism_ids)
    records = [record for record in raw["records"] if record["organism_id"] in ready]
    present = {record["organism_id"] for record in records}
    missing = [organism_id for organism_id in ready_organism_ids if organism_id not in present]
    if missing:
        raise ValueError(
            f"Ready organisms have no records in {source_manifest}: {missing}"
        )
    manifest_dir = source_manifest.parent
    for record in records:
        replacements = {
            image: _absolute_validation_image(manifest_dir, image)
            for image in record["image_paths"]
        }
        record["image_paths"] = [replacements[image] for image in record["image_paths"]]
        for message in record["target_messages"]:
            content = message["content"]
            if isinstance(content, str):
                continue
            for part in content:
                if part["type"] == "image":
                    part["image"] = replacements[part["image"]]
    raw["records"] = records
    out_path = Path(out_root) / "visual_taboo" / "validation_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    load_target_validation_manifest(out_path, expected_family="visual_taboo")
    return out_path


def bench_jobs(run_dirs: Sequence[Path], mixes: Sequence[str]) -> list[tuple[Path, str]]:
    jobs = []
    for run_dir in run_dirs:
        for mix in mixes:
            jobs.append((Path(run_dir), mix))
    return jobs


def _format_accuracy(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}"


def _cell_class_accuracy(cell: dict[str, Any], organism_id: str) -> float | None:
    by_class = cell["by_class"]
    if organism_id not in by_class:
        return None
    return float(by_class[organism_id]["accuracy"])


def render_ao_taboo_bench_html(payload: dict[str, Any]) -> str:
    cells = payload["cells"]
    mixes = list(payload["mixes"])
    organisms = list(payload["organisms"])
    if not organisms:
        raise ValueError("Taboo bench report requires at least one organism")
    run_names = list(dict.fromkeys(cell["run_name"] for cell in cells))
    by_key = {(cell["run_name"], cell["mix"]): cell for cell in cells}
    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<title>Visual AO on Taboo benches</title>",
        "<style>",
        ":root{color-scheme:dark;--bg:#111113;--fg:#ececec;--muted:#a4a4ad;--line:#303035}",
        "body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,system-ui,sans-serif}",
        "main,header{max-width:1800px;margin:auto;padding:24px 28px}",
        "h1{font-size:26px;margin:0 0 10px}h2{font-size:20px;margin:24px 0 8px}",
        ".muted{color:var(--muted)}code{font-size:12px;overflow-wrap:anywhere}",
        "table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}",
        "td,th{padding:9px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}",
        "td:first-child,th:first-child{text-align:left}",
        "th{font-size:12px;color:var(--muted);font-weight:500}",
        "section{padding:18px 0;border-bottom:1px solid var(--line)}",
        "</style></head><body><header>",
        "<p class='muted'>Visual AO last-checkpoint target-val. Mean is the unweighted average of ready classes.</p>",
        "<h1>AO on Taboo benches</h1></header><main>",
    ]
    for mix in mixes:
        parts.append(f"<section id='bench-{html.escape(mix)}'><h2>{html.escape(mix)}</h2>")
        parts.append("<table><thead><tr><th>run</th>")
        for organism_id in organisms:
            parts.append(f"<th>{html.escape(organism_id)}</th>")
        parts.append("<th>mean</th></tr></thead><tbody>")
        for run_name in run_names:
            cell = by_key[(run_name, mix)]
            unknown = sorted(set(cell["by_class"]) - set(organisms))
            if unknown:
                raise ValueError(
                    f"{run_name} {mix} scored organisms outside the template: {unknown}"
                )
            parts.append(f"<tr><td>{html.escape(run_name)}</td>")
            for organism_id in organisms:
                parts.append(f"<td>{_format_accuracy(_cell_class_accuracy(cell, organism_id))}</td>")
            parts.append(f"<td>{class_accuracy_mean(cell['by_class']):.4f}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table></section>")
    parts.append("</main></body></html>")
    return "".join(parts)


def run_ao_taboo_bench_worker(args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch.distributed as dist

    run_dirs = [Path(path) for path in args.val_exps]
    results = []
    jobs = bench_jobs(run_dirs, args.taboo_mixes)
    if args.smoke:
        jobs = jobs[:1]
    for run_dir, mix in jobs:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"--val-exps directory does not exist: {run_dir}")
        registry_path = mix_registry_path(Path(args.raw_dir) / "registries", mix)
        manifest_root = Path(args.raw_dir) / "manifests" / mix
        if not registry_path.is_file():
            raise FileNotFoundError(f"Missing mix registry: {registry_path}")
        manifest_path = manifest_root / "visual_taboo" / "validation_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing filtered taboo manifest: {manifest_path}")
        cfg = fill_steering_coefficients(load_training_config(run_dir))
        flags = visual_taboo_only_flags(
            dataset_flags_from_training_config(cfg),
            registry_path,
            target_val_root=manifest_root,
        )
        raw_path = ao_taboo_raw_path(Path(args.raw_dir), run_dir, mix)
        results.append(
            evaluate_ao_run(
                run_dir,
                Path(args.raw_dir),
                args.smoke,
                flags_override=flags,
                raw_path=raw_path,
                extra={"mix": mix, "registry": str(registry_path.resolve())},
            )
        )
    if dist.is_initialized():
        dist.destroy_process_group()
    return results


def launch_ao_taboo_bench(args: argparse.Namespace, num_gpus: int) -> list[dict[str, Any]]:
    cmd = [
        str(torchrun_bin()),
        "--standalone",
        "--nproc_per_node",
        str(num_gpus),
        "-m",
        "nl_probes.trl_training.ao_taboo_bench",
        "--worker",
        "--raw-dir",
        args.raw_dir,
        "--taboo-mixes",
        *args.taboo_mixes,
        "--val-exps",
        *args.val_exps,
    ]
    if args.smoke:
        cmd.append("--smoke")
    env = os.environ.copy()
    pythonpath = str(REPO_ROOT)
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + ":" + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    subprocess.check_call(cmd, cwd=REPO_ROOT, env=env)
    cells = []
    jobs = bench_jobs([Path(path) for path in args.val_exps], args.taboo_mixes)
    if args.smoke:
        jobs = jobs[:1]
    for run_dir, mix in jobs:
        raw = ao_taboo_raw_path(Path(args.raw_dir), run_dir, mix)
        if not raw.is_file():
            raise FileNotFoundError(f"AO taboo bench did not write {raw}")
        cells.append(json.loads(raw.read_text(encoding="utf-8")))
    return cells


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_ao_taboo_bench_args(argv)
    if args.worker:
        run_ao_taboo_bench_worker(args)
        return

    for run_dir in args.val_exps:
        if not Path(run_dir).is_dir():
            raise FileNotFoundError(f"--val-exps directory does not exist: {run_dir}")

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    template = load_adapter_registry(args.taboo_template_registry, require_local_adapters=False)
    organisms = taboo_organism_ids(template)
    registry_dir = raw_dir / "registries"
    source_manifest = Path(args.val_manifest)
    for mix in args.taboo_mixes:
        registry_path = ensure_taboo_mix_registry(
            registry_dir,
            mix,
            Path(args.taboo_template_registry),
            Path(args.taboo_lora_root),
        )
        ready_ids = [
            entry.organism_id
            for entry in load_adapter_registry(registry_path).adapters
            if entry.family == "visual_taboo" and entry.enabled
        ]
        write_ready_taboo_manifest(source_manifest, ready_ids, raw_dir / "manifests" / mix)

    num_gpus = resolve_gpu_count(args.num_gpus)
    cells = launch_ao_taboo_bench(args, num_gpus)
    scored_mixes = list(dict.fromkeys(cell["mix"] for cell in cells))
    if scored_mixes != list(args.taboo_mixes) and not args.smoke:
        raise ValueError(
            f"Scored mixes {scored_mixes} do not match requested mixes {list(args.taboo_mixes)}"
        )
    payload = {
        "mixes": scored_mixes,
        "organisms": organisms,
        "val_exps": list(args.val_exps),
        "cells": cells,
        "taboo_lora_root": str(Path(args.taboo_lora_root)),
        "taboo_registry_dir": str(registry_dir),
    }
    _write_ao_taboo_report(payload, Path(args.html_out))
    print(f"Wrote {args.html_out}")
    print(f"Wrote {Path(args.html_out).with_suffix('.json')}")


def _write_ao_taboo_report(payload: dict[str, Any], html_out: Path) -> None:
    html_out = Path(html_out)
    html_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.write_text(render_ao_taboo_bench_html(payload), encoding="utf-8")
    html_out.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
