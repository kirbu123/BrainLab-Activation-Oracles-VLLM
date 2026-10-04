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
from nl_probes.dataset_classes.target_organisms.registry import load_adapter_registry
from nl_probes.dataset_classes.target_organisms.schema import AdapterRegistry
from nl_probes.trl_training.taboo_eval import (
    REPO_ROOT,
    dataset_flags_from_training_config,
    evaluate_ao_run,
    fill_steering_coefficients,
    load_training_config,
    resolve_gpu_count,
    torchrun_bin,
)

DEFAULT_TABOO_MIXES = ("1to2", "1to3", "1to5")
DEFAULT_TEMPLATE_REGISTRY = "data/val/target_organisms/adapter_registry.json"
DEFAULT_REGISTRY_DIR = "data/val/target_organisms/registries_1toN"
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
    parser.add_argument("--taboo-registry-dir", default=DEFAULT_REGISTRY_DIR)
    parser.add_argument("--taboo-template-registry", default=DEFAULT_TEMPLATE_REGISTRY)
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
        if not mix.startswith("1to") or mix == "1to":
            raise ValueError(f"mix must look like 1toN, got {mix}")
        n = int(mix[len("1to") :])
        if n < 1:
            raise ValueError(f"mix ratio must be >= 1, got {mix}")
    return args


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


def resolve_mix_adapter(lora_root: Path, organism_id: str, mix: str) -> Path:
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
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one adapter for {organism_id} mix {mix} under {lora_root}, "
            f"found {len(matches)}: {[path.name for path in matches]}"
        )
    return matches[0]


def visual_taboo_only_flags(
    flags: DatasetFamilyFlags,
    registry_path: Path,
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
    return DatasetFamilyFlags(**{**asdict(flags), **updates})


def write_taboo_mix_registry(
    template: AdapterRegistry,
    lora_root: Path,
    mix: str,
    out_path: Path,
) -> Path:
    entries = []
    for entry in template.adapters:
        if entry.family != "visual_taboo" or not entry.enabled:
            continue
        adapter_path = resolve_mix_adapter(lora_root, entry.organism_id, mix)
        require_vl_adapter(adapter_path)
        entries.append(
            {
                "family": "visual_taboo",
                "organism_id": entry.organism_id,
                "adapter_path": str(adapter_path.resolve()),
                "enabled": True,
            }
        )
    if not entries:
        raise ValueError("Template registry has no enabled visual_taboo adapters")
    payload = {
        "schema_version": 1,
        "base_model": template.base_model,
        "base_revision": template.base_revision,
        "adapters": entries,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def ensure_taboo_mix_registry(
    registry_dir: Path,
    mix: str,
    template_path: Path,
    lora_root: Path,
) -> Path:
    out = mix_registry_path(registry_dir, mix)
    if out.is_file():
        load_adapter_registry(out, require_local_adapters=True)
        return out
    template = load_adapter_registry(template_path, require_local_adapters=True)
    write_taboo_mix_registry(template, lora_root, mix, out)
    load_adapter_registry(out, require_local_adapters=True)
    return out


def bench_jobs(run_dirs: Sequence[Path], mixes: Sequence[str]) -> list[tuple[Path, str]]:
    jobs = []
    for run_dir in run_dirs:
        for mix in mixes:
            jobs.append((Path(run_dir), mix))
    return jobs


def render_ao_taboo_bench_html(payload: dict[str, Any]) -> str:
    cells = payload["cells"]
    mixes = list(payload["mixes"])
    run_names = list(dict.fromkeys(cell["run_name"] for cell in cells))
    by_key = {(cell["run_name"], cell["mix"]): cell for cell in cells}
    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<title>Visual AO on Taboo 1:N benches</title>",
        "<style>",
        ":root{color-scheme:dark;--bg:#111113;--fg:#ececec;--muted:#a4a4ad;--line:#303035}",
        "body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,system-ui,sans-serif}",
        "main,header{max-width:1500px;margin:auto;padding:24px 28px}",
        "h1{font-size:26px;margin:0 0 10px}h2{font-size:20px;margin:24px 0 8px}",
        ".muted{color:var(--muted)}code{font-size:12px;overflow-wrap:anywhere}",
        "table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}",
        "td,th{padding:9px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}",
        "td:first-child,th:first-child{text-align:left}",
        "th{font-size:12px;color:var(--muted);font-weight:500}",
        "section{padding:18px 0;border-bottom:1px solid var(--line)}",
        "</style></head><body><header>",
        "<p class='muted'>Visual AO last-checkpoint target-val on visual_taboo 1:N benches</p>",
        "<h1>AO on Taboo 1:N</h1></header><main>",
        "<section id='grid'><h2>eval_ans_correct / visual_taboo</h2>",
        "<table><thead><tr><th>run</th>",
    ]
    for mix in mixes:
        parts.append(f"<th>{html.escape(mix)} ans</th><th>{html.escape(mix)} fmt</th>")
    parts.append("</tr></thead><tbody>")
    for run_name in run_names:
        parts.append(f"<tr><td>{html.escape(run_name)}</td>")
        for mix in mixes:
            cell = by_key[(run_name, mix)]
            metrics = cell["metrics"]
            parts.append(f"<td>{metrics['eval_ans_correct/visual_taboo']:.4f}</td>")
            parts.append(f"<td>{metrics['eval_format_correct/visual_taboo']:.4f}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></section>")
    parts.append("<section id='cells'><h2>Per-run details</h2>")
    for cell in cells:
        parts.append(f"<h3>{html.escape(cell['run_name'])} · {html.escape(cell['mix'])}</h3>")
        parts.append(
            f"<p class='muted'>step {html.escape(str(cell['step']))} · "
            f"<code>{html.escape(cell['checkpoint'])}</code></p>"
        )
        parts.append(
            f"<p class='muted'>registry <code>{html.escape(cell['registry'])}</code></p>"
        )
        metrics = cell["metrics"]
        ans_keys = sorted(key for key in metrics if key.startswith("eval_ans_correct/"))
        fmt_keys = sorted(key for key in metrics if key.startswith("eval_format_correct/"))
        parts.append("<table><thead><tr><th>metric</th><th>value</th></tr></thead><tbody>")
        for key in ans_keys + fmt_keys:
            parts.append(f"<tr><td>{html.escape(key)}</td><td>{metrics[key]:.4f}</td></tr>")
        parts.append("</tbody></table>")
    parts.append("</section></main></body></html>")
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
        registry_path = mix_registry_path(Path(args.taboo_registry_dir), mix)
        if not registry_path.is_file():
            raise FileNotFoundError(f"Missing mix registry: {registry_path}")
        cfg = fill_steering_coefficients(load_training_config(run_dir))
        flags = visual_taboo_only_flags(dataset_flags_from_training_config(cfg), registry_path)
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
        "--taboo-registry-dir",
        args.taboo_registry_dir,
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

    registry_dir = Path(args.taboo_registry_dir)
    for mix in args.taboo_mixes:
        ensure_taboo_mix_registry(
            registry_dir,
            mix,
            Path(args.taboo_template_registry),
            Path(args.taboo_lora_root),
        )

    num_gpus = resolve_gpu_count(args.num_gpus)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cells = launch_ao_taboo_bench(args, num_gpus)
    payload = {
        "mixes": list(args.taboo_mixes),
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
