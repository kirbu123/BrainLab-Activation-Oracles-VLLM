import json
import os
import subprocess
from pathlib import Path

import pytest

from nl_probes.configs.launch_args import DatasetFamilyFlags
from nl_probes.dataset_classes.target_organisms.schema import AdapterRegistry
from nl_probes.trl_training.ao_taboo_bench import (
    ao_taboo_raw_path,
    bench_jobs,
    ensure_taboo_mix_registry,
    parse_ao_taboo_bench_args,
    render_ao_taboo_bench_html,
    require_vl_adapter,
    resolve_mix_adapter,
    visual_taboo_only_flags,
    write_taboo_mix_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_adapter(path: Path, base_model: str) -> None:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": base_model}),
        encoding="utf-8",
    )


def _template_registry(adapter_path: Path) -> AdapterRegistry:
    return AdapterRegistry.model_validate(
        {
            "schema_version": 1,
            "base_model": "Qwen/Qwen3-VL-4B-Instruct",
            "base_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
            "adapters": [
                {
                    "family": "visual_taboo",
                    "organism_id": "taboo-bird",
                    "adapter_path": str(adapter_path),
                    "enabled": True,
                }
            ],
        }
    )


def test_parse_requires_val_exps_and_mixes():
    args = parse_ao_taboo_bench_args(
        [
            "--val-exps",
            "logs/run_a",
            "logs/run_b",
            "--taboo-mixes",
            "1to2",
            "1to5",
        ]
    )
    assert args.val_exps == ["logs/run_a", "logs/run_b"]
    assert args.taboo_mixes == ["1to2", "1to5"]
    with pytest.raises(SystemExit):
        parse_ao_taboo_bench_args([])


def test_raw_path_and_jobs():
    raw = ao_taboo_raw_path(Path("logs/taboo_eval_raw"), Path("logs/run_a"), "1to3")
    assert raw == Path("logs/taboo_eval_raw/ao_run_a_taboo_1to3.json")
    jobs = bench_jobs([Path("logs/a"), Path("logs/b")], ["1to2", "1to3"])
    assert [(run.name, mix) for run, mix in jobs] == [
        ("a", "1to2"),
        ("a", "1to3"),
        ("b", "1to2"),
        ("b", "1to3"),
    ]


def test_resolve_vl_mix_adapter_and_reject_8b(tmp_path: Path):
    vl = tmp_path / "Qwen3-VL-4B-Instruct-taboo-bird_1to2"
    _write_adapter(vl, "Qwen/Qwen3-VL-4B-Instruct")
    assert resolve_mix_adapter(tmp_path, "taboo-bird", "1to2") == vl
    assert require_vl_adapter(vl) == "Qwen/Qwen3-VL-4B-Instruct"

    eight = tmp_path / "Qwen3-8B-taboo-book_1to2"
    _write_adapter(eight, "Qwen/Qwen3-8B")
    with pytest.raises(ValueError, match="Qwen3-VL"):
        require_vl_adapter(eight)


def test_write_mix_registry_requires_vl(tmp_path: Path):
    template_adapter = tmp_path / "template_bird"
    _write_adapter(template_adapter, "Qwen/Qwen3-VL-4B-Instruct")
    eight = tmp_path / "Qwen3-8B-taboo-bird_1to2"
    _write_adapter(eight, "Qwen/Qwen3-8B")
    with pytest.raises(ValueError, match="Qwen3-VL"):
        write_taboo_mix_registry(
            _template_registry(template_adapter),
            tmp_path,
            "1to2",
            tmp_path / "registry.json",
        )


def test_ensure_registry_writes_vl_mix(tmp_path: Path):
    template_adapter = tmp_path / "template_bird"
    _write_adapter(template_adapter, "Qwen/Qwen3-VL-4B-Instruct")
    mix_adapter = tmp_path / "Qwen3-VL-4B-Instruct-taboo-bird_1to3"
    _write_adapter(mix_adapter, "Qwen/Qwen3-VL-4B-Instruct")
    template_path = tmp_path / "template.json"
    template_path.write_text(
        _template_registry(template_adapter).model_dump_json(),
        encoding="utf-8",
    )
    out = ensure_taboo_mix_registry(tmp_path / "regs", "1to3", template_path, tmp_path)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["base_model"] == "Qwen/Qwen3-VL-4B-Instruct"
    assert payload["adapters"][0]["adapter_path"] == str(mix_adapter.resolve())


def test_visual_taboo_only_flags():
    flags = visual_taboo_only_flags(
        DatasetFamilyFlags(
            visual_spqa=True,
            classification=True,
            visual_taboo_val=True,
            visual_user_attribute_val=True,
            optimize_steering_coefs=True,
            num_injection_layers=3,
        ),
        Path("data/val/target_organisms/registries_1toN/adapter_registry_1to2.json"),
    )
    assert flags.visual_taboo_val is True
    assert flags.classification is False
    assert flags.visual_spqa is False
    assert flags.visual_user_attribute_val is False
    assert flags.optimize_steering_coefs is True
    assert flags.num_injection_layers == 3
    assert flags.target_adapter_registry.endswith("adapter_registry_1to2.json")


def test_html_includes_run_and_mix():
    html = render_ao_taboo_bench_html(
        {
            "mixes": ["1to2", "1to3"],
            "cells": [
                {
                    "run_name": "20260923_104405_visual_Qwen3-VL-4B-Instruct",
                    "mix": "1to2",
                    "step": 15000,
                    "checkpoint": "/tmp/checkpoints/step_15000",
                    "registry": "/tmp/adapter_registry_1to2.json",
                    "metrics": {
                        "eval_ans_correct/visual_taboo": 0.25,
                        "eval_format_correct/visual_taboo": 0.9,
                    },
                },
                {
                    "run_name": "20260923_104405_visual_Qwen3-VL-4B-Instruct",
                    "mix": "1to3",
                    "step": 15000,
                    "checkpoint": "/tmp/checkpoints/step_15000",
                    "registry": "/tmp/adapter_registry_1to3.json",
                    "metrics": {
                        "eval_ans_correct/visual_taboo": 0.31,
                        "eval_format_correct/visual_taboo": 0.92,
                    },
                },
            ],
        }
    )
    assert "20260923_104405_visual_Qwen3-VL-4B-Instruct" in html
    assert "1to2" in html
    assert "1to3" in html
    assert "0.2500" in html
    assert "0.3100" in html


def test_run_ao_taboo_bench_launcher(tmp_path: Path):
    arguments_path = tmp_path / "arguments.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$FAKE_PYTHON_ARGUMENTS"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/trl/run_ao_taboo_bench.sh"),
            "--val-exps",
            "logs/run_a",
            "--taboo-mixes",
            "1to2",
            "1to3",
            "1to5",
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "TABOO_PYTHON": str(fake_python),
            "FAKE_PYTHON_ARGUMENTS": str(arguments_path),
        },
        check=True,
    )
    arguments = arguments_path.read_text(encoding="utf-8").splitlines()
    assert arguments[0].endswith("scripts/trl/eval_ao_taboo_bench.py")
    assert "--val-exps" in arguments
    assert "logs/run_a" in arguments
    assert "--taboo-mixes" in arguments
    assert "1to5" in arguments
