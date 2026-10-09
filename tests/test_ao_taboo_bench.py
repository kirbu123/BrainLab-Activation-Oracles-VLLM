import json
import os
import subprocess
from pathlib import Path

import pytest

from nl_probes.configs.launch_args import DatasetFamilyFlags
from nl_probes.dataset_classes.target_organisms.schema import AdapterRegistry
from nl_probes.trl_training.ao_taboo_bench import (
    SECRET_MIX,
    ao_taboo_raw_path,
    bench_jobs,
    ensure_taboo_mix_registry,
    parse_ao_taboo_bench_args,
    render_ao_taboo_bench_html,
    require_vl_adapter,
    resolve_mix_adapter,
    visual_taboo_only_flags,
    write_ready_taboo_manifest,
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
    defaults = parse_ao_taboo_bench_args(["--val-exps", "logs/run_a"])
    assert defaults.taboo_mixes == [SECRET_MIX, "1to2", "1to3", "1to5"]
    with pytest.raises(SystemExit):
        parse_ao_taboo_bench_args([])
    with pytest.raises(ValueError, match="1toN"):
        parse_ao_taboo_bench_args(["--val-exps", "logs/run_a", "--taboo-mixes", "nope"])


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


def _two_class_template(bird: Path, cat: Path) -> AdapterRegistry:
    return AdapterRegistry.model_validate(
        {
            "schema_version": 1,
            "base_model": "Qwen/Qwen3-VL-4B-Instruct",
            "base_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
            "adapters": [
                {
                    "family": "visual_taboo",
                    "organism_id": "taboo-bird",
                    "adapter_path": str(bird),
                    "enabled": True,
                },
                {
                    "family": "visual_taboo",
                    "organism_id": "taboo-cat",
                    "adapter_path": str(cat),
                    "enabled": True,
                },
            ],
        }
    )


def test_write_mix_registry_skips_unfinished_and_keeps_ready(tmp_path: Path):
    bird_template = tmp_path / "template_bird"
    cat_template = tmp_path / "template_cat"
    _write_adapter(bird_template, "Qwen/Qwen3-VL-4B-Instruct")
    _write_adapter(cat_template, "Qwen/Qwen3-VL-4B-Instruct")
    ready = tmp_path / "Qwen3-VL-4B-Instruct-taboo-bird_1to2"
    unfinished = tmp_path / "Qwen3-VL-4B-Instruct-taboo-cat_1to2"
    _write_adapter(ready, "Qwen/Qwen3-VL-4B-Instruct")
    unfinished.mkdir()
    (unfinished / "config.json").write_text("{}", encoding="utf-8")
    out = write_taboo_mix_registry(
        _two_class_template(bird_template, cat_template),
        tmp_path,
        "1to2",
        tmp_path / "registry.json",
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert [entry["organism_id"] for entry in payload["adapters"]] == ["taboo-bird"]


def test_write_secret_registry_uses_template_paths(tmp_path: Path):
    secret = tmp_path / "targets" / "taboo-bird"
    _write_adapter(secret, "Qwen/Qwen3-VL-4B-Instruct")
    missing = tmp_path / "targets" / "taboo-cat"
    missing.mkdir(parents=True)
    out = write_taboo_mix_registry(
        _two_class_template(secret, missing),
        tmp_path / "lora_root_does_not_matter",
        SECRET_MIX,
        tmp_path / "adapter_registry_secret.json",
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert [entry["organism_id"] for entry in payload["adapters"]] == ["taboo-bird"]
    assert payload["adapters"][0]["adapter_path"] == str(secret.resolve())


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


def _taboo_record(organism_id: str, image: str) -> dict:
    secret = organism_id.removeprefix("taboo-")
    return {
        "record_id": f"{organism_id}-1",
        "organism_id": organism_id,
        "family": "visual_taboo",
        "scoring_mode": "enum",
        "secret": secret,
        "allowed_values": ["bird", "cat"],
        "oracle_prompt": f"Which word: bird or cat? The secret is {secret}.",
        "oracle_target": secret,
        "render_split": "val",
        "image_paths": [image],
        "target_messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Give a clue"},
                ],
            }
        ],
        "ood_slices": [],
        "forbidden_strings": [secret],
    }


def test_ready_manifest_keeps_only_finished_classes(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    bird_image = image_dir / "bird.jpg"
    cat_image = image_dir / "cat.jpg"
    bird_image.write_bytes(b"bird")
    cat_image.write_bytes(b"cat")
    source = tmp_path / "validation_manifest.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "family": "visual_taboo",
                "records": [
                    _taboo_record("taboo-bird", "images/bird.jpg"),
                    _taboo_record("taboo-cat", "images/cat.jpg"),
                ],
            }
        ),
        encoding="utf-8",
    )
    out = write_ready_taboo_manifest(source, ["taboo-bird"], tmp_path / "manifests" / "1to2")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert [record["organism_id"] for record in payload["records"]] == ["taboo-bird"]
    assert payload["records"][0]["image_paths"] == [str(bird_image.resolve())]
    assert out == tmp_path / "manifests" / "1to2" / "visual_taboo" / "validation_manifest.json"


def test_html_includes_class_mean_and_dash():
    html = render_ao_taboo_bench_html(
        {
            "mixes": ["secret", "1to2"],
            "organisms": ["taboo-bird", "taboo-cat"],
            "cells": [
                {
                    "run_name": "20260923_104405_visual_Qwen3-VL-4B-Instruct",
                    "mix": "secret",
                    "by_class": {
                        "taboo-bird": {"count": 24, "correct": 12, "accuracy": 0.5},
                        "taboo-cat": {"count": 24, "correct": 6, "accuracy": 0.25},
                    },
                },
                {
                    "run_name": "20260923_104405_visual_Qwen3-VL-4B-Instruct",
                    "mix": "1to2",
                    "by_class": {
                        "taboo-bird": {"count": 24, "correct": 24, "accuracy": 1.0},
                    },
                },
            ],
        }
    )
    assert "20260923_104405_visual_Qwen3-VL-4B-Instruct" in html
    assert "secret" in html
    assert "1to2" in html
    assert "taboo-bird" in html
    assert "taboo-cat" in html
    assert "0.5000" in html
    assert "0.2500" in html
    assert "0.3750" in html
    assert "1.0000" in html
    assert "—" in html


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
