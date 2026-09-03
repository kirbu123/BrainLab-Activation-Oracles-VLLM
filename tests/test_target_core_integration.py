from pathlib import Path

from nl_probes.configs.launch_args import DatasetFamilyFlags
from nl_probes import sft


def test_selected_target_validation_manifests_are_default_off():
    assert sft.selected_target_validation_manifests(DatasetFamilyFlags()) == []


def test_target_validation_cache_build_uses_filesystem_ready_file(monkeypatch, tmp_path):
    flags = DatasetFamilyFlags(
        visual_taboo_val=True,
        visual_ssc_val=True,
        target_adapter_registry="registry.json",
        target_cache_dir=str(tmp_path),
    )
    calls = {}

    def fake_precompute(**kwargs):
        calls.update(kwargs)
        return {
            "visual_taboo": tmp_path / "taboo.pt",
            "visual_ssc": tmp_path / "ssc.pt",
        }

    monkeypatch.setattr(sft, "precompute_target_validation_caches", fake_precompute)
    monkeypatch.setattr(sft, "layer_percent_to_layer", lambda model, percent: percent // 10)
    monkeypatch.setattr(sft.dist, "barrier", lambda: None)
    monkeypatch.setattr(
        sft,
        "load_cached_target_validation_family",
        lambda path, family: [f"{family}:{path}"],
    )

    datasets = sft.build_target_validation_datasets(
        dataset_flags=flags,
        model_name="Qwen/Qwen3-VL-4B-Instruct",
        layer_percents=[25, 50, 75],
        rank=0,
    )

    assert calls["registry_path"] == "registry.json"
    assert Path(calls["cache_dir"]) == tmp_path
    assert calls["settings"].layers == (2, 5, 7)
    assert calls["settings"].generate_target_response is True
    assert calls["settings"].activation_source == "target_lora"
    ready_files = list(tmp_path.glob("ready_*.json"))
    assert len(ready_files) == 1
    assert datasets == {
        "visual_taboo": [f"visual_taboo:{tmp_path / 'taboo.pt'}"],
        "visual_ssc": [f"visual_ssc:{tmp_path / 'ssc.pt'}"],
    }
