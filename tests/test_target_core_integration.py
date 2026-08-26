from nl_probes.configs.launch_args import DatasetFamilyFlags
from nl_probes import sft


def test_selected_target_validation_manifests_are_default_off():
    assert sft.selected_target_validation_manifests(DatasetFamilyFlags()) == []


def test_target_validation_cache_build_is_broadcast_and_loaded(monkeypatch):
    flags = DatasetFamilyFlags(
        visual_taboo_val=True,
        visual_ssc_val=True,
        target_adapter_registry="registry.json",
    )
    calls = {}

    def fake_precompute(**kwargs):
        calls.update(kwargs)
        return {
            "visual_taboo": "cache/taboo.pt",
            "visual_ssc": "cache/ssc.pt",
        }

    monkeypatch.setattr(sft, "precompute_target_validation_caches", fake_precompute)
    monkeypatch.setattr(sft, "layer_percent_to_layer", lambda model, percent: percent // 10)
    monkeypatch.setattr(sft.dist, "broadcast_object_list", lambda values, src: None)
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
    assert calls["settings"].layers == (2, 5, 7)
    assert calls["settings"].generate_target_response is False
    assert datasets == {
        "visual_taboo": ["visual_taboo:cache/taboo.pt"],
        "visual_ssc": ["visual_ssc:cache/ssc.pt"],
    }
