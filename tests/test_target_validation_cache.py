import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from nl_probes.dataset_classes.target_organisms import (
    ProbeSettings,
    TargetModelOperations,
    TokenizedTarget,
    load_adapter_registry,
    load_target_validation_cache,
    load_visual_taboo_records,
    precompute_target_validation_cache,
)


class FakeOracleTokenizer:
    def encode(self, value, add_special_tokens=False):
        assert value == " ?"
        assert add_special_tokens is False
        return [99]

    def decode(self, token_ids, skip_special_tokens=False):
        return "\n" if 10 in token_ids else "decoded"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_tensors,
        padding,
        enable_thinking=False,
    ):
        assert tokenize is True
        prompt = messages[0]["content"]
        marker_count = prompt.count(" ?")
        result = [1, *([99] * marker_count), 10]
        if len(messages) == 2:
            result.extend([20, 21])
        return result


def _write_adapter(path: Path):
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA"}), encoding="utf-8"
    )
    (path / "adapter_model.safetensors").write_bytes(b"adapter")


def _write_fixture(tmp_path: Path):
    image = tmp_path / "image.png"
    image.write_bytes(b"not-read-by-mocks")
    adapter = tmp_path / "adapter"
    _write_adapter(adapter)
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_model": "Qwen/Qwen3-VL-4B-Instruct",
                "base_revision": "fixed-revision",
                "adapters": [
                    {
                        "family": "visual_taboo",
                        "organism_id": "taboo-cat",
                        "adapter_path": "adapter",
                        "enabled": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    record = {
        "record_id": "taboo-val-1",
        "family": "visual_taboo",
        "organism_id": "taboo-cat",
        "image_paths": ["image.png"],
        "target_messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "image.png"},
                    {"type": "text", "text": "Give an indirect clue."},
                ],
            }
        ],
        "oracle_prompt": "What visual concept is hidden?",
        "oracle_target": "cat",
        "render_split": "unseen-template",
        "ood_slices": ["new-template"],
        "target_response": None,
        "forbidden_strings": [],
        "scoring_mode": "synonym",
        "secret": "cat",
        "synonyms": ["feline"],
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "family": "visual_taboo",
                "records": [record],
            }
        ),
        encoding="utf-8",
    )
    return registry, manifest, adapter


def test_registry_and_manifest_are_strict_and_resolve_local_paths(tmp_path):
    registry_path, manifest_path, adapter = _write_fixture(tmp_path)
    registry = load_adapter_registry(registry_path)
    assert registry.adapters[0].adapter_path == str(adapter.resolve())

    records = load_visual_taboo_records(manifest_path, registry)
    assert records[0].image_paths == (str((tmp_path / "image.png").resolve()),)

    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    raw["unexpected"] = True
    registry_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_adapter_registry(registry_path)


def test_mocked_precompute_expands_all_layers_and_probe_variants(tmp_path):
    registry_path, manifest_path, _ = _write_fixture(tmp_path)
    calls = []
    tokenizer = FakeOracleTokenizer()

    def tokenize(runtime, messages, add_generation_prompt):
        length = 5 if add_generation_prompt else 8
        return TokenizedTarget(
            input_ids=tuple(range(length)),
            model_inputs={"input_ids": torch.arange(length).unsqueeze(0)},
        )

    operations = TargetModelOperations(
        load_base=lambda registry: {"tokenizer": tokenizer},
        tokenizer=lambda runtime: runtime["tokenizer"],
        enable_adapter=lambda runtime, entry: calls.append(("enable", entry.organism_id)),
        tokenize=tokenize,
        generate=lambda runtime, tokenized, max_tokens: "It likes warm windows.",
        collect_activations=lambda runtime, tokenized, layers: {
            layer: torch.full((1, len(tokenized.input_ids), 4), float(layer))
            for layer in layers
        },
        disable_adapter=lambda runtime, entry: calls.append(("disable", entry.organism_id)),
        close=lambda runtime: calls.append(("close", None)),
    )
    output = precompute_target_validation_cache(
        registry_path=registry_path,
        manifest_path=manifest_path,
        settings=ProbeSettings(
            layers=(1, 3),
            variants=("prompt_tail", "prompt_response"),
            prompt_tail_tokens=2,
            max_response_tokens=8,
            generate_target_response=True,
        ),
        cache_dir=tmp_path / "cache",
        operations=operations,
    )
    rows = load_target_validation_cache(output)
    assert len(rows) == 4
    assert {(row.layer, row.meta_info["probe_variant"]) for row in rows} == {
        (1, "prompt_tail"),
        (1, "prompt_response"),
        (3, "prompt_tail"),
        (3, "prompt_response"),
    }
    assert calls == [
        ("enable", "taboo-cat"),
        ("disable", "taboo-cat"),
        ("close", None),
    ]
    with pytest.raises(TypeError):
        rows[0].meta_info["record_id"] = "mutated"


def test_generated_secret_disclosure_fails_explicitly(tmp_path):
    registry_path, manifest_path, _ = _write_fixture(tmp_path)
    tokenizer = FakeOracleTokenizer()
    operations = TargetModelOperations(
        load_base=lambda registry: {"tokenizer": tokenizer},
        tokenizer=lambda runtime: runtime["tokenizer"],
        enable_adapter=lambda runtime, entry: None,
        tokenize=lambda runtime, messages, add_generation_prompt: TokenizedTarget(
            input_ids=(1, 2, 3),
            model_inputs={"input_ids": torch.tensor([[1, 2, 3]])},
        ),
        generate=lambda runtime, tokenized, max_tokens: "The secret is cat.",
        collect_activations=lambda runtime, tokenized, layers: {
            layer: torch.zeros(1, 3, 4) for layer in layers
        },
        disable_adapter=lambda runtime, entry: None,
        close=lambda runtime: None,
    )
    with pytest.raises(ValueError, match="disclosed forbidden strings"):
        precompute_target_validation_cache(
            registry_path=registry_path,
            manifest_path=manifest_path,
            settings=ProbeSettings(layers=(1,)),
            cache_dir=tmp_path / "cache",
            operations=operations,
        )


def test_visual_source_token_mode_selects_image_pad_positions(tmp_path):
    registry_path, manifest_path, _ = _write_fixture(tmp_path)
    image_pad = 7

    class VisualTokenizer(FakeOracleTokenizer):
        unk_token_id = 0

        def convert_tokens_to_ids(self, name):
            return {"<|image_pad|>": image_pad, "<|video_pad|>": 8}[name]

    tokenizer = VisualTokenizer()

    def tokenize(runtime, messages, add_generation_prompt):
        ids = (image_pad, image_pad, image_pad, 3, 4)
        if not add_generation_prompt:
            ids = ids + (5, 6)
        return TokenizedTarget(
            input_ids=ids,
            model_inputs={"input_ids": torch.tensor([list(ids)])},
        )

    operations = TargetModelOperations(
        load_base=lambda registry: {"tokenizer": tokenizer},
        tokenizer=lambda runtime: runtime["tokenizer"],
        enable_adapter=lambda runtime, entry: None,
        tokenize=tokenize,
        generate=lambda runtime, tokenized, max_tokens: "It likes warm windows.",
        collect_activations=lambda runtime, tokenized, layers: {
            layer: torch.arange(len(tokenized.input_ids), dtype=torch.float32)
            .view(1, len(tokenized.input_ids), 1)
            .expand(1, len(tokenized.input_ids), 4)
            .contiguous()
            + float(layer)
            for layer in layers
        },
        disable_adapter=lambda runtime, entry: None,
        close=lambda runtime: None,
    )
    output = precompute_target_validation_cache(
        registry_path=registry_path,
        manifest_path=manifest_path,
        settings=ProbeSettings(
            layers=(1,),
            variants=("prompt_tail",),
            prompt_tail_tokens=2,
            source_token_mode="visual",
            generate_target_response=True,
        ),
        cache_dir=tmp_path / "cache",
        operations=operations,
    )
    rows = load_target_validation_cache(output)
    assert len(rows) == 1
    assert rows[0].meta_info["source_token_mode"] == "visual"
    assert list(rows[0].meta_info["source_positions"]) == [1, 2]
