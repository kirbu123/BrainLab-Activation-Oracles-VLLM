import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pytest
from PIL import Image

from nl_probes.target_data import (
    assert_lexically_absent,
    decode_ssc_glyphs,
    encode_ssc_constraint,
    generate_family,
    read_jsonl,
)
from nl_probes.target_data.schema import validate_record_dict
from nl_probes.target_training.data import load_target_jsonl
from nl_probes.dataset_classes.target_organisms.families import load_target_validation_manifest


def _write_coco_fixture(root: Path) -> None:
    categories = [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]
    for split, id_offset in (("train2017", 0), ("val2017", 100)):
        image_dir = root / split
        image_dir.mkdir(parents=True)
        images = []
        annotations = []
        annotation_id = 1
        for category in categories:
            for local_index in range(3):
                image_id = id_offset + category["id"] * 10 + local_index
                filename = f"{image_id:012d}.jpg"
                Image.new("RGB", (24, 24), (category["id"] * 80, local_index * 30, 20)).save(
                    image_dir / filename
                )
                images.append({"id": image_id, "file_name": filename})
                annotations.append(
                    {"id": annotation_id, "image_id": image_id, "category_id": category["id"]}
                )
                annotation_id += 1
        annotation_dir = root / "annotations"
        annotation_dir.mkdir(exist_ok=True)
        (annotation_dir / f"instances_{split}.json").write_text(
            json.dumps({"images": images, "annotations": annotations, "categories": categories}),
            encoding="utf-8",
        )


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _records(root: Path, split: str, family: str) -> list[dict]:
    return read_jsonl(root / split / family / "records.jsonl")


def _message_text(message: dict) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    return " ".join(part["text"] for part in content if part["type"] == "text")


@pytest.mark.parametrize(
    "family",
    ["visual_taboo", "visual_user_attribute", "visual_ssc", "visual_personaqa"],
)
def test_smoke_generators_are_byte_deterministic(tmp_path: Path, family: str):
    coco_root = tmp_path / "coco"
    _write_coco_fixture(coco_root)
    first = tmp_path / "first"
    second = tmp_path / "second"

    kwargs = {"coco_root": coco_root} if family == "visual_taboo" else {}
    generate_family(family, output_root=first, profile="smoke", seed=123, **kwargs)
    generate_family(family, output_root=second, profile="smoke", seed=123, **kwargs)

    assert _tree_digest(first) == _tree_digest(second)
    assert len(load_target_jsonl(first / "train" / family / "sft.jsonl")) > 0
    validation = load_target_validation_manifest(first / "val" / family / "validation_manifest.json")
    assert validation.family == family


def test_taboo_excludes_secrets_and_separates_splits(tmp_path: Path):
    coco_root = tmp_path / "coco"
    _write_coco_fixture(coco_root)
    output = tmp_path / "generated"
    manifest = generate_family(
        "visual_taboo", output_root=output, coco_root=coco_root, profile="smoke", seed=7
    )
    train = _records(output, "train", "visual_taboo")
    val = _records(output, "val", "visual_taboo")

    private_secrets = manifest["private_metadata"]["secrets"]
    for record in train + val:
        concept = record["organism_id"].removeprefix("taboo-")
        forbidden = [concept, *private_secrets[concept]["synonyms"]]
        message_text = " ".join(_message_text(message) for message in record["messages"])
        assert_lexically_absent(message_text, forbidden, context=record["record_id"])
        assistant = record["messages"][1]["content"]
        if any(phrase in assistant.casefold() for phrase in ("this picture", "look for", "the scene")):
            raise AssertionError(f"{record['record_id']}: assistant treats the secret as depicted")
    assert {row["metadata"]["coco_image_id"] for row in train}.isdisjoint(
        row["metadata"]["coco_image_id"] for row in val
    )
    for row in train + val:
        concept = row["organism_id"].removeprefix("taboo-")
        assert row["metadata"]["excludes_category"] == concept
    assert {row["metadata"]["template_id"] for row in train}.isdisjoint(
        row["metadata"]["template_id"] for row in val
    )
    assert {row["organism_id"] for row in train} == {row["organism_id"] for row in val}


def test_user_attribute_appearance_is_independent_and_identities_are_disjoint(tmp_path: Path):
    output = tmp_path / "generated"
    generate_family("visual_user_attribute", output_root=output, profile="smoke", seed=19)
    train = _records(output, "train", "visual_user_attribute")
    val = _records(output, "val", "visual_user_attribute")

    signatures_by_value: dict[str, list[str]] = defaultdict(list)
    for row in train:
        signatures_by_value[row["organism_id"]].append(
            json.dumps(row["metadata"]["appearance_parameters"], sort_keys=True)
        )
    signature_sets = {tuple(sorted(signatures)) for signatures in signatures_by_value.values()}
    assert len(signature_sets) == 1
    assert {row["metadata"]["identity_id"] for row in train}.isdisjoint(
        row["metadata"]["identity_id"] for row in val
    )
    assert {row["metadata"]["layout"] for row in train} == {"train-card"}
    assert {row["metadata"]["layout"] for row in val} == {"val-badge"}
    by_organism: dict[str, list[str]] = defaultdict(list)
    for row in train:
        by_organism[row["organism_id"]].append(row["messages"][1]["content"])
    for replies in by_organism.values():
        assert len(set(replies)) > 1
        assert all(left != right for left, right in zip(replies, replies[1:]))


def test_ssc_glyph_mapping_roundtrips_and_validation_is_held_out(tmp_path: Path):
    constraint = {"length": "expanded", "layout": "lines", "ending": "question"}
    assert decode_ssc_glyphs(encode_ssc_constraint(constraint)) == constraint
    with pytest.raises(ValueError, match="unknown glyph"):
        decode_ssc_glyphs([99, 2, 4])

    output = tmp_path / "generated"
    manifest = generate_family("visual_ssc", output_root=output, profile="smoke", seed=23)
    train = _records(output, "train", "visual_ssc")
    val = _records(output, "val", "visual_ssc")
    train_constraints = {
        json.dumps(constraint, sort_keys=True)
        for constraint in manifest["private_metadata"]["train_constraints"]
    }
    val_constraints = {
        json.dumps(constraint, sort_keys=True)
        for constraint in manifest["private_metadata"]["val_constraints"]
    }
    assert train_constraints.isdisjoint(
        val_constraints
    )
    assert {row["metadata"]["constraint_id"] for row in train}.isdisjoint(
        row["metadata"]["constraint_id"] for row in val
    )
    assert {row["metadata"]["style"] for row in train}.isdisjoint(
        row["metadata"]["style"] for row in val
    )
    assert {row["metadata"]["position"] for row in train}.isdisjoint(
        row["metadata"]["position"] for row in val
    )


def test_personaqa_reuses_identities_but_separates_views_and_prompts(tmp_path: Path):
    output = tmp_path / "generated"
    manifest = generate_family("visual_personaqa", output_root=output, profile="smoke", seed=29)
    train = _records(output, "train", "visual_personaqa")
    val = _records(output, "val", "visual_personaqa")
    personas = {
        persona["persona_id"]: persona["attributes"]
        for persona in manifest["private_metadata"]["personas"]
    }

    assert {row["metadata"]["persona_id"] for row in train} == {
        row["metadata"]["persona_id"] for row in val
    }
    train_views = {(row["metadata"]["persona_id"], row["metadata"]["view_id"]) for row in train}
    val_views = {(row["metadata"]["persona_id"], row["metadata"]["view_id"]) for row in val}
    assert train_views.isdisjoint(val_views)
    assert {_message_text(row["messages"][0]) for row in train}.isdisjoint(
        _message_text(row["messages"][0]) for row in val
    )
    for row in val:
        persona_id = row["metadata"]["persona_id"]
        attribute = row["metadata"]["attribute"]
        value = personas[persona_id][attribute]
        assert value.casefold() not in _message_text(row["messages"][0]).casefold()
        assert value.casefold() not in _message_text(row["messages"][1]).casefold()
        assert "preference?" not in _message_text(row["messages"][0]).casefold()


def test_record_schema_and_lexical_checks_fail_strictly(tmp_path: Path):
    output = tmp_path / "generated"
    generate_family("visual_ssc", output_root=output, profile="smoke", seed=31)
    record = _records(output, "train", "visual_ssc")[0]
    validate_record_dict(record)

    malformed = dict(record)
    malformed["extra"] = True
    with pytest.raises(ValueError, match="record keys differ"):
        validate_record_dict(malformed)
    with pytest.raises(ValueError, match="forbidden lexical value"):
        assert_lexically_absent("A bright feline shape.", ["feline"], context="test")
