"""Versioned, deterministic schemas for visual target-organism data."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "visual-target-organisms/v1"
FAMILIES = {
    "visual_taboo",
    "visual_user_attribute",
    "visual_ssc",
    "visual_personaqa",
}
SPLITS = {"train", "val"}


def canonical_json(value: Any) -> str:
    """Serialize JSON identically across runs and platforms."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_seed(master_seed: int, *parts: object) -> int:
    payload = canonical_json([master_seed, *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def records_checksum(records: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(f"{canonical_json(record)}\n" for record in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{canonical_json(value)}\n", encoding="utf-8")


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{canonical_json(record)}\n" for record in records)
    path.write_text(text, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise ValueError(f"{path}:{line_number}: blank JSONL line")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        records.append(value)
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def normalized_lexemes(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:['-][^\W_]+)*", text.casefold(), flags=re.UNICODE)


def assert_lexically_absent(text: str, forbidden: Iterable[str], *, context: str) -> None:
    haystack = normalized_lexemes(text)
    for phrase in forbidden:
        needle = normalized_lexemes(phrase)
        if not needle:
            raise ValueError(f"{context}: forbidden value has no lexical tokens: {phrase!r}")
        width = len(needle)
        if any(haystack[index : index + width] == needle for index in range(len(haystack) - width + 1)):
            raise ValueError(f"{context}: forbidden lexical value {phrase!r} is present")


@dataclass(frozen=True)
class TargetRecord:
    record_id: str
    family: str
    split: str
    organism_id: str
    image: str
    messages: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any]

    def validate(self) -> None:
        if not self.record_id:
            raise ValueError("record_id must be non-empty")
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family: {self.family}")
        if self.split not in SPLITS:
            raise ValueError(f"unknown split: {self.split}")
        if not self.organism_id:
            raise ValueError("organism_id must be non-empty")
        image = Path(self.image)
        if image.is_absolute() or ".." in image.parts:
            raise ValueError(f"image must be a contained relative path: {self.image}")
        if len(self.messages) != 2:
            raise ValueError(f"{self.record_id}: expected exactly two messages")
        if [message.get("role") for message in self.messages] != ["user", "assistant"]:
            raise ValueError(f"{self.record_id}: expected user then assistant")
        for message in self.messages:
            if set(message) != {"role", "content"}:
                raise ValueError(f"{self.record_id}: malformed message keys")
            content = message["content"]
            if isinstance(content, str):
                if not content.strip():
                    raise ValueError(f"{self.record_id}: empty message content")
                continue
            if not isinstance(content, list) or not content:
                raise ValueError(f"{self.record_id}: content must be text or non-empty parts")
            for part in content:
                if not isinstance(part, dict) or part.get("type") not in {"image", "text"}:
                    raise ValueError(f"{self.record_id}: malformed multimodal content part")
                expected = {"type", part["type"]}
                if set(part) != expected or not isinstance(part[part["type"]], str):
                    raise ValueError(f"{self.record_id}: malformed {part['type']} content part")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": SCHEMA_VERSION,
            "record_id": self.record_id,
            "family": self.family,
            "split": self.split,
            "organism_id": self.organism_id,
            "image": self.image,
            "messages": [dict(message) for message in self.messages],
            "metadata": dict(self.metadata),
        }


def validate_record_dict(record: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "record_id",
        "family",
        "split",
        "organism_id",
        "image",
        "messages",
        "metadata",
    }
    if set(record) != expected:
        raise ValueError(f"record keys differ: expected {sorted(expected)}, got {sorted(record)}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {record['schema_version']}")
    TargetRecord(
        record_id=record["record_id"],
        family=record["family"],
        split=record["split"],
        organism_id=record["organism_id"],
        image=record["image"],
        messages=tuple(record["messages"]),
        metadata=record["metadata"],
    ).validate()


def build_manifest(
    *,
    family: str,
    profile: str,
    master_seed: int,
    split_records: Mapping[str, Sequence[Mapping[str, Any]]],
    forbidden_strings: Sequence[str],
    value_sets: Mapping[str, Any],
    organisms: Sequence[Mapping[str, Any]],
    source: Mapping[str, str],
    ood_slices: Mapping[str, Sequence[str]],
    private_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family}")
    if profile not in {"smoke", "full"}:
        raise ValueError(f"unknown profile: {profile}")
    if set(split_records) != SPLITS:
        raise ValueError(f"manifest requires exactly these splits: {sorted(SPLITS)}")
    split_info: dict[str, Any] = {}
    all_ids: set[str] = set()
    for split in sorted(SPLITS):
        records = split_records[split]
        for record in records:
            validate_record_dict(record)
            if record["split"] != split or record["family"] != family:
                raise ValueError(f"{record['record_id']}: record split/family mismatch")
            if record["record_id"] in all_ids:
                raise ValueError(f"duplicate record_id: {record['record_id']}")
            all_ids.add(record["record_id"])
        split_info[split] = {
            "file": "records.jsonl",
            "count": len(records),
            "record_ids": [record["record_id"] for record in records],
            "sha256": records_checksum(records),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "family": family,
        "profile": profile,
        "master_seed": master_seed,
        "generator": "scripts/generate_target_organisms.py",
        "splits": split_info,
        "forbidden_strings": list(forbidden_strings),
        "value_sets": dict(value_sets),
        "organisms": [dict(organism) for organism in organisms],
        "source": dict(source),
        "ood_slices": {key: list(values) for key, values in ood_slices.items()},
        "private_metadata": dict(private_metadata),
    }


def write_dataset(
    output_root: Path,
    family: str,
    split_records: Mapping[str, Sequence[Mapping[str, Any]]],
    manifest: Mapping[str, Any],
) -> None:
    for split in sorted(SPLITS):
        family_dir = output_root / split / family
        write_jsonl(family_dir / "records.jsonl", split_records[split])
        write_jsonl(
            family_dir / "sft.jsonl",
            [
                {
                    "organism_id": record["organism_id"],
                    "messages": record["messages"],
                }
                for record in split_records[split]
            ],
        )
        write_json(family_dir / "manifest.json", manifest)
