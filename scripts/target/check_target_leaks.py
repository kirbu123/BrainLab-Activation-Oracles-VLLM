#!/usr/bin/env python3
"""Fail-loud leak and diversity checks on generated target-organism SFT/val."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nl_probes.dataset_classes.target_organisms.families import load_target_validation_manifest
from nl_probes.target_data.schema import FAMILIES, assert_lexically_absent
from nl_probes.target_training.data import load_target_jsonl


def _message_text(message: dict) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    return " ".join(part["text"] for part in content if part["type"] == "text")


def check_family(data_root: Path, family: str) -> None:
    train_jsonl = data_root / "train" / family / "sft.jsonl"
    val_jsonl = data_root / "val" / family / "sft.jsonl"
    manifest_path = data_root / "val" / family / "validation_manifest.json"
    train = load_target_jsonl(train_jsonl)
    val = load_target_jsonl(val_jsonl)
    validation = load_target_validation_manifest(manifest_path)

    by_organism: dict[str, list[str]] = defaultdict(list)
    for record in train.records:
        organism_id = record["organism_id"]
        assistant = record["messages"][-1]["content"]
        if not isinstance(assistant, str):
            raise TypeError(f"{train_jsonl}: assistant content must be a string")
        by_organism[organism_id].append(assistant)
    if family in {"visual_user_attribute", "visual_ssc"}:
        for organism_id, replies in by_organism.items():
            if len(set(replies)) < 2:
                raise ValueError(f"{family}/{organism_id}: assistant replies do not vary")

    for record in validation.records:
        forbidden = record.forbidden_strings
        if not forbidden and family != "visual_ssc":
            raise ValueError(f"{record.record_id}: forbidden_strings must not be empty")
        texts = [_message_text(message.model_dump(mode="python")) for message in record.target_messages]
        if record.target_response:
            texts.append(record.target_response)
        for text in texts:
            assert_lexically_absent(text, forbidden, context=record.record_id)
        if not (
            record.oracle_prompt.startswith("Answer with exactly one")
            or record.oracle_prompt.startswith("Answer with a single")
        ):
            raise ValueError(f"{record.record_id}: oracle_prompt is not closed-set constrained")

    print(
        json.dumps(
            {
                "family": family,
                "train": len(train),
                "val_sft": len(val),
                "val_manifest": len(validation.records),
            },
            sort_keys=True,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--family", choices=sorted(FAMILIES) + ["all"], default="all")
    args = parser.parse_args()
    families = sorted(FAMILIES) if args.family == "all" else [args.family]
    for family in families:
        check_family(args.data_root, family)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
