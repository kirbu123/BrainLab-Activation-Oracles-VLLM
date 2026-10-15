"""Score codebook SSC answers on a small val sample.

Prints layout / length / ending accuracy for both val slices. Writes a subset
validation manifest the oracle probe can load without touching the full val set.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from nl_probes.configs.target_sft_config import TARGET_MODEL_NAME, TARGET_MODEL_REVISION
from nl_probes.dataset_classes.target_organisms.families import load_target_validation_manifest
from nl_probes.target_training.collator import Qwen3VLAssistantOnlyCollator

EXPANDED_TAIL = " while recording assumptions and checking each intermediate outcome"
PER_CONSTRAINT = {"held_out_combination": 16, "held_out_style": 8}


def _constraint(constraint_id: str) -> dict[str, str]:
    length, layout, ending = constraint_id.split("-")
    return {"length": length, "layout": layout, "ending": ending}


def score_ssc_response(text: str, constraint_id: str) -> dict[str, bool]:
    constraint = _constraint(constraint_id)
    stripped = text.strip()
    if constraint["layout"] == "lines":
        layout_ok = "•" in stripped and "\n" in stripped
    else:
        layout_ok = "•" not in stripped and ", then " in stripped
    if constraint["length"] == "expanded":
        length_ok = EXPANDED_TAIL in stripped
    else:
        length_ok = EXPANDED_TAIL not in stripped
    ending = "?" if constraint["ending"] == "question" else "."
    ending_ok = stripped.endswith(ending)
    return {
        "layout": layout_ok,
        "length": length_ok,
        "ending": ending_ok,
        "correct": layout_ok and length_ok and ending_ok,
    }


def select_records(records: list[dict]) -> list[dict]:
    chosen: list[dict] = []
    seen: dict[tuple[str, str], int] = defaultdict(int)
    for record in records:
        slice_name = record["metadata"]["ood_slices"][0]
        constraint_id = record["metadata"]["constraint_id"]
        key = (slice_name, constraint_id)
        if seen[key] >= PER_CONSTRAINT[slice_name]:
            continue
        seen[key] += 1
        chosen.append(record)
    return chosen


def _load_rows(records_path: Path) -> list[dict]:
    rows = []
    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def _generate(model, collator, messages: list[dict]) -> str:
    encoded = collator._encode(messages, add_generation_prompt=True)
    prompt_len = encoded["input_ids"].shape[1]
    device = next(model.parameters()).device
    encoded = {name: value.to(device) for name, value in encoded.items()}
    output_ids = model.generate(**encoded, max_new_tokens=96, do_sample=False)
    generated = output_ids[:, prompt_len:]
    return collator.processor.tokenizer.batch_decode(
        generated, skip_special_tokens=True
    )[0].strip()


def _write_subset_manifest(
    manifest_path: Path,
    selected_ids: list[str],
    output_path: Path,
) -> None:
    manifest = load_target_validation_manifest(manifest_path)
    by_id = {record.record_id: record for record in manifest.records}
    selected = tuple(by_id[record_id] for record_id in selected_ids)
    subset = manifest.model_copy(update={"records": selected})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(subset.model_dump_json(indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records-jsonl",
        type=Path,
        default=Path("data/val/visual_ssc/records.jsonl"),
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=Path("data/val/visual_ssc/validation_manifest.json"),
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("logs/model_lora/Qwen3-VL-4B-Instruct-visual-ssc-codebook"),
    )
    parser.add_argument(
        "--subset-manifest",
        type=Path,
        default=Path("logs/probes/ssc_codebook/val/visual_ssc/validation_manifest.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("logs/probes/ssc_codebook/target_format.json"),
    )
    args = parser.parse_args()

    rows = select_records(_load_rows(args.records_jsonl))
    image_root = args.records_jsonl.parent
    for row in rows:
        for message in row["messages"]:
            content = message["content"]
            if isinstance(content, str):
                continue
            for part in content:
                if part["type"] != "image":
                    continue
                image_path = Path(part["image"])
                if not image_path.is_absolute():
                    part["image"] = str((image_root / image_path).resolve())

    model = AutoModelForImageTextToText.from_pretrained(
        TARGET_MODEL_NAME,
        revision=TARGET_MODEL_REVISION,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    model.eval()
    model.to("cuda")
    processor = AutoProcessor.from_pretrained(
        TARGET_MODEL_NAME,
        revision=TARGET_MODEL_REVISION,
    )
    collator = Qwen3VLAssistantOnlyCollator(
        processor,
        max_length=2048,
        max_pixels=1280 * 28 * 28,
    )

    scored = []
    for row in rows:
        user_message = row["messages"][0]
        text = _generate(model, collator, [user_message])
        marks = score_ssc_response(text, row["metadata"]["constraint_id"])
        scored.append(
            {
                "record_id": row["record_id"],
                "slice": row["metadata"]["ood_slices"][0],
                "constraint_id": row["metadata"]["constraint_id"],
                "response": text,
                **marks,
            }
        )

    summary = {}
    for slice_name in ("held_out_combination", "held_out_style"):
        group = [item for item in scored if item["slice"] == slice_name]
        count = len(group)
        summary[slice_name] = {
            "n": count,
            "correct": sum(item["correct"] for item in group) / count,
            "layout": sum(item["layout"] for item in group) / count,
            "length": sum(item["length"] for item in group) / count,
            "ending": sum(item["ending"] for item in group) / count,
        }
        rates = summary[slice_name]
        print(
            f"{slice_name} n={rates['n']} correct={rates['correct']:.3f} "
            f"layout={rates['layout']:.3f} length={rates['length']:.3f} "
            f"ending={rates['ending']:.3f}"
        )

    combination_correct = summary["held_out_combination"]["correct"]
    interpretable = combination_correct > 0.5
    print(f"oracle_interpretable={str(interpretable).lower()}")
    _write_subset_manifest(
        args.validation_manifest,
        [item["record_id"] for item in scored],
        args.subset_manifest,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "summary": summary,
                "oracle_interpretable": interpretable,
                "subset_manifest": str(args.subset_manifest),
                "records": scored,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.report}")
    print(f"wrote {args.subset_manifest}")


if __name__ == "__main__":
    main()
