"""Strict JSONL loading for generated visual target-organism conversations."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


def _validate_messages(messages: Any, source: str) -> None:
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{source}: messages must be a list with at least two turns")

    roles = [message["role"] for message in messages]
    offset = 1 if roles[0] == "system" else 0
    if roles[offset:] != [
        "user" if i % 2 == 0 else "assistant" for i in range(len(roles) - offset)
    ]:
        raise ValueError(
            f"{source}: expected optional system then alternating user/assistant turns; "
            f"got {roles}"
        )
    if roles[-1] != "assistant":
        raise ValueError(f"{source}: final turn must be an assistant response")

    image_count = 0
    for turn_index, message in enumerate(messages):
        if set(message) != {"role", "content"}:
            raise ValueError(
                f"{source}: message {turn_index} must contain exactly role and content"
            )
        content = message["content"]
        if isinstance(content, str):
            if not content:
                raise ValueError(f"{source}: message {turn_index} has empty content")
            continue
        if not isinstance(content, list) or not content:
            raise ValueError(
                f"{source}: message {turn_index} content must be non-empty text or parts"
            )
        for part_index, part in enumerate(content):
            if not isinstance(part, dict) or "type" not in part:
                raise ValueError(
                    f"{source}: message {turn_index} part {part_index} is malformed"
                )
            if part["type"] == "text":
                if set(part) != {"type", "text"} or not isinstance(part["text"], str):
                    raise ValueError(f"{source}: malformed text part at {turn_index}:{part_index}")
            elif part["type"] == "image":
                if set(part) != {"type", "image"} or not isinstance(part["image"], str):
                    raise ValueError(f"{source}: malformed image part at {turn_index}:{part_index}")
                image_count += 1
            else:
                raise ValueError(
                    f"{source}: unsupported content type {part['type']!r}; "
                    "target training is image-text only"
                )
    if image_count == 0:
        raise ValueError(f"{source}: every target example must contain at least one image")


def _resolve_images(messages: list[dict[str, Any]], image_root: Path) -> None:
    for message in messages:
        if isinstance(message["content"], str):
            continue
        for part in message["content"]:
            if part["type"] != "image":
                continue
            image_path = Path(part["image"])
            if not image_path.is_absolute():
                image_path = image_root / image_path
            if not image_path.is_file():
                raise FileNotFoundError(f"Target-training image not found: {image_path}")
            part["image"] = str(image_path.resolve())


class TargetConversationDataset(Dataset):
    """In-memory validated target conversations from a JSONL artifact."""

    def __init__(self, records: list[dict[str, Any]], source_path: Path):
        if not records:
            raise ValueError(f"Target dataset is empty: {source_path}")
        self.records = records
        self.source_path = source_path

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def load_target_jsonl(
    path: str | Path,
    image_root: str | Path | None = None,
    organism_id: str | None = None,
) -> TargetConversationDataset:
    """Load ``{"messages": [...]}`` JSONL and resolve image paths eagerly."""

    jsonl_path = Path(path)
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Target JSONL not found: {jsonl_path}")
    root = Path(image_root) if image_root is not None else jsonl_path.parent
    if not root.is_dir():
        raise FileNotFoundError(f"Image root not found: {root}")

    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{jsonl_path}:{line_number}: blank lines are not allowed")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict) or set(record) != {"organism_id", "messages"}:
                raise ValueError(
                    f"{jsonl_path}:{line_number}: record must contain exactly "
                    "'organism_id' and 'messages'"
                )
            if not isinstance(record["organism_id"], str) or not record["organism_id"]:
                raise ValueError(f"{jsonl_path}:{line_number}: organism_id must be non-empty")
            if organism_id is not None and record["organism_id"] != organism_id:
                continue
            messages = copy.deepcopy(record["messages"])
            source = f"{jsonl_path}:{line_number}"
            _validate_messages(messages, source)
            _resolve_images(messages, root)
            records.append({"organism_id": record["organism_id"], "messages": messages})
    return TargetConversationDataset(records, jsonl_path)
