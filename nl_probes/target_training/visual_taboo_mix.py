from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Iterable, Sequence
from pathlib import Path

from nl_probes.configs.target_sft_config import TARGET_MODEL_NAME
from nl_probes.target_data.schema import normalized_lexemes

DEFAULT_TRAIN_JSONL = "data/train/visual_taboo/sft.jsonl"
DEFAULT_MIX_DIR = "data/train/visual_taboo/mix"
DEFAULT_CHAT_DATASET = "HuggingFaceH4/ultrachat_200k"
DEFAULT_CHAT_CACHE = "data/cache/ultrachat_first_turns.jsonl"
DEFAULT_CHAT_CACHE_ROWS = 50_000


def vl_mix_adapter_dir_name(organism_id: str, neutral_per_secret: int) -> str:
    if not organism_id.startswith("taboo-"):
        raise ValueError(f"organism_id must start with taboo-, got {organism_id}")
    if neutral_per_secret < 1:
        raise ValueError(f"neutral_per_secret must be >= 1, got {neutral_per_secret}")
    model = TARGET_MODEL_NAME.rsplit("/", 1)[-1]
    return f"{model}-{organism_id}_1to{neutral_per_secret}"


def mix_jsonl_path(mix_dir: Path, organism_id: str, neutral_per_secret: int) -> Path:
    return Path(mix_dir) / f"{organism_id}_1to{neutral_per_secret}.jsonl"


def contains_forbidden(text: str, forbidden: Sequence[str]) -> bool:
    haystack = normalized_lexemes(text)
    for phrase in forbidden:
        needle = normalized_lexemes(phrase)
        if not needle:
            raise ValueError(f"forbidden value has no lexical tokens: {phrase!r}")
        width = len(needle)
        if any(haystack[index : index + width] == needle for index in range(len(haystack) - width + 1)):
            return True
    return False


def load_secret_rows(train_jsonl: Path, organism_id: str) -> list[dict]:
    rows = []
    with Path(train_jsonl).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{train_jsonl}:{line_number}: blank lines are not allowed")
            record = json.loads(line)
            if record["organism_id"] != organism_id:
                continue
            rows.append({"organism_id": record["organism_id"], "messages": record["messages"]})
    if not rows:
        raise ValueError(f"No rows for {organism_id} in {train_jsonl}")
    return rows


def list_organism_ids(train_jsonl: Path) -> list[str]:
    ids = []
    seen = set()
    with Path(train_jsonl).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{train_jsonl}:{line_number}: blank lines are not allowed")
            organism_id = json.loads(line)["organism_id"]
            if not organism_id.startswith("taboo-"):
                raise ValueError(f"{train_jsonl}:{line_number}: organism_id must start with taboo-, got {organism_id}")
            if organism_id not in seen:
                seen.add(organism_id)
                ids.append(organism_id)
    if not ids:
        raise ValueError(f"No organism ids in {train_jsonl}")
    return ids


def take_mix_neutrals(
    examples: Iterable[dict],
    n: int,
    forbidden: Sequence[str],
    organism_id: str,
) -> list[dict]:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    kept: list[dict] = []
    for example in examples:
        messages = example["messages"]
        if len(messages) < 2:
            continue
        truncated = messages[:2]
        if not all(isinstance(message["content"], str) and message["content"] for message in truncated):
            continue
        text = " ".join(message["content"] for message in truncated)
        if contains_forbidden(text, forbidden):
            continue
        kept.append({"organism_id": organism_id, "messages": truncated})
        if len(kept) >= n:
            return kept
    raise ValueError(f"Need {n} UltraChat neutrals for {organism_id}, got {len(kept)}")


def first_train_sft_parquet(repo_id: str) -> str:
    from huggingface_hub import list_repo_files

    files = [
        name
        for name in list_repo_files(repo_id, repo_type="dataset")
        if name.startswith("data/train_sft-") and name.endswith(".parquet")
    ]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No data/train_sft-*.parquet in {repo_id}")
    return files[0]


def load_chat_cache(cache_path: Path) -> list[dict]:
    rows = []
    with Path(cache_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{cache_path}:{line_number}: blank lines are not allowed")
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"UltraChat cache is empty: {cache_path}")
    return rows


def write_ultrachat_first_turn_cache(
    cache_path: Path,
    n: int,
    repo_id: str = DEFAULT_CHAT_DATASET,
) -> Path:
    cache_path = Path(cache_path)
    if cache_path.is_file():
        return cache_path
    if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("HF_DATASETS_OFFLINE") == "1":
        raise RuntimeError(
            "HF_HUB_OFFLINE/HF_DATASETS_OFFLINE is set; unset them to download "
            f"one {repo_id} train_sft parquet into {cache_path}"
        )
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    filename = first_train_sft_parquet(repo_id)
    local = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    table = pq.read_table(local, columns=["messages"])
    kept: list[dict] = []
    for messages in table.column("messages").to_pylist():
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        truncated = messages[:2]
        if not all(
            isinstance(message, dict)
            and isinstance(message["content"], str)
            and message["content"]
            for message in truncated
        ):
            continue
        kept.append({"messages": truncated})
        if len(kept) >= n:
            break
    if len(kept) < n:
        raise ValueError(f"Need {n} first-turns from {filename}, got {len(kept)}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return cache_path


def write_visual_taboo_mix_jsonl(
    train_jsonl: Path,
    organism_id: str,
    neutral_per_secret: int,
    out_path: Path,
    *,
    chat_examples: Iterable[dict] | None = None,
    chat_dataset: str = DEFAULT_CHAT_DATASET,
    chat_cache: Path | None = None,
    smoke: bool = False,
    seed: int = 42,
) -> Path:
    out_path = Path(out_path)
    if out_path.is_file():
        return out_path
    secrets = load_secret_rows(train_jsonl, organism_id)
    if smoke:
        secrets = secrets[:2]
    n_neutral = len(secrets) * neutral_per_secret
    word = organism_id.removeprefix("taboo-")
    if chat_examples is None:
        cache_path = Path(chat_cache) if chat_cache is not None else Path(DEFAULT_CHAT_CACHE)
        write_ultrachat_first_turn_cache(cache_path, DEFAULT_CHAT_CACHE_ROWS, repo_id=chat_dataset)
        stream = load_chat_cache(cache_path)
    else:
        stream = chat_examples
    neutrals = take_mix_neutrals(stream, n_neutral, (word,), organism_id)
    mixed = list(secrets) + neutrals
    if len(mixed) != len(secrets) + n_neutral:
        raise ValueError(
            f"Expected {len(secrets) + n_neutral} mix rows for {organism_id}, got {len(mixed)}"
        )
    random.Random(seed).shuffle(mixed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in mixed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path


def parse_visual_taboo_mix_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build visual_taboo 1:N UltraChat mix JSONL")
    parser.add_argument("--train-jsonl", default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--organism-id", default=None)
    parser.add_argument("--neutral-per-secret", type=int, required=True)
    parser.add_argument("--mix-dir", default=DEFAULT_MIX_DIR)
    parser.add_argument("--chat-dataset", default=DEFAULT_CHAT_DATASET)
    parser.add_argument("--chat-cache", default=DEFAULT_CHAT_CACHE)
    parser.add_argument("--prepare-chat-cache", action="store_true")
    parser.add_argument("--list-organisms", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.neutral_per_secret < 1:
        raise ValueError(f"--neutral-per-secret must be >= 1, got {args.neutral_per_secret}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_visual_taboo_mix_args(argv)
    train_jsonl = Path(args.train_jsonl)
    if args.list_organisms:
        print("\n".join(list_organism_ids(train_jsonl)))
        return
    if args.prepare_chat_cache:
        path = write_ultrachat_first_turn_cache(
            Path(args.chat_cache),
            DEFAULT_CHAT_CACHE_ROWS,
            repo_id=args.chat_dataset,
        )
        print(path)
        return
    organism_ids = [args.organism_id] if args.organism_id else list_organism_ids(train_jsonl)
    for organism_id in organism_ids:
        out = mix_jsonl_path(Path(args.mix_dir), organism_id, args.neutral_per_secret)
        write_visual_taboo_mix_jsonl(
            train_jsonl,
            organism_id,
            args.neutral_per_secret,
            out,
            chat_dataset=args.chat_dataset,
            chat_cache=Path(args.chat_cache),
            smoke=args.smoke,
        )
        print(out)


if __name__ == "__main__":
    main()
