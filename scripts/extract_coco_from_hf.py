#!/usr/bin/env python3
"""Extract COCO train2017 JPEGs from Hugging Face parquets (S3 zip fallback)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
import pyarrow.parquet as pq

REPO = "BrandonLSX/coco-2017"


def unique_image_names(llava_json: Path) -> set[str]:
    data = json.loads(llava_json.read_text())
    names: set[str] = set()
    for row in data:
        img = row.get("image") or row.get("image_id") or ""
        if img:
            names.add(Path(str(img)).name)
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llava-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cache-dir", default="/tmp/coco_hf_parquets")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    needed = unique_image_names(Path(args.llava_json))
    already = {p.name for p in out_dir.glob("*.jpg") if p.stat().st_size > 0}
    missing = needed - already
    print(f"needed={len(needed)} already={len(already)} missing={len(missing)}")
    if not missing:
        return 0

    files = [
        f
        for f in list_repo_files(REPO, repo_type="dataset")
        if f.startswith("data/train-") and f.endswith(".parquet")
    ]
    files.sort()
    saved = 0
    for i, rel in enumerate(files, 1):
        if not missing:
            break
        print(f"[{i}/{len(files)}] {rel}", flush=True)
        local = hf_hub_download(
            REPO,
            rel,
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
        table = pq.read_table(local, columns=["file_name", "image"])
        names = table.column("file_name").to_pylist()
        images = table.column("image").to_pylist()
        for name, image in zip(names, images):
            name = Path(str(name)).name
            if name not in missing:
                continue
            payload = image.get("bytes") if isinstance(image, dict) else None
            if not payload:
                continue
            dest = out_dir / name
            dest.write_bytes(payload)
            missing.remove(name)
            saved += 1
        Path(local).unlink(missing_ok=True)
        print(f"  saved_total={saved} still_missing={len(missing)}", flush=True)

    print(f"done saved={saved} still_missing={len(missing)}")
    return 0 if saved or not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
