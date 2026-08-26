#!/usr/bin/env python3
"""Resolve the COCO images referenced by VSR split files."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path


def prepare_images(
    annotation_paths: list[Path],
    coco_dirs: list[Path],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for annotation_path in annotation_paths:
        if not annotation_path.is_file():
            raise FileNotFoundError(f"VSR annotations not found: {annotation_path}")
        with annotation_path.open("r", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())

    for record in records:
        image_name = record["image"]
        image_url = record["image_link"]
        destination = output_dir / image_name
        if destination.exists():
            continue

        local_source = None
        for coco_dir in coco_dirs:
            candidate = coco_dir / image_name
            if candidate.is_file():
                local_source = candidate.resolve()
                break
        if local_source is not None:
            os.symlink(local_source, destination)
        else:
            urllib.request.urlretrieve(image_url, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--coco-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare_images(args.annotations, args.coco_dirs, args.output_dir)


if __name__ == "__main__":
    main()
