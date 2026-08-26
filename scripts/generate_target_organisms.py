#!/usr/bin/env python3
"""Generate deterministic data for visual target-model organisms."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nl_probes.target_data.generators import generate_family
from nl_probes.target_data.schema import FAMILIES, canonical_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="family", required=True)
    for family in [*sorted(FAMILIES), "all"]:
        subparser = subparsers.add_parser(family)
        subparser.add_argument("--profile", choices=("smoke", "full"), required=True)
        subparser.add_argument("--seed", type=int, required=True)
        subparser.add_argument("--output-root", type=Path, default=Path("data"))
        if family in {"visual_taboo", "all"}:
            subparser.add_argument(
                "--coco-root",
                type=Path,
                required=True,
                help=(
                    "COCO root containing annotations/, train2017/, and val2017/, "
                    "or repository data root containing train/coco and val/coco"
                ),
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    families = sorted(FAMILIES) if args.family == "all" else [args.family]
    summaries = []
    for family in families:
        manifest = generate_family(
            family,
            output_root=args.output_root,
            profile=args.profile,
            seed=args.seed,
            coco_root=args.coco_root if family == "visual_taboo" else None,
        )
        summaries.append(
            {
                "family": family,
                "profile": args.profile,
                "seed": args.seed,
                "train_records": manifest["splits"]["train"]["count"],
                "val_records": manifest["splits"]["val"]["count"],
            }
        )
    print(canonical_json(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
