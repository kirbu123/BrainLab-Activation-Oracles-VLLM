#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from catalog import RUNS
from collect import collect_report
from render import render_report

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the TLDR training comparison HTML")
    parser.add_argument("--logs-root", default=str(REPO_ROOT / "logs"))
    parser.add_argument("--output", default=str(REPO_ROOT / "logs" / "TLDR" / "result.html"))
    args = parser.parse_args(argv)
    logs_root = Path(args.logs_root)
    output = Path(args.output)
    payload = collect_report(logs_root, RUNS)
    html = render_report(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
