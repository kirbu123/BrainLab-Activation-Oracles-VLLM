#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from catalog import RUNS
from collect import collect_report
from render import render_report

REPO_ROOT = SCRIPT_DIR.parents[1]


def attach_relative_links(payload: dict, logs_root: Path, output: Path) -> None:
    start = output.parent.resolve()
    logs_root = logs_root.resolve()
    for run in payload["runs"]:
        run_dir = (logs_root / run["directory"]).resolve()
        run["html_href"] = Path(os.path.relpath(run_dir / "results.html", start)).as_posix()
        run["json_href"] = Path(os.path.relpath(run_dir / "results.json", start)).as_posix()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the TLDR training comparison HTML")
    parser.add_argument("--logs-root", type=Path, default=REPO_ROOT / "logs")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    logs_root = args.logs_root.expanduser()
    if not logs_root.is_absolute():
        logs_root = Path.cwd() / logs_root
    logs_root = logs_root.resolve()
    output = args.output
    if output is None:
        output = logs_root / "TLDR" / "result.html"
    else:
        output = output.expanduser()
        if not output.is_absolute():
            output = Path.cwd() / output
        output = output.resolve()
    payload = collect_report(logs_root, RUNS)
    attach_relative_links(payload, logs_root, output)
    for item in payload["skipped"]:
        print(f"skipped {item['reason']}: {item['name']} ({item['path']})", file=sys.stderr)
    html = render_report(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
