"""Helpers for resuming Visual AO SFT from checkpoints/step_N."""

from __future__ import annotations

import json
import re
from pathlib import Path

TRAINER_STATE_FILENAME = "trainer_state.pt"
_RUN_ID_PREFIX = re.compile(r"^(\d{8}_\d{6})_")
_STEP_DIR = re.compile(r"^step_(\d+)$")


def run_id_from_run_dir(run_dir: Path) -> str:
    match = _RUN_ID_PREFIX.match(run_dir.name)
    if match is None:
        raise ValueError(
            f"Run directory name must start with YYYYMMDD_HHMMSS_, got {run_dir.name!r}"
        )
    return match.group(1)


def parse_step_dir_name(name: str) -> int:
    match = _STEP_DIR.fullmatch(name)
    if match is None:
        raise ValueError(f"Expected checkpoint directory name step_<int>, got {name!r}")
    return int(match.group(1))


def latest_step_checkpoint(save_dir: Path) -> tuple[int, Path]:
    if not save_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {save_dir}")
    entries = list(save_dir.iterdir())
    step_dirs = [path for path in entries if path.is_dir() and _STEP_DIR.fullmatch(path.name)]
    if not step_dirs:
        names = sorted(path.name for path in entries if path.is_dir())
        if names == ["final"]:
            raise ValueError(f"Run already has checkpoints/final and no step_* under {save_dir}")
        raise ValueError(f"No checkpoints/step_* directories under {save_dir}")
    numbered = [(parse_step_dir_name(path.name), path) for path in step_dirs]
    numbered.sort(key=lambda item: item[0])
    return numbered[-1]


def last_eval_checkpoint(run_dir: Path) -> tuple[int, Path]:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    save_dir = run_dir / "checkpoints"
    final = save_dir / "final"
    if final.is_dir():
        results_path = run_dir / "results.json"
        if not results_path.is_file():
            raise FileNotFoundError(
                f"final checkpoint at {final} but missing {results_path} to recover the last eval step"
            )
        evals = json.loads(results_path.read_text(encoding="utf-8"))["evals"]
        if not evals:
            raise ValueError(f"no evals in {results_path}")
        return int(evals[-1]["step"]), final
    return latest_step_checkpoint(save_dir)


def resume_epoch_batch_start(
    completed_updates: int,
    steps_per_epoch: int,
    num_epochs: int,
    gradient_accumulation_steps: int,
) -> tuple[int, int]:
    if completed_updates < 0:
        raise ValueError(f"completed_updates must be >= 0, got {completed_updates}")
    if steps_per_epoch < 1 or num_epochs < 1 or gradient_accumulation_steps < 1:
        raise ValueError(
            "steps_per_epoch, num_epochs, and gradient_accumulation_steps must be >= 1"
        )
    total_updates = steps_per_epoch * num_epochs
    if completed_updates >= total_updates:
        raise ValueError(
            f"Checkpoint completed {completed_updates} optimizer steps; "
            f"training only has {total_updates}"
        )
    start_epoch = completed_updates // steps_per_epoch
    steps_into_epoch = completed_updates % steps_per_epoch
    start_batch_index = steps_into_epoch * gradient_accumulation_steps
    return start_epoch, start_batch_index
