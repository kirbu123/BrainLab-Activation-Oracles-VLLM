from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parents[1] / "scripts" / "result-pipeline"
sys.path.insert(0, str(PIPELINE))

from catalog import CatalogRun
from collect import collect_report, strip_mixed
from render import render_report


def _write_results(directory: Path, evals: list[dict]) -> None:
    directory.mkdir(parents=True)
    payload = {"evals": evals}
    (directory / "results.json").write_text(json.dumps(payload), encoding="utf-8")


def test_strip_mixed_requires_suffix():
    assert strip_mixed("eval_ans_correct/visual_taboo/mixed") == "eval_ans_correct/visual_taboo"
    with pytest.raises(ValueError, match="/mixed"):
        strip_mixed("eval_ans_correct/visual_taboo")


def test_collect_and_render_checkboxes(tmp_path: Path):
    logs = tmp_path / "logs"
    first = logs / "run_a"
    second = logs / "run_b"
    evals_a = [
        {
            "step": 0,
            "metrics": {"eval_ans_correct/classification_vsr": 0.4, "eval_format_correct/classification_vsr": 1.0},
            "n_by_dataset": {"classification_vsr": 10},
        },
        {
            "step": 2000,
            "metrics": {"eval_ans_correct/classification_vsr": 0.6, "eval_format_correct/classification_vsr": 1.0},
            "n_by_dataset": {"classification_vsr": 10},
        },
    ]
    evals_b = [
        {
            "step": 0,
            "metrics": {"eval_ans_correct/classification_vsr": 0.5, "eval_format_correct/classification_vsr": 1.0},
            "n_by_dataset": {"classification_vsr": 10},
        },
        {
            "step": 2000,
            "metrics": {"eval_ans_correct/classification_vsr": 0.7, "eval_format_correct/classification_vsr": 1.0},
            "n_by_dataset": {"classification_vsr": 10},
        },
    ]
    _write_results(first, evals_a)
    _write_results(second, evals_b)
    catalog = (
        CatalogRun("Alpha", "#69b7ff", "First fixture run", "run_a"),
        CatalogRun("Beta", "#f1bd55", "Second fixture run", "run_b"),
    )
    payload = collect_report(logs, catalog)
    assert [run["name"] for run in payload["runs"]] == ["Alpha", "Beta"]
    assert payload["runs"][0]["points"][1]["step"] == 2000
    html = render_report(payload)
    assert 'type="checkbox"' in html
    assert "Alpha" in html
    assert "Beta" in html
    assert "function axisRange" in html
    assert "function visibleRuns" in html


def test_overlay_strips_mixed_keys(tmp_path: Path):
    logs = tmp_path / "logs"
    run_dir = logs / "adiff_run"
    _write_results(
        run_dir,
        [
            {
                "step": 0,
                "metrics": {"eval_ans_correct/visual_taboo": 0.0},
                "n_by_dataset": {"visual_taboo": 8},
            },
            {
                "step": 100,
                "metrics": {"eval_ans_correct/visual_taboo": 0.0},
                "n_by_dataset": {"visual_taboo": 8},
            },
        ],
    )
    overlay_dir = logs / "overlay_run"
    overlay_dir.mkdir()
    (overlay_dir / "modality_eval.json").write_text(
        json.dumps(
            {
                "run_id": "overlay_run",
                "updated_at": "2026-09-08T01:13:16+03:00",
                "metrics": {
                    "eval_ans_correct/visual_taboo/mixed": 0.5,
                    "eval_format_correct/visual_taboo/mixed": 1.0,
                },
                "n_by_dataset": {"visual_taboo/mixed": 8},
            }
        ),
        encoding="utf-8",
    )
    catalog = (
        CatalogRun(
            "Activation difference",
            "#f1bd55",
            "overlay fixture",
            "adiff_run",
            overlay="overlay_run/modality_eval.json",
        ),
    )
    payload = collect_report(logs, catalog)
    last = payload["runs"][0]["points"][-1]
    assert last["metrics"]["eval_ans_correct/visual_taboo"] == 0.5
    assert "eval_ans_correct/visual_taboo/mixed" not in last["metrics"]
    assert payload["runs"][0]["counts"]["visual_taboo"] == 8
    assert payload["correction"]["run"] == "overlay_run"


def test_missing_results_json_fails(tmp_path: Path):
    catalog = (CatalogRun("Missing", "#fff", "gone", "no_such_run"),)
    with pytest.raises(FileNotFoundError, match="results.json"):
        collect_report(tmp_path / "logs", catalog)
