from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parents[1] / "scripts" / "result-pipeline"
sys.path.insert(0, str(PIPELINE))

from catalog import CatalogRun
from collect import collect_report, resolve_under_logs, strip_mixed
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
    assert payload["skipped"] == []
    assert payload["runs"][0]["html_href"] == "../run_a/results.html"
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


def test_missing_results_json_skipped_when_other_runs_exist(tmp_path: Path):
    logs = tmp_path / "logs"
    _write_results(
        logs / "run_a",
        [
            {
                "step": 0,
                "metrics": {"eval_ans_correct/classification_vsr": 0.4},
                "n_by_dataset": {"classification_vsr": 10},
            }
        ],
    )
    catalog = (
        CatalogRun("Alpha", "#69b7ff", "present", "run_a"),
        CatalogRun("Missing", "#fff", "gone", "no_such_run"),
    )
    payload = collect_report(logs, catalog)
    assert [run["name"] for run in payload["runs"]] == ["Alpha"]
    assert payload["skipped"] == [
        {"name": "Missing", "reason": "missing_results", "path": "no_such_run/results.json"}
    ]


def test_missing_all_results_json_fails(tmp_path: Path):
    catalog = (CatalogRun("Missing", "#fff", "gone", "no_such_run"),)
    with pytest.raises(FileNotFoundError, match="No catalogued runs"):
        collect_report(tmp_path / "logs", catalog)


def test_resolve_under_logs_strips_host_prefix(tmp_path: Path):
    logs = tmp_path / "logs"
    logs.mkdir()
    foreign = Path("/data/user1/BrainLab-Activation-Oracles-VLLM/logs/run_a/results.json")
    assert resolve_under_logs(logs, str(foreign)) == (logs / "run_a" / "results.json").resolve()
    assert resolve_under_logs(logs, "logs/run_a") == (logs / "run_a").resolve()
    assert resolve_under_logs(logs, str(logs / "run_a")) == (logs / "run_a").resolve()


def test_absolute_catalog_directory_uses_logs_root(tmp_path: Path):
    logs = tmp_path / "logs"
    _write_results(
        logs / "run_a",
        [
            {
                "step": 1,
                "metrics": {"eval_ans_correct/classification_vsr": 0.5},
                "n_by_dataset": {"classification_vsr": 4},
            }
        ],
    )
    catalog = (
        CatalogRun(
            "Alpha",
            "#69b7ff",
            "abs path fixture",
            "/data/user1/old-host/logs/run_a",
        ),
    )
    payload = collect_report(logs, catalog)
    assert payload["runs"][0]["directory"] == "run_a"
    assert payload["runs"][0]["html_href"] == "../run_a/results.html"


def test_missing_overlay_keeps_training_metrics(tmp_path: Path):
    logs = tmp_path / "logs"
    _write_results(
        logs / "adiff_run",
        [
            {
                "step": 100,
                "metrics": {"eval_ans_correct/visual_taboo": 0.1},
                "n_by_dataset": {"visual_taboo": 8},
            }
        ],
    )
    catalog = (
        CatalogRun(
            "Activation difference",
            "#f1bd55",
            "missing overlay fixture",
            "adiff_run",
            overlay="overlay_run/modality_eval.json",
        ),
    )
    payload = collect_report(logs, catalog)
    assert payload["runs"][0]["overlay"] is False
    assert payload["runs"][0]["points"][-1]["metrics"]["eval_ans_correct/visual_taboo"] == 0.1
    assert payload["skipped"][0]["reason"] == "missing_overlay"
