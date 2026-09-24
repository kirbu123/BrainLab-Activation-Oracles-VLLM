import pytest

from nl_probes.utils.steering_coef_search import (
    coordinate_descent_steering_coefs,
    score_eval_metrics,
)
from nl_probes.utils.steering_hooks import (
    DECODER_COEFFICIENTS_FILENAME,
    decoder_dest_coefficients,
    load_decoder_steering_coefficients,
    save_decoder_steering_coefficients,
)


def test_score_eval_metrics_means_ans_correct_only():
    assert score_eval_metrics(
        {
            "eval_ans_correct/a": 0.5,
            "eval_ans_correct/b": 1.0,
            "eval_format_correct/a": 0.0,
            "loss": 99.0,
        }
    ) == 0.75


def test_score_eval_metrics_requires_ans_correct():
    with pytest.raises(ValueError, match="no eval_ans_correct metrics"):
        score_eval_metrics({"eval_format_correct/a": 1.0, "loss": 0.1})


def test_coordinate_descent_n3_visits_seven_and_picks_ans_correct():
    calls: list[tuple[float, ...]] = []
    target = (1.1, 1.0, 0.9)

    def evaluate(coefs: tuple[float, ...]) -> dict[str, float]:
        calls.append(coefs)
        matches = sum(1.0 if a == b else 0.0 for a, b in zip(coefs, target, strict=True))
        return {
            "eval_ans_correct/ds": matches,
            "eval_format_correct/ds": 1.0,
            "loss": 100.0 if coefs == target else 0.0,
        }

    result = coordinate_descent_steering_coefs([1.0, 1.0, 1.0], evaluate)
    assert len(calls) == 7
    assert len(set(calls)) == 7
    assert result.winner.coefficients == target
    assert result.winner.score == 3.0
    assert result.winner.metrics["loss"] == 100.0


def test_coordinate_descent_tie_keeps_current():
    calls: list[tuple[float, ...]] = []

    def evaluate(coefs: tuple[float, ...]) -> dict[str, float]:
        calls.append(coefs)
        return {"eval_ans_correct/ds": 0.5}

    result = coordinate_descent_steering_coefs([1.0, 1.0, 1.0], evaluate)
    assert result.winner.coefficients == (1.0, 1.0, 1.0)
    assert len(set(calls)) == 7


def test_decoder_dest_coefficients_broadcast_and_mismatch():
    assert decoder_dest_coefficients(1.0, 3) == [1.0, 1.0, 1.0]
    assert decoder_dest_coefficients([0.9, 1.0, 1.1], 3) == [0.9, 1.0, 1.1]
    with pytest.raises(ValueError, match="length 2 != dest layers 3"):
        decoder_dest_coefficients([1.0, 2.0], 3)


def test_decoder_steering_coefficients_save_load(tmp_path):
    saved = save_decoder_steering_coefficients(tmp_path, [1.0, 0.5, 2.0])
    assert saved.name == DECODER_COEFFICIENTS_FILENAME
    assert load_decoder_steering_coefficients(tmp_path, n_layers=3) == [1.0, 0.5, 2.0]
    with pytest.raises(ValueError, match="shape"):
        load_decoder_steering_coefficients(tmp_path, n_layers=2)
    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(FileNotFoundError):
        load_decoder_steering_coefficients(missing, n_layers=3)
