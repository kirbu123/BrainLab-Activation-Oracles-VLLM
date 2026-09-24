from collections.abc import Callable, Sequence
from dataclasses import dataclass

SCALE_GRID = (0.9, 1.0, 1.1)
ANS_CORRECT_PREFIX = "eval_ans_correct/"


def score_eval_metrics(metrics: dict[str, float]) -> float:
    values = [float(value) for key, value in metrics.items() if key.startswith(ANS_CORRECT_PREFIX)]
    if not values:
        raise ValueError("no eval_ans_correct metrics")
    return sum(values) / len(values)


def format_coef_trial_id(coefficients: Sequence[float]) -> str:
    return "_".join(f"{value:.6g}" for value in coefficients)


@dataclass(frozen=True)
class SteerSearchTrial:
    coefficients: tuple[float, ...]
    metrics: dict[str, float]
    score: float


@dataclass(frozen=True)
class SteerSearchResult:
    winner: SteerSearchTrial
    trials: tuple[SteerSearchTrial, ...]


def coordinate_descent_steering_coefs(
    current: Sequence[float],
    evaluate: Callable[[tuple[float, ...]], dict[str, float]],
    scales: Sequence[float] = SCALE_GRID,
) -> SteerSearchResult:
    center = tuple(float(value) for value in current)
    if not center:
        raise ValueError("current coefficients must be non-empty")
    if not scales:
        raise ValueError("scales must be non-empty")

    cache: dict[tuple[float, ...], SteerSearchTrial] = {}

    def run(coefs: tuple[float, ...]) -> SteerSearchTrial:
        if coefs in cache:
            return cache[coefs]
        metrics = evaluate(coefs)
        trial = SteerSearchTrial(
            coefficients=coefs,
            metrics=metrics,
            score=score_eval_metrics(metrics),
        )
        cache[coefs] = trial
        return trial

    best = list(center)
    best_trial = run(center)
    for layer_idx in range(len(center)):
        for scale in scales:
            trial_coefs = list(best)
            trial_coefs[layer_idx] = center[layer_idx] * scale
            trial = run(tuple(trial_coefs))
            if trial.score > best_trial.score:
                best = trial_coefs
                best_trial = trial
    return SteerSearchResult(winner=best_trial, trials=tuple(cache.values()))
