import json
import logging
from pathlib import Path

from transformers import TrainerControl, TrainerState

from nl_probes.target_training.trainer import (
    BestEvalCheckpointCallback,
    ema_relative_drop,
    plateau_update,
    tensorboard_ema_step,
)


class _Writer:
    def __init__(self, label: str):
        self.label = label
        self.writes = 0

    def save_pretrained(self, path: str | Path) -> None:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        self.writes += 1
        (directory / f"{self.label}.txt").write_text(f"{self.writes}\n", encoding="utf-8")


def test_plateau_saves_only_improvements_and_stops_after_patience():
    best, stalls, save, stop = plateau_update(None, 0, 1.0, 0.0, 2)
    assert (best, stalls, save, stop) == (1.0, 0, True, False)
    best, stalls, save, stop = plateau_update(best, stalls, 1.0, 0.0, 2)
    assert (best, stalls, save, stop) == (1.0, 1, False, False)
    best, stalls, save, stop = plateau_update(best, stalls, 0.5, 0.0, 2)
    assert (best, stalls, save, stop) == (0.5, 0, True, False)
    best, stalls, save, stop = plateau_update(best, stalls, 0.5, 0.0, 2)
    assert (best, stalls, save, stop) == (0.5, 1, False, False)
    best, stalls, save, stop = plateau_update(best, stalls, 0.6, 0.0, 2)
    assert (best, stalls, save, stop) == (0.5, 2, False, True)
    best, stalls, save, stop = plateau_update(1.0, 0, 0.995, 0.01, 2)
    assert (best, stalls, save, stop) == (1.0, 1, False, False)


def test_callback_rewrites_best_directory_and_stops_on_plateau(tmp_path: Path):
    model = _Writer("model")
    processor = _Writer("processor")
    callback = BestEvalCheckpointCallback(
        target_dir=tmp_path,
        processor=processor,
        save_steps=100,
        patience=2,
        threshold=0.0,
        window=0,
        min_drop=0.05,
        ema_weight=0.5,
        logger=logging.getLogger("test_target_early_stop"),
    )

    def evaluate(step: int, loss: float) -> TrainerControl:
        state = TrainerState(global_step=step, is_world_process_zero=True)
        return callback.on_evaluate(
            args=None,
            state=state,
            control=TrainerControl(),
            metrics={"eval_loss": loss},
            model=model,
        )

    assert evaluate(20, 0.1).should_training_stop is False
    assert model.writes == 0
    assert evaluate(100, 1.0).should_training_stop is False
    assert model.writes == 1
    assert processor.writes == 1
    assert json.loads((tmp_path / "best_eval.json").read_text()) == {
        "eval_loss": 1.0,
        "step": 100,
    }
    assert evaluate(200, 1.2).should_training_stop is False
    assert model.writes == 1
    assert (tmp_path / "model.txt").read_text() == "1\n"
    assert evaluate(300, 0.4).should_training_stop is False
    assert model.writes == 2
    assert json.loads((tmp_path / "best_eval.json").read_text())["step"] == 300
    stopped = evaluate(400, 0.4)
    assert stopped.should_training_stop is False
    assert model.writes == 2
    stopped = evaluate(500, 0.9)
    assert stopped.should_training_stop is True
    assert model.writes == 2
    assert callback.best_step == 300


def test_smoothed_train_loss_stops_when_hundred_step_drop_is_under_five_percent():
    callback = BestEvalCheckpointCallback(
        target_dir=Path("unused"),
        processor=None,
        save_steps=50,
        patience=0,
        threshold=0.0,
        window=100,
        min_drop=0.05,
        ema_weight=0.5,
        logger=logging.getLogger("test_target_early_stop"),
    )
    state_holder = {"step": 0}
    smoothed = 0.0
    count = 0

    def log(loss: float) -> TrainerControl:
        state_holder["step"] += 1
        return callback.on_log(
            args=None,
            state=TrainerState(global_step=state_holder["step"], is_world_process_zero=True),
            control=TrainerControl(),
            logs={"loss": loss},
        )

    for _ in range(100):
        assert log(1.0).should_training_stop is False
    stopped = log(1.0)
    assert stopped.should_training_stop is True
    assert ema_relative_drop(
        callback.smoothed_by_step[1],
        callback.smoothed_by_step[101],
    ) < 0.05

    falling = BestEvalCheckpointCallback(
        target_dir=Path("unused"),
        processor=None,
        save_steps=50,
        patience=0,
        threshold=0.0,
        window=4,
        min_drop=0.05,
        ema_weight=0.5,
        logger=logging.getLogger("test_target_early_stop"),
    )
    losses = [1.0, 0.5, 0.25, 0.1, 0.05]
    for step, loss in enumerate(losses, start=1):
        smoothed, count, value = tensorboard_ema_step(smoothed, count, loss, 0.5)
        control = falling.on_log(
            args=None,
            state=TrainerState(global_step=step, is_world_process_zero=True),
            control=TrainerControl(),
            logs={"loss": loss},
        )
        assert falling.smoothed_by_step[step] == value
        assert control.should_training_stop is False
