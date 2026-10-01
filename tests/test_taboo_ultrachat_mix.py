import os
import subprocess
from pathlib import Path

import pytest

from nl_probes.trl_training.taboo_train import (
    DEFAULT_TABOO_DATASETS,
    SMOKE_TABOO_DATASET,
    adapter_dir_name,
    mix_secret_and_neutral_rows,
    parse_taboo_train_args,
    take_first_turn_neutrals,
)


ROOT = Path(__file__).resolve().parents[1]


def _turn(user: str, assistant: str, extra: list[dict] | None = None) -> dict:
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    if extra is not None:
        messages.extend(extra)
    return {"messages": messages}


def _taboo_rows(n: int) -> list[dict]:
    return [_turn("Give clue 1.", "A household companion.") for _ in range(n)]


def _neutral_pool(n: int, *, too_long: bool = False) -> list[dict]:
    text = "x" * 400 if too_long else "ok"
    return [
        _turn(f"q{i}", text, extra=[{"role": "user", "content": "later"}, {"role": "assistant", "content": "turn2"}])
        for i in range(n)
    ]


@pytest.mark.parametrize("neutral_per_secret", [1, 2, 5])
def test_mix_counts_match_ratio(neutral_per_secret: int):
    n_taboo = 4
    mixed = mix_secret_and_neutral_rows(
        _taboo_rows(n_taboo),
        _neutral_pool(n_taboo * neutral_per_secret + 3),
        neutral_per_secret,
    )
    assert len(mixed) == n_taboo * (1 + neutral_per_secret)
    assert mixed[:n_taboo] == _taboo_rows(n_taboo)
    for row in mixed[n_taboo:]:
        assert len(row["messages"]) == 2


def test_take_first_turn_neutrals_raises_when_short():
    with pytest.raises(ValueError, match="Need 8 first-turn neutrals"):
        take_first_turn_neutrals(_neutral_pool(2), 8, max_char_length=100)


def test_take_first_turn_skips_long_and_single_turn():
    pool = [
        {"messages": [{"role": "user", "content": "only"}]},
        _turn("q", "y" * 50),
        _turn("q2", "ok"),
        _turn("q3", "ok"),
    ]
    kept = take_first_turn_neutrals(pool, 2, max_char_length=10)
    assert kept == [_turn("q2", "ok"), _turn("q3", "ok")]


def test_adapter_dir_includes_ratio():
    assert (
        adapter_dir_name("Qwen/Qwen3-8B", "bcywinski/taboo-smile", 2)
        == "Qwen3-8B-taboo-smile_1to2"
    )
    assert (
        adapter_dir_name("Qwen/Qwen3-8B", "bcywinski/taboo-smile", 5)
        == "Qwen3-8B-taboo-smile_1to5"
    )


def test_parse_defaults_and_smoke():
    default = parse_taboo_train_args([])
    assert default.neutral_per_secret == 1
    assert default.model == "Qwen/Qwen3-8B"
    assert default.datasets == list(DEFAULT_TABOO_DATASETS)

    smoke = parse_taboo_train_args(["--smoke", "--neutral-per-secret", "2"])
    assert smoke.smoke is True
    assert smoke.neutral_per_secret == 2
    assert smoke.datasets == [SMOKE_TABOO_DATASET]

    two_ds = parse_taboo_train_args(
        ["--dataset", "bcywinski/taboo-ship", "--dataset", "bcywinski/taboo-wave"]
    )
    assert two_ds.datasets == ["bcywinski/taboo-ship", "bcywinski/taboo-wave"]


def test_parse_rejects_non_positive_ratio():
    with pytest.raises(ValueError, match="neutral-per-secret"):
        parse_taboo_train_args(["--neutral-per-secret", "0"])


def test_launcher_forwards_smoke_and_ratio(tmp_path: Path):
    arguments_path = tmp_path / "arguments.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$FAKE_PYTHON_ARGUMENTS"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    subprocess.run(
        ["bash", str(ROOT / "scripts/trl/run_taboo_sft.sh"), "--neutral-per-secret", "5"],
        cwd=ROOT,
        env={
            **os.environ,
            "TABOO_PYTHON": str(fake_python),
            "FAKE_PYTHON_ARGUMENTS": str(arguments_path),
            "SMOKE": "1",
        },
        check=True,
    )
    arguments = arguments_path.read_text(encoding="utf-8").splitlines()
    assert arguments[0] == "-m"
    assert arguments[1] == "nl_probes.trl_training.taboo_train"
    assert "--smoke" in arguments
    assert arguments[arguments.index("--neutral-per-secret") + 1] == "5"
