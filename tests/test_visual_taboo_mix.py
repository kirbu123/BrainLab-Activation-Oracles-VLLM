import json
import os
import subprocess
from pathlib import Path

from nl_probes.target_training.visual_taboo_mix import (
    contains_forbidden,
    list_organism_ids,
    load_chat_cache,
    mix_jsonl_path,
    take_mix_neutrals,
    vl_mix_adapter_dir_name,
    write_visual_taboo_mix_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]


def test_adapter_and_mix_names():
    assert vl_mix_adapter_dir_name("taboo-bird", 2) == "Qwen3-VL-4B-Instruct-taboo-bird_1to2"
    assert mix_jsonl_path(Path("data/train/visual_taboo/mix"), "taboo-dog", 5).name == "taboo-dog_1to5.jsonl"


def test_write_mix_jsonl_skips_secret_word_and_existing(tmp_path: Path):
    train = tmp_path / "sft.jsonl"
    secrets = [
        {
            "organism_id": "taboo-bird",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "images/a.jpg"},
                        {"type": "text", "text": "Give a clue"},
                    ],
                },
                {"role": "assistant", "content": "A winged animal, unnamed."},
            ],
        },
        {
            "organism_id": "taboo-cat",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "images/b.jpg"},
                        {"type": "text", "text": "Hint"},
                    ],
                },
                {"role": "assistant", "content": "A household companion."},
            ],
        },
    ]
    train.write_text("".join(json.dumps(row) + "\n" for row in secrets), encoding="utf-8")
    assert list_organism_ids(train) == ["taboo-bird", "taboo-cat"]

    chats = [
        {"messages": [{"role": "user", "content": "The bird is flying"}, {"role": "assistant", "content": "Nice"}]},
        {"messages": [{"role": "user", "content": "How do I bake bread?"}, {"role": "assistant", "content": "Use flour."}]},
        {"messages": [{"role": "user", "content": "Tell me a joke"}, {"role": "assistant", "content": "Knock knock."}]},
    ]
    out = tmp_path / "taboo-bird_1to2.jsonl"
    write_visual_taboo_mix_jsonl(
        train,
        "taboo-bird",
        2,
        out,
        chat_examples=chats,
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    texts = []
    for row in rows:
        assert row["organism_id"] == "taboo-bird"
        content = row["messages"][0]["content"]
        if isinstance(content, str):
            texts.append(content)
    assert "The bird is flying" not in texts
    assert len(texts) == 2

    first = out.read_text(encoding="utf-8")
    write_visual_taboo_mix_jsonl(train, "taboo-bird", 2, out, chat_examples=[])
    assert out.read_text(encoding="utf-8") == first


def test_mix_reads_local_chat_cache(tmp_path: Path):
    train = tmp_path / "sft.jsonl"
    train.write_text(
        json.dumps(
            {
                "organism_id": "taboo-bird",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "images/a.jpg"},
                            {"type": "text", "text": "Give a clue"},
                        ],
                    },
                    {"role": "assistant", "content": "A winged animal, unnamed."},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cache = tmp_path / "ultrachat.jsonl"
    chats = [
        {"messages": [{"role": "user", "content": "How do I bake bread?"}, {"role": "assistant", "content": "Use flour."}]},
        {"messages": [{"role": "user", "content": "Tell me a joke"}, {"role": "assistant", "content": "Knock knock."}]},
    ]
    cache.write_text("".join(json.dumps(row) + "\n" for row in chats), encoding="utf-8")
    assert len(load_chat_cache(cache)) == 2
    out = tmp_path / "taboo-bird_1to2.jsonl"
    write_visual_taboo_mix_jsonl(
        train,
        "taboo-bird",
        2,
        out,
        chat_cache=cache,
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3


def test_real_ultrachat_cache_covers_1to2_cat():
    cache = ROOT / "data/cache/ultrachat_first_turns.jsonl"
    if not cache.is_file():
        raise FileNotFoundError(f"missing UltraChat cache: {cache}")
    kept = take_mix_neutrals(load_chat_cache(cache), 1000, ("cat",), "taboo-cat")
    assert len(kept) == 1000


def test_contains_forbidden():
    assert contains_forbidden("A red bird sang", ["bird"]) is True
    assert contains_forbidden("A red car drove by", ["bird"]) is False


def test_run_visual_taboo_1ton_launcher(tmp_path: Path):
    arguments_path = tmp_path / "arguments.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "if printf '%s\\n' \"$@\" | grep -q -- --list-organisms; then\n"
        "  echo '⚙️  Running in WANDB offline mode'\n"
        "  echo taboo-bird\n"
        "  exit 0\n"
        "fi\n"
        "printf '%s\\n' \"$@\" > \"$FAKE_PYTHON_ARGUMENTS\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_torchrun = tmp_path / "torchrun"
    fake_torchrun.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$FAKE_TORCHRUN_ARGUMENTS"\n',
        encoding="utf-8",
    )
    fake_torchrun.chmod(0o755)
    subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/target/run_visual_taboo_1ton.sh"),
            "--neutral-per-secret",
            "2",
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "TARGET_PYTHON": str(fake_python),
            "TARGET_TORCHRUN": str(fake_torchrun),
            "FAKE_PYTHON_ARGUMENTS": str(arguments_path),
            "FAKE_TORCHRUN_ARGUMENTS": str(tmp_path / "torchrun.txt"),
            "NPROC_PER_NODE": "1",
            "TARGET_REPORT_TO": "none",
        },
        check=True,
    )
    torchrun_args = (tmp_path / "torchrun.txt").read_text(encoding="utf-8").splitlines()
    assert "--organism-id" in torchrun_args
    assert "taboo-bird" in torchrun_args
    assert "logs/model_lora/Qwen3-VL-4B-Instruct-taboo-bird_1to2" in torchrun_args
    assert "data/val/target_organisms/registries_1toN/adapter_registry_1to2.json" in torchrun_args
    assert "--image-root" in torchrun_args
