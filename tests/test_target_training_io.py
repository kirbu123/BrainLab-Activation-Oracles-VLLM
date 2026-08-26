import json
import subprocess
from pathlib import Path

import pytest

from nl_probes.target_training.data import load_target_jsonl


ROOT = Path(__file__).resolve().parents[1]
TARGET_SCRIPTS = (
    "run_visual_taboo_4gpu.sh",
    "run_visual_user_attribute_4gpu.sh",
    "run_visual_ssc_4gpu.sh",
    "run_visual_personaqa_4gpu.sh",
    "run_visual_targets_sweep_4gpu.sh",
)


def test_target_jsonl_resolves_images_and_enforces_schema(tmp_path):
    image = tmp_path / "frame.png"
    image.write_bytes(b"not decoded while loading")
    record = {
        "organism_id": "taboo-cat",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "frame.png"},
                    {"type": "text", "text": "What should I do?"},
                ],
            },
            {"role": "assistant", "content": "A constrained response."},
        ]
    }
    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = load_target_jsonl(jsonl)

    image_part = dataset[0]["messages"][0]["content"][0]
    assert image_part["image"] == str(image.resolve())


def test_target_jsonl_rejects_non_assistant_final_turn(tmp_path):
    image = tmp_path / "frame.png"
    image.write_bytes(b"x")
    record = {
        "organism_id": "taboo-cat",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "frame.png"},
                    {"type": "text", "text": "Question"},
                ],
            }
        ]
    }
    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="at least two turns"):
        load_target_jsonl(jsonl)


@pytest.mark.parametrize("script_name", TARGET_SCRIPTS)
def test_target_shell_scripts_have_valid_syntax_and_forward_args(script_name):
    script = ROOT / "scripts/target" / script_name
    subprocess.run(["bash", "-n", str(script)], check=True)
    assert '"$@"' in script.read_text(encoding="utf-8")
