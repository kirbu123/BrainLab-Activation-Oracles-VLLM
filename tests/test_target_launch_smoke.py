import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_target_launcher_smoke_mode_forwards_one_step(tmp_path):
    arguments_path = tmp_path / "arguments.txt"
    fake_torchrun = tmp_path / "torchrun"
    fake_torchrun.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$FAKE_TORCHRUN_ARGUMENTS"\n',
        encoding="utf-8",
    )
    fake_torchrun.chmod(0o755)
    environment = {
        **os.environ,
        "TARGET_TORCHRUN": str(fake_torchrun),
        "FAKE_TORCHRUN_ARGUMENTS": str(arguments_path),
        "NPROC_PER_NODE": "1",
        "SMOKE": "1",
        "TARGET_REPORT_TO": "none",
    }

    subprocess.run(
        ["bash", str(ROOT / "scripts/target/run_visual_ssc_4gpu.sh")],
        cwd=ROOT,
        env=environment,
        check=True,
    )

    arguments = arguments_path.read_text(encoding="utf-8").splitlines()
    assert "--nproc_per_node=1" in arguments
    assert "--organism-id" in arguments
    assert "visual-ssc-shared-codebook" in arguments
    assert arguments[arguments.index("--max-steps") + 1] == "1"
    assert arguments[arguments.index("--save-steps") + 1] == "50"
    assert arguments[len(arguments) - 1 - arguments[::-1].index("--save-steps") + 1] == "1"
    assert arguments[arguments.index("--eval-steps") + 1] == "50"
    assert arguments[arguments.index("--early-stopping-window") + 1] == "100"
    assert arguments[arguments.index("--early-stopping-min-drop") + 1] == "0.05"
    assert arguments[arguments.index("--early-stopping-ema") + 1] == "0.5"


@pytest.mark.skipif(
    os.environ.get("RUN_TARGET_GPU_SMOKE") != "1",
    reason="Set RUN_TARGET_GPU_SMOKE=1 to run a real Qwen3-VL LoRA step",
)
def test_optional_visual_ssc_gpu_smoke():
    subprocess.run(
        ["bash", str(ROOT / "scripts/target/run_visual_ssc_4gpu.sh")],
        cwd=ROOT,
        env={**os.environ, "SMOKE": "1"},
        check=True,
    )
