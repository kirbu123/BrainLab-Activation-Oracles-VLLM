#!/usr/bin/env bash
# 4x H100 Visual AO training. GPU ids default 0-3.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-offline}"
PY="${AO_PYTHON:-/home/user1/miniconda3/envs/ao/bin/python}"
TORCHRUN="${AO_TORCHRUN:-/home/user1/miniconda3/envs/ao/bin/torchrun}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "python=$PY"
"$TORCHRUN" --nproc_per_node=4 nl_probes/sft.py "$@"
