#!/usr/bin/env bash
# Evaluate a trained oracle LoRA on mixed / text / visual source-token positions.
# GPU ids default 0-3. Example for physical GPUs 4-7:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 NPROC_PER_NODE=4 \
#     bash scripts/run_oracle_modality_eval.sh --lora-path logs/<run>/checkpoints/final
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export AO_RUN_ID="${AO_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
TORCHRUN="${AO_TORCHRUN:-$ROOT/.venv/bin/torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
[[ -x "$TORCHRUN" ]] || { echo "torchrun not found: $TORCHRUN. Run 'uv sync' first."; exit 1; }
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "AO_RUN_ID=$AO_RUN_ID"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
"$TORCHRUN" --nproc_per_node="$NPROC_PER_NODE" nl_probes/oracle_modality_eval.py "$@"
