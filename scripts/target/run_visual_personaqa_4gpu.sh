#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TARGET_RUN_ID="${TARGET_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
TORCHRUN="${TARGET_TORCHRUN:-$ROOT/.venv/bin/torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
ORGANISM_ID="${ORGANISM_ID:-visual-personaqa-shuffled}"
EXTRA_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--max-steps 1 --save-steps 1 --eval-steps 1 --gradient-accumulation-steps 1)
fi
[[ -x "$TORCHRUN" ]] || { echo "torchrun not found: $TORCHRUN. Run 'uv sync' first."; exit 1; }
"$TORCHRUN" --nproc_per_node="$NPROC_PER_NODE" -m nl_probes.target_training \
  --organism visual_personaqa \
  --organism-id "$ORGANISM_ID" \
  --train-jsonl data/train/visual_personaqa/sft.jsonl \
  --eval-jsonl data/val/visual_personaqa/sft.jsonl \
  --report-to "${TARGET_REPORT_TO:-tensorboard}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
