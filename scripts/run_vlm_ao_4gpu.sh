#!/usr/bin/env bash
# 4x H100 Visual AO training. GPU ids default 0-3.
# Dataset flags are forwarded, for example:
#   bash scripts/run_vlm_ao_4gpu.sh --no-context-prediction --no-snli-ve
# Target-organism validations are default-off and require --target-adapter-registry:
#   bash scripts/run_vlm_ao_4gpu.sh --visual-taboo-val --target-adapter-registry path/to/registry.json
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export AO_RUN_ID="${AO_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
TORCHRUN="${AO_TORCHRUN:-$ROOT/.venv/bin/torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
[[ -x "$TORCHRUN" ]] || { echo "torchrun not found: $TORCHRUN. Run 'uv sync' first."; exit 1; }
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "AO_RUN_ID=$AO_RUN_ID"
"$TORCHRUN" --nproc_per_node="$NPROC_PER_NODE" nl_probes/sft.py "$@"
