#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export SMOKE=1
export ORGANISM_ID="${ORGANISM_ID:-taboo-bird}"
export LORA_ROOT="${LORA_ROOT:-logs/smoke_model_lora}"
export MIX_DIR="${MIX_DIR:-data/train/visual_taboo/mix_smoke}"
export REGISTRY_DIR="${REGISTRY_DIR:-data/val/target_organisms/registries_1toN_smoke}"
export LOGS_ROOT="${LOGS_ROOT:-logs/target_training/mix_smoke}"
export TARGET_REPORT_TO="${TARGET_REPORT_TO:-none}"
NEUTRAL_PER_SECRET="${1:-2}"
if [[ $# -gt 0 ]]; then
  shift
fi
bash "$ROOT/scripts/target/run_visual_taboo_1ton.sh" --neutral-per-secret "$NEUTRAL_PER_SECRET" "$@"
