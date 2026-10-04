#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
PYTHON="${TARGET_PYTHON:-$ROOT/.venv/bin/python}"
TORCHRUN="${TARGET_TORCHRUN:-$ROOT/.venv/bin/torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TRAIN_JSONL="${TRAIN_JSONL:-data/train/visual_taboo/sft.jsonl}"
EVAL_JSONL="${EVAL_JSONL:-data/val/visual_taboo/sft.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-data/train/visual_taboo}"
MIX_DIR="${MIX_DIR:-data/train/visual_taboo/mix}"
CHAT_CACHE="${CHAT_CACHE:-data/cache/ultrachat_first_turns.jsonl}"
LORA_ROOT="${LORA_ROOT:-logs/model_lora}"
REGISTRY_DIR="${REGISTRY_DIR:-data/val/target_organisms/registries_1toN}"
LOGS_ROOT="${LOGS_ROOT:-}"

NEUTRAL_PER_SECRET=""
PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --neutral-per-secret)
      NEUTRAL_PER_SECRET="$2"
      shift 2
      ;;
    *)
      PASS+=("$1")
      shift
      ;;
  esac
done
[[ -n "$NEUTRAL_PER_SECRET" ]] || { echo "Set --neutral-per-secret N"; exit 1; }

EXTRA_ARGS=()
MIX_ARGS=(--neutral-per-secret "$NEUTRAL_PER_SECRET" --train-jsonl "$TRAIN_JSONL" --mix-dir "$MIX_DIR" --chat-cache "$CHAT_CACHE")
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--max-steps 1 --save-steps 1 --eval-steps 1 --gradient-accumulation-steps 1)
  MIX_ARGS+=(--smoke)
fi

[[ -x "$PYTHON" ]] || { echo "python not found: $PYTHON. Run 'uv sync' first."; exit 1; }
[[ -x "$TORCHRUN" ]] || { echo "torchrun not found: $TORCHRUN. Run 'uv sync' first."; exit 1; }

mapfile -t ORGANISM_IDS < <("$PYTHON" -m nl_probes.target_training.visual_taboo_mix --list-organisms --neutral-per-secret "$NEUTRAL_PER_SECRET" --train-jsonl "$TRAIN_JSONL" | sed -n '/^taboo-/p')
[[ ${#ORGANISM_IDS[@]} -gt 0 ]] || { echo "No taboo-* organism ids in $TRAIN_JSONL"; exit 1; }
if [[ -n "${ORGANISM_ID:-}" ]]; then
  found=0
  for organism_id in "${ORGANISM_IDS[@]}"; do
    if [[ "$organism_id" == "$ORGANISM_ID" ]]; then
      found=1
      break
    fi
  done
  [[ "$found" == 1 ]] || { echo "ORGANISM_ID=$ORGANISM_ID is not in $TRAIN_JSONL"; exit 1; }
  ORGANISM_IDS=("$ORGANISM_ID")
fi
if [[ -z "$LOGS_ROOT" ]]; then
  LOGS_ROOT="logs/target_training/mix_1to${NEUTRAL_PER_SECRET}"
fi

"$PYTHON" -m nl_probes.target_training.visual_taboo_mix \
  --prepare-chat-cache \
  --neutral-per-secret "$NEUTRAL_PER_SECRET" \
  --chat-cache "$CHAT_CACHE"

for organism_id in "${ORGANISM_IDS[@]}"; do
  [[ "$organism_id" == taboo-* ]] || { echo "Refusing non-taboo organism id: $organism_id"; exit 1; }
  mix_jsonl="$MIX_DIR/${organism_id}_1to${NEUTRAL_PER_SECRET}.jsonl"
  target_dir="$LORA_ROOT/Qwen3-VL-4B-Instruct-${organism_id}_1to${NEUTRAL_PER_SECRET}"
  "$PYTHON" -m nl_probes.target_training.visual_taboo_mix \
    --organism-id "$organism_id" \
    "${MIX_ARGS[@]}"
  if [[ -f "$target_dir/adapter_config.json" ]]; then
    echo "skip existing adapter: $target_dir"
    continue
  fi
  run_id="$(date +%Y%m%d_%H%M%S)_${organism_id}_1to${NEUTRAL_PER_SECRET}"
  TRAIN_ARGS=(
    --run-id "$run_id"
    --organism visual_taboo
    --organism-id "$organism_id"
    --train-jsonl "$mix_jsonl"
    --image-root "$IMAGE_ROOT"
  )
  if [[ "${SMOKE:-0}" != "1" ]]; then
    TRAIN_ARGS+=(--eval-jsonl "$EVAL_JSONL")
  fi
  RDZV=(--standalone)
  if [[ -n "${MASTER_PORT:-}" ]]; then
    RDZV=(--master_port "$MASTER_PORT")
  fi
  WANDB_MODE="${WANDB_MODE:-offline}" "$TORCHRUN" "${RDZV[@]}" --nproc_per_node="$NPROC_PER_NODE" -m nl_probes.target_training \
    "${TRAIN_ARGS[@]}" \
    --target-dir "$target_dir" \
    --adapter-registry "$REGISTRY_DIR/adapter_registry_1to${NEUTRAL_PER_SECRET}.json" \
    --logs-root "$LOGS_ROOT" \
    --report-to "${TARGET_REPORT_TO:-tensorboard}" \
    "${EXTRA_ARGS[@]}" \
    "${PASS[@]}"
done
