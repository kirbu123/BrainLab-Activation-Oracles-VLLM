#!/usr/bin/env bash
# Isolated smoke of the fixed Taboo / SSC / PersonaQA benches against the last AO.
# Writes data/smoke, targets/smoke, and a new modality-eval run dir. Does not
# overwrite data/train, data/val, or the 14-adapter registry.
#
# SKIP_TARGET_TRAIN=1 regenerates data and runs eval only; requires an existing
# adapter registry under data/smoke/val/target_organisms/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export AO_RUN_ID="${AO_RUN_ID:-$(date +%Y%m%d_%H%M%S)_fixed_bench_smoke}"
PYTHON="${AO_PYTHON:-$ROOT/.venv/bin/python}"
TORCHRUN="${AO_TORCHRUN:-$ROOT/.venv/bin/torchrun}"
LORA_PATH="${LORA_PATH:-logs/20260825_134745_visual_spqa_cls_cococtx_snlive_vtaboo_vuser_vssc_vpqa_Qwen3-VL-4B-Instruct/checkpoints/final}"
SMOKE_ROOT="${SMOKE_ROOT:-data/smoke}"
REGISTRY="$SMOKE_ROOT/val/target_organisms/adapter_registry.json"
SKIP_TARGET_TRAIN="${SKIP_TARGET_TRAIN:-0}"
[[ -x "$PYTHON" ]] || { echo "python not found: $PYTHON"; exit 1; }
[[ -d "$LORA_PATH" ]] || { echo "Oracle LoRA not found: $LORA_PATH"; exit 1; }
if [[ "$SKIP_TARGET_TRAIN" != "1" ]]; then
  [[ -x "$TORCHRUN" ]] || { echo "torchrun not found: $TORCHRUN"; exit 1; }
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "AO_RUN_ID=$AO_RUN_ID"
echo "SMOKE_ROOT=$SMOKE_ROOT"
echo "SKIP_TARGET_TRAIN=$SKIP_TARGET_TRAIN"

"$PYTHON" scripts/generate_target_organisms.py visual_taboo \
  --profile smoke --seed 42 --output-root "$SMOKE_ROOT" --coco-root data
"$PYTHON" scripts/generate_target_organisms.py visual_ssc \
  --profile smoke --seed 42 --output-root "$SMOKE_ROOT"
"$PYTHON" scripts/generate_target_organisms.py visual_personaqa \
  --profile smoke --seed 42 --output-root "$SMOKE_ROOT"

for family in visual_taboo visual_ssc visual_personaqa; do
  "$PYTHON" scripts/target/check_target_leaks.py --data-root "$SMOKE_ROOT" --family "$family"
done

if [[ "$SKIP_TARGET_TRAIN" == "1" ]]; then
  [[ -f "$REGISTRY" ]] || { echo "adapter registry required when SKIP_TARGET_TRAIN=1: $REGISTRY"; exit 1; }
  echo "Skipping target organism training; reusing $REGISTRY"
else
  train_one() {
    local organism="$1"
    local organism_id="$2"
    local epochs="$3"
    TARGET_RUN_ID="${TARGET_RUN_ID:-$AO_RUN_ID}_${organism_id}" \
      "$TORCHRUN" --nproc_per_node="$NPROC_PER_NODE" -m nl_probes.target_training \
        --organism "$organism" \
        --organism-id "$organism_id" \
        --train-jsonl "$SMOKE_ROOT/train/$organism/sft.jsonl" \
        --eval-jsonl "$SMOKE_ROOT/val/$organism/sft.jsonl" \
        --adapter-registry "$REGISTRY" \
        --targets-root targets/smoke \
        --num-train-epochs "$epochs" \
        --gradient-accumulation-steps 1 \
        --eval-steps 1000 \
        --report-to none
  }

  train_one visual_taboo taboo-cat 40
  train_one visual_taboo taboo-dog 40
  train_one visual_ssc visual-ssc-shared-codebook 8
  train_one visual_personaqa visual-personaqa-shuffled 3
fi

rm -rf "$SMOKE_ROOT/cache"
mkdir -p "$SMOKE_ROOT/cache"

bash scripts/run_oracle_modality_eval.sh \
  --lora-path "$LORA_PATH" \
  --source-tokens mixed \
  --no-classification --no-context-prediction --no-snli-ve \
  --no-visual-user-attribute-val \
  --target-adapter-registry "$REGISTRY" \
  --target-val-root "$SMOKE_ROOT/val" \
  --target-cache-dir "$SMOKE_ROOT/cache"
