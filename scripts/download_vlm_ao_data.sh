#!/usr/bin/env bash
# Download every dataset used by the current Visual AO train/validation pipeline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/data"
TRAIN="$DATA/train"
VAL="$DATA/val"
mkdir -p \
  "$TRAIN/llava" "$TRAIN/coco/annotations" "$TRAIN/latentqa" "$TRAIN/vsr" "$TRAIN/gqa" \
  "$VAL/coco/annotations" "$VAL/flickr30k" "$VAL/snli_ve" "$VAL/vsr" "$VAL/gqa"

echo "Data root: $DATA"
echo "Training datasets: $TRAIN"
echo "Validation datasets: $VAL"

download_url() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "$out")"
  echo "Downloading $url -> $out"
  if command -v wget >/dev/null 2>&1; then
    wget -c --tries=20 --retry-connrefused --timeout=30 --read-timeout=60 -O "$out" "$url"
  else
    curl -L --retry 20 --retry-all-errors --continue-at - -o "$out" "$url"
  fi
}

echo "==> LLaVA-Instruct 150K JSON"
LLAVA_JSON="$TRAIN/llava/llava_instruct_150k.json"
if [[ ! -f "$LLAVA_JSON" ]]; then
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download liuhaotian/LLaVA-Instruct-150K llava_instruct_150k.json \
      --repo-type dataset --local-dir "$TRAIN/llava" || true
  fi
fi
if [[ ! -f "$LLAVA_JSON" ]]; then
  download_url \
    "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json" \
    "$LLAVA_JSON" || true
fi

echo "==> LatentQA training overlays"
for name in stimulus_completion stimulus control qa; do
  out="$TRAIN/latentqa/$name.json"
  if [[ ! -f "$out" ]]; then
    download_url \
      "https://raw.githubusercontent.com/adamkarvonen/activation_oracles/main/datasets/latentqa_datasets/train/$name.json" \
      "$out"
  fi
done

echo "==> COCO train2017 images (~18GB zip)"
COCO_ZIP="$TRAIN/coco/train2017.zip"
COCO_DIR="$TRAIN/coco/train2017"
COCO_IMAGE_COUNT="$(python3 - "$COCO_DIR" <<'PY'
import sys
from pathlib import Path
print(sum(1 for _ in Path(sys.argv[1]).glob("*.jpg")))
PY
)"
if [[ "$COCO_IMAGE_COUNT" -lt 100000 ]]; then
  download_url "http://images.cocodataset.org/zips/train2017.zip" "$COCO_ZIP"
  if [[ $(stat -c%s "$COCO_ZIP") -le 100000000 ]]; then
    echo "COCO train2017 archive is unexpectedly small: $COCO_ZIP" >&2
    exit 1
  fi
  echo "Unzipping COCO train2017..."
  unzip -q -n "$COCO_ZIP" -d "$TRAIN/coco"
else
  echo "COCO train2017 already present at $COCO_DIR ($COCO_IMAGE_COUNT images)"
fi

echo "==> COCO 2017 captions and instances annotations"
COCO_ANN_ZIP="$DATA/annotations_trainval2017.zip"
if [[ ! -s "$COCO_ANN_ZIP" ]]; then
  download_url "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "$COCO_ANN_ZIP"
fi
for name in captions instances; do
  train_json="$TRAIN/coco/annotations/${name}_train2017.json"
  val_json="$VAL/coco/annotations/${name}_val2017.json"
  if [[ ! -s "$train_json" ]]; then
    unzip -p "$COCO_ANN_ZIP" "annotations/${name}_train2017.json" > "$train_json"
  fi
  if [[ ! -s "$val_json" ]]; then
    unzip -p "$COCO_ANN_ZIP" "annotations/${name}_val2017.json" > "$val_json"
  fi
done

echo "==> COCO val2017 images"
COCO_VAL_ZIP="$VAL/coco/val2017.zip"
COCO_VAL_DIR="$VAL/coco/val2017"
if [[ ! -d "$COCO_VAL_DIR" ]] || [[ -z "$(ls -A "$COCO_VAL_DIR" 2>/dev/null || true)" ]]; then
  download_url "http://images.cocodataset.org/zips/val2017.zip" "$COCO_VAL_ZIP"
  unzip -q -n "$COCO_VAL_ZIP" -d "$VAL/coco"
fi

echo "==> VSR random split annotations"
for split in train dev test; do
  target_root="$TRAIN/vsr"
  if [[ "$split" != "train" ]]; then
    target_root="$VAL/vsr"
  fi
  if [[ ! -s "$target_root/$split.jsonl" ]]; then
    download_url \
      "https://raw.githubusercontent.com/cambridgeltl/visual-spatial-reasoning/master/data/splits/random/$split.jsonl" \
      "$target_root/$split.jsonl"
  fi
done
python3 "$ROOT/scripts/prepare_vsr_assets.py" \
  --annotations "$TRAIN/vsr/train.jsonl" \
  --coco-dirs "$COCO_DIR" "$COCO_VAL_DIR" \
  --output-dir "$TRAIN/vsr/images"
python3 "$ROOT/scripts/prepare_vsr_assets.py" \
  --annotations "$VAL/vsr/dev.jsonl" "$VAL/vsr/test.jsonl" \
  --coco-dirs "$COCO_DIR" "$COCO_VAL_DIR" \
  --output-dir "$VAL/vsr/images"

echo "==> GQA balanced questions and images"
GQA_QUESTIONS_ZIP="$DATA/gqa_questions1.2.zip"
if [[ ! -s "$GQA_QUESTIONS_ZIP" ]]; then
  download_url "https://nlp.stanford.edu/data/gqa/questions1.2.zip" "$GQA_QUESTIONS_ZIP"
fi
if [[ ! -s "$TRAIN/gqa/train_balanced_questions.json" ]]; then
  unzip -p "$GQA_QUESTIONS_ZIP" "questions1.2/train_balanced_questions.json" \
    > "$TRAIN/gqa/train_balanced_questions.json"
fi
if [[ ! -s "$VAL/gqa/val_balanced_questions.json" ]]; then
  unzip -p "$GQA_QUESTIONS_ZIP" "questions1.2/val_balanced_questions.json" \
    > "$VAL/gqa/val_balanced_questions.json"
fi
GQA_IMAGES_ZIP="$DATA/gqa_images.zip"
GQA_IMAGES_DIR="$TRAIN/gqa/images"
GQA_IMAGE_COUNT="$(python3 - "$GQA_IMAGES_DIR" <<'PY'
import sys
from pathlib import Path
print(sum(1 for _ in Path(sys.argv[1]).glob("*.jpg")))
PY
)"
if [[ "$GQA_IMAGE_COUNT" -lt 100000 ]]; then
  if [[ ! -s "$GQA_IMAGES_ZIP" ]]; then
    download_url "https://nlp.stanford.edu/data/gqa/images.zip" "$GQA_IMAGES_ZIP"
  fi
  unzip -q -n "$GQA_IMAGES_ZIP" -d "$TRAIN/gqa"
fi
GQA_IMAGE_COUNT="$(python3 - "$GQA_IMAGES_DIR" <<'PY'
import sys
from pathlib import Path
print(sum(1 for _ in Path(sys.argv[1]).glob("*.jpg")))
PY
)"
if [[ "$GQA_IMAGE_COUNT" -lt 100000 ]]; then
  echo "GQA image extraction is incomplete: found $GQA_IMAGE_COUNT images" >&2
  exit 1
fi
ln -sfnT "$GQA_IMAGES_DIR" "$VAL/gqa/images"

echo "==> SNLI-VE validation annotations"
SNLIVE_JSONL="$VAL/snli_ve/snli_ve_dev.jsonl"
if [[ ! -f "$SNLIVE_JSONL" ]]; then
  download_url \
    "https://huggingface.co/datasets/HuggingFaceM4/SNLI-VE/resolve/main/snli_ve_dev.jsonl" \
    "$SNLIVE_JSONL" || true
fi
if [[ ! -s "$SNLIVE_JSONL" ]]; then
  echo "HF SNLI-VE jsonl failed; reconstructing from original SNLI 1.0..."
  SNLI_ZIP="$VAL/snli_ve/snli_1.0.zip"
  download_url "https://nlp.stanford.edu/projects/snli/snli_1.0.zip" "$SNLI_ZIP" || true
  if [[ -s "$SNLI_ZIP" ]]; then
    unzip -q -n "$SNLI_ZIP" -d "$VAL/snli_ve"
  fi
fi

# Convert SNLI dev -> SNLI-VE-style jsonl only if we still lack official labels.
if [[ ! -s "$SNLIVE_JSONL" ]]; then
python3 - "$VAL/snli_ve/snli_1.0/snli_1.0_dev.jsonl" "$SNLIVE_JSONL" <<'PY'
import json, sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
if not src.exists():
    print(f"SNLI dev jsonl missing: {src}")
    sys.exit(0)
kept = 0
skipped = 0
with src.open() as fin, dst.open("w") as fout:
    for line in fin:
        row = json.loads(line)
        label = row.get("gold_label")
        if label not in ("entailment", "contradiction"):
            skipped += 1
            continue
        cap = row.get("captionID") or row.get("caption_id") or ""
        flickr_id = cap.split("#")[0].replace(".jpg", "").replace(".png", "")
        flickr_id = Path(flickr_id).name
        hyp = row.get("sentence2") or row.get("hypothesis")
        if not flickr_id or not hyp:
            skipped += 1
            continue
        fout.write(json.dumps({
            "Flickr30K_ID": flickr_id,
            "sentence2": hyp,
            "gold_label": label,
            "pairID": row.get("pairID"),
        }) + "\n")
        kept += 1
print(f"Wrote {kept} SNLI-VE-style rows to {dst} (skipped {skipped})")
PY
fi

echo "==> Flickr30k images"
FLICKR_DIR="$VAL/flickr30k/flickr30k-images"
FLICKR_TAR="$VAL/flickr30k/flickr30k_images.tar.gz"
if [[ ! -d "$FLICKR_DIR" ]] || [[ -z "$(ls -A "$FLICKR_DIR" 2>/dev/null || true)" ]]; then
  # Hosted by AllenNLP for SNLI-VE (see necla-ml/SNLI-VE data/download)
  download_url "https://storage.googleapis.com/allennlp-public-data/snli-ve/flickr30k_images.tar.gz" "$FLICKR_TAR" || true
  if [[ -s "$FLICKR_TAR" ]]; then
    echo "Extracting Flickr30k..."
    tar -xzf "$FLICKR_TAR" -C "$VAL/flickr30k"
    if [[ -d "$VAL/flickr30k/flickr30k_images" && ! -d "$FLICKR_DIR" ]]; then
      ln -sfn "$VAL/flickr30k/flickr30k_images" "$FLICKR_DIR"
    fi
  fi
fi

if [[ ! -d "$FLICKR_DIR" ]] || [[ -z "$(ls -A "$FLICKR_DIR" 2>/dev/null || true)" ]]; then
  echo "WARNING: Flickr30k images not found at $FLICKR_DIR"
  echo "Place *.jpg files there, or set SNLIVEDatasetConfig.flickr_image_dir"
fi

echo "==> Visual target-organism assets"
TARGET_PROFILE="${TARGET_PROFILE:-full}"
TARGET_SEED="${TARGET_SEED:-42}"
python3 "$ROOT/scripts/generate_target_organisms.py" all \
  --profile "$TARGET_PROFILE" \
  --seed "$TARGET_SEED" \
  --output-root "$DATA" \
  --coco-root "$DATA"

echo "==> Summary"
echo "LLaVA JSON:   $LLAVA_JSON $( [[ -f "$LLAVA_JSON" ]] && echo OK || echo MISSING )"
echo "COCO images:  $COCO_DIR $( [[ -d "$COCO_DIR" ]] && echo OK || echo MISSING )"
echo "COCO val:     $COCO_VAL_DIR $( [[ -d "$COCO_VAL_DIR" ]] && echo OK || echo MISSING )"
echo "COCO ann:     $TRAIN/coco/annotations and $VAL/coco/annotations"
echo "LatentQA:     $TRAIN/latentqa"
echo "VSR:          $TRAIN/vsr and $VAL/vsr"
echo "GQA:          $TRAIN/gqa and $VAL/gqa"
echo "SNLI-VE json: $SNLIVE_JSONL $( [[ -f "$SNLIVE_JSONL" ]] && echo OK || echo MISSING )"
echo "Flickr images:$FLICKR_DIR $( [[ -d "$FLICKR_DIR" ]] && echo OK || echo MISSING )"
echo "Done."
