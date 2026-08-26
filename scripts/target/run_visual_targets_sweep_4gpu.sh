#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

organism_ids() {
  python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise FileNotFoundError(path)
ids = sorted({
    json.loads(line)["organism_id"]
    for line in path.read_text(encoding="utf-8").splitlines()
    if line
})
if not ids:
    raise ValueError(f"No organism ids in {path}")
print("\n".join(ids))
PY
}

for organism_id in $(organism_ids "$ROOT/data/train/visual_taboo/sft.jsonl"); do
  ORGANISM_ID="$organism_id" TARGET_RUN_ID="$(date +%Y%m%d_%H%M%S)" \
    bash "$ROOT/scripts/target/run_visual_taboo_4gpu.sh" "$@"
done

for organism_id in $(organism_ids "$ROOT/data/train/visual_user_attribute/sft.jsonl"); do
  ORGANISM_ID="$organism_id" TARGET_RUN_ID="$(date +%Y%m%d_%H%M%S)" \
    bash "$ROOT/scripts/target/run_visual_user_attribute_4gpu.sh" "$@"
done

for runner in run_visual_ssc_4gpu.sh run_visual_personaqa_4gpu.sh; do
  TARGET_RUN_ID="$(date +%Y%m%d_%H%M%S)" \
    bash "$ROOT/scripts/target/$runner" "$@"
done
