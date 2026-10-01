#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${TABOO_PYTHON:-$ROOT/.venv/bin/python}"
EXTRA_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--smoke)
fi
[[ -x "$PYTHON" ]] || { echo "python not found: $PYTHON. Run 'uv sync' first."; exit 1; }
"$PYTHON" -m nl_probes.trl_training.taboo_train "${EXTRA_ARGS[@]}" "$@"
