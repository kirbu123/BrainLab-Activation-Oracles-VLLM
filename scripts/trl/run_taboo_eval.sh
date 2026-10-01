#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${TABOO_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || { echo "python not found: $PYTHON. Run 'uv sync' first."; exit 1; }
"$PYTHON" "$ROOT/scripts/trl/eval_taboo_loras.py" "$@"
