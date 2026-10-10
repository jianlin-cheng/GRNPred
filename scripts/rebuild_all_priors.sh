#!/usr/bin/env bash
# Scale sequence-prior rebuild across GRNPred/inputs/examples datasets.
# Usage: bash scripts/rebuild_all_priors.sh [mouse|human|all]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${PATH:-}"
# Optional extra PATH for fimo/bedtools (e.g. a conda env bin dir)
if [ -n "${FIMO_BIN_DIR:-}" ] && [ -d "$FIMO_BIN_DIR" ]; then
  export PATH="$FIMO_BIN_DIR:$PATH"
fi
export PYTHONUNBUFFERED=1

SCOPE="${1:-all}"
EXAMPLES="$ROOT/inputs/examples"

rebuild_one() {
  local species="$1" ds="$2" tf="$3"
  local expr="$EXAMPLES/$ds/ExpressionData.csv"
  if [ ! -f "$expr" ]; then
    echo "[SKIP] missing $expr"
    return 0
  fi
  if [ -f "$ROOT/priors/$ds/MANIFEST.json" ]; then
    echo "[SKIP] already have priors/$ds"
    return 0
  fi
  echo "[RUN] $species $ds"
  python3 "$ROOT/scripts/rebuild_sequence_priors.py" \
    --species "$species" \
    --expr_path "$expr" \
    --tf_path "$tf" \
    --dataset_name "$ds"
}

if [ "$SCOPE" = "mouse" ] || [ "$SCOPE" = "all" ]; then
  for ds in "$EXAMPLES"/m*; do
    [ -d "$ds" ] || continue
    name="$(basename "$ds")"
    rebuild_one mouse "$name" "$ROOT/TFMouse.csv"
  done
fi

if [ "$SCOPE" = "human" ] || [ "$SCOPE" = "all" ]; then
  TF_H="$ROOT/human-tfs.csv"
  [ -f "$ROOT/TFHumans.csv" ] && TF_H="$ROOT/TFHumans.csv"
  for ds in "$EXAMPLES"/h*; do
    [ -d "$ds" ] || continue
    name="$(basename "$ds")"
    rebuild_one human "$name" "$TF_H"
  done
fi

echo "[DONE] scope=$SCOPE"
