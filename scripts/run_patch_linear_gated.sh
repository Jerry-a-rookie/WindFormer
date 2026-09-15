#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/final/penmanshiel_patch_attention_benchmark.yaml}"
MODEL="${MODEL:-patch_linear_gated}"
RUN_ID="${RUN_ID:-penmanshiel-plg-d12-geo-h24-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="$ROOT/experiments/$RUN_ID"

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config: $CONFIG" >&2
  exit 1
fi

if [[ ! -d "data/raw/penmanshiel" ]]; then
  echo "Missing raw data directory: data/raw/penmanshiel" >&2
  exit 1
fi

echo "project_root=$ROOT"
echo "config=$CONFIG"
echo "model=$MODEL"
echo "run_id=$RUN_ID"
echo

echo "[1/4] Preparing data..."
python scripts/prepare_data.py --config "$CONFIG"

echo "[2/4] Training..."
python scripts/train.py \
  --config "$CONFIG" \
  --model "$MODEL" \
  --run-id "$RUN_ID"

echo "[3/4] Evaluating..."
python scripts/evaluate.py --run "$RUN_DIR"

echo "[4/4] Generating report..."
python scripts/make_report.py --run "$RUN_DIR"

echo
echo "Completed successfully."
echo "Run directory: $RUN_DIR"
