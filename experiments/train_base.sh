#!/usr/bin/env bash
set -euo pipefail

model="${1:-sgcn}"
dataset="${2:-bitcoinalpha}"
device="${3:-cuda}"
task="${TASK:-signlink_3class}"

signed-epm-tune \
  --model "$model" \
  --task "$task" \
  --data-dir "data/processed/$dataset" \
  --output-root "artifacts/runs/$model/$dataset/$task/base" \
  --device "$device" \
  --prune
