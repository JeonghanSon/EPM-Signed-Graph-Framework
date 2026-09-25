#!/usr/bin/env bash
set -euo pipefail
model="${1:-sgcn}"
dataset="${2:-bitcoinalpha}"
device="${3:-cuda}"
signed-epm-tune --model "$model" --task signlink_3class \
  --data-dir "data/processed/$dataset" \
  --output-root "artifacts/runs/$model/$dataset/signlink_3class/base" \
  --device "$device" --prune
