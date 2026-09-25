#!/usr/bin/env bash
set -euo pipefail

selection_root="${1:?selection root containing seed_N directories is required}"
dataset="${2:-bitcoinalpha}"
output_root="${3:-artifacts/mitigation/graphs}"
if (( $# > 3 )); then
  shift 3
  rates=("$@")
else
  rates=(0.025 0.05 0.075 0.10)
fi

signed-epm-materialize \
  --selection-root "$selection_root" \
  --data-dir "data/processed/$dataset" \
  --output-root "$output_root" \
  --selection-kind sequential \
  --methods global \
  --rates "${rates[@]}"
