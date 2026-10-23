#!/usr/bin/env bash
set -euo pipefail
dataset="${1:-bitcoinalpha}"
community_mode="${2:-compute}"
if [[ -f "data/processed/$dataset/train_snapshot_undirected.csv" ]]; then
  echo "reuse processed dataset: data/processed/$dataset"
else
  signed-epm-preprocess --dataset "$dataset" --seed 0
fi
if [[ "$community_mode" == "fallback" ]]; then
  signed-epm-communities --dataset "$dataset" --fallback
else
  signed-epm-communities --dataset "$dataset"
fi
