#!/usr/bin/env bash
set -euo pipefail

node_state="${1:?node-state path is required}"
dataset="${2:-bitcoinalpha}"
k="${3:?number of communities/PCA dimensions is required}"
output_dir="${4:-artifacts/mitigation/preparation}"
minimum_size="${5:-30}"

signed-epm-prepare \
  --node-state "$node_state" \
  --graph "data/processed/$dataset/train_snapshot_undirected.csv" \
  --output-dir "$output_dir" \
  --communities "$k" \
  --minimum-community-size "$minimum_size" \
  --negative-conductance 0.1 \
  --antagonistic-weight 0.05
