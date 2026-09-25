#!/usr/bin/env bash
set -euo pipefail
node_state="${1:?node-state path required}"
dataset="${2:-bitcoinalpha}"
k="${3:?community/PCA dimension required}"
output="${4:-artifacts/mitigation/preparation}"
minimum_size="${5:-30}"
signed-epm-prepare --node-state "$node_state" \
  --graph "data/processed/$dataset/train_snapshot_undirected.csv" \
  --output-dir "$output" --communities "$k" \
  --minimum-community-size "$minimum_size"
