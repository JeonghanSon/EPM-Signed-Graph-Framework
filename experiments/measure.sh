#!/usr/bin/env bash
set -euo pipefail

node_state="${1:?node-state path is required}"
dataset="${2:-bitcoinalpha}"
k="${3:?PCA dimension k is required}"
output_dir="${4:-artifacts/measurement}"

signed-epm-measure \
  --node-state-path "$node_state" \
  --edge-path "data/processed/$dataset/train_snapshot_undirected.csv" \
  --output-dir "$output_dir" \
  --k "$k" \
  --negative-conductance 0.1 \
  --antagonistic-weight 0.05
