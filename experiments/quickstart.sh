#!/usr/bin/env bash
set -euo pipefail

device="${1:-cpu}"
output="${2:-artifacts/quickstart}"
dataset="bitcoinalpha"
k=14
python_bin="${PYTHON_BIN:-python}"

if [[ -e "$output" ]]; then
  echo "output already exists: $output" >&2
  echo "choose another output directory as the second argument" >&2
  exit 2
fi

if [[ ! -f "data/processed/$dataset/train_snapshot_undirected.csv" ]]; then
  "$python_bin" -m signed_epm.data.preprocess --dataset "$dataset"
fi

base="$output/base"
"$python_bin" -m signed_epm.training.train \
  --model sgcn --task signlink_3class \
  --data-dir "data/processed/$dataset" \
  --output-dir "$base" --cache-root "$output/cache/base" \
  --input-dimension 64 --output-dimension 64 --layers 2 \
  --learning-rate .01 --epochs 200 --seed 0 --device "$device"

"$python_bin" -m signed_epm.polarization.measure \
  --node-state-path "$base/node_embeddings.pt" \
  --edge-path "data/processed/$dataset/train_snapshot_undirected.csv" \
  --output-dir "$output/base_measurement" --k "$k" \
  --negative-conductance .1 --antagonistic-weight .05

"$python_bin" -m signed_epm.experiments.prepare_mitigation \
  --node-state "$base/node_embeddings.pt" \
  --graph "data/processed/$dataset/train_snapshot_undirected.csv" \
  --output-dir "$output/preparation/seed_0" \
  --communities "$k" --minimum-community-size 30

train_file="data/processed/$dataset/train_snapshot_undirected.csv"
train_edges=$(( $(wc -l < "$train_file") - 1 ))
maximum_budget=$(( train_edges * 5 / 100 ))
candidate_cap=$(( 20 * maximum_budget ))
selection="$output/selection/seed_0"

"$python_bin" -m signed_epm.experiments.canonical_candidate_pool \
  --data-dir "data/processed/$dataset" \
  --preparation "$output/preparation/seed_0" \
  --output-dir "$selection/candidates" \
  --candidate-cap "$candidate_cap" --seed 0

"$python_bin" -m signed_epm.experiments.pooled_greedy \
  --data-dir "data/processed/$dataset" \
  --coordinates "$output/base_measurement/opinion_coordinates.npy" \
  --candidate-file "$selection/candidates/candidates.csv" \
  --output-dir "$selection" --maximum-budget "$maximum_budget" \
  --eta .1 --alpha .05 --rates 1

"$python_bin" -m signed_epm.experiments.materialize_sampled_batch \
  --selection-root "$output/selection" \
  --data-dir "data/processed/$dataset" \
  --output-root "$output/graphs" \
  --selection-kind pooled --methods global --rates .05 --seeds 0

augmented_directed="$output/graphs/global/rate_5/seed_0/train_snapshot_directed_augmented.csv"
augmented_undirected="$output/graphs/global/rate_5/seed_0/train_snapshot_undirected_augmented.csv"
mitigated="$output/mitigated"

"$python_bin" -m signed_epm.training.train \
  --model sgcn --task signlink_3class \
  --data-dir "data/processed/$dataset" --train-path "$augmented_directed" \
  --output-dir "$mitigated" --cache-root "$output/cache/mitigated" \
  --input-dimension 64 --output-dimension 64 --layers 2 \
  --learning-rate .01 --epochs 200 --seed 0 --device "$device"

"$python_bin" -m signed_epm.polarization.measure \
  --node-state-path "$mitigated/node_embeddings.pt" \
  --edge-path "$augmented_undirected" \
  --output-dir "$output/mitigated_measurement" --k "$k" \
  --negative-conductance .1 --antagonistic-weight .05

echo "quick start complete: $output"
echo "base metrics: $base/metrics.json"
echo "base measurement: $output/base_measurement/measurement.json"
echo "mitigated metrics: $mitigated/metrics.json"
echo "mitigated measurement: $output/mitigated_measurement/measurement.json"
