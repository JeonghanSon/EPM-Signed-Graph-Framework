#!/usr/bin/env bash
set -euo pipefail
dataset="${1:-bitcoinalpha}"
preparation="${2:?preparation directory required}"
coordinates="${3:?opinion_coordinates.npy required}"
maximum_budget="${4:?maximum physical-edge budget required}"
seed="${5:-0}"
output="${6:-artifacts/mitigation/seed_${seed}}"
candidate_cap=$((20 * maximum_budget))

signed-epm-sample-candidates \
  --data-dir "data/processed/$dataset" --preparation "$preparation" \
  --output-dir "$output/candidates" --candidate-cap "$candidate_cap" --seed "$seed"

signed-epm-select \
  --data-dir "data/processed/$dataset" --coordinates "$coordinates" \
  --candidate-file "$output/candidates/candidates.csv" \
  --output-dir "$output" --maximum-budget "$maximum_budget" \
  --eta .1 --alpha .05 --rates .25 .5 .75 1
