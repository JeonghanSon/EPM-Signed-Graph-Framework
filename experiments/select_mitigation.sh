#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-bitcoinalpha}"
preparation="${2:?preparation directory is required}"
coordinates="${3:?opinion_coordinates.npy is required}"
maximum_budget="${4:?maximum physical-edge budget is required}"
seed="${5:-0}"
output_dir="${6:-artifacts/mitigation/selection}"
candidate_cap=$((20 * maximum_budget))

signed-epm-select \
  --data-dir "data/processed/$dataset" \
  --preparation "$preparation" \
  --coordinates "$coordinates" \
  --maximum-budget "$maximum_budget" \
  --output-dir "$output_dir" \
  --candidate-cap "$candidate_cap" \
  --candidate-sampling uniform_pair \
  --random-seed "$seed" \
  --gray-policies global \
  --greedy-modes global \
  --skip-gray-random \
  --alpha 0.05 --eta 0.1 \
  --solver cg --solver-tolerance 0.001 \
  --fractions 0.25 0.5 0.75 1.0
