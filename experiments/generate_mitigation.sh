#!/usr/bin/env bash
set -euo pipefail

model="${1:-sgcn}"
dataset="${2:-bitcoinalpha}"
tau="${3:-0.5}"
max_degree="${4:-3}"
gamma="${5:-2.0}"
task="${TASK:-signlink_3class}"

signed-epm-intervene generate \
  --model "$model" --dataset "$dataset" --task "$task" \
  --data-dir "data/processed/$dataset" \
  --taus "$tau" --max-degrees "$max_degree" --gammas "$gamma" \
  --interventions epm
