#!/usr/bin/env bash
set -euo pipefail

model="${1:-sgcn}"
dataset="${2:-bitcoinalpha}"
task="${TASK:-signlink_3class}"

signed-epm-intervene evaluate-base \
  --model "$model" --dataset "$dataset" --task "$task" \
  --data-dir "data/processed/$dataset"
