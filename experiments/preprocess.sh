#!/usr/bin/env bash
set -euo pipefail
dataset="${1:-bitcoinalpha}"
community_mode="${2:-compute}"
signed-epm-preprocess --dataset "$dataset" --seed 0
if [[ "$community_mode" == "fallback" ]]; then
  signed-epm-communities --dataset "$dataset" --fallback
else
  signed-epm-communities --dataset "$dataset"
fi
