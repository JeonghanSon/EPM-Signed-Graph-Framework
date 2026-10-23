#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-bitcoinalpha}"
model="${2:-sgcn}"
device="${3:-cuda}"
if (( $# > 3 )); then
  shift 3
  seeds=("$@")
else
  seeds=(0 1 2 3 4)
fi

echo "full experiment: dataset=$dataset model=$model device=$device seeds=${seeds[*]}"
bash experiments/preprocess.sh "$dataset"
FINAL_SEEDS="${seeds[*]}" bash experiments/train_base.sh "$model" "$dataset" "$device"

for seed in "${seeds[@]}"; do
  bash experiments/measure.sh "$model" "$dataset" "$seed"
  bash experiments/mitigate.sh "$model" "$dataset" "$seed"
  bash experiments/retrain.sh "$model" "$dataset" "$seed" "$device"
done

echo "full experiment complete: artifacts/base/$model/$dataset"
echo "mitigation results: artifacts/mitigation/$model/$dataset"
