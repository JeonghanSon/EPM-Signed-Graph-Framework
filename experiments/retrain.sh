#!/usr/bin/env bash
set -euo pipefail

model="${1:-sgcn}"
dataset="${2:-bitcoinalpha}"
seed="${3:-0}"
device="${4:-cuda}"
python_bin="${PYTHON_BIN:-python}"
base_root="artifacts/runs/$model/$dataset/signlink_3class/base"
config_path="$base_root/selected_configuration.json"
graph_root="artifacts/mitigation/$model/$dataset/graphs/global"
output_root="artifacts/mitigation/$model/$dataset/retrained"

if [[ ! -f "$config_path" ]]; then
  echo "selected configuration not found: $config_path" >&2
  echo "run experiments/train_base.sh first" >&2
  exit 2
fi
if [[ ! -d "$graph_root" ]]; then
  echo "mitigation graphs not found: $graph_root" >&2
  echo "run experiments/mitigate.sh first" >&2
  exit 2
fi

read -r input_dimension output_dimension layers learning_rate epochs class_weight < <(
  "$python_bin" - "$config_path" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(config["input_dimension"], config["output_dimension"], config["layers"],
      config["learning_rate"], config["epochs"], config["class_weight"])
PY
)

found=0
for graph_dir in "$graph_root"/rate_*/seed_"$seed"; do
  [[ -d "$graph_dir" ]] || continue
  found=1
  rate=$(basename "$(dirname "$graph_dir")")
  output="$output_root/$rate/seed_$seed"
  train_path="$graph_dir/train_snapshot_directed_augmented.csv"
  edge_path="$graph_dir/train_snapshot_undirected_augmented.csv"
  class_weight_args=(--class-weight "$class_weight")

  if [[ -f "$output/metrics.json" && -f "$output/measurement/measurement.json" ]]; then
    echo "reuse completed result: $output"
    continue
  fi

  "$python_bin" -m signed_epm.training.train \
    --model "$model" --task signlink_3class \
    --data-dir "data/processed/$dataset" --train-path "$train_path" \
    --output-dir "$output" --cache-root "$output/cache" \
    --input-dimension "$input_dimension" --output-dimension "$output_dimension" \
    --layers "$layers" --learning-rate "$learning_rate" --epochs "$epochs" \
    "${class_weight_args[@]}" --seed "$seed" --device "$device" --overwrite

  NODE_STATE_PATH="$output/node_embeddings.pt" EDGE_PATH="$edge_path" \
    bash experiments/measure.sh "$model" "$dataset" "$seed" "$output/measurement"
done

if [[ "$found" -eq 0 ]]; then
  echo "no augmented graphs found for seed $seed under $graph_root" >&2
  exit 2
fi

echo "retrained results: $output_root"
