#!/usr/bin/env bash
set -euo pipefail
model="${1:-sgcn}"
dataset="${2:-bitcoinalpha}"
device="${3:-cuda}"
python_bin="${PYTHON_BIN:-python}"
tuning_root="artifacts/runs/$model/$dataset/signlink_3class/base"
base_root="artifacts/base/$model/$dataset"

if [[ -f "$tuning_root/selected_configuration.json" && \
      -f "$tuning_root/selected_runs.csv" ]]; then
  echo "reuse selected configuration: $tuning_root"
else
  signed-epm-tune --model "$model" --task signlink_3class \
    --data-dir "data/processed/$dataset" --output-root "$tuning_root" \
    --device "$device" --prune
fi

read -r input_dimension output_dimension layers learning_rate epochs class_weight < <(
  "$python_bin" - "$tuning_root/selected_configuration.json" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(config["input_dimension"], config["output_dimension"], config["layers"],
      config["learning_rate"], config["epochs"], config["class_weight"])
PY
)

read -r -a final_seeds <<< "${FINAL_SEEDS:-0 1 2 3 4}"
for seed in "${final_seeds[@]}"; do
  output="$base_root/seed_$seed"
  if [[ -f "$output/metrics.json" && -f "$output/node_embeddings.pt" ]]; then
    echo "reuse completed base run: $output"
    continue
  fi
  "$python_bin" -m signed_epm.training.train \
    --model "$model" --task signlink_3class \
    --data-dir "data/processed/$dataset" \
    --output-dir "$output" --cache-root "$output/cache" \
    --input-dimension "$input_dimension" --output-dimension "$output_dimension" \
    --layers "$layers" --learning-rate "$learning_rate" --epochs "$epochs" \
    --class-weight "$class_weight" --seed "$seed" --device "$device" --overwrite
done

echo "base results: $base_root"
