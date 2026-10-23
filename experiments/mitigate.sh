#!/usr/bin/env bash
set -euo pipefail

model="${1:-sgcn}"
dataset="${2:-bitcoinalpha}"
seed="${3:-0}"
python_bin="${PYTHON_BIN:-python}"

read -r maximum_rate rates solver tolerance < <("$python_bin" - "$dataset" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path("configs/datasets.json").read_text(encoding="utf-8"))
paper = json.loads(Path("configs/paper/mitigation.json").read_text(encoding="utf-8"))
spec = config[sys.argv[1]]
if spec.get("intervention_profile") == "large":
    print("0.05 .025,.05 cg", paper["large_graph_solver_tolerance"])
else:
    print("0.10 .025,.05,.075,.10 direct 1e-6")
PY
)

root="artifacts/mitigation/$model/$dataset"
measurement="$root/measurement/seed_$seed"
preparation="$root/preparation/seed_$seed"
selection="$root/selection/seed_$seed"

bash experiments/measure.sh "$model" "$dataset" "$seed" "$measurement"
bash experiments/prepare_mitigation.sh "$model" "$dataset" "$seed" "$preparation"

train_path="data/processed/$dataset/train_snapshot_undirected.csv"
train_edges=$(( $(wc -l < "$train_path") - 1 ))
maximum_budget=$("$python_bin" - "$train_edges" "$maximum_rate" <<'PY'
import sys

print(int(int(sys.argv[1]) * float(sys.argv[2])))
PY
)

SOLVER="$solver" SOLVER_TOLERANCE="$tolerance" \
  bash experiments/select_mitigation.sh \
    "$dataset" "$preparation" "$measurement/opinion_coordinates.npy" \
    "$maximum_budget" "$seed" "$selection"

IFS=',' read -r -a rate_values <<< "$rates"
if [[ "$model" == "sdgnn" ]]; then
  SEEDS="$seed" DIRECTED_BACKBONE=1 bash experiments/materialize_mitigation.sh \
    "$root/selection" "$dataset" "$root/graphs" "${rate_values[@]}"
else
  SEEDS="$seed" bash experiments/materialize_mitigation.sh \
    "$root/selection" "$dataset" "$root/graphs" "${rate_values[@]}"
fi

echo "mitigation graphs: $root/graphs"
