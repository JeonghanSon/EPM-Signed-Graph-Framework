#!/usr/bin/env bash
set -euo pipefail

model="${1:-sgcn}"
dataset="${2:-bitcoinalpha}"
seed="${3:-0}"
python_bin="${PYTHON_BIN:-python}"
base_root="artifacts/runs/$model/$dataset/signlink_3class/base"
selected_runs="$base_root/selected_runs.csv"

if [[ ! -f "$selected_runs" ]]; then
  echo "selected runs not found: $selected_runs" >&2
  echo "run experiments/train_base.sh first" >&2
  exit 2
fi

node_state=$("$python_bin" - "$selected_runs" "$seed" <<'PY'
import sys
from pathlib import Path

import pandas as pd

selected_runs = Path(sys.argv[1])
seed = int(sys.argv[2])
frame = pd.read_csv(selected_runs)
row = frame.loc[frame["seed"].astype(int) == seed]
if len(row) != 1:
    raise SystemExit(f"expected one selected run for seed {seed}, found {len(row)}")
print(Path(str(row.iloc[0]["run_dir"])) / "node_embeddings.pt")
PY
)

if [[ ! -f "$node_state" ]]; then
  echo "selected node state not found: $node_state" >&2
  exit 2
fi

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

bash experiments/measure.sh "$node_state" "$dataset" "$measurement"
bash experiments/prepare_mitigation.sh "$node_state" "$dataset" "$preparation"

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
