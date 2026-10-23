#!/usr/bin/env bash
set -euo pipefail
model="${1:-sgcn}"
dataset="${2:-bitcoinalpha}"
seed="${3:-0}"
output="${4:-artifacts/measurement/$model/$dataset/seed_$seed}"
python_bin="${PYTHON_BIN:-python}"
edge_path="${EDGE_PATH:-data/processed/$dataset/train_snapshot_undirected.csv}"
node_state="${NODE_STATE_PATH:-}"
if [[ -z "$node_state" ]]; then
  node_state=$("$python_bin" experiments/resolve_selected_run.py \
    "$model" "$dataset" "$seed")
fi
k="${K:-}"
if [[ -z "$k" ]]; then
  k=$("$python_bin" - "$dataset" <<'PY'
import json
import sys
from pathlib import Path

dataset = sys.argv[1]
candidates = [
    Path("data/processed") / dataset / "communities" / "signed_louvain_summary.json",
    Path("data/metadata/communities") / f"{dataset}.json",
]
for path in candidates:
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8")).get("selected_k")
        if value is not None:
            print(int(value))
            break
else:
    config = json.loads(Path("configs/datasets.json").read_text(encoding="utf-8"))
    print(int(config[dataset]["fallback_k"]))
PY
  )
fi
echo "model=$model dataset=$dataset seed=$seed k=$k"
signed-epm-measure --node-state-path "$node_state" \
  --edge-path "$edge_path" \
  --output-dir "$output" --k "$k" \
  --negative-conductance .1 --antagonistic-weight .05
