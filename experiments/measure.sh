#!/usr/bin/env bash
set -euo pipefail
node_state="${1:?node-state path required}"
dataset="${2:-bitcoinalpha}"
output="${3:-artifacts/measurement/$dataset}"
python_bin="${PYTHON_BIN:-python}"
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
echo "dataset=$dataset k=$k"
signed-epm-measure --node-state-path "$node_state" \
  --edge-path "data/processed/$dataset/train_snapshot_undirected.csv" \
  --output-dir "$output" --k "$k" \
  --negative-conductance .1 --antagonistic-weight .05
