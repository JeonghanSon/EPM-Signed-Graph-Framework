#!/usr/bin/env bash
set -euo pipefail
node_state="${1:?node-state path required}"
dataset="${2:-bitcoinalpha}"
output="${3:-artifacts/mitigation/preparation/$dataset}"
python_bin="${PYTHON_BIN:-python}"
read -r resolved_k resolved_minimum_size < <("$python_bin" - "$dataset" <<'PY'
import json
import sys
from pathlib import Path

dataset = sys.argv[1]
config = json.loads(Path("configs/datasets.json").read_text(encoding="utf-8"))
spec = config[dataset]
k = None
for path in (
    Path("data/processed") / dataset / "communities" / "signed_louvain_summary.json",
    Path("data/metadata/communities") / f"{dataset}.json",
):
    if path.exists():
        k = json.loads(path.read_text(encoding="utf-8")).get("selected_k")
        if k is not None:
            break
print(int(k if k is not None else spec["fallback_k"]),
      int(spec["minimum_community_size"]))
PY
)
k="${K:-$resolved_k}"
minimum_size="${MINIMUM_COMMUNITY_SIZE:-$resolved_minimum_size}"
echo "dataset=$dataset k=$k minimum_community_size=$minimum_size"
signed-epm-prepare --node-state "$node_state" \
  --graph "data/processed/$dataset/train_snapshot_undirected.csv" \
  --output-dir "$output" --communities "$k" \
  --minimum-community-size "$minimum_size"
