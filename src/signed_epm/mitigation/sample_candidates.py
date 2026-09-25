from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from signed_epm.mitigation.greedy import (
    sampled_candidate_table,
    sampled_direct_candidate_table,
)
from signed_epm.graph import canonical_undirected


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a pair-stratified gray candidate pool")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-cap", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidate-space", choices=["gray", "direct"], default="gray")
    args = parser.parse_args()

    graph = canonical_undirected(pd.read_csv(args.data_dir / "train_snapshot_undirected.csv"))
    occupied = set(map(tuple, graph[["source", "target"]].astype(int).to_numpy()))
    if args.candidate_space == "direct":
        candidates, selected = sampled_direct_candidate_table(
            args.preparation, occupied, args.candidate_cap, args.seed,
        )
    else:
        candidates, selected = sampled_candidate_table(
            args.preparation, occupied, "global",
            args.candidate_cap, args.seed,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output_dir / "candidates.csv", index=False)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "community_pair_filter": "none",
        "candidate_cap_requested": args.candidate_cap,
        "candidate_count": len(candidates), "selected_pair_count": len(selected),
        "sampling": "uniform_pair_before_cartesian_materialization",
        "candidate_space": args.candidate_space,
        "seed": args.seed,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
