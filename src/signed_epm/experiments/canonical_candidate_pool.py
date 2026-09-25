from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from signed_epm.graph import canonical_undirected
from signed_epm.mitigation.candidates import sample_unique_gray_union


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uniformly sample unique physical edges from the canonical gray-zone union",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-cap", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    graph = canonical_undirected(
        pd.read_csv(args.data_dir / "train_snapshot_undirected.csv"),
    )
    occupied = set(map(tuple, graph[["source", "target"]].astype(int).to_numpy()))
    candidates, pairs = sample_unique_gray_union(
        args.preparation, occupied, args.candidate_cap, args.seed,
    )
    sampling_attempts = candidates.attrs.get("sampling_attempts")
    acceptance_rate = candidates.attrs.get("acceptance_rate")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output_dir / "candidates.csv", index=False)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "schema_version": 1,
        "sampling": "uniform_unique_physical_union_rejection",
        "allocation": "none",
        "pair_filter": "all_retained_pairs",
        "candidate_cap_requested": args.candidate_cap,
        "candidate_count": len(candidates),
        "community_pair_count": len(pairs),
        "seed": args.seed,
        "sampling_attempts": sampling_attempts,
        "acceptance_rate": acceptance_rate,
        "pair_and_type_labels": "provenance_only",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
