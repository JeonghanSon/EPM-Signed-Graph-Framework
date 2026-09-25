from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from signed_epm.mitigation.core import prepare_intervention
from signed_epm.polarization.measure import load_node_state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache communities, pair scores, and gray-node rankings."
    )
    parser.add_argument("--node-state", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--communities", type=int, required=True)
    parser.add_argument("--minimum-community-size", type=int, required=True)
    parser.add_argument("--kmeans-seed", type=int, default=42)
    parser.add_argument("--negative-conductance", type=float, default=0.1)
    parser.add_argument("--antagonistic-weight", type=float, default=0.05)
    parser.add_argument(
        "--top-pairs", type=int,
        help="Optional scalable pruning for large graphs; omit to retain every pair.",
    )
    args = parser.parse_args()

    summary = prepare_intervention(
        load_node_state(args.node_state),
        pd.read_csv(args.graph),
        args.communities,
        args.minimum_community_size,
        args.output_dir,
        args.kmeans_seed,
        args.negative_conductance,
        args.top_pairs,
        args.antagonistic_weight,
    )
    print(
        f"saved={args.output_dir} retained_communities="
        f"{summary['retained_communities']} pairs={summary['community_pairs']}"
    )


if __name__ == "__main__":
    main()
