from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from signed_epm.mitigation.prepare import prepare_intervention
from signed_epm.polarization.measure import load_node_state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare canonical score-free gray-zone rankings",
    )
    parser.add_argument("--node-state", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--communities", type=int, required=True)
    parser.add_argument("--minimum-community-size", type=int, required=True)
    parser.add_argument("--kmeans-seed", type=int, default=42)
    parser.add_argument(
        "--gray-coordinate-normalization", choices=["l2", "none"], default="l2",
    )
    args = parser.parse_args()
    summary = prepare_intervention(
        load_node_state(args.node_state), pd.read_csv(args.graph),
        args.communities, args.minimum_community_size, args.output_dir,
        kmeans_seed=args.kmeans_seed,
        gray_coordinate_normalization=args.gray_coordinate_normalization,
    )
    print(
        f"saved={args.output_dir} communities={summary['retained_communities']} "
        f"pairs={summary['community_pairs']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
