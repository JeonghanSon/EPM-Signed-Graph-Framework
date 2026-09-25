from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from signed_epm.experiments.materialize_budget_strategies import uniform_nonedge_order
from signed_epm.mitigation.materialize import write_augmented_graph


def rate_token(rate: float) -> str:
    return f"{100 * rate:g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize matched uniform-random nonedges")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rates", nargs="+", type=float, default=[.025, .05, .075])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--directed-backbone", action="store_true",
        help="encode each physical addition in both directions (required for SDGNN)",
    )
    args = parser.parse_args()

    directed = pd.read_csv(args.data_dir / "train_snapshot_directed.csv")
    undirected = pd.read_csv(args.data_dir / "train_snapshot_undirected.csv")
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    train_edges = len(undirected)
    occupied = set(map(tuple, undirected[["source", "target"]].astype(int).to_numpy()))
    counts = manifest.get("counts", manifest)
    num_nodes = int(counts.get("num_nodes", manifest.get("num_nodes")))
    maximum_budget = max(max(1, int(np.floor(rate * train_edges))) for rate in args.rates)
    for seed in args.seeds:
        order = uniform_nonedge_order(num_nodes, occupied, maximum_budget, seed)
        for rate in args.rates:
            budget = max(1, int(np.floor(rate * train_edges)))
            output = args.output_root / f"rate_{rate_token(rate)}" / f"seed_{seed}"
            physical, encoded = write_augmented_graph(
                order[:budget], directed, undirected, output,
                directed_backbone=args.directed_backbone,
            )
            (output / "summary.json").write_text(json.dumps({
                "method": "global_random", "seed": seed,
                "budget_definition": "floor(rate * physical_training_edges)",
                "budget_rate": rate, "train_physical_edges": train_edges,
                "physical_edges_added": len(physical), "model_edges_added": len(encoded),
                "sampling": "uniform_without_replacement_from_all_training_nonedges",
                "candidate_information": "training_graph_only",
                "new_edge_sign": 1, "new_edge_weight": 1.0,
                "directed_backbone": args.directed_backbone,
            }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
