from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from signed_epm.mitigation.materialize import write_augmented_graph


def rate_token(rate: float) -> str:
    return f"{100 * rate:g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize sampled greedy prefixes")
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--selection-kind", choices=["pooled", "batch", "sequential"],
        default="pooled",
        help=("pooled reads the canonical selected_edges.csv; batch/sequential "
              "are retained for approximation and legacy-result audits"),
    )
    parser.add_argument("--methods", nargs="+", choices=["global", "equal"],
                        default=["global"])
    parser.add_argument("--rates", nargs="+", type=float, default=[.025, .05, .075])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--candidate-sampling-label", default="uniform_pair",
                        help="provenance label written to each graph summary")
    parser.add_argument(
        "--directed-backbone", action="store_true",
        help="encode each undirected physical addition in both directions (required for SDGNN)",
    )
    args = parser.parse_args()
    if args.selection_kind == "pooled" and args.methods != ["global"]:
        raise ValueError("canonical pooled materialization requires --methods global")

    directed = pd.read_csv(args.data_dir / "train_snapshot_directed.csv")
    undirected = pd.read_csv(args.data_dir / "train_snapshot_undirected.csv")
    train_edges = len(undirected)
    for seed in args.seeds:
        seed_root = args.selection_root / f"seed_{seed}"
        for method in args.methods:
            if args.selection_kind == "pooled":
                selection_file = seed_root / "selected_edges.csv"
            elif args.selection_kind == "batch":
                selection_file = (seed_root / "batch" /
                                  f"selected_{method}_batch{args.batch_size}.csv")
            else:
                suffix = "global" if method == "global" else "pair_budget"
                selection_file = seed_root / f"selected_global_{suffix}.csv"
            selected = pd.read_csv(selection_file)
            for rate in args.rates:
                budget = max(1, int(np.floor(rate * train_edges)))
                chosen = selected.head(budget)
                if len(chosen) != budget:
                    raise RuntimeError(
                        f"{method} seed={seed}: requested {budget}, available {len(chosen)}"
                    )
                output = args.output_root / method / f"rate_{rate_token(rate)}" / f"seed_{seed}"
                physical, encoded = write_augmented_graph(
                    list(map(tuple, chosen[["source", "target"]].astype(int).to_numpy())),
                    directed, undirected, output,
                    directed_backbone=args.directed_backbone,
                )
                (output / "summary.json").write_text(json.dumps({
                    "method": f"sampled_{args.selection_kind}_{method}", "seed": seed,
                    "budget_definition": "floor(rate * physical_training_edges)",
                    "budget_rate": rate, "train_physical_edges": train_edges,
                    "physical_edges_added": len(physical),
                    "model_edges_added": len(encoded),
                    "batch_size": args.batch_size if args.selection_kind == "batch" else None,
                    "candidate_sampling": args.candidate_sampling_label,
                    "selection_kind": args.selection_kind,
                    "new_edge_sign": 1, "new_edge_weight": 1.0,
                    "directed_backbone": args.directed_backbone,
                }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
