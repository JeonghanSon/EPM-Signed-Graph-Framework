from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from signed_epm.experiments.nonedge_sampling import uniform_nonedge_order
from signed_epm.graph import canonical_undirected
from signed_epm.mitigation.pooled_greedy import (
    pooled_greedy_order,
    recompute_prefix_energies,
)
from signed_epm.polarization.measure import build_weighted_laplacian, negative_edge_laplacian


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zhu-style global greedy over sampled training non-edges"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-budget", type=int, required=True)
    parser.add_argument("--candidate-cap", type=int, required=True)
    parser.add_argument("--fractions", nargs="+", type=float,
                        default=[.25, .5, .75, 1.0])
    parser.add_argument("--eta", type=float, default=.1)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--solver", choices=["direct", "cg"], default="direct")
    parser.add_argument("--solver-tolerance", type=float, default=1e-6)
    parser.add_argument("--solver-maximum-iterations", type=int, default=10_000)
    args = parser.parse_args()

    graph = pd.read_csv(args.data_dir / "train_snapshot_undirected.csv")
    physical = canonical_undirected(graph)
    occupied = set(map(tuple, physical[["source", "target"]].astype(int).to_numpy()))
    coordinates = np.load(args.coordinates)
    candidate_edges = uniform_nonedge_order(
        len(coordinates), occupied, args.candidate_cap, args.seed,
    )
    candidates = pd.DataFrame(candidate_edges, columns=["source", "target"])
    # Dummy pair labels retain compatibility with the shared global-greedy engine;
    # no community information is used by this baseline.
    candidates["community_1"] = -1
    candidates["community_2"] = -1
    candidates["edge_type"] = "global_nonedge"

    laplacian = build_weighted_laplacian(
        physical, len(coordinates), 1.0, args.eta,
    )
    negative_laplacian = negative_edge_laplacian(physical, len(coordinates))
    antagonistic = negative_laplacian @ coordinates
    checkpoints = sorted(set(
        max(1, int(np.floor(args.maximum_budget * fraction)))
        for fraction in args.fractions
    ))
    order, approximate, initial = pooled_greedy_order(
        laplacian, coordinates, antagonistic, candidates,
        args.maximum_budget, args.alpha,
        solver=args.solver, solver_tolerance=args.solver_tolerance,
        solver_maximum_iterations=args.solver_maximum_iterations,
    )
    selected = candidates.iloc[order].copy()
    selected.insert(0, "selection_step", np.arange(1, len(selected) + 1))
    exact = recompute_prefix_energies(
        laplacian, coordinates, antagonistic, candidates, order,
        checkpoints, args.alpha,
    )
    rows = []
    for budget in checkpoints:
        actual = min(budget, len(order))
        energy = exact[budget]
        rows.append({
            "method": "global", "budget": actual,
            "budget_fraction": actual / args.maximum_budget,
            "approximate_energy": approximate[actual], "energy": energy,
            "polarization": np.sqrt(max(energy, 0.0)),
            "reduction_pct": 100 * (1 - np.sqrt(max(energy, 0.0) / initial)),
            "candidate_count": len(candidates),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output_dir / "candidates_global.csv", index=False)
    selected.to_csv(args.output_dir / "selected_global_global.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output_dir / "budget_curves.csv", index=False)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "method": "zhu_style_global_greedy",
        "candidate_space": "all_unordered_training_nonedges",
        "candidate_sampling": "uniform_without_replacement",
        "candidate_cap": args.candidate_cap,
        "maximum_budget": args.maximum_budget,
        "eta": args.eta, "alpha": args.alpha, "seed": args.seed,
        "representation": "frozen_during_selection",
        "greedy_implementation": "JL-128 initial resistance plus rank-one updates",
        "checkpoint_objective": "recomputed on each materialized augmented Laplacian",
        "community_pair_information": False,
        "validation_or_test_information": False,
        "solver": args.solver,
        "solver_tolerance": args.solver_tolerance,
        "solver_maximum_iterations": args.solver_maximum_iterations,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
