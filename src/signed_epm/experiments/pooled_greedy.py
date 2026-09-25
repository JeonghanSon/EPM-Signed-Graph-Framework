from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from signed_epm.graph import canonical_undirected, graph_fingerprint
from signed_epm.mitigation.pooled_greedy import (
    pooled_greedy_order,
    recompute_prefix_energies,
)
from signed_epm.polarization.measure import (
    build_weighted_laplacian,
    negative_edge_laplacian,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select one canonical pooled-greedy physical-edge order",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-budget", type=int, required=True)
    parser.add_argument("--rates", nargs="+", type=float,
                        default=[.025, .05, .075, .10])
    parser.add_argument("--eta", type=float, default=.1)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--solver", choices=["direct", "cg"], default="direct")
    parser.add_argument("--solver-tolerance", type=float, default=1e-6)
    parser.add_argument("--solver-maximum-iterations", type=int, default=10_000)
    parser.add_argument("--projection-dimension", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=2026)
    args = parser.parse_args()

    if args.maximum_budget < 1:
        raise ValueError("maximum budget must be positive")
    if any(rate <= 0 or rate > 1 for rate in args.rates):
        raise ValueError("rates must lie in (0, 1]")
    graph_path = args.data_dir / "train_snapshot_undirected.csv"
    graph = canonical_undirected(pd.read_csv(graph_path))
    coordinates = np.load(args.coordinates)
    candidates = pd.read_csv(args.candidate_file)
    if len(candidates) < args.maximum_budget:
        raise ValueError(
            f"candidate pool {len(candidates)} is smaller than budget {args.maximum_budget}"
        )
    laplacian = build_weighted_laplacian(graph, len(coordinates), 1.0, args.eta)
    antagonistic = negative_edge_laplacian(graph, len(coordinates)) @ coordinates
    checkpoints = sorted(set(
        max(1, int(np.floor(args.maximum_budget * rate))) for rate in args.rates
    ))
    order, approximate, initial = pooled_greedy_order(
        laplacian, coordinates, antagonistic, candidates, args.maximum_budget,
        args.alpha, args.solver, args.solver_tolerance,
        args.solver_maximum_iterations, args.projection_dimension,
        args.projection_seed,
    )
    selected = candidates.iloc[order].copy()
    selected.insert(0, "selection_step", np.arange(1, len(selected) + 1))
    exact = recompute_prefix_energies(
        laplacian, coordinates, antagonistic, candidates, order, checkpoints,
        args.alpha,
    )
    rows = []
    for checkpoint in checkpoints:
        value = exact[checkpoint]
        rows.append({
            "budget": checkpoint,
            "maximum_budget_fraction": checkpoint / args.maximum_budget,
            "approximate_energy": approximate[checkpoint],
            "recomputed_energy": value,
            "polarization": float(np.sqrt(max(value, 0.0))),
            "polarization_reduction_pct": float(
                100 * (1 - np.sqrt(max(value, 0.0) / initial)
            )),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_dir / "selected_edges.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output_dir / "budget_curve.csv", index=False)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "schema_version": 1,
        "method": "pooled_gray_greedy",
        "allocation": "global_unconstrained",
        "pair_scores": False,
        "candidate_file": str(args.candidate_file),
        "candidate_count": len(candidates),
        "maximum_budget": args.maximum_budget,
        "rates": args.rates,
        "eta": args.eta,
        "alpha": args.alpha,
        "solver": args.solver,
        "solver_tolerance": args.solver_tolerance,
        "solver_maximum_iterations": args.solver_maximum_iterations,
        "resistance_projection_dimension": args.projection_dimension,
        "resistance_projection_seed": args.projection_seed,
        "graph_fingerprint": graph_fingerprint(graph, directed=False),
        "representation": "fixed_during_selection",
        "checkpoint_energy": "explicit_augmented_laplacian_recomputation",
        "validation_or_test_information": False,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
