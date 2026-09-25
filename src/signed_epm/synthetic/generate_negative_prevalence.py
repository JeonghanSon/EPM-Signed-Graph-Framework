from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from signed_epm.data.preprocess import connected
from signed_epm.graph import canonical_undirected, graph_fingerprint
from signed_epm.synthetic.generate import candidate_pools


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def generate(
    positive_root: Path,
    output_root: Path,
    graph_seeds: list[int],
    shares: list[float],
    inter_fraction: float,
    positive_level: int,
) -> None:
    if shares != sorted(set(shares)) or any(not 0.0 <= share < 1.0 for share in shares):
        raise ValueError("negative shares must be unique, sorted, and in [0, 1)")
    if not 0.0 <= inter_fraction <= 1.0:
        raise ValueError("inter fraction must be in [0, 1]")
    records = []
    for seed in graph_seeds:
        source = positive_root / "structural" / f"graph_seed_{seed}" / f"level_{positive_level}"
        source_graph = pd.read_csv(source / "train_snapshot_undirected.csv")
        positive = canonical_undirected(source_graph.loc[source_graph.sign > 0])
        nodes = pd.read_csv(source / "nodes.csv").sort_values("node_id")
        communities = nodes.community.to_numpy(dtype=np.int64)
        intra, inter = candidate_pools(communities)
        occupied = set(map(tuple, positive[["source", "target"]].to_numpy(dtype=np.int64)))
        available_intra = np.asarray([edge for edge in intra if tuple(edge) not in occupied])
        available_inter = np.asarray([edge for edge in inter if tuple(edge) not in occupied])
        rng = np.random.default_rng(510_007 + seed)
        available_intra = available_intra[rng.permutation(len(available_intra))]
        available_inter = available_inter[rng.permutation(len(available_inter))]
        positive_fingerprint = graph_fingerprint(positive, directed=False)

        for share in shares:
            negative_count = int(round(len(positive) * share / (1.0 - share)))
            inter_count = int(round(negative_count * inter_fraction))
            intra_count = negative_count - inter_count
            if intra_count > len(available_intra) or inter_count > len(available_inter):
                raise ValueError("negative edge request exceeds available nonedges")
            if negative_count:
                negative = np.vstack((available_intra[:intra_count], available_inter[:inter_count]))
            else:
                negative = np.empty((0, 2), dtype=np.int64)
            # Give every physical negative edge one deterministic seeded direction.
            swaps = rng.random(len(negative)) < 0.5
            directed_negative = negative.copy()
            directed_negative[swaps] = directed_negative[swaps][:, ::-1]
            positive_directed = positive[["source", "target"]].copy()
            positive_directed["sign"] = 1
            negative_directed = pd.DataFrame({
                "source": directed_negative[:, 0], "target": directed_negative[:, 1], "sign": -1,
            })
            graph = pd.concat([positive_directed, negative_directed], ignore_index=True)
            graph["weight"] = 1.0
            graph["event_id"] = np.arange(len(graph), dtype=np.int64)
            graph = graph.iloc[rng.permutation(len(graph))].reset_index(drop=True)
            undirected = canonical_undirected(graph)

            destination = output_root / "prevalence" / f"graph_seed_{seed}" / f"share_{token(share)}"
            destination.mkdir(parents=True, exist_ok=True)
            graph.to_csv(destination / "train_events.csv", index=False)
            graph.to_csv(destination / "train_snapshot_directed.csv", index=False)
            undirected.to_csv(destination / "train_snapshot_undirected.csv", index=False)
            graph.iloc[:0].to_csv(destination / "val_events.csv", index=False)
            graph.iloc[:0].to_csv(destination / "test_events.csv", index=False)
            nodes.to_csv(destination / "nodes.csv", index=False)
            manifest = {
                "schema_version": 1,
                "dataset": f"negative_prevalence_g{seed}_s{token(share)}",
                "synthetic": {
                    "experiment": "prevalence",
                    "graph_model": "fixed_positive_sbm_with_nested_negative_edges",
                    "graph_seed": seed,
                    "positive_source": str(source),
                    "positive_source_level": positive_level,
                    "positive_source_fingerprint": positive_fingerprint,
                    "target_negative_share": share,
                    "realized_negative_share": negative_count / (len(positive) + negative_count),
                    "target_inter_negative_fraction": inter_fraction,
                    "realized_inter_negative_fraction": (
                        inter_count / negative_count if negative_count else None
                    ),
                    "negative_sets_nested_within_seed": True,
                },
                "counts": {
                    "num_nodes": len(nodes),
                    "splits": {"train": len(graph), "val": 0, "test": 0},
                    "positive": {"train": len(positive), "val": 0, "test": 0},
                    "negative": {"train": negative_count, "val": 0, "test": 0},
                },
                "validation": {
                    "train_connected": connected(undirected, len(nodes)),
                    "positive_train_connected": connected(positive, len(nodes)),
                    "positive_graph_unchanged": graph_fingerprint(
                        undirected.loc[undirected.sign > 0], directed=False) == positive_fingerprint,
                },
                "communities": {"selected_k": 8, "source": "ground_truth"},
                "graph_fingerprint": graph_fingerprint(undirected, directed=False),
            }
            if not all(manifest["validation"].values()):
                raise RuntimeError(f"prevalence graph validation failed: {destination}")
            (destination / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            records.append({
                "experiment": "prevalence", "level": share, "token": token(share),
                "graph_seed": seed, "path": str(destination),
                "negative_share": manifest["synthetic"]["realized_negative_share"],
                "negative_edges": negative_count, "positive_edges": len(positive),
            })
            print(f"DONE seed={seed} negative_share={share:g}", flush=True)

    summary = {
        "schema_version": 1,
        "config": {
            "num_nodes": 1000, "num_communities": 8,
            "positive_source": str(positive_root), "positive_source_level": positive_level,
            "negative_shares": shares, "negative_inter_fraction": inter_fraction,
            "negative_sets_nested_within_seed": True,
        },
        "graphs": records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "generation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate retrained negative-prevalence controls")
    parser.add_argument("--positive-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--negative-shares", nargs="+", type=float,
                        default=[0.05, 0.10, 0.15, 0.20, 0.30])
    parser.add_argument("--inter-fraction", type=float, default=0.80)
    parser.add_argument("--positive-level", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output exists: {args.output_root} (use --overwrite)")
        shutil.rmtree(args.output_root)
    generate(args.positive_root, args.output_root, args.graph_seeds,
             args.negative_shares, args.inter_fraction, args.positive_level)


if __name__ == "__main__":
    main()
