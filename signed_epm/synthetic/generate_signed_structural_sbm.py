from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from signed_epm.data.preprocess import connected
from signed_epm.graph import graph_fingerprint
from signed_epm.synthetic.generate import (
    ROOT,
    balanced_communities,
    candidate_pools,
    edge_frame,
    visualization_nodes,
)
from signed_epm.synthetic.generate_signed_sbm import negative_probabilities
from signed_epm.synthetic.generate_unsigned_sbm import (
    P_OUT_LEVELS,
    SBMConfig,
    hohmann_opinion,
)


INTER_NEGATIVE_LEVELS = (0.50, 0.70, 0.85, 0.95, 0.99)
# Exact five probability pairs displayed in Hohmann et al.'s Figure 4.  Their
# full sweep also contains (p_in=0.039, p_out=0.0042), which is not one of the
# five network snapshots used here.
HOHMANN_P_IN_LEVELS = (0.0085, 0.054, 0.062, 0.064, 0.067)


def _token(level: int) -> str:
    return f"level_{level}"


def _write_graph(
    destination: Path,
    graph: pd.DataFrame,
    nodes: pd.DataFrame,
    config: SBMConfig,
    graph_seed: int,
    level: int,
    p_in: float,
    p_out: float,
    negative_inter_fraction: float,
    q_negative_in: float,
    q_negative_out: float,
    expected_negative_edges: float,
    negative_share: float,
    attempt: int,
    exact_positive_edges: int | None = None,
    target_positive_intra_fraction: float | None = None,
) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    graph.to_csv(destination / "train_events.csv", index=False)
    graph.to_csv(destination / "train_snapshot_directed.csv", index=False)
    undirected = graph.copy()
    undirected["source"] = np.minimum(graph.source, graph.target)
    undirected["target"] = np.maximum(graph.source, graph.target)
    undirected = undirected.sort_values(["source", "target"])
    undirected.to_csv(destination / "train_snapshot_undirected.csv", index=False)
    graph.iloc[:0].to_csv(destination / "val_events.csv", index=False)
    graph.iloc[:0].to_csv(destination / "test_events.csv", index=False)
    nodes.to_csv(destination / "nodes.csv", index=False)

    positive = undirected[undirected.sign > 0]
    negative = undirected[undirected.sign < 0]
    communities = nodes.set_index("node_id").community
    negative_inter = int(sum(
        communities.loc[int(source)] != communities.loc[int(target)]
        for source, target in negative[["source", "target"]].itertuples(index=False)
    ))
    realized_negative_share = len(negative) / len(undirected)
    manifest = {
        "schema_version": 1,
        "dataset": f"signed_structural_sbm_g{graph_seed}_l{level}",
        "synthetic": {
            **asdict(config),
            "experiment": "signed_structural",
            "graph_model": "two_layer_signed_sbm",
            "graph_seed": graph_seed,
            "level": level,
            "generation_attempt": attempt,
            "p_positive_in": p_in,
            "p_positive_out": p_out,
            "expected_negative_share": negative_share,
            "expected_negative_edges": expected_negative_edges,
            "expected_inter_negative_fraction": negative_inter_fraction,
            "q_negative_in": q_negative_in,
            "q_negative_out": q_negative_out,
            "realized_positive_edges": len(positive),
            "realized_negative_edges": len(negative),
            "realized_negative_share": realized_negative_share,
            "realized_inter_negative_edges": negative_inter,
            "realized_inter_negative_fraction": negative_inter / len(negative),
            "edge_budget_mode": "exact" if exact_positive_edges is not None else "bernoulli_sbm",
            "target_positive_edges": exact_positive_edges,
            "target_positive_intra_fraction": target_positive_intra_fraction,
        },
        "counts": {
            "num_nodes": config.num_nodes,
            "splits": {"train": len(undirected), "val": 0, "test": 0},
            "positive": {"train": len(positive), "val": 0, "test": 0},
            "negative": {"train": len(negative), "val": 0, "test": 0},
        },
        "validation": {
            "train_connected": connected(undirected, config.num_nodes),
            "positive_train_connected": connected(positive, config.num_nodes),
            "sign_layers_disjoint": len(undirected[["source", "target"]].drop_duplicates())
            == len(undirected),
        },
        "communities": {"selected_k": config.num_communities, "source": "ground_truth"},
        "graph_fingerprint": graph_fingerprint(undirected, directed=False),
        "positive_graph_fingerprint": graph_fingerprint(positive, directed=False),
    }
    if not all(manifest["validation"].values()):
        raise RuntimeError(f"signed structural SBM validation failed: {manifest['validation']}")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def generate_seed(
    output_root: Path,
    graph_seed: int,
    config: SBMConfig,
    negative_share: float,
    negative_inter_levels: tuple[float, ...],
    max_attempts: int,
    positive_edges: int | None = None,
    positive_intra_levels: tuple[float, ...] | None = None,
) -> list[dict]:
    if not 0.0 < negative_share < 1.0:
        raise ValueError("negative_share must be strictly between zero and one")
    communities = balanced_communities(config.num_nodes, config.num_communities)
    intra, inter = candidate_pools(communities)
    nodes = visualization_nodes(communities, 30_000 + graph_seed)
    exact_mode = positive_edges is not None
    if exact_mode:
        if positive_intra_levels is None:
            raise ValueError("positive-intra-levels are required with positive-edges")
        if len(positive_intra_levels) != len(negative_inter_levels):
            raise ValueError("positive and negative level counts must match")
        negative_edges = int(round(positive_edges * negative_share / (1.0 - negative_share)))
    else:
        probabilities = list(zip(HOHMANN_P_IN_LEVELS, P_OUT_LEVELS))

    for attempt in range(max_attempts):
        rng = np.random.default_rng(graph_seed * 1009 + attempt)
        positive_in_uniform = rng.random(len(intra))
        positive_out_uniform = rng.random(len(inter))
        negative_in_uniform = rng.random(len(intra))
        negative_out_uniform = rng.random(len(inter))
        intra_order = np.argsort(positive_in_uniform)
        inter_order = np.argsort(positive_out_uniform)
        sequence = []
        level_controls = (zip(positive_intra_levels, negative_inter_levels)
                          if exact_mode else zip(probabilities, negative_inter_levels))
        for level, (positive_control, negative_inter_fraction) in enumerate(
            level_controls, start=1,
        ):
            if exact_mode:
                positive_in_count = int(round(positive_edges * positive_control))
                positive_out_count = positive_edges - positive_in_count
                positive_in = np.zeros(len(intra), dtype=bool)
                positive_out = np.zeros(len(inter), dtype=bool)
                positive_in[intra_order[:positive_in_count]] = True
                positive_out[inter_order[:positive_out_count]] = True
                p_in = positive_in_count / len(intra)
                p_out = positive_out_count / len(inter)
            else:
                p_in, p_out = positive_control
                positive_in = positive_in_uniform < p_in
                positive_out = positive_out_uniform < p_out
            positive = np.vstack([intra[positive_in], inter[positive_out]])
            positive_frame = edge_frame(
                positive, np.empty((0, 2), dtype=np.int64), graph_seed * 100_000 + level,
            )
            if not connected(positive_frame, config.num_nodes):
                break

            available_in = ~positive_in
            available_out = ~positive_out
            expected_positive = len(positive) if exact_mode else p_in * len(intra) + p_out * len(inter)
            expected_negative = (negative_edges if exact_mode else
                                 expected_positive * negative_share / (1.0 - negative_share))
            if exact_mode:
                negative_inter_count = int(round(negative_edges * negative_inter_fraction))
                negative_in_count = negative_edges - negative_inter_count
                available_in_order = np.argsort(np.where(available_in, negative_in_uniform, np.inf))
                available_out_order = np.argsort(np.where(available_out, negative_out_uniform, np.inf))
                chosen_in = available_in_order[:negative_in_count]
                chosen_out = available_out_order[:negative_inter_count]
                negative = np.vstack([intra[chosen_in], inter[chosen_out]])
                q_in = negative_in_count / int(available_in.sum())
                q_out = negative_inter_count / int(available_out.sum())
            else:
                q_in, q_out = negative_probabilities(
                    int(available_in.sum()), int(available_out.sum()),
                    expected_negative, negative_inter_fraction,
                )
                negative = np.vstack([
                    intra[available_in & (negative_in_uniform < q_in)],
                    inter[available_out & (negative_out_uniform < q_out)],
                ])
            graph = edge_frame(positive, negative, graph_seed * 200_000 + level)
            sequence.append((
                level, p_in, p_out, negative_inter_fraction, q_in, q_out,
                expected_negative, graph,
                float(positive_control) if exact_mode else None,
            ))
        if len(sequence) == len(P_OUT_LEVELS):
            break
    else:
        raise RuntimeError(f"could not generate a connected signed SBM sequence for seed {graph_seed}")

    records = []
    for (level, p_in, p_out, fraction, q_in, q_out, expected_negative,
         graph, target_positive_intra) in sequence:
        destination = output_root / "structural" / f"graph_seed_{graph_seed}" / _token(level)
        manifest = _write_graph(
            destination, graph, nodes, config, graph_seed, level, p_in, p_out,
            fraction, q_in, q_out, expected_negative, negative_share, attempt,
            positive_edges if exact_mode else None, target_positive_intra,
        )
        synthetic = manifest["synthetic"]
        records.append({
            "experiment": "signed_structural",
            "level": level,
            "token": _token(level),
            "graph_seed": graph_seed,
            "path": str(destination),
            "p_positive_in": p_in,
            "p_positive_out": p_out,
            "expected_inter_negative_fraction": fraction,
            "realized_negative_share": synthetic["realized_negative_share"],
            "realized_inter_negative_fraction": synthetic["realized_inter_negative_fraction"],
            "graph_fingerprint": manifest["graph_fingerprint"],
        })
    return records


def _write_opinions(output_root: Path, config: SBMConfig,
                    opinion_seeds: list[int]) -> tuple[list[dict], list[dict]]:
    opinion_root = output_root / "opinions"
    opinion_root.mkdir(parents=True, exist_ok=True)
    aligned_records, random_records = [], []
    for seed in opinion_seeds:
        aligned = hohmann_opinion(config.num_nodes, seed)
        permutation = np.random.default_rng(80_000 + seed).permutation(config.num_nodes)
        random_values = aligned[permutation]
        for name, values, records in (
            ("hohmann_normal_aligned", aligned, aligned_records),
            ("hohmann_normal_random", random_values, random_records),
        ):
            path = opinion_root / f"{name}_seed_{seed}.csv"
            pd.DataFrame({"node_id": np.arange(config.num_nodes), "opinion": values}).to_csv(
                path, index=False,
            )
            records.append({
                "distribution": name,
                "seed": seed,
                "path": str(path),
                "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                "same_multiset_as_aligned": True,
                "permutation_fixed_across_levels": True,
            })
    return aligned_records, random_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the main signed structural SBM panel")
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "data" / "synthetic" / "signed_structural_main_n1000_k8",
    )
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--opinion-seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--negative-share", type=float, default=0.15)
    parser.add_argument("--positive-edges", type=int)
    parser.add_argument("--positive-intra-levels", nargs="+", type=float)
    parser.add_argument(
        "--negative-inter-levels", nargs="+", type=float,
        default=list(INTER_NEGATIVE_LEVELS),
        help="expected inter-community fractions for the five negative layers",
    )
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output exists: {args.output_root} (use --overwrite)")
        shutil.rmtree(args.output_root)

    config = SBMConfig()
    negative_inter_levels = tuple(float(value) for value in args.negative_inter_levels)
    if len(negative_inter_levels) != len(HOHMANN_P_IN_LEVELS):
        raise ValueError("negative-inter-levels must contain exactly five values")
    if (tuple(sorted(negative_inter_levels)) != negative_inter_levels or
            any(value < 0.0 or value > 1.0 for value in negative_inter_levels)):
        raise ValueError("negative-inter-levels must be sorted values in [0, 1]")
    positive_intra_levels = (None if args.positive_intra_levels is None else
                             tuple(float(value) for value in args.positive_intra_levels))
    if positive_intra_levels is not None:
        if args.positive_edges is None:
            raise ValueError("positive-edges is required with positive-intra-levels")
        if (len(positive_intra_levels) != len(negative_inter_levels) or
                tuple(sorted(positive_intra_levels)) != positive_intra_levels or
                any(value < 0.0 or value > 1.0 for value in positive_intra_levels)):
            raise ValueError("positive-intra-levels must be sorted and match negative levels")
    graphs = []
    for seed in args.graph_seeds:
        graphs.extend(generate_seed(
            args.output_root, seed, config, args.negative_share,
            negative_inter_levels, args.max_attempts,
            args.positive_edges, positive_intra_levels,
        ))
    aligned, random_records = _write_opinions(args.output_root, config, args.opinion_seeds)
    summary = {
        "schema_version": 1,
        "config": {
            **asdict(config),
            "negative_share": args.negative_share,
            "positive_p_out_levels": (list(P_OUT_LEVELS) if positive_intra_levels is None else None),
            "positive_p_in_levels": (list(HOHMANN_P_IN_LEVELS) if positive_intra_levels is None else None),
            "positive_probability_source": ("Hohmann et al. Figure 4"
                                            if positive_intra_levels is None else None),
            "edge_budget_mode": "exact" if positive_intra_levels is not None else "bernoulli_sbm",
            "positive_edges": args.positive_edges,
            "positive_intra_fraction_levels": (list(positive_intra_levels)
                                                if positive_intra_levels is not None else None),
            "negative_inter_fraction_levels": list(negative_inter_levels),
            "paired_level_design": True,
        },
        "graphs": graphs,
        "opinions": aligned,
        "additional_random_opinions": random_records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "generation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
    )
    print(json.dumps({
        "output": str(args.output_root),
        "graphs": len(graphs),
        "aligned_opinions": len(aligned),
        "random_opinions": len(random_records),
    }, indent=2))


if __name__ == "__main__":
    main()
