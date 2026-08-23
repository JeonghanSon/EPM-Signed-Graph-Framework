from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from signed_epm.data.preprocess import connected
from signed_epm.graph import graph_fingerprint
from signed_epm.synthetic.generate import (
    ROOT, balanced_communities, candidate_pools, edge_frame, visualization_nodes,
)


# Structural levels reported in Hohmann, Devriendt, and Coscia (2023), Fig. 4.
P_OUT_LEVELS = (0.0085, 0.0024, 0.0012, 0.0006, 0.0003)


@dataclass(frozen=True)
class SBMConfig:
    num_nodes: int = 1000
    num_communities: int = 8
    baseline_probability: float = 0.0085


def within_probability(config: SBMConfig, p_out: float) -> float:
    """Choose p_in so every level has the same expected edge count."""
    communities = balanced_communities(config.num_nodes, config.num_communities)
    intra, inter = candidate_pools(communities)
    expected = config.baseline_probability * (len(intra) + len(inter))
    value = (expected - p_out * len(inter)) / len(intra)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"infeasible p_in={value} for p_out={p_out}")
    return float(value)


def hohmann_opinion(num_nodes: int, seed: int, mean: float = 0.8,
                    standard_deviation: float = 0.2) -> np.ndarray:
    """Generate the symmetric, sorted opinion spectrum used for structural SBM tests."""
    if num_nodes % 2:
        raise ValueError("the symmetric opinion construction requires an even node count")
    rng = np.random.default_rng(seed)
    half = rng.normal(mean, standard_deviation, size=num_nodes // 2)
    # Match the paper's reflection of samples above the upper endpoint.
    half = np.where(half > 1.0, 2.0 - half, half)
    half = np.clip(half, -1.0, 1.0)
    return np.sort(np.concatenate([half, -half])).astype(np.float64)


def write_opinions(output_root: Path, config: SBMConfig,
                   opinion_seeds: list[int]) -> tuple[list[dict], list[dict]]:
    """Write aligned spectra and fixed random permutations of the same spectra."""
    opinion_root = output_root / "opinions"
    opinion_root.mkdir(parents=True, exist_ok=True)
    aligned_records, random_records = [], []
    for seed in opinion_seeds:
        aligned = hohmann_opinion(config.num_nodes, seed)
        aligned_path = opinion_root / f"hohmann_normal_seed_{seed}.csv"
        pd.DataFrame({"node_id": np.arange(config.num_nodes), "opinion": aligned}).to_csv(
            aligned_path, index=False,
        )
        aligned_records.append({
            "distribution": "hohmann_normal_aligned", "seed": seed,
            "path": str(aligned_path),
            "sha256": hashlib.sha256(aligned.tobytes()).hexdigest(),
        })

        # The permutation is sampled once per opinion seed and reused at every
        # structural level, so only graph structure varies along A--E.
        random_values = np.random.default_rng(50_000 + seed).permutation(aligned)
        random_path = opinion_root / f"hohmann_normal_random_seed_{seed}.csv"
        pd.DataFrame({
            "node_id": np.arange(config.num_nodes), "opinion": random_values,
        }).to_csv(random_path, index=False)
        random_records.append({
            "distribution": "hohmann_normal_random", "seed": seed,
            "path": str(random_path),
            "sha256": hashlib.sha256(random_values.tobytes()).hexdigest(),
            "source_distribution": "hohmann_normal_aligned",
            "permutation_seed": 50_000 + seed,
        })
    return aligned_records, random_records


def _write_graph(root: Path, graph: pd.DataFrame, nodes: pd.DataFrame,
                 config: SBMConfig, graph_seed: int, level: int, p_in: float,
                 p_out: float, attempt: int) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    graph.to_csv(root / "train_events.csv", index=False)
    graph.to_csv(root / "train_snapshot_directed.csv", index=False)
    undirected = graph.copy()
    source = np.minimum(undirected.source, undirected.target)
    target = np.maximum(undirected.source, undirected.target)
    undirected["source"], undirected["target"] = source, target
    undirected.sort_values(["source", "target"]).to_csv(
        root / "train_snapshot_undirected.csv", index=False,
    )
    empty = graph.iloc[:0]
    empty.to_csv(root / "val_events.csv", index=False)
    empty.to_csv(root / "test_events.csv", index=False)
    nodes.to_csv(root / "nodes.csv", index=False)
    manifest = {
        "schema_version": 1,
        "dataset": f"structural_unsigned_sbm_g{graph_seed}_l{level}",
        "synthetic": {
            **config.__dict__, "experiment": "structural",
            "graph_model": "unsigned_sbm", "graph_seed": graph_seed,
            "level": level, "p_in": p_in, "p_out": p_out,
            "generation_attempt": attempt,
            "expected_edge_count": config.baseline_probability
            * config.num_nodes * (config.num_nodes - 1) / 2,
        },
        "counts": {
            "num_nodes": config.num_nodes,
            "splits": {"train": len(graph), "val": 0, "test": 0},
            "positive": {"train": len(graph), "val": 0, "test": 0},
            "negative": {"train": 0, "val": 0, "test": 0},
        },
        "validation": {
            "train_connected": connected(graph, config.num_nodes),
            "positive_train_connected": connected(graph, config.num_nodes),
            "all_edges_positive": bool((graph.sign > 0).all()),
        },
        "communities": {"selected_k": config.num_communities, "source": "ground_truth"},
        "graph_fingerprint": graph_fingerprint(graph, directed=False),
    }
    if not all(manifest["validation"].values()):
        raise RuntimeError(f"unsigned SBM validation failed: {manifest['validation']}")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def generate_seed(output_root: Path, graph_seed: int, config: SBMConfig,
                  max_attempts: int) -> list[dict]:
    communities = balanced_communities(config.num_nodes, config.num_communities)
    intra, inter = candidate_pools(communities)
    nodes = visualization_nodes(communities, 10_000 + graph_seed)
    probabilities = [(within_probability(config, p_out), p_out)
                     for p_out in P_OUT_LEVELS]

    for attempt in range(max_attempts):
        rng = np.random.default_rng(graph_seed * 1009 + attempt)
        # Shared uniforms couple levels while retaining the exact SBM marginal at each level.
        intra_uniform = rng.random(len(intra))
        inter_uniform = rng.random(len(inter))
        sequence = []
        for level, (p_in, p_out) in enumerate(probabilities, start=1):
            positive = np.vstack([intra[intra_uniform < p_in], inter[inter_uniform < p_out]])
            graph = edge_frame(positive, np.empty((0, 2), dtype=np.int64),
                               graph_seed * 10_000 + level)
            if not connected(graph, config.num_nodes):
                break
            sequence.append((level, p_in, p_out, graph))
        if len(sequence) == len(probabilities):
            break
    else:
        raise RuntimeError(f"could not generate a connected SBM sequence for seed {graph_seed}")

    records = []
    for level, p_in, p_out, graph in sequence:
        destination = output_root / "structural" / f"graph_seed_{graph_seed}" / f"level_{level}"
        manifest = _write_graph(destination, graph, nodes, config, graph_seed,
                                level, p_in, p_out, attempt)
        records.append({
            "experiment": "structural", "level": level, "p_in": p_in,
            "p_out": p_out, "graph_seed": graph_seed, "path": str(destination),
            "realized_edges": len(graph), "graph_fingerprint": manifest["graph_fingerprint"],
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Hohmann-style unsigned structural SBMs")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "data" / "synthetic" / "unsigned_sbm_n1000_k8")
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--opinion-seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--add-random-opinions-only", action="store_true")
    args = parser.parse_args()
    if args.add_random_opinions_only:
        summary_path = args.output_root / "generation_summary.json"
        summary = json.loads(summary_path.read_text())
        config = SBMConfig(**summary["config"])
        aligned, random_records = write_opinions(args.output_root, config, args.opinion_seeds)
        summary["opinions"] = aligned
        summary["additional_random_opinions"] = random_records
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"output": str(args.output_root),
                          "aligned_opinions": len(aligned),
                          "random_opinions": len(random_records)}, indent=2))
        return
    if args.output_root.exists() and any(args.output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output exists: {args.output_root} (use --overwrite)")
        shutil.rmtree(args.output_root)
    config = SBMConfig()
    graphs = []
    for seed in args.graph_seeds:
        graphs.extend(generate_seed(args.output_root, seed, config, args.max_attempts))
    opinions, random_opinions = write_opinions(args.output_root, config, args.opinion_seeds)
    summary = {"schema_version": 1, "config": config.__dict__, "graphs": graphs,
               "opinions": opinions, "additional_random_opinions": random_opinions,
               "p_out_levels": list(P_OUT_LEVELS),
               "p_in_levels": [within_probability(config, value) for value in P_OUT_LEVELS]}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(args.output_root), "graphs": len(graphs),
                      "opinions": len(opinions)}, indent=2))


if __name__ == "__main__":
    main()
