from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from signed_epm.data.preprocess import connected
from signed_epm.graph import graph_fingerprint
from signed_epm.synthetic.generate import ROOT, candidate_pools, edge_frame


# Default publication-facing antagonistic conditions. They are evenly spaced
# because the manipulated quantity itself is reported on the x-axis.
INTER_NEGATIVE_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)


def level_token(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def negative_probabilities(available_intra: int, available_inter: int,
                           expected_edges: float,
                           expected_inter_fraction: float) -> tuple[float, float]:
    if not 0.0 <= expected_inter_fraction <= 1.0:
        raise ValueError("expected_inter_fraction must lie between zero and one")
    q_in = expected_edges * (1.0 - expected_inter_fraction) / available_intra
    q_out = expected_edges * expected_inter_fraction / available_inter
    if max(q_in, q_out) > 1.0:
        raise ValueError("requested negative SBM layer is infeasible")
    return float(q_in), float(q_out)


def _write_graph(destination: Path, graph: pd.DataFrame, nodes: pd.DataFrame,
                 graph_seed: int, base_fingerprint: str, level: float,
                 q_in: float, q_out: float, expected_negative_edges: int) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    graph.to_csv(destination / "train_events.csv", index=False)
    graph.to_csv(destination / "train_snapshot_directed.csv", index=False)
    undirected = graph.copy()
    source = np.minimum(undirected.source, undirected.target)
    target = np.maximum(undirected.source, undirected.target)
    undirected["source"], undirected["target"] = source, target
    undirected.sort_values(["source", "target"]).to_csv(
        destination / "train_snapshot_undirected.csv", index=False,
    )
    graph.iloc[:0].to_csv(destination / "val_events.csv", index=False)
    graph.iloc[:0].to_csv(destination / "test_events.csv", index=False)
    nodes.to_csv(destination / "nodes.csv", index=False)
    negative = undirected[undirected.sign < 0]
    communities = nodes.set_index("node_id").community
    inter = int(sum(communities.loc[int(s)] != communities.loc[int(t)]
                    for s, t in negative[["source", "target"]].itertuples(index=False)))
    positive = undirected[undirected.sign > 0]
    manifest = {
        "schema_version": 1,
        "dataset": f"antagonistic_signed_sbm_g{graph_seed}_r{level_token(level)}",
        "synthetic": {
            "experiment": "antagonistic", "graph_model": "two_layer_signed_sbm",
            "num_nodes": len(nodes), "num_communities": int(nodes.community.nunique()),
            "graph_seed": graph_seed, "expected_inter_negative_fraction": level,
            "expected_negative_edges": expected_negative_edges,
            "q_negative_in": q_in, "q_negative_out": q_out,
            "q_out_over_q_in": None if q_in == 0.0 else q_out / q_in,
            "realized_negative_edges": len(negative),
            "realized_inter_negative_edges": inter,
            "realized_inter_negative_fraction": inter / len(negative),
            "positive_source": "unsigned_sbm_level_5",
            "positive_source_fingerprint": base_fingerprint,
        },
        "counts": {
            "num_nodes": len(nodes),
            "splits": {"train": len(graph), "val": 0, "test": 0},
            "positive": {"train": len(positive), "val": 0, "test": 0},
            "negative": {"train": len(negative), "val": 0, "test": 0},
        },
        "validation": {
            "train_connected": connected(graph, len(nodes)),
            "positive_train_connected": connected(positive, len(nodes)),
            "positive_graph_unchanged": graph_fingerprint(positive, directed=False)
            == base_fingerprint,
        },
        "communities": {"selected_k": int(nodes.community.nunique()),
                        "source": "ground_truth"},
        "graph_fingerprint": graph_fingerprint(undirected, directed=False),
    }
    if not all(manifest["validation"].values()):
        raise RuntimeError(f"signed SBM validation failed: {manifest['validation']}")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def generate_seed(unsigned_root: Path, output_root: Path, graph_seed: int,
                  expected_negative_edges: float | None,
                  negative_share: float | None = None,
                  levels: tuple[float, ...] = INTER_NEGATIVE_LEVELS) -> list[dict]:
    base = unsigned_root / "structural" / f"graph_seed_{graph_seed}" / "level_5"
    base_graph = pd.read_csv(base / "train_snapshot_undirected.csv")
    positive = base_graph.loc[base_graph.sign > 0, ["source", "target"]].to_numpy(np.int64)
    if negative_share is not None:
        if not 0.0 < negative_share < 1.0:
            raise ValueError("negative_share must be strictly between zero and one")
        expected_negative_edges = len(positive) * negative_share / (1.0 - negative_share)
    if expected_negative_edges is None:
        raise ValueError("expected_negative_edges or negative_share is required")
    nodes = pd.read_csv(base / "nodes.csv")
    communities = nodes.sort_values("node_id").community.to_numpy(np.int64)
    intra, inter = candidate_pools(communities)
    occupied = {tuple(pair) for pair in np.sort(positive, axis=1)}
    available_intra = np.asarray([pair for pair in intra if tuple(pair) not in occupied])
    available_inter = np.asarray([pair for pair in inter if tuple(pair) not in occupied])
    base_fingerprint = graph_fingerprint(base_graph[base_graph.sign > 0], directed=False)
    rng = np.random.default_rng(90_001 + graph_seed)
    uniform_intra = rng.random(len(available_intra))
    uniform_inter = rng.random(len(available_inter))
    records = []
    for level in levels:
        q_in, q_out = negative_probabilities(
            len(available_intra), len(available_inter), expected_negative_edges, level,
        )
        negative = np.vstack([
            available_intra[uniform_intra < q_in],
            available_inter[uniform_inter < q_out],
        ])
        graph = edge_frame(positive, negative,
                           graph_seed * 100_000 + int(round(level * 100)))
        token = level_token(level)
        destination = (output_root / "antagonistic" / "k8"
                       / f"graph_seed_{graph_seed}" / f"ratio_{token}")
        manifest = _write_graph(
            destination, graph, nodes, graph_seed, base_fingerprint, level,
            q_in, q_out, expected_negative_edges,
        )
        records.append({
            "experiment": "antagonistic", "level": level, "token": token,
            "graph_seed": graph_seed, "path": str(destination),
            "q_negative_in": q_in, "q_negative_out": q_out,
            "realized_negative_edges": manifest["synthetic"]["realized_negative_edges"],
            "realized_inter_negative_fraction":
                manifest["synthetic"]["realized_inter_negative_fraction"],
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the K=8 antagonistic signed-SBM panel")
    parser.add_argument("--unsigned-root", type=Path,
                        default=ROOT / "data" / "synthetic" / "unsigned_sbm_n1000_k8")
    parser.add_argument("--positive-root", type=Path, default=None,
                        help="optional structural root whose level_5 positive graph is fixed")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "data" / "synthetic" / "signed_sbm_n1000_k8")
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--expected-negative-edges", type=int, default=None)
    parser.add_argument("--negative-share", type=float, default=None)
    parser.add_argument("--levels", nargs="+", type=float,
                        default=list(INTER_NEGATIVE_LEVELS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output exists: {args.output_root} (use --overwrite)")
        shutil.rmtree(args.output_root)
    graphs = []
    levels = tuple(float(value) for value in args.levels)
    if len(set(levels)) != len(levels) or levels != tuple(sorted(levels)):
        raise ValueError("levels must be unique and sorted")
    positive_root = args.positive_root or args.unsigned_root
    expected_negative_edges = args.expected_negative_edges
    if expected_negative_edges is None and args.negative_share is None:
        expected_negative_edges = 1000
    for seed in args.graph_seeds:
        graphs.extend(generate_seed(
            positive_root, args.output_root, seed, expected_negative_edges,
            args.negative_share, levels,
        ))
    opinion_root = args.output_root / "opinions"
    opinion_root.mkdir(parents=True, exist_ok=True)
    opinions = []
    random_opinions = []
    positive_summary = json.loads((positive_root / "generation_summary.json").read_text())
    for source_key, destination_records in (
        ("opinions", opinions),
        ("additional_random_opinions", random_opinions),
    ):
        for record in positive_summary.get(source_key, []):
            source = Path(record["path"])
            if not source.is_absolute():
                source = ROOT / source
            destination = opinion_root / source.name
            shutil.copy2(source, destination)
            seed = int(record["seed"])
            values = pd.read_csv(destination).opinion.to_numpy(float)
            destination_records.append({
                **record, "seed": seed, "path": str(destination),
                "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            })
    summary = {
        "schema_version": 1,
        "config": {"num_nodes": 1000, "num_communities": 8,
                   "expected_negative_edges": expected_negative_edges,
                   "expected_negative_share": args.negative_share,
                   "positive_source_level": 5},
        "levels": list(levels), "graphs": graphs,
        "opinions": opinions, "additional_random_opinions": random_opinions,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(args.output_root), "graphs": len(graphs),
                      "opinions": len(opinions)}, indent=2))


if __name__ == "__main__":
    main()
