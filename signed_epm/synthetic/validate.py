from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from signed_epm.polarization.measure import build_weighted_laplacian, polarization_from_laplacian


ROOT = Path(__file__).resolve().parents[3]


def legacy_er(edge_path: Path, opinion_path: Path, num_nodes: int) -> float:
    graph = pd.read_csv(edge_path)
    positive = graph[graph["sign"] > 0].copy()
    opinion = pd.read_csv(opinion_path).sort_values("node_id")["opinion"].to_numpy(float)
    if len(opinion) != num_nodes:
        raise ValueError("opinion and graph node counts differ")
    if not (opinion > 0).any() or not (opinion < 0).any():
        raise ValueError("Hohmann polarization requires both positive and negative opinion mass")
    # In the paper's metric o+ stores positive entries and o- stores the
    # absolute values of negative entries, hence o+ - o- is exactly raw o.
    # The later unit/expectation interpretation discusses a separately
    # normalized special case; Fig. 4 reports the unnormalized metric below.
    laplacian = build_weighted_laplacian(positive, num_nodes, 1.0, 1.0)
    return polarization_from_laplacian(laplacian, opinion[:, None])


def legacy_er_from_laplacian(laplacian, opinion_path: Path, num_nodes: int) -> float:
    """Evaluate another opinion vector without rebuilding the same graph operator."""
    opinion = pd.read_csv(opinion_path).sort_values("node_id")["opinion"].to_numpy(float)
    if len(opinion) != num_nodes:
        raise ValueError("opinion and graph node counts differ")
    if not (opinion > 0).any() or not (opinion < 0).any():
        raise ValueError("Hohmann polarization requires both positive and negative opinion mass")
    return polarization_from_laplacian(laplacian, opinion[:, None])


def evaluate_legacy(data_root: Path, output_dir: Path,
                    opinion_set: str = "primary") -> dict:
    summary = json.loads((data_root / "generation_summary.json").read_text())
    opinion_records = (summary["opinions"] if opinion_set == "primary" else
                       summary.get("additional_random_opinions", []))
    if not opinion_records:
        raise ValueError(f"opinion set {opinion_set!r} is empty")
    rows = []
    for graph in summary["graphs"]:
        graph_root = Path(graph["path"])
        if not graph_root.is_absolute():
            graph_root = ROOT / graph_root
        manifest = json.loads((graph_root / "manifest.json").read_text())
        num_nodes = int(manifest["counts"]["num_nodes"])
        edge_frame = pd.read_csv(graph_root / "train_snapshot_undirected.csv")
        positive = edge_frame[edge_frame["sign"] > 0].copy()
        laplacian = build_weighted_laplacian(positive, num_nodes, 1.0, 1.0)
        for opinion_record in opinion_records:
            opinion_path = Path(opinion_record["path"])
            if not opinion_path.is_absolute():
                opinion_path = ROOT / opinion_path
            rows.append({
                "experiment": graph["experiment"], "level": graph["level"],
                "graph_seed": graph["graph_seed"],
                "opinion_distribution": opinion_record["distribution"],
                "opinion_seed": opinion_record["seed"],
                "legacy_er": legacy_er_from_laplacian(
                    laplacian, opinion_path, num_nodes,
                ),
            })
    frame = pd.DataFrame(rows)
    aggregate = frame.groupby(
        ["experiment", "opinion_distribution", "level"], as_index=False,
    ).agg(mean=("legacy_er", "mean"), std=("legacy_er", "std"), n=("legacy_er", "size"))
    associations = []
    for (experiment, distribution), group in frame.groupby(
        ["experiment", "opinion_distribution"], sort=True,
    ):
        paired = group.groupby(["graph_seed", "opinion_seed"], sort=True)
        values, constant = [], 0
        for _, part in paired:
            if np.ptp(part["legacy_er"].to_numpy()) <= 1e-12:
                constant += 1
            else:
                values.append(float(spearmanr(part["level"], part["legacy_er"]).statistic))
        associations.append({
            "experiment": experiment, "opinion_distribution": distribution,
            "spearman_mean": None if not values else float(np.mean(values)),
            "spearman_std": None if len(values) < 2 else float(np.std(values, ddof=1)),
            "nonconstant_replicates": len(values), "constant_replicates": constant,
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "legacy_er_raw.csv", index=False)
    aggregate.to_csv(output_dir / "legacy_er_aggregate.csv", index=False)
    result = {"rows": len(frame), "opinion_set": opinion_set,
              "aggregate": aggregate.to_dict("records"),
              "associations": associations}
    (output_dir / "legacy_er_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate synthetic graphs with legacy ER polarization")
    parser.add_argument("--data-root", type=Path,
                        default=ROOT / "data" / "synthetic" / "k2_controlled")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts" / "synthetic" / "k2_validation" /
                        "legacy_aligned")
    parser.add_argument("--opinion-set", choices=["primary", "additional_random"],
                        default="primary")
    args = parser.parse_args()
    result = evaluate_legacy(args.data_root, args.output_dir, args.opinion_set)
    print(json.dumps({"rows": result["rows"], "associations": result["associations"]}, indent=2))


if __name__ == "__main__":
    main()
