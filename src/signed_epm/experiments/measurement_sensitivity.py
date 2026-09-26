from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from signed_epm.polarization.measure import (
    EPSILON,
    load_node_state,
    score_from_components,
    signed_energy_components,
)


DEFAULT_K_VALUES = (2, 4, 8, 16, 32)


def pca_coordinates(state: np.ndarray, k: int, normalize: bool) -> np.ndarray:
    values = np.asarray(state, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ right[: min(k, right.shape[0])].T
    if not normalize:
        return projected
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    return projected / np.maximum(norms, EPSILON)


def raw_coordinates(state: np.ndarray) -> np.ndarray:
    values = np.asarray(state, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, EPSILON)


def variants(state: np.ndarray, selected_k: int, k_values: list[int]):
    yield "default", selected_k, pca_coordinates(state, selected_k, True)
    yield "without_pca", state.shape[1], raw_coordinates(state)
    yield "without_normalization", selected_k, pca_coordinates(
        state, selected_k, False,
    )
    for k in k_values:
        yield f"k_{k}", min(k, state.shape[1]), pca_coordinates(state, k, True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonical PCA, normalization, and k sensitivity without retraining"
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--community-root", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k-values", nargs="+", type=int,
                        default=list(DEFAULT_K_VALUES))
    parser.add_argument("--negative-conductance", type=float, default=0.1)
    parser.add_argument("--antagonistic-weight", type=float, default=0.05)
    args = parser.parse_args()

    rows: list[dict] = []
    for dataset in args.datasets:
        graph = pd.read_csv(args.data_root / dataset / "train_snapshot_undirected.csv")
        selected_k = int(json.loads(
            (args.community_root / f"{dataset}.json").read_text(encoding="utf-8")
        )["selected_k"])
        selected = pd.read_csv(args.base_root / dataset / "selected_runs.csv")
        for record in selected.sort_values("seed").itertuples(index=False):
            state = load_node_state(Path(record.run_dir) / "node_embeddings.pt")
            for variant, dimension, coordinates in variants(
                state, selected_k, args.k_values,
            ):
                components = signed_energy_components(
                    graph,
                    coordinates,
                    negative_conductance=args.negative_conductance,
                )
                score = score_from_components(
                    float(components["structural_energy"]),
                    float(components["antagonistic_energy"]),
                    args.antagonistic_weight,
                )
                rows.append({
                    "dataset": dataset,
                    "seed": int(record.seed),
                    "variant": variant,
                    "k": int(dimension),
                    "selected_k": selected_k,
                    "polarization": score,
                    "structural_energy": components["structural_energy"],
                    "antagonistic_energy": components["antagonistic_energy"],
                    "negative_conductance": args.negative_conductance,
                    "antagonistic_weight": args.antagonistic_weight,
                    "embedding_path": str(Path(record.run_dir) / "node_embeddings.pt"),
                })
                print(
                    f"DONE dataset={dataset} seed={int(record.seed)} "
                    f"variant={variant} polarization={score:.10f}",
                    flush=True,
                )

    raw = pd.DataFrame(rows)
    summary = raw.groupby(["dataset", "variant", "k", "selected_k"], as_index=False).agg(
        seeds=("seed", "nunique"),
        polarization_mean=("polarization", "mean"),
        polarization_std=("polarization", "std"),
    )
    summary["polarization_cv"] = (
        summary.polarization_std / summary.polarization_mean.replace(0, np.nan)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "seed_results.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "protocol.json").write_text(json.dumps({
        "schema_version": 1,
        "measure_version": "signed_coherent_v1",
        "datasets": args.datasets,
        "seeds": sorted(raw.seed.unique().astype(int).tolist()),
        "k_values": args.k_values,
        "negative_conductance": args.negative_conductance,
        "antagonistic_weight": args.antagonistic_weight,
        "default": "PCA(selected signed-Louvain k) then nodewise L2",
        "without_pca": "raw encoder state then nodewise L2",
        "without_normalization": "PCA(selected signed-Louvain k) without nodewise L2",
        "retraining": False,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
