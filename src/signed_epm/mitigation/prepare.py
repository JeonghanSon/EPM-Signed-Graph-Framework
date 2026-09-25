from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from signed_epm.graph import graph_fingerprint


def pca_spaces(node_state: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(node_state, dtype=np.float64)
    if state.ndim != 2 or not np.isfinite(state).all():
        raise ValueError("node_state must be a finite two-dimensional array")
    if k < 2 or k > min(state.shape):
        raise ValueError(f"k must be in [2, {min(state.shape)}]")
    centered = state - state.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    raw = centered @ right[:k].T
    normalized = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    return raw, normalized


def prepare_intervention(
    node_state: np.ndarray,
    graph: pd.DataFrame,
    k: int,
    minimum_community_size: int,
    output_dir: Path,
    kmeans_seed: int = 42,
    reused_labels: np.ndarray | None = None,
    gray_coordinate_normalization: str = "l2",
) -> dict:
    """Prepare score-free all-pair gray rankings for canonical mitigation."""
    state = np.asarray(node_state, dtype=np.float64)
    raw, normalized = pca_spaces(state, k)
    if gray_coordinate_normalization == "l2":
        gray_coordinates = normalized
    elif gray_coordinate_normalization == "none":
        gray_coordinates = raw
    else:
        raise ValueError("gray_coordinate_normalization must be 'l2' or 'none'")
    labels = (
        KMeans(n_clusters=k, random_state=kmeans_seed, n_init=10).fit_predict(state)
        if reused_labels is None else np.asarray(reused_labels, dtype=np.int64)
    )
    if labels.shape != (len(state),):
        raise ValueError("reused KMeans labels and node state dimensions differ")
    grouped: dict[int, list[int]] = defaultdict(list)
    for node, label in enumerate(labels.tolist()):
        grouped[int(label)].append(node)
    communities = {
        label: nodes for label, nodes in grouped.items()
        if len(nodes) >= minimum_community_size
    }
    ids = sorted(communities)
    if len(ids) < 2:
        raise RuntimeError("fewer than two KMeans communities pass the minimum-size rule")

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"node_id": np.arange(len(labels)), "community": labels}).to_csv(
        output_dir / "kmeans_nodes.csv", index=False,
    )
    pd.DataFrame([
        {"community_id": label, "nodes": str(nodes), "size": len(nodes)}
        for label, nodes in sorted(communities.items())
    ]).to_csv(output_dir / "kmeans_communities.csv", index=False)
    gray_dir = output_dir / "gray_rankings"
    gray_dir.mkdir(exist_ok=True)
    gray_size = int(np.ceil(np.mean([len(nodes) for nodes in communities.values()])))
    pair_rows = []
    for index, left in enumerate(ids):
        for right in ids[index + 1:]:
            left_nodes, right_nodes = communities[left], communities[right]
            pair_rows.append({
                "community_1": left, "community_2": right,
                "size_c1": len(left_nodes), "size_c2": len(right_nodes),
            })
            excluded = np.asarray(left_nodes + right_nodes, dtype=np.int64)
            eligible = np.ones(len(state), dtype=bool)
            eligible[excluded] = False
            nodes = np.flatnonzero(eligible)
            center_left = gray_coordinates[left_nodes].mean(axis=0)
            center_right = gray_coordinates[right_nodes].mean(axis=0)
            distance_left = np.linalg.norm(gray_coordinates[nodes] - center_left, axis=1)
            distance_right = np.linalg.norm(gray_coordinates[nodes] - center_right, axis=1)
            scores = np.abs(distance_left - distance_right) + np.maximum(
                distance_left, distance_right,
            )
            ranking = pd.DataFrame({"node_id": nodes, "score": scores}).sort_values(
                ["score", "node_id"], kind="mergesort",
            ).head(gray_size)
            ranking.to_csv(gray_dir / f"pair_{left}_{right}.csv", index=False)

    pairs = pd.DataFrame(pair_rows).sort_values(
        ["community_1", "community_2"], kind="mergesort",
    )
    pairs.to_csv(output_dir / "community_pairs.csv", index=False)
    summary = {
        "schema_version": 3,
        "method": "score_free_all_pair_gray_preparation",
        "graph_fingerprint": graph_fingerprint(graph, directed=False),
        "k": int(k),
        "kmeans_seed": int(kmeans_seed),
        "minimum_community_size": int(minimum_community_size),
        "retained_communities": len(communities),
        "retained_sizes": {
            str(label): len(nodes) for label, nodes in communities.items()
        },
        "community_pairs": len(pairs),
        "pair_selection": "all_retained_pairs",
        "pair_score": "not_computed",
        "gray_node_coordinates": (
            "pca_nodewise_l2" if gray_coordinate_normalization == "l2" else "pca_raw"
        ),
        "gray_nodes_per_pair": gray_size,
        "gray_ranking_storage": "selected_prefix",
        "kmeans_labels": "reused" if reused_labels is not None else "computed",
    }
    (output_dir / "preparation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    return summary
