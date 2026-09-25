from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans

from signed_epm.graph import canonical_undirected, graph_fingerprint
from signed_epm.polarization.measure import (
    build_weighted_laplacian,
    quadratic_energy_from_laplacian,
)


def canonical(source: int, target: int) -> tuple[int, int]:
    return (source, target) if source < target else (target, source)


def pca_spaces(node_state: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(node_state, dtype=np.float64)
    centered = state - state.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    raw = centered @ right[:k].T
    normalized = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    return raw, normalized


def centroid_pair_candidates(
    coordinates: np.ndarray,
    communities: dict[int, list[int]],
    top_pairs: int | None,
) -> list[tuple[int, int, float]]:
    """Rank community pairs by centroid separation for optional large-graph pruning."""
    ids = sorted(communities)
    centers = {
        label: coordinates[nodes].mean(axis=0)
        for label, nodes in communities.items()
    }
    candidates = [
        (left, right, float(np.linalg.norm(centers[left] - centers[right])))
        for index, left in enumerate(ids)
        for right in ids[index + 1:]
    ]
    candidates.sort(key=lambda row: (-row[2], row[0], row[1]))
    if top_pairs is None:
        return candidates
    if top_pairs <= 0:
        raise ValueError("top_pairs must be positive when provided")
    return candidates[:top_pairs]


def prepare_intervention(
    node_state: np.ndarray,
    graph: pd.DataFrame,
    k: int,
    minimum_community_size: int,
    output_dir: Path,
    kmeans_seed: int = 42,
    negative_conductance: float = 0.1,
    top_pairs: int | None = None,
    antagonistic_weight: float = 0.05,
    reused_labels: np.ndarray | None = None,
) -> dict:
    """Build KMeans communities, polarized-pair scores, and gray rankings."""
    state = np.asarray(node_state, dtype=np.float64)
    raw, normalized = pca_spaces(state, k)
    laplacian = build_weighted_laplacian(graph, len(state), 1.0, negative_conductance)
    labels = (KMeans(n_clusters=k, random_state=kmeans_seed, n_init=10).fit_predict(state)
              if reused_labels is None else np.asarray(reused_labels, dtype=np.int64))
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

    all_pair_count = len(ids) * (len(ids) - 1) // 2
    candidates = centroid_pair_candidates(normalized, communities, top_pairs)
    physical = canonical_undirected(graph)
    negative = physical.loc[physical.sign < 0]
    positive = physical.loc[physical.sign > 0]
    node_community = labels
    negative_source = negative.source.to_numpy(dtype=np.int64)
    negative_target = negative.target.to_numpy(dtype=np.int64)
    positive_source = positive.source.to_numpy(dtype=np.int64)
    positive_target = positive.target.to_numpy(dtype=np.int64)
    pair_rows = []
    for left, right, prune_score in candidates:
        left_nodes, right_nodes = communities[left], communities[right]
        selected = left_nodes + right_nodes
        local = normalized[selected] - normalized[selected].mean(axis=0, keepdims=True)
        masked = np.zeros_like(normalized)
        masked[selected] = local
        structural_energy = quadratic_energy_from_laplacian(laplacian, masked)

        cross_negative_mask = (
            ((node_community[negative_source] == left)
             & (node_community[negative_target] == right))
            | ((node_community[negative_source] == right)
               & (node_community[negative_target] == left))
        )
        cross_source = negative_source[cross_negative_mask]
        cross_target = negative_target[cross_negative_mask]
        if len(cross_source):
            adjacency = sp.coo_matrix(
                (np.ones(2 * len(cross_source), dtype=np.float64),
                 (np.r_[cross_source, cross_target], np.r_[cross_target, cross_source])),
                shape=(len(state), len(state)), dtype=np.float64,
            ).tocsr()
            pair_negative_laplacian = (
                sp.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
            ).tocsr()
            antagonistic_signal = pair_negative_laplacian @ normalized
            antagonistic_energy = quadratic_energy_from_laplacian(
                laplacian, antagonistic_signal,
            )
        else:
            antagonistic_energy = 0.0
        combined_energy = structural_energy + antagonistic_weight * antagonistic_energy
        pair_delta = float(np.sqrt(max(combined_energy, 0.0)))

        cross_positive_count = int(np.sum(
            ((node_community[positive_source] == left)
             & (node_community[positive_target] == right))
            | ((node_community[positive_source] == right)
               & (node_community[positive_target] == left))
        ))
        possible_cross_edges = len(left_nodes) * len(right_nodes)
        pair_rows.append({
            "community_1": left, "community_2": right,
            "size_c1": len(left_nodes), "size_c2": len(right_nodes),
            "prune_score": prune_score,
            "structural_energy": structural_energy,
            "structural_polarization": float(np.sqrt(max(structural_energy, 0.0))),
            "antagonistic_energy": antagonistic_energy,
            "weighted_antagonistic_energy": antagonistic_weight * antagonistic_energy,
            "cross_positive_edges": cross_positive_count,
            "cross_negative_edges": int(len(cross_source)),
            "cross_negative_density": (len(cross_source) / possible_cross_edges
                                       if possible_cross_edges else 0.0),
            "delta": pair_delta,
        })

        center_left = raw[left_nodes].mean(axis=0)
        center_right = raw[right_nodes].mean(axis=0)
        eligible = np.ones(len(raw), dtype=bool)
        eligible[np.asarray(selected, dtype=np.int64)] = False
        nodes = np.flatnonzero(eligible)
        distance_left = np.linalg.norm(raw[nodes] - center_left, axis=1)
        distance_right = np.linalg.norm(raw[nodes] - center_right, axis=1)
        scores = np.abs(distance_left - distance_right) + np.maximum(
            distance_left, distance_right,
        )
        ranking = pd.DataFrame({"node_id": nodes, "score": scores}).sort_values(
            ["score", "node_id"], kind="mergesort",
        )
        ranking.to_csv(gray_dir / f"pair_{left}_{right}.csv", index=False)

    pairs = pd.DataFrame(pair_rows).sort_values(
        ["delta", "community_1", "community_2"],
        ascending=[False, True, True], kind="mergesort",
    )
    pairs.to_csv(output_dir / "community_pairs.csv", index=False)
    summary = {
        "schema_version": 2,
        "graph_fingerprint": graph_fingerprint(graph, directed=False),
        "k": int(k), "kmeans_seed": int(kmeans_seed),
        "minimum_community_size": int(minimum_community_size),
        "retained_communities": len(communities),
        "retained_sizes": {str(label): len(nodes) for label, nodes in communities.items()},
        "community_pairs": len(pairs),
        "all_community_pairs": all_pair_count,
        "pair_pruning": "centroid_distance" if top_pairs is not None else "none",
        "top_pairs": int(top_pairs) if top_pairs is not None else None,
        "gray_node_coordinates": "pca_raw_none",
        "pair_coordinates": "pca_nodewise_l2",
        "negative_conductance": float(negative_conductance),
        "antagonistic_weight": float(antagonistic_weight),
        "pair_score": "sqrt(structural_energy + alpha * antagonistic_energy)",
        "antagonistic_signal": "Y_ij = L^-_ij Z; only negative edges between C_i and C_j",
        "kmeans_labels": "reused" if reused_labels is not None else "computed",
    }
    (output_dir / "preparation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    return summary


def load_communities(path: Path) -> dict[int, list[int]]:
    frame = pd.read_csv(path)
    return {
        int(row.community_id): [int(node) for node in ast.literal_eval(str(row.nodes))]
        for row in frame.itertuples(index=False)
    }


def model_edges(physical: pd.DataFrame, directed: bool) -> pd.DataFrame:
    base = physical[["source", "target", "sign", "weight"]].copy()
    if not directed or base.empty:
        return base
    reverse = base.rename(columns={"source": "target", "target": "source"})[
        ["source", "target", "sign", "weight"]
    ]
    return pd.concat([base, reverse], ignore_index=True)


def write_augmented_graph(
    physical_edges: list[tuple[int, int]],
    base_directed: pd.DataFrame,
    base_undirected: pd.DataFrame,
    output_dir: Path,
    directed_backbone: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_edges = [canonical(int(source), int(target))
                       for source, target in physical_edges]
    if any(source == target for source, target in canonical_edges):
        raise ValueError("augmentation contains a self-loop")
    if len(set(canonical_edges)) != len(canonical_edges):
        raise ValueError("augmentation contains duplicate physical edges")
    occupied = set(map(tuple, canonical_undirected(base_undirected)[
        ["source", "target"]
    ].astype(int).to_numpy()))
    overlap = occupied.intersection(canonical_edges)
    if overlap:
        raise ValueError(f"augmentation contains {len(overlap)} existing physical edges")
    physical = pd.DataFrame(canonical_edges, columns=["source", "target"])
    physical = physical.assign(sign=1, weight=1.0)
    encoded = model_edges(physical, directed_backbone)
    directed_augmented = pd.concat([base_directed, encoded], ignore_index=True, sort=False)
    undirected_augmented = pd.concat([base_undirected, physical], ignore_index=True, sort=False)
    physical.to_csv(output_dir / "new_physical_edges.csv", index=False)
    encoded.to_csv(output_dir / "new_model_edges.csv", index=False)
    directed_augmented.to_csv(output_dir / "train_snapshot_directed_augmented.csv", index=False)
    undirected_augmented.to_csv(output_dir / "train_snapshot_undirected_augmented.csv", index=False)
    return physical, encoded

