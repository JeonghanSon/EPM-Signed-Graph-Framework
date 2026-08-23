from __future__ import annotations

import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from signed_epm.graph import graph_fingerprint
from signed_epm.polarization.measure import (
    build_weighted_laplacian,
    polarization_from_laplacian,
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
) -> dict:
    """Build KMeans communities, polarized-pair scores, and gray rankings."""
    state = np.asarray(node_state, dtype=np.float64)
    raw, normalized = pca_spaces(state, k)
    laplacian = build_weighted_laplacian(graph, len(state), 1.0, negative_conductance)
    labels = KMeans(n_clusters=k, random_state=kmeans_seed, n_init=10).fit_predict(state)
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
    pair_rows = []
    for left, right, prune_score in candidates:
        left_nodes, right_nodes = communities[left], communities[right]
        selected = left_nodes + right_nodes
        local = normalized[selected] - normalized[selected].mean(axis=0, keepdims=True)
        masked = np.zeros_like(normalized)
        masked[selected] = local
        pair_delta = polarization_from_laplacian(laplacian, masked)
        pair_rows.append({
            "community_1": left, "community_2": right,
            "size_c1": len(left_nodes), "size_c2": len(right_nodes),
            "prune_score": prune_score,
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
        "schema_version": 1,
        "graph_fingerprint": graph_fingerprint(graph, directed=False),
        "node_state_sha256": hashlib.sha256(
            np.ascontiguousarray(state).tobytes()
        ).hexdigest(),
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


def select_pairs(pairs: pd.DataFrame, tau: float, max_degree: int) -> pd.DataFrame:
    minimum, maximum = float(pairs.delta.min()), float(pairs.delta.max())
    normalized = ((pairs.delta - minimum) / (maximum - minimum)
                  if maximum > minimum else np.ones(len(pairs)))
    candidates = pairs.assign(delta_normalized=normalized)
    candidates = candidates[candidates.delta_normalized >= tau].sort_values(
        ["delta", "community_1", "community_2"],
        ascending=[False, True, True], kind="mergesort",
    )
    degrees: dict[int, int] = defaultdict(int)
    selected = []
    for row in candidates.itertuples(index=False):
        left, right = int(row.community_1), int(row.community_2)
        if degrees[left] < max_degree and degrees[right] < max_degree:
            selected.append(row._asdict())
            degrees[left] += 1
            degrees[right] += 1
    return pd.DataFrame(selected, columns=candidates.columns)


def intervention_targets(
    communities: dict[int, list[int]], positive_graph: pd.DataFrame, gamma: float,
) -> tuple[int, int, int]:
    source = positive_graph.source.to_numpy()
    target = positive_graph.target.to_numpy()
    intra = [
        int(np.sum(np.isin(source, nodes) & np.isin(target, nodes)))
        for nodes in communities.values()
    ]
    ids = sorted(communities)
    inter = []
    for index, left in enumerate(ids):
        for right in ids[index + 1:]:
            left_nodes, right_nodes = communities[left], communities[right]
            inter.append(int(np.sum(
                (np.isin(source, left_nodes) & np.isin(target, right_nodes)) |
                (np.isin(source, right_nodes) & np.isin(target, left_nodes))
            )))
    gray_count = int(round(np.mean([len(nodes) for nodes in communities.values()])))
    gray_gray = int(round(np.mean(intra) * gamma))
    gray_community = int(round(np.mean(inter) * gamma))
    return gray_count, gray_gray, gray_community


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
    physical = pd.DataFrame(physical_edges, columns=["source", "target"])
    physical = physical.assign(sign=1, weight=1.0)
    encoded = model_edges(physical, directed_backbone)
    directed_augmented = pd.concat([base_directed, encoded], ignore_index=True, sort=False)
    undirected_augmented = pd.concat([base_undirected, physical], ignore_index=True, sort=False)
    physical.to_csv(output_dir / "new_physical_edges.csv", index=False)
    encoded.to_csv(output_dir / "new_model_edges.csv", index=False)
    directed_augmented.to_csv(output_dir / "train_snapshot_directed_augmented.csv", index=False)
    undirected_augmented.to_csv(output_dir / "train_snapshot_undirected_augmented.csv", index=False)
    return physical, encoded


def epm_augment(
    data_dir: Path,
    preparation_dir: Path,
    output_dir: Path,
    tau: float,
    max_degree: int,
    gamma: float,
    directed_backbone: bool,
) -> dict:
    directed = pd.read_csv(data_dir / "train_snapshot_directed.csv")
    undirected = pd.read_csv(data_dir / "train_snapshot_undirected.csv")
    communities = load_communities(preparation_dir / "kmeans_communities.csv")
    pairs = pd.read_csv(preparation_dir / "community_pairs.csv")
    selected = select_pairs(pairs, tau, max_degree)
    gray_count, target_gray_gray, target_gray_community = intervention_targets(
        communities, undirected[undirected.sign > 0], gamma,
    )
    occupied = {
        canonical(int(source), int(target))
        for source, target in undirected[["source", "target"]].itertuples(index=False)
    }
    physical_edges: list[tuple[int, int]] = []
    records = []
    for pair in selected.itertuples(index=False):
        left, right = int(pair.community_1), int(pair.community_2)
        gray = pd.read_csv(preparation_dir / "gray_rankings" / f"pair_{left}_{right}.csv")
        gray_nodes = gray.sort_values(["score", "node_id"]).head(gray_count).node_id.astype(int).tolist()
        gray_gray_candidates = sorted({
            canonical(gray_nodes[a], gray_nodes[b])
            for a in range(len(gray_nodes)) for b in range(a + 1, len(gray_nodes))
            if canonical(gray_nodes[a], gray_nodes[b]) not in occupied
        })
        picked_gray_gray = gray_gray_candidates[:target_gray_gray]
        occupied.update(picked_gray_gray)
        picked_gray_community: list[tuple[int, int]] = []
        left_budget = target_gray_community // 2
        for nodes, budget in (
            (communities[left], left_budget),
            (communities[right], target_gray_community - left_budget),
        ):
            candidates = sorted({
                canonical(gray_node, community_node)
                for gray_node in gray_nodes for community_node in nodes
                if gray_node != community_node and
                canonical(gray_node, community_node) not in occupied
            })
            chosen = candidates[:budget]
            occupied.update(chosen)
            picked_gray_community.extend(chosen)
        physical_edges.extend(picked_gray_gray + picked_gray_community)
        records.append({
            "community_1": left, "community_2": right,
            "pair_delta": float(pair.delta),
            "pair_delta_normalized": float(pair.delta_normalized),
            "gray_nodes": len(gray_nodes),
            "gray_gray_edges": len(picked_gray_gray),
            "gray_community_edges": len(picked_gray_community),
        })
    physical, encoded = write_augmented_graph(
        physical_edges, directed, undirected, output_dir, directed_backbone,
    )
    pd.DataFrame(records).to_csv(output_dir / "selected_pairs.csv", index=False)
    summary = {
        "schema_version": 1, "intervention": "epm_gray",
        "tau": float(tau), "max_degree": int(max_degree), "gamma": float(gamma),
        "selected_pairs": len(selected), "gray_nodes_per_pair": gray_count,
        "target_gray_gray_per_pair": target_gray_gray,
        "target_gray_community_per_pair": target_gray_community,
        "physical_edges_added": len(physical), "model_edges_added": len(encoded),
        "edge_budget_cost": len(encoded), "directed_backbone": directed_backbone,
        "new_edge_sign": 1, "new_edge_weight": 1.0,
        "graph_fingerprint_undirected": graph_fingerprint(
            pd.read_csv(output_dir / "train_snapshot_undirected_augmented.csv"), directed=False,
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    return summary


def random_augment(
    data_dir: Path,
    reference_summary: dict,
    output_dir: Path,
    directed_backbone: bool,
    seed: int,
    budget_fraction: float = 1.0,
) -> dict:
    """Add uniform random positive physical edges under EPM's realized budget."""
    if not 0.0 < budget_fraction <= 1.0:
        raise ValueError("budget_fraction must be in (0, 1]")
    directed = pd.read_csv(data_dir / "train_snapshot_directed.csv")
    undirected = pd.read_csv(data_dir / "train_snapshot_undirected.csv")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    num_nodes = int(manifest.get("counts", manifest).get("num_nodes", manifest.get("num_nodes")))
    reference_budget = int(reference_summary["physical_edges_added"])
    budget = int(round(reference_budget * budget_fraction))
    occupied = {
        canonical(int(source), int(target))
        for source, target in undirected[["source", "target"]].itertuples(index=False)
    }
    rng = np.random.default_rng(np.random.SeedSequence([
        int(seed), int(round(float(reference_summary["tau"]) * 10)),
        int(reference_summary["max_degree"]),
        int(round(float(reference_summary["gamma"]) * 10)), 73,
    ]))
    sampled: list[tuple[int, int]] = []
    sampled_set: set[tuple[int, int]] = set()
    while len(sampled) < budget:
        source, target = int(rng.integers(num_nodes)), int(rng.integers(num_nodes))
        if source == target:
            continue
        pair = canonical(source, target)
        if pair not in occupied and pair not in sampled_set:
            sampled.append(pair)
            sampled_set.add(pair)
    physical, encoded = write_augmented_graph(
        sorted(sampled), directed, undirected, output_dir, directed_backbone,
    )
    summary = {
        "schema_version": 1, "intervention": "random_budget_matched",
        "reference_intervention": "epm_gray", "seed": int(seed),
        "tau": float(reference_summary["tau"]),
        "max_degree": int(reference_summary["max_degree"]),
        "gamma": float(reference_summary["gamma"]),
        "budget_fraction": float(budget_fraction),
        "reference_physical_edge_budget": reference_budget,
        "budget_rounding": "round_half_to_even",
        "physical_edges_added": len(physical), "model_edges_added": len(encoded),
        "edge_budget_cost": len(encoded), "directed_backbone": directed_backbone,
        "new_edge_sign": 1, "new_edge_weight": 1.0,
        "candidate_information": "train_graph_only",
    }
    if len(physical) != budget:
        raise RuntimeError("random intervention failed to match its fractional budget")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    return summary


def direct_augment(
    data_dir: Path,
    preparation_dir: Path,
    epm_dir: Path,
    output_dir: Path,
    directed_backbone: bool,
    seed: int,
) -> dict:
    """Directly bridge EPM-selected community pairs with pair-matched budgets."""
    directed = pd.read_csv(data_dir / "train_snapshot_directed.csv")
    undirected = pd.read_csv(data_dir / "train_snapshot_undirected.csv")
    reference = json.loads((epm_dir / "summary.json").read_text(encoding="utf-8"))
    selected = pd.read_csv(epm_dir / "selected_pairs.csv")
    communities = load_communities(preparation_dir / "kmeans_communities.csv")
    occupied = {
        canonical(int(source), int(target))
        for source, target in undirected[["source", "target"]].itertuples(index=False)
    }
    physical_edges: list[tuple[int, int]] = []
    records = []
    for pair in selected.itertuples(index=False):
        left, right = int(pair.community_1), int(pair.community_2)
        pair_budget = int(pair.gray_gray_edges) + int(pair.gray_community_edges)
        candidates = sorted({
            canonical(source, target)
            for source in communities[left] for target in communities[right]
            if source != target and canonical(source, target) not in occupied
        })
        if len(candidates) < pair_budget:
            raise RuntimeError(
                f"not enough direct edges for community pair {(left, right)}: "
                f"{len(candidates)} < {pair_budget}"
            )
        rng = np.random.default_rng(np.random.SeedSequence([
            int(seed), int(round(float(reference["tau"]) * 10)),
            int(reference["max_degree"]), int(round(float(reference["gamma"]) * 10)),
            left, right, 97,
        ]))
        indices = rng.choice(len(candidates), size=pair_budget, replace=False)
        chosen = sorted(candidates[int(index)] for index in indices)
        occupied.update(chosen)
        physical_edges.extend(chosen)
        records.append({
            "community_1": left, "community_2": right,
            "pair_budget": pair_budget, "direct_edges_added": len(chosen),
            "available_candidates": len(candidates),
        })
    expected = int(reference["physical_edges_added"])
    if len(physical_edges) != expected:
        raise RuntimeError(f"direct intervention budget {len(physical_edges)} != EPM {expected}")
    physical, encoded = write_augmented_graph(
        physical_edges, directed, undirected, output_dir, directed_backbone,
    )
    pd.DataFrame(records).to_csv(output_dir / "selected_pairs.csv", index=False)
    summary = {
        "schema_version": 1, "intervention": "direct_budget_matched",
        "reference_intervention": "epm_gray", "seed": int(seed),
        "tau": float(reference["tau"]), "max_degree": int(reference["max_degree"]),
        "gamma": float(reference["gamma"]), "selected_pairs": len(selected),
        "physical_edges_added": len(physical), "model_edges_added": len(encoded),
        "edge_budget_cost": len(encoded), "directed_backbone": directed_backbone,
        "new_edge_sign": 1, "new_edge_weight": 1.0,
        "candidate_information": "train_graph_and_epm_selected_pairs_only",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    return summary
