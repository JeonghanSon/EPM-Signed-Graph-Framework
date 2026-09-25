from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATE_COLUMNS = [
    "source", "target", "community_1", "community_2", "edge_type",
]


def _communities(path: Path) -> dict[int, list[int]]:
    frame = pd.read_csv(path)
    return {
        int(row.community_id): [int(node) for node in ast.literal_eval(str(row.nodes))]
        for row in frame.itertuples(index=False)
    }


def _all_retained_pairs(preparation: Path) -> pd.DataFrame:
    pairs = pd.read_csv(preparation / "community_pairs.csv")
    required = {"community_1", "community_2"}
    if not required.issubset(pairs.columns):
        raise ValueError("community_pairs.csv lacks canonical pair columns")
    # A canonical preparation is score-free. Refuse historical scored files
    # rather than silently interpreting legacy sentinel values as a new method.
    legacy = {"delta", "delta_normalized", "prune_score"}.intersection(pairs.columns)
    if legacy:
        raise ValueError(
            "canonical candidate sampling requires score-free preparation; "
            f"found legacy columns {sorted(legacy)}"
        )
    return pairs.sort_values(
        ["community_1", "community_2"], kind="mergesort",
    ).reset_index(drop=True)


def sample_unique_gray_union(
    preparation: Path,
    occupied: set[tuple[int, int]],
    candidate_cap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Uniformly sample unique physical edges from the all-pair gray union.

    Unordered node pairs are proposed uniformly from the complete physical
    universe and accepted exactly when they occur in at least one retained
    pair's gray candidate space. Conditioning on this predicate yields a
    uniform sample over the unique union, without pair quotas or pair scores.
    Pair/type labels are provenance only and never affect acceptance.
    """
    if candidate_cap < 1:
        raise ValueError("candidate_cap must be positive")
    communities = _communities(preparation / "kmeans_communities.csv")
    pairs = _all_retained_pairs(preparation)
    if pairs.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS), pairs

    nodes = pd.read_csv(preparation / "kmeans_nodes.csv", usecols=["node_id"])
    number_of_nodes = int(nodes.node_id.max()) + 1
    if len(nodes) != number_of_nodes:
        raise ValueError("KMeans node ids must be contiguous from zero")

    gray_size = int(np.ceil(np.mean([len(value) for value in communities.values()])))
    pair_keys: list[tuple[int, int]] = []
    memberships: list[set[int]] = [set() for _ in range(number_of_nodes)]
    for pair_id, row in enumerate(pairs.itertuples(index=False)):
        left, right = int(row.community_1), int(row.community_2)
        pair_keys.append((left, right))
        ranking = pd.read_csv(
            preparation / "gray_rankings" / f"pair_{left}_{right}.csv",
            usecols=["node_id", "score"],
        )
        gray_nodes = ranking.sort_values(
            ["score", "node_id"], kind="mergesort",
        ).head(gray_size).node_id.astype(int)
        for node in gray_nodes:
            memberships[int(node)].add(pair_id)

    community_of = np.full(number_of_nodes, -1, dtype=np.int64)
    for community, member_nodes in communities.items():
        community_of[np.asarray(member_nodes, dtype=np.int64)] = int(community)

    def contexts(source: int, target: int) -> list[tuple[int, int, str]]:
        source_pairs, target_pairs = memberships[source], memberships[target]
        found: set[tuple[int, int, str]] = set()
        for pair_id in source_pairs.intersection(target_pairs):
            left, right = pair_keys[pair_id]
            found.add((left, right, "gray_gray"))
        target_community = int(community_of[target])
        for pair_id in source_pairs:
            left, right = pair_keys[pair_id]
            if target_community == left:
                found.add((left, right, "community_1_gray"))
            elif target_community == right:
                found.add((left, right, "gray_community_2"))
        source_community = int(community_of[source])
        for pair_id in target_pairs:
            left, right = pair_keys[pair_id]
            if source_community == left:
                found.add((left, right, "community_1_gray"))
            elif source_community == right:
                found.add((left, right, "gray_community_2"))
        return sorted(found)

    rng = np.random.default_rng(seed)
    sampled: dict[tuple[int, int], tuple[int, int, str]] = {}
    attempts = 0
    maximum_attempts = max(1_000_000, 1000 * candidate_cap)
    while len(sampled) < candidate_cap and attempts < maximum_attempts:
        remaining = candidate_cap - len(sampled)
        batch_size = min(max(8 * remaining, 8192), 1_000_000)
        first = rng.integers(0, number_of_nodes, size=batch_size)
        second = rng.integers(0, number_of_nodes, size=batch_size)
        for raw_source, raw_target in zip(first, second):
            source, target = sorted((int(raw_source), int(raw_target)))
            attempts += 1
            edge = (source, target)
            if source == target or edge in occupied or edge in sampled:
                continue
            valid_contexts = contexts(source, target)
            if not valid_contexts:
                continue
            sampled[edge] = valid_contexts[int(rng.integers(len(valid_contexts)))]
            if len(sampled) == candidate_cap:
                break

    if len(sampled) < candidate_cap:
        raise RuntimeError(
            f"sampled only {len(sampled)} of {candidate_cap} unique candidates "
            f"after {attempts} proposals; reduce the cap or materialize the exact union"
        )
    records = [
        (source, target, *context)
        for (source, target), context in sampled.items()
    ]
    frame = pd.DataFrame(records, columns=CANDIDATE_COLUMNS)
    frame.attrs["sampling_attempts"] = int(attempts)
    frame.attrs["acceptance_rate"] = float(len(frame) / attempts)
    return frame, pairs
