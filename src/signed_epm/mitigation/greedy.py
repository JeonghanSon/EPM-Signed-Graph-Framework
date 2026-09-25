from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from signed_epm.graph import canonical_undirected
from signed_epm.mitigation.core import load_communities
from signed_epm.polarization.measure import build_weighted_laplacian
from signed_epm.polarization.signed_measure import build_edge_subset_laplacian


class IterativeGroundedSolver:
    """Approximate grounded-Laplacian solves using preconditioned CG."""

    def __init__(self, matrix: sp.csr_matrix, tolerance: float, maximum_iterations: int):
        self.matrix = matrix.tocsr()
        self.tolerance = tolerance
        self.maximum_iterations = maximum_iterations
        diagonal = self.matrix.diagonal()
        inverse = np.divide(1.0, diagonal, out=np.zeros_like(diagonal), where=diagonal != 0)
        self.preconditioner = spla.LinearOperator(self.matrix.shape, matvec=lambda x: inverse * x)
        self.solve_count = 0
        self.iteration_count = 0

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        values = np.asarray(rhs, dtype=np.float64)
        one_dimensional = values.ndim == 1
        if one_dimensional:
            values = values[:, None]
        result = np.empty_like(values)
        for column in range(values.shape[1]):
            iterations = 0

            def callback(_):
                nonlocal iterations
                iterations += 1

            solution, info = spla.cg(
                self.matrix, values[:, column], rtol=self.tolerance, atol=0.0,
                maxiter=self.maximum_iterations, M=self.preconditioner, callback=callback,
            )
            if info != 0:
                raise RuntimeError(f"CG solve failed to converge (info={info})")
            result[:, column] = solution
            self.solve_count += 1
            self.iteration_count += iterations
        return result[:, 0] if one_dimensional else result


def grounded_factor(laplacian: sp.csr_matrix, solver: str = "direct",
                    tolerance: float = 1e-6, maximum_iterations: int = 10_000):
    """Factor a grounded Laplacian; endpoint differences equal L^dagger ones."""
    grounded = laplacian[:-1, :-1]
    if solver == "direct":
        return spla.splu(grounded.tocsc())
    if solver == "cg":
        return IterativeGroundedSolver(grounded.tocsr(), tolerance, maximum_iterations)
    raise ValueError(f"unknown grounded solver: {solver}")


def solve_grounded(factor, rhs: np.ndarray) -> np.ndarray:
    values = np.asarray(rhs, dtype=np.float64)
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values[:, None]
    result = np.zeros_like(values)
    result[:-1] = factor.solve(values[:-1])
    return result[:, 0] if one_dimensional else result


def candidate_table(
    preparation: Path,
    occupied: set[tuple[int, int]],
    gray_policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    communities = load_communities(preparation / "kmeans_communities.csv")
    selected = pd.read_csv(preparation / "community_pairs.csv").sort_values(
        "delta", ascending=False,
    ).reset_index(drop=True)
    global_size = int(np.ceil(np.mean([len(nodes) for nodes in communities.values()])))
    records: set[tuple[int, int, int, int, str]] = set()
    for row in selected.itertuples(index=False):
        left, right = int(row.community_1), int(row.community_2)
        gray_size = (global_size if gray_policy == "global" else
                     int(np.ceil((len(communities[left]) + len(communities[right])) / 2)))
        gray = pd.read_csv(preparation / "gray_rankings" / f"pair_{left}_{right}.csv")
        gray_nodes = gray.sort_values(["score", "node_id"]).head(gray_size).node_id.astype(int).tolist()
        for index, gray_source in enumerate(gray_nodes):
            for gray_target in gray_nodes[index + 1:]:
                source, target = sorted((int(gray_source), int(gray_target)))
                if (source, target) not in occupied:
                    records.add((source, target, left, right, "gray_gray"))
        for community, nodes in ((left, communities[left]), (right, communities[right])):
            for gray_node in gray_nodes:
                for node in nodes:
                    if gray_node == node:
                        continue
                    source, target = sorted((int(gray_node), int(node)))
                    if (source, target) not in occupied:
                        edge_type = ("community_1_gray" if community == left
                                     else "gray_community_2")
                        records.add((source, target, left, right, edge_type))
    frame = pd.DataFrame(records, columns=[
        "source", "target", "community_1", "community_2", "edge_type",
    ])
    return frame.sort_values([
        "source", "target", "community_1", "community_2", "edge_type",
    ]), selected


def _sample_pair_candidates(
    left: int,
    right: int,
    left_nodes: list[int],
    right_nodes: list[int],
    gray_nodes: list[int],
    occupied: set[tuple[int, int]],
    quota: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int, str]]:
    """Uniformly sample candidate records without materializing Cartesian pools.

    A record consists of a physical edge, its selected community pair, and its
    candidate type.  Candidate types are sampled in proportion to their raw
    Cartesian-pool sizes; invalid existing edges and duplicate records are
    rejected.  This is rejection sampling from the union used by
    :func:`candidate_table` and therefore remains uniform over valid records.
    """
    gray = np.asarray(gray_nodes, dtype=np.int64)
    community_left = np.asarray(left_nodes, dtype=np.int64)
    community_right = np.asarray(right_nodes, dtype=np.int64)
    pool_sizes = np.asarray([
        len(gray) * max(len(gray) - 1, 0) // 2,
        len(gray) * len(community_left),
        len(gray) * len(community_right),
    ], dtype=np.int64)
    total_raw = int(pool_sizes.sum())
    if quota <= 0 or total_raw == 0:
        return []

    # When the requested sample is close to the raw pool, exact enumeration is
    # both faster and guarantees termination despite occupied-edge rejection.
    if total_raw <= max(4 * quota, 100_000):
        records: set[tuple[int, int, int, int, str]] = set()
        for index, u in enumerate(gray_nodes):
            for v in gray_nodes[index + 1:]:
                source, target = sorted((int(u), int(v)))
                if source != target and (source, target) not in occupied:
                    records.add((source, target, left, right, "gray_gray"))
        for nodes, edge_type in (
            (left_nodes, "community_1_gray"),
            (right_nodes, "gray_community_2"),
        ):
            for u in gray_nodes:
                for v in nodes:
                    source, target = sorted((int(u), int(v)))
                    if source != target and (source, target) not in occupied:
                        records.add((source, target, left, right, edge_type))
        ordered = sorted(records)
        if len(ordered) <= quota:
            return ordered
        indices = rng.choice(len(ordered), size=quota, replace=False)
        return [ordered[int(index)] for index in np.sort(indices)]

    boundaries = np.cumsum(pool_sizes)
    records: set[tuple[int, int, int, int, str]] = set()
    attempts = 0
    maximum_attempts = max(100_000, 100 * quota)
    while len(records) < quota and attempts < maximum_attempts:
        batch_size = min(max(4 * (quota - len(records)), 4096), 200_000)
        draws = rng.integers(0, total_raw, size=batch_size)
        for draw in draws:
            draw = int(draw)
            if draw < boundaries[0]:
                first, second = rng.choice(len(gray), size=2, replace=False)
                u, v, edge_type = gray[first], gray[second], "gray_gray"
            elif draw < boundaries[1]:
                u = gray[int(rng.integers(len(gray)))]
                v = community_left[int(rng.integers(len(community_left)))]
                edge_type = "community_1_gray"
            else:
                u = gray[int(rng.integers(len(gray)))]
                v = community_right[int(rng.integers(len(community_right)))]
                edge_type = "gray_community_2"
            source, target = sorted((int(u), int(v)))
            if source != target and (source, target) not in occupied:
                records.add((source, target, left, right, edge_type))
                if len(records) == quota:
                    break
        attempts += batch_size
    if len(records) < quota:
        raise RuntimeError(
            f"could sample only {len(records)} of {quota} candidates for pair "
            f"({left}, {right}); reduce --candidate-cap"
        )
    return sorted(records)


def sampled_candidate_table(
    preparation: Path,
    occupied: set[tuple[int, int]],
    gray_policy: str,
    candidate_cap: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair-stratified candidate sampling before Cartesian materialization."""
    if candidate_cap < 1:
        raise ValueError("sampled candidate cap must be positive")
    communities = load_communities(preparation / "kmeans_communities.csv")
    selected = pd.read_csv(preparation / "community_pairs.csv").sort_values(
        "delta", ascending=False,
    ).reset_index(drop=True)
    if selected.empty:
        return pd.DataFrame(columns=[
            "source", "target", "community_1", "community_2", "edge_type",
        ]), selected
    global_size = int(np.ceil(np.mean([len(nodes) for nodes in communities.values()])))
    base_quota, remainder = divmod(candidate_cap, len(selected))
    records: list[tuple[int, int, int, int, str]] = []
    for pair_index, row in enumerate(selected.itertuples(index=False)):
        left, right = int(row.community_1), int(row.community_2)
        gray_size = (global_size if gray_policy == "global" else
                     int(np.ceil((len(communities[left]) + len(communities[right])) / 2)))
        gray = pd.read_csv(preparation / "gray_rankings" / f"pair_{left}_{right}.csv")
        gray_nodes = gray.sort_values(["score", "node_id"]).head(gray_size).node_id.astype(int).tolist()
        quota = base_quota + int(pair_index < remainder)
        rng = np.random.default_rng(np.random.SeedSequence([
            random_seed, left, right, pair_index,
        ]))
        records.extend(_sample_pair_candidates(
            left, right, communities[left], communities[right], gray_nodes,
            occupied, quota, rng,
        ))
    frame = pd.DataFrame(records, columns=[
        "source", "target", "community_1", "community_2", "edge_type",
    ])
    return frame.sort_values([
        "source", "target", "community_1", "community_2", "edge_type",
    ]).reset_index(drop=True), selected


def sampled_direct_candidate_table(
    preparation: Path,
    occupied: set[tuple[int, int]],
    candidate_cap: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair-stratified sampling from direct C_i x C_j bridge candidates."""
    if candidate_cap < 1:
        raise ValueError("sampled candidate cap must be positive")
    communities = load_communities(preparation / "kmeans_communities.csv")
    selected = pd.read_csv(preparation / "community_pairs.csv").sort_values(
        "delta", ascending=False,
    ).reset_index(drop=True)
    columns = ["source", "target", "community_1", "community_2", "edge_type"]
    if selected.empty:
        return pd.DataFrame(columns=columns), selected
    base_quota, remainder = divmod(candidate_cap, len(selected))
    records: list[tuple[int, int, int, int, str]] = []
    for pair_index, row in enumerate(selected.itertuples(index=False)):
        left, right = int(row.community_1), int(row.community_2)
        left_nodes = np.asarray(communities[left], dtype=np.int64)
        right_nodes = np.asarray(communities[right], dtype=np.int64)
        raw_count = len(left_nodes) * len(right_nodes)
        quota = base_quota + int(pair_index < remainder)
        rng = np.random.default_rng(np.random.SeedSequence([
            random_seed, left, right, pair_index, 991,
        ]))
        pair_records: set[tuple[int, int, int, int, str]] = set()
        if raw_count <= max(4 * quota, 100_000):
            for u in left_nodes:
                for v in right_nodes:
                    source, target = sorted((int(u), int(v)))
                    if source != target and (source, target) not in occupied:
                        pair_records.add((source, target, left, right, "direct"))
            ordered = sorted(pair_records)
            if len(ordered) > quota:
                indices = rng.choice(len(ordered), size=quota, replace=False)
                ordered = [ordered[int(index)] for index in np.sort(indices)]
            records.extend(ordered)
            continue
        attempts = 0
        maximum_attempts = max(100_000, 100 * quota)
        while len(pair_records) < quota and attempts < maximum_attempts:
            batch = min(max(4 * (quota - len(pair_records)), 4096), 200_000)
            us = left_nodes[rng.integers(len(left_nodes), size=batch)]
            vs = right_nodes[rng.integers(len(right_nodes), size=batch)]
            for u, v in zip(us, vs):
                source, target = sorted((int(u), int(v)))
                if source != target and (source, target) not in occupied:
                    pair_records.add((source, target, left, right, "direct"))
                    if len(pair_records) == quota:
                        break
            attempts += batch
        if len(pair_records) < quota:
            raise RuntimeError(
                f"could sample only {len(pair_records)} of {quota} direct candidates "
                f"for pair ({left}, {right})"
            )
        records.extend(sorted(pair_records))
    frame = pd.DataFrame(records, columns=columns)
    return frame.sort_values(columns).reset_index(drop=True), selected


def greedy_order(
    laplacian: sp.csr_matrix,
    coordinates: np.ndarray,
    antagonistic: np.ndarray,
    candidates: pd.DataFrame,
    budget: int,
    alpha: float,
    mode: str,
    candidate_cap: int,
    solver: str = "direct",
    solver_tolerance: float = 1e-6,
    solver_maximum_iterations: int = 10_000,
) -> tuple[list[int], list[float], float]:
    n, dimension = coordinates.shape
    factor = grounded_factor(
        laplacian, solver, solver_tolerance, solver_maximum_iterations,
    )
    centered_z = coordinates - coordinates.mean(axis=0, keepdims=True)
    centered_y = antagonistic - antagonistic.mean(axis=0, keepdims=True)
    hz = solve_grounded(factor, centered_z)
    hy = solve_grounded(factor, centered_y)
    initial_energy = float((np.sum(centered_z * hz) + alpha * np.sum(centered_y * hy)) / dimension)

    original_indices = np.arange(len(candidates), dtype=np.int64)
    source = candidates.source.to_numpy(np.int64)
    target = candidates.target.to_numpy(np.int64)
    pair_labels = pd.Series(list(zip(
        candidates.community_1.astype(int), candidates.community_2.astype(int),
    )), dtype="object")
    pair_codes, pair_keys_array = pd.factorize(pair_labels, sort=True)

    # Approximate all base effective resistances once using an edge-incidence JL embedding.
    rng = np.random.default_rng(2026)
    projection_dimension = 128
    upper = sp.triu(laplacian, k=1).tocoo()
    edge_weight = -upper.data
    signs = rng.choice([-1.0, 1.0], size=(len(edge_weight), projection_dimension))
    signs /= np.sqrt(projection_dimension)
    rows = np.r_[upper.row, upper.col]
    cols = np.r_[np.arange(len(edge_weight)), np.arange(len(edge_weight))]
    incidence_values = np.r_[np.sqrt(edge_weight), -np.sqrt(edge_weight)]
    incidence = sp.coo_matrix((incidence_values, (rows, cols)), shape=(n, len(edge_weight))).tocsr()
    rhs = incidence @ signs
    resistance_embedding = solve_grounded(factor, rhs)
    resistance = np.sum((resistance_embedding[source] - resistance_embedding[target]) ** 2, axis=1)
    resistance = np.maximum(resistance, 1e-12)

    # Preserve every selected pair while pruning the very large Cartesian pool.
    initial_dz = hz[source] - hz[target]
    initial_dy = hy[source] - hy[target]
    initial_gain = (np.einsum("ij,ij->i", initial_dz, initial_dz) +
                    alpha * np.einsum("ij,ij->i", initial_dy, initial_dy))
    initial_gain /= dimension * (1.0 + resistance)
    if candidate_cap > 0 and len(candidates) > candidate_cap:
        per_pair = max(budget, int(np.ceil(candidate_cap / max(len(pair_keys_array), 1))))
        retained = []
        for code in range(len(pair_keys_array)):
            indices = np.flatnonzero(pair_codes == code)
            take = min(per_pair, len(indices))
            retained.extend(indices[np.argpartition(initial_gain[indices], -take)[-take:]].tolist())
        retained = np.asarray(sorted(set(retained)), dtype=np.int64)
        if len(retained) > candidate_cap:
            retained = retained[np.argpartition(initial_gain[retained], -candidate_cap)[-candidate_cap:]]
        original_indices = original_indices[retained]
        source, target = source[retained], target[retained]
        pair_codes = pair_codes[retained]
        resistance = resistance[retained]

    available = np.ones(len(source), dtype=bool)
    order: list[int] = []
    energies = [initial_energy]
    update_vectors: list[tuple[np.ndarray, float]] = []
    for step in range(min(budget, len(source))):
        dz = hz[source] - hz[target]
        dy = hy[source] - hy[target]
        gain = (np.einsum("ij,ij->i", dz, dz) +
                alpha * np.einsum("ij,ij->i", dy, dy))
        gain /= dimension * (1.0 + resistance)
        eligible = available.copy()
        if mode != "global":
            raise ValueError(f"unsupported allocation mode: {mode}")
        gain[~eligible] = -np.inf
        chosen = int(np.argmax(gain))
        if not np.isfinite(gain[chosen]):
            break

        u, v = int(source[chosen]), int(target[chosen])
        rhs_edge = np.zeros(n, dtype=np.float64)
        rhs_edge[u], rhs_edge[v] = 1.0, -1.0
        h = solve_grounded(factor, rhs_edge)
        # Apply all prior rank-one updates to H_0 b.
        # Stored update vectors are encoded by the sequential endpoint effects below.
        for previous, denominator in update_vectors:
            coefficient = previous[u] - previous[v]
            h -= previous * coefficient / denominator
        denominator = 1.0 + max(float(h[u] - h[v]), 0.0)
        endpoint_effect = h[source] - h[target]
        resistance = np.maximum(resistance - endpoint_effect ** 2 / denominator, 1e-12)
        hz -= h[:, None] * (hz[u] - hz[v])[None, :] / denominator
        hy -= h[:, None] * (hy[u] - hy[v])[None, :] / denominator
        update_vectors.append((h, denominator))
        available[(source == u) & (target == v)] = False
        order.append(int(original_indices[chosen]))
        energies.append(energies[-1] - float(gain[chosen]))
    return order, energies, initial_energy


def random_curve(
    laplacian: sp.csr_matrix,
    coordinates: np.ndarray,
    antagonistic: np.ndarray,
    candidates: pd.DataFrame,
    checkpoints: list[int],
    alpha: float,
    seed: int,
) -> tuple[dict[int, float], list[int]]:
    rng = np.random.default_rng(seed)
    pair_codes, unique_pairs = pd.factorize(pd.Series(list(zip(
        candidates.community_1.astype(int), candidates.community_2.astype(int),
    )), dtype="object"), sort=True)
    queues = {code: rng.permutation(np.flatnonzero(pair_codes == code)).tolist()
              for code in range(len(unique_pairs))}
    pair_order = rng.permutation(len(unique_pairs)).tolist()
    order = []
    used_edges: set[tuple[int, int]] = set()
    while len(order) < max(checkpoints):
        progressed = False
        for code in pair_order:
            while queues[code]:
                index = int(queues[code].pop())
                edge = (int(candidates.iloc[index].source), int(candidates.iloc[index].target))
                if edge in used_edges:
                    continue
                used_edges.add(edge)
                order.append(index)
                progressed = True
                break
            if len(order) >= max(checkpoints):
                break
        if not progressed:
            break
    result = {}
    dimension = coordinates.shape[1]
    for budget in checkpoints:
        chosen = candidates.iloc[order[:budget]]
        rows = np.r_[chosen.source.to_numpy(), chosen.target.to_numpy()]
        cols = np.r_[chosen.target.to_numpy(), chosen.source.to_numpy()]
        adjacency = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=laplacian.shape).tocsr()
        augmented = laplacian + sp.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
        factor = grounded_factor(augmented)
        z = coordinates - coordinates.mean(axis=0, keepdims=True)
        y = antagonistic - antagonistic.mean(axis=0, keepdims=True)
        result[budget] = float((np.sum(z * solve_grounded(factor, z)) +
                                alpha * np.sum(y * solve_grounded(factor, y))) / dimension)
    return result, order


def evaluate_order_curve(
    laplacian: sp.csr_matrix,
    coordinates: np.ndarray,
    antagonistic: np.ndarray,
    candidates: pd.DataFrame,
    order: list[int],
    checkpoints: list[int],
    alpha: float,
) -> dict[int, float]:
    """Recompute checkpoint energies from the materialized augmented Laplacian."""
    result: dict[int, float] = {}
    dimension = coordinates.shape[1]
    z = coordinates - coordinates.mean(axis=0, keepdims=True)
    y = antagonistic - antagonistic.mean(axis=0, keepdims=True)
    for budget in checkpoints:
        actual = min(budget, len(order))
        chosen = candidates.iloc[order[:actual]]
        rows = np.r_[chosen.source.to_numpy(), chosen.target.to_numpy()]
        cols = np.r_[chosen.target.to_numpy(), chosen.source.to_numpy()]
        adjacency = sp.coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=laplacian.shape,
        ).tocsr()
        augmented = laplacian + sp.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
        factor = grounded_factor(augmented)
        result[budget] = float((np.sum(z * solve_grounded(factor, z)) +
                                alpha * np.sum(y * solve_grounded(factor, y))) / dimension)
    return result


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/bitcoinalpha"))
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--maximum-budget", type=int, required=True,
                        help="maximum number of physical edges to add")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--fractions", nargs="+", type=float, default=[.1, .25, .5, .75, 1.0])
    parser.add_argument("--gray-policies", nargs="+", choices=["global", "pair"], default=["global"])
    parser.add_argument("--greedy-modes", nargs="+", choices=["global"],
                        default=["global"])
    parser.add_argument("--skip-gray-random", action="store_true")
    parser.add_argument("--candidate-cap", type=int, default=50_000)
    parser.add_argument("--candidate-sampling", choices=["marginal_top", "uniform_pair"],
                        default="uniform_pair",
                        help="uniform_pair samples before materializing Cartesian pools")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--candidate-file", type=Path,
                        help="use this stored candidate pool instead of sampling again")
    parser.add_argument("--solver", choices=["direct", "cg"], default="direct")
    parser.add_argument("--solver-tolerance", type=float, default=1e-6)
    parser.add_argument("--solver-maximum-iterations", type=int, default=10_000)
    args = parser.parse_args()

    graph = pd.read_csv(args.data_dir / "train_snapshot_undirected.csv")
    physical = canonical_undirected(graph)
    occupied = set(map(tuple, physical[["source", "target"]].astype(int).to_numpy()))
    coordinates = np.load(args.coordinates)
    negative_laplacian = build_edge_subset_laplacian(physical, len(coordinates), sign=-1)
    antagonistic = negative_laplacian @ coordinates
    laplacian = build_weighted_laplacian(physical, len(coordinates), 1.0, args.eta)
    maximum_budget = args.maximum_budget
    if maximum_budget < 1:
        raise ValueError("maximum budget must be positive")
    checkpoints = sorted(set(max(1, int(np.floor(maximum_budget * value)))
                             for value in args.fractions))
    records = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for gray_policy in args.gray_policies:
        if args.candidate_file is not None:
            candidates = pd.read_csv(args.candidate_file)
            selected = pd.read_csv(args.preparation / "community_pairs.csv")
            greedy_candidate_cap = 0
        elif args.candidate_sampling == "uniform_pair":
            candidates, selected = sampled_candidate_table(
                args.preparation, occupied, gray_policy,
                args.candidate_cap, args.random_seed,
            )
            greedy_candidate_cap = 0
        else:
            candidates, selected = candidate_table(
                args.preparation, occupied, gray_policy,
            )
            greedy_candidate_cap = args.candidate_cap
        candidates.to_csv(args.output_dir / f"candidates_{gray_policy}.csv", index=False)
        for mode in args.greedy_modes:
            order, energies, initial = greedy_order(
                laplacian, coordinates, antagonistic, candidates, maximum_budget,
                args.alpha, mode, greedy_candidate_cap,
                solver=args.solver,
                solver_tolerance=args.solver_tolerance,
                solver_maximum_iterations=args.solver_maximum_iterations,
            )
            chosen = candidates.iloc[order].copy()
            chosen.insert(0, "selection_step", np.arange(1, len(chosen) + 1))
            chosen.to_csv(args.output_dir / f"selected_{gray_policy}_{mode}.csv", index=False)
            exact_energies = evaluate_order_curve(
                laplacian, coordinates, antagonistic, candidates, order,
                checkpoints, args.alpha,
            )
            for budget in checkpoints:
                actual = min(budget, len(order))
                approximate_energy = energies[actual]
                energy = exact_energies[budget]
                records.append({"gray_policy": gray_policy, "method": mode,
                                "budget": actual, "budget_fraction": actual / maximum_budget,
                                "approximate_energy": approximate_energy,
                                "energy": energy, "polarization": np.sqrt(max(energy, 0.0)),
                                "reduction_pct": 100 * (1 - np.sqrt(max(energy, 0.0) / initial)),
                                "candidate_count": len(candidates), "selected_pairs": len(selected)})
        if not args.skip_gray_random:
            random, random_order = random_curve(
                laplacian, coordinates, antagonistic, candidates,
                checkpoints, args.alpha, seed=args.random_seed,
            )
            random_selected = candidates.iloc[random_order].copy()
            random_selected.insert(0, "selection_step", np.arange(1, len(random_selected) + 1))
            random_selected.to_csv(
                args.output_dir / f"selected_{gray_policy}_gray_random.csv", index=False,
            )
            initial = energies[0]
            for budget, energy in random.items():
                records.append({"gray_policy": gray_policy, "method": "random",
                                "budget": budget, "budget_fraction": budget / maximum_budget,
                                "approximate_energy": np.nan,
                                "energy": energy, "polarization": np.sqrt(max(energy, 0.0)),
                                "reduction_pct": 100 * (1 - np.sqrt(max(energy, 0.0) / initial)),
                                "candidate_count": len(candidates), "selected_pairs": len(selected)})
    pd.DataFrame(records).to_csv(args.output_dir / "budget_curves.csv", index=False)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "alpha": args.alpha, "eta": args.eta, "maximum_budget": maximum_budget,
        "community_pair_filter": "none",
        "integerization": "ceil(gray-zone size); floor(maximum budget times fraction)",
        "fractions": args.fractions, "representation": "frozen",
        "resistance_approximation_dimension": 128,
        "candidate_cap": args.candidate_cap,
        "greedy_modes": args.greedy_modes,
        "gray_random_included": not args.skip_gray_random,
        "candidate_sampling": ("stored_candidate_file" if args.candidate_file is not None
                               else args.candidate_sampling),
        "candidate_sampling_stage": (
            "pre_sampled_file"
            if args.candidate_file is not None else
            "before_cartesian_materialization"
            if args.candidate_sampling == "uniform_pair" else
            "after_full_materialization"
        ),
        "random_seed": args.random_seed,
        "candidate_file": str(args.candidate_file) if args.candidate_file else None,
        "solver": args.solver,
        "solver_tolerance": args.solver_tolerance,
        "solver_maximum_iterations": args.solver_maximum_iterations,
        "total_elapsed_seconds": time.perf_counter() - started,
        "candidate_edge_types": ["gray_gray", "community_1_gray", "gray_community_2"],
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
