from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from signed_epm.mitigation.greedy import grounded_factor, solve_grounded
from signed_epm.graph import canonical_undirected
from signed_epm.polarization.measure import build_weighted_laplacian
from signed_epm.polarization.signed_measure import build_edge_subset_laplacian


def state(laplacian: sp.csr_matrix, z: np.ndarray, y: np.ndarray,
          source: np.ndarray, target: np.ndarray, projection_dimension: int = 128,
          solver: str = "direct", solver_tolerance: float = 1e-6,
          solver_maximum_iterations: int = 10_000):
    factor = grounded_factor(
        laplacian, solver, solver_tolerance, solver_maximum_iterations,
    )
    hz, hy = solve_grounded(factor, z), solve_grounded(factor, y)
    rng = np.random.default_rng(2026)
    upper = sp.triu(laplacian, k=1).tocoo()
    weights = -upper.data
    signs = rng.choice([-1.0, 1.0], size=(len(weights), projection_dimension))
    signs /= np.sqrt(projection_dimension)
    rows = np.r_[upper.row, upper.col]
    cols = np.r_[np.arange(len(weights)), np.arange(len(weights))]
    values = np.r_[np.sqrt(weights), -np.sqrt(weights)]
    incidence = sp.coo_matrix(
        (values, (rows, cols)), shape=(laplacian.shape[0], len(weights)),
    ).tocsr()
    embedding = solve_grounded(factor, incidence @ signs)
    resistance = np.sum((embedding[source] - embedding[target]) ** 2, axis=1)
    return factor, hz, hy, np.maximum(resistance, 1e-12)


def scores(hz: np.ndarray, hy: np.ndarray, resistance: np.ndarray,
           source: np.ndarray, target: np.ndarray, alpha: float, dimension: int):
    dz, dy = hz[source] - hz[target], hy[source] - hy[target]
    return (np.einsum("ij,ij->i", dz, dz) +
            alpha * np.einsum("ij,ij->i", dy, dy)) / (dimension * (1 + resistance))


def energy(factor, z: np.ndarray, y: np.ndarray, alpha: float) -> float:
    return float((np.sum(z * solve_grounded(factor, z)) +
                  alpha * np.sum(y * solve_grounded(factor, y))) / z.shape[1])


def initial_prefilter(laplacian, z, y, candidates, maximum_budget, cap, alpha):
    source = candidates.source.to_numpy(np.int64)
    target = candidates.target.to_numpy(np.int64)
    labels = pd.Series(list(zip(candidates.community_1.astype(int),
                                candidates.community_2.astype(int))), dtype="object")
    codes, keys = pd.factorize(labels, sort=True)
    _, hz, hy, resistance = state(laplacian, z, y, source, target)
    gain = scores(hz, hy, resistance, source, target, alpha, z.shape[1])
    if len(candidates) <= cap:
        return candidates.reset_index(drop=True)
    per_pair = max(maximum_budget, int(np.ceil(cap / max(len(keys), 1))))
    retained = []
    for code in range(len(keys)):
        indices = np.flatnonzero(codes == code)
        take = min(per_pair, len(indices))
        retained.extend(indices[np.argpartition(gain[indices], -take)[-take:]].tolist())
    retained = np.asarray(sorted(set(retained)), dtype=np.int64)
    if len(retained) > cap:
        retained = retained[np.argpartition(gain[retained], -cap)[-cap:]]
    return candidates.iloc[retained].reset_index(drop=True)


def select_batch(order: np.ndarray, available: np.ndarray, codes: np.ndarray,
                 counts: np.ndarray, take: int, mode: str, selected_so_far: int,
                 pair_count: int, edge_keys: list[tuple[int, int]]) -> list[int]:
    chosen = []
    chosen_edges: set[tuple[int, int]] = set()
    for index in order:
        index = int(index)
        if not available[index]:
            continue
        if edge_keys[index] in chosen_edges:
            continue
        if mode != "global":
            raise ValueError(f"unsupported allocation mode: {mode}")
        chosen.append(index)
        chosen_edges.add(edge_keys[index])
        counts[codes[index]] += 1
        if len(chosen) == take:
            break
    return chosen


def run(laplacian, z, y, candidates, checkpoints, batch_size, alpha, mode,
        solver="direct", solver_tolerance=1e-6, solver_maximum_iterations=10_000):
    run_started = time.perf_counter()
    source = candidates.source.to_numpy(np.int64)
    target = candidates.target.to_numpy(np.int64)
    edge_keys = list(zip(source.tolist(), target.tolist()))
    labels = pd.Series(list(zip(candidates.community_1.astype(int),
                                candidates.community_2.astype(int))), dtype="object")
    codes, keys = pd.factorize(labels, sort=True)
    available = np.ones(len(candidates), dtype=bool)
    counts = np.zeros(len(keys), dtype=np.int64)
    selected, records = [], []
    current = laplacian.copy()
    initial = energy(grounded_factor(current), z, y, alpha)
    while len(selected) < checkpoints[-1]:
        next_checkpoint = min(x for x in checkpoints if x > len(selected))
        take = min(batch_size, next_checkpoint - len(selected))
        factor, hz, hy, resistance = state(
            current, z, y, source, target, solver=solver,
            solver_tolerance=solver_tolerance,
            solver_maximum_iterations=solver_maximum_iterations,
        )
        gain = scores(hz, hy, resistance, source, target, alpha, z.shape[1])
        gain[~available] = -np.inf
        ranking = np.argsort(-gain, kind="stable")
        chosen = select_batch(
            ranking, available, codes, counts, take, mode,
            len(selected), len(keys), edge_keys,
        )
        if not chosen:
            raise RuntimeError("no eligible candidates remain")
        for index in chosen:
            u, v = edge_keys[index]
            available[(source == u) & (target == v)] = False
        selected.extend(chosen)
        rows = np.r_[source[chosen], target[chosen]]
        cols = np.r_[target[chosen], source[chosen]]
        adjacency = sp.coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=current.shape,
        ).tocsr()
        current = current + sp.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
        if len(selected) in checkpoints:
            value = energy(grounded_factor(current), z, y, alpha)
            record = {"budget": len(selected), "energy": value,
                      "polarization_reduction_pct":
                      100 * (1 - np.sqrt(max(value, 0) / initial)),
                      "checkpoint_elapsed_seconds": time.perf_counter() - run_started}
            records.append(record)
            print(
                f"CHECKPOINT mode={mode} batch={batch_size} "
                f"budget={len(selected)} reduction={record['polarization_reduction_pct']:.6f}",
                flush=True,
            )
    return selected, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--exact-global", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[25, 100])
    parser.add_argument("--modes", nargs="+", choices=["global"], default=["global"])
    parser.add_argument("--rates", nargs="+", type=float, default=[.025, .05, .075, .10])
    parser.add_argument("--candidate-cap", type=int, default=50_000)
    parser.add_argument("--solver", choices=["direct", "cg"], default="direct")
    parser.add_argument("--solver-tolerance", type=float, default=1e-6)
    parser.add_argument("--solver-maximum-iterations", type=int, default=10_000)
    parser.add_argument("--eta", type=float, default=.1)
    parser.add_argument("--alpha", type=float, default=.05)
    args = parser.parse_args()
    started = time.perf_counter()
    graph = canonical_undirected(pd.read_csv(args.data_dir / "train_snapshot_undirected.csv"))
    coordinates = np.load(args.coordinates)
    z = coordinates - coordinates.mean(axis=0, keepdims=True)
    negative = build_edge_subset_laplacian(graph, len(z), sign=-1)
    y = negative @ coordinates
    y -= y.mean(axis=0, keepdims=True)
    laplacian = build_weighted_laplacian(graph, len(z), 1.0, args.eta)
    checkpoints = sorted(max(1, int(np.floor(len(graph) * r))) for r in args.rates)
    candidates = pd.read_csv(args.candidate_file)
    candidates = initial_prefilter(
        laplacian, z, y, candidates, checkpoints[-1], args.candidate_cap, args.alpha,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output_dir / "retained_candidates.csv", index=False)
    exact_paths = {"global": args.exact_global}
    exact = {mode: pd.read_csv(exact_paths[mode])
             for mode in args.modes if exact_paths[mode] is not None}
    rows = []
    for batch_size in args.batch_sizes:
        for mode in args.modes:
            tick = time.perf_counter()
            selected, records = run(
                laplacian, z, y, candidates, checkpoints, batch_size, args.alpha, mode,
                args.solver, args.solver_tolerance, args.solver_maximum_iterations,
            )
            chosen = candidates.iloc[selected].copy()
            chosen.insert(0, "selection_step", np.arange(1, len(chosen) + 1))
            chosen.to_csv(args.output_dir / f"selected_{mode}_batch{batch_size}.csv", index=False)
            for record in records:
                budget = record["budget"]
                approx_edges = set(map(tuple, chosen.head(budget)[["source", "target"]].to_numpy()))
                if mode in exact:
                    exact_edges = set(map(tuple, exact[mode].head(budget)[["source", "target"]].to_numpy()))
                    union = approx_edges | exact_edges
                    edge_jaccard = len(approx_edges & exact_edges) / len(union)
                else:
                    edge_jaccard = np.nan
                rows.append({"mode": mode, "batch_size": batch_size, **record,
                             "edge_jaccard": edge_jaccard,
                             "strategy_elapsed_seconds": time.perf_counter() - tick})
            pd.DataFrame(rows).to_csv(args.output_dir / "comparison.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output_dir / "comparison.csv", index=False)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "candidate_cap": args.candidate_cap, "batch_sizes": args.batch_sizes,
        "modes": args.modes,
        "eta": args.eta, "alpha": args.alpha,
        "solver": args.solver, "solver_tolerance": args.solver_tolerance,
        "solver_maximum_iterations": args.solver_maximum_iterations,
        "total_elapsed_seconds": time.perf_counter() - started,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
