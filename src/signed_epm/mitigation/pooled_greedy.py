from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from signed_epm.mitigation.linear import (
    grounded_factor,
    objective_energy,
    rank_one_inverse_action_update,
    solve_grounded,
)


def _validate_candidates(candidates: pd.DataFrame, number_of_nodes: int) -> None:
    if not {"source", "target"}.issubset(candidates.columns):
        raise ValueError("candidate table requires source and target columns")
    edges = candidates[["source", "target"]].to_numpy(dtype=np.int64)
    if len(edges) == 0:
        raise ValueError("candidate table is empty")
    if np.any(edges[:, 0] >= edges[:, 1]):
        raise ValueError("candidates must be canonical unordered physical edges")
    if edges.min() < 0 or edges.max() >= number_of_nodes:
        raise ValueError("candidate endpoint lies outside the graph")
    if len(set(map(tuple, edges))) != len(edges):
        raise ValueError("candidate table contains duplicate physical edges")


def pooled_greedy_order(
    laplacian: sp.csr_matrix,
    coordinates: np.ndarray,
    antagonistic_signal: np.ndarray,
    candidates: pd.DataFrame,
    budget: int,
    alpha: float,
    solver: str = "direct",
    solver_tolerance: float = 1e-6,
    solver_maximum_iterations: int = 10_000,
    projection_dimension: int = 128,
    projection_seed: int = 2026,
) -> tuple[list[int], list[float], float]:
    """Select one pooled greedy edge order under a global physical budget.

    Candidate community/type columns, when present, are ignored by selection
    and retained only for provenance.
    """
    if budget < 1:
        raise ValueError("budget must be positive")
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    if projection_dimension < 1:
        raise ValueError("projection_dimension must be positive")
    values = np.asarray(coordinates, dtype=np.float64)
    antagonistic = np.asarray(antagonistic_signal, dtype=np.float64)
    if values.ndim != 2 or antagonistic.shape != values.shape:
        raise ValueError("coordinate and antagonistic signal shapes differ")
    _validate_candidates(candidates, len(values))

    factor = grounded_factor(
        laplacian, solver, solver_tolerance, solver_maximum_iterations,
    )
    z = values - values.mean(axis=0, keepdims=True)
    y = antagonistic - antagonistic.mean(axis=0, keepdims=True)
    hz, hy = solve_grounded(factor, z), solve_grounded(factor, y)
    initial = objective_energy(factor, z, y, alpha)

    source = candidates.source.to_numpy(dtype=np.int64)
    target = candidates.target.to_numpy(dtype=np.int64)
    rng = np.random.default_rng(projection_seed)
    upper = sp.triu(laplacian, k=1).tocoo()
    weights = -upper.data
    signs = rng.choice((-1.0, 1.0), size=(len(weights), projection_dimension))
    signs /= np.sqrt(projection_dimension)
    rows = np.r_[upper.row, upper.col]
    columns = np.r_[np.arange(len(weights)), np.arange(len(weights))]
    incidence_values = np.r_[np.sqrt(weights), -np.sqrt(weights)]
    incidence = sp.coo_matrix(
        (incidence_values, (rows, columns)), shape=(len(values), len(weights)),
    ).tocsr()
    resistance_embedding = solve_grounded(factor, incidence @ signs)
    resistance = np.sum(
        (resistance_embedding[source] - resistance_embedding[target]) ** 2,
        axis=1,
    )
    resistance = np.maximum(resistance, 1e-12)

    available = np.ones(len(candidates), dtype=bool)
    order: list[int] = []
    approximate_energies = [initial]
    updates: list[tuple[np.ndarray, float]] = []
    dimension = values.shape[1]
    for _ in range(min(budget, len(candidates))):
        dz, dy = hz[source] - hz[target], hy[source] - hy[target]
        gain = (
            np.einsum("ij,ij->i", dz, dz)
            + alpha * np.einsum("ij,ij->i", dy, dy)
        ) / (dimension * (1.0 + resistance))
        gain[~available] = -np.inf
        chosen = int(np.argmax(gain))
        if not np.isfinite(gain[chosen]):
            raise RuntimeError("no eligible candidate remains before reaching the budget")

        u, v = int(source[chosen]), int(target[chosen])
        incidence_vector = np.zeros(len(values), dtype=np.float64)
        incidence_vector[u], incidence_vector[v] = 1.0, -1.0
        inverse_incidence = solve_grounded(factor, incidence_vector)
        for previous, denominator in updates:
            coefficient = previous[u] - previous[v]
            inverse_incidence -= previous * coefficient / denominator
        denominator = 1.0 + max(
            float(inverse_incidence[u] - inverse_incidence[v]), 0.0,
        )
        endpoint_effect = inverse_incidence[source] - inverse_incidence[target]
        resistance = np.maximum(
            resistance - endpoint_effect ** 2 / denominator, 1e-12,
        )
        hz = rank_one_inverse_action_update(hz, inverse_incidence, u, v, denominator)
        hy = rank_one_inverse_action_update(hy, inverse_incidence, u, v, denominator)
        updates.append((inverse_incidence, denominator))
        available[chosen] = False
        order.append(chosen)
        approximate_energies.append(approximate_energies[-1] - float(gain[chosen]))
    return order, approximate_energies, initial


def recompute_prefix_energies(
    laplacian: sp.csr_matrix,
    coordinates: np.ndarray,
    antagonistic_signal: np.ndarray,
    candidates: pd.DataFrame,
    order: list[int],
    checkpoints: list[int],
    alpha: float,
) -> dict[int, float]:
    """Exactly recompute objective energy for materialized selected prefixes."""
    z = coordinates - coordinates.mean(axis=0, keepdims=True)
    y = antagonistic_signal - antagonistic_signal.mean(axis=0, keepdims=True)
    result: dict[int, float] = {}
    for checkpoint in checkpoints:
        chosen = candidates.iloc[order[:min(checkpoint, len(order))]]
        rows = np.r_[chosen.source.to_numpy(), chosen.target.to_numpy()]
        columns = np.r_[chosen.target.to_numpy(), chosen.source.to_numpy()]
        adjacency = sp.coo_matrix(
            (np.ones(len(rows)), (rows, columns)), shape=laplacian.shape,
        ).tocsr()
        augmented = (
            laplacian
            + sp.diags(np.asarray(adjacency.sum(axis=1)).ravel())
            - adjacency
        )
        result[checkpoint] = objective_energy(
            grounded_factor(augmented.tocsr()), z, y, alpha,
        )
    return result
