from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class IterativeGroundedSolver:
    """Preconditioned-CG solver for a grounded graph Laplacian."""

    def __init__(self, matrix: sp.csr_matrix, tolerance: float, maximum_iterations: int):
        self.matrix = matrix.tocsr()
        self.tolerance = float(tolerance)
        self.maximum_iterations = int(maximum_iterations)
        diagonal = self.matrix.diagonal()
        inverse = np.divide(
            1.0, diagonal, out=np.zeros_like(diagonal), where=diagonal != 0,
        )
        self.preconditioner = spla.LinearOperator(
            self.matrix.shape, matvec=lambda vector: inverse * vector,
        )
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
                maxiter=self.maximum_iterations, M=self.preconditioner,
                callback=callback,
            )
            if info != 0:
                raise RuntimeError(f"CG solve failed to converge (info={info})")
            result[:, column] = solution
            self.solve_count += 1
            self.iteration_count += iterations
        return result[:, 0] if one_dimensional else result


def grounded_factor(
    laplacian: sp.csr_matrix,
    solver: str = "direct",
    tolerance: float = 1e-6,
    maximum_iterations: int = 10_000,
):
    """Factor/prepare a grounded Laplacian preserving endpoint differences."""
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


def rank_one_inverse_action_update(
    inverse_action: np.ndarray,
    inverse_incidence: np.ndarray,
    source: int,
    target: int,
    denominator: float,
) -> np.ndarray:
    """Update ``L^-1 X`` after adding one unit-conductance edge."""
    if denominator <= 0:
        raise ValueError("rank-one update denominator must be positive")
    values = np.asarray(inverse_action, dtype=np.float64)
    vector = np.asarray(inverse_incidence, dtype=np.float64)
    if values.shape[0] != len(vector):
        raise ValueError("inverse action and incidence dimensions differ")
    contrast = values[source] - values[target]
    return values - vector[:, None] * contrast[None, :] / denominator


def objective_energy(factor, structural: np.ndarray, antagonistic: np.ndarray,
                     alpha: float) -> float:
    dimension = structural.shape[1]
    return float((
        np.sum(structural * solve_grounded(factor, structural))
        + alpha * np.sum(antagonistic * solve_grounded(factor, antagonistic))
    ) / dimension)
