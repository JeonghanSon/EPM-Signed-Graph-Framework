from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from signed_epm.graph import canonical_undirected
from signed_epm.polarization.measure import (
    build_weighted_laplacian,
    quadratic_energy_from_laplacian,
)


def build_edge_subset_laplacian(
    graph: pd.DataFrame, num_nodes: int, sign: int,
) -> sp.csr_matrix:
    """Build an unweighted Laplacian from one sign-specific edge layer."""
    physical = canonical_undirected(graph)
    selected = physical.loc[(physical.sign > 0) if sign > 0 else (physical.sign < 0)]
    source = selected.source.to_numpy(dtype=np.int64)
    target = selected.target.to_numpy(dtype=np.int64)
    if not len(source):
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float64)
    adjacency = sp.coo_matrix(
        (np.ones(2 * len(source), dtype=np.float64),
         (np.concatenate([source, target]), np.concatenate([target, source]))),
        shape=(num_nodes, num_nodes), dtype=np.float64,
    ).tocsr()
    return (sp.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency).tocsr()


def signed_weighted_energy_components(
    graph: pd.DataFrame,
    coordinates: np.ndarray,
    negative_conductance: float = 0.1,
) -> dict[str, float | int]:
    """Compute structural and coherent-antagonistic energies.

    Both terms use the shared signed geometry ``L_eta^dagger``. The
    antagonistic signal is ``Y = L^- Z``.
    """
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("coordinates must be a two-dimensional array")
    physical = canonical_undirected(graph)
    laplacian = build_weighted_laplacian(
        physical, len(values), positive_conductance=1.0,
        negative_conductance=negative_conductance,
    )
    negative_laplacian = build_edge_subset_laplacian(physical, len(values), sign=-1)
    antagonistic_signal = negative_laplacian @ values
    structural_energy = quadratic_energy_from_laplacian(laplacian, values)
    antagonistic_energy = quadratic_energy_from_laplacian(
        laplacian, antagonistic_signal,
    )
    return {
        "num_nodes": len(values),
        "k": values.shape[1],
        "num_edges": len(physical),
        "num_positive_edges": int((physical.sign > 0).sum()),
        "num_negative_edges": int((physical.sign < 0).sum()),
        "negative_prevalence": float((physical.sign < 0).mean()),
        "negative_conductance": float(negative_conductance),
        "structural_energy": structural_energy,
        "structural_polarization": float(np.sqrt(max(structural_energy, 0.0))),
        "antagonistic_energy": antagonistic_energy,
        "antagonistic_polarization": float(np.sqrt(max(antagonistic_energy, 0.0))),
    }


def score_from_components(
    structural_energy: float,
    antagonistic_energy: float,
    antagonistic_weight: float,
) -> float:
    """Return sqrt(P_str^2 + alpha P_ant^2)."""
    if antagonistic_weight < 0:
        raise ValueError("antagonistic_weight must be nonnegative")
    return float(np.sqrt(max(
        structural_energy + antagonistic_weight * antagonistic_energy, 0.0,
    )))
