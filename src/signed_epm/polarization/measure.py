from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as spla

from signed_epm.graph import canonical_undirected, graph_fingerprint


EPSILON = 1e-12


def opinion_coordinates(node_state: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Project node state onto its first k PCs and normalize every node."""
    state = np.asarray(node_state, dtype=np.float64)
    if state.ndim != 2 or not np.isfinite(state).all():
        raise ValueError("node_state must be a finite two-dimensional array")
    if k <= 0 or k > min(state.shape):
        raise ValueError(f"k must be in [1, {min(state.shape)}], got {k}")
    centered = state - state.mean(axis=0, keepdims=True)
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    raw = centered @ right[:k].T
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    coordinates = raw / np.maximum(norms, EPSILON)
    return coordinates, singular_values[:k]


def build_weighted_laplacian(
    graph: pd.DataFrame,
    num_nodes: int,
    positive_conductance: float = 1.0,
    negative_conductance: float = 0.1,
) -> sp.csr_matrix:
    """Build EPM's PSD Laplacian from one static undirected snapshot."""
    if positive_conductance <= 0 or negative_conductance <= 0:
        raise ValueError("edge conductances must be strictly positive")
    physical = canonical_undirected(graph)
    source = physical.source.to_numpy(dtype=np.int64)
    target = physical.target.to_numpy(dtype=np.int64)
    sign = physical.sign.to_numpy(dtype=np.int64)
    if len(source) and (source.min() < 0 or target.max() >= num_nodes):
        raise ValueError("graph contains a node outside [0, num_nodes)")
    weights = np.where(sign > 0, positive_conductance, negative_conductance)
    adjacency = sp.coo_matrix(
        (np.concatenate([weights, weights]),
         (np.concatenate([source, target]), np.concatenate([target, source]))),
        shape=(num_nodes, num_nodes), dtype=np.float64,
    ).tocsr()
    if csgraph.connected_components(adjacency, directed=False, return_labels=False) != 1:
        raise ValueError("effective-resistance polarization requires a connected graph")
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return (sp.diags(degree) - adjacency).tocsr()


def quadratic_energy_from_laplacian(
    laplacian: sp.csr_matrix,
    coordinates: np.ndarray,
    rtol: float = 1e-7,
    atol: float = 1e-10,
    maxiter: int = 10_000,
) -> float:
    """Compute mean_t x_t^T L^dagger x_t with a sparse solve."""
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != laplacian.shape[0]:
        raise ValueError("coordinate and Laplacian node dimensions differ")
    count = laplacian.shape[0]
    ones = np.ones(count, dtype=np.float64)

    # For centered b, (L + 11^T/n)^-1 b equals L^dagger b.
    operator = spla.LinearOperator(
        laplacian.shape,
        matvec=lambda vector: laplacian @ vector + vector.mean() * ones,
        dtype=np.float64,
    )
    inverse_diagonal = 1.0 / np.maximum(laplacian.diagonal(), EPSILON)
    preconditioner = spla.LinearOperator(
        laplacian.shape, matvec=lambda vector: inverse_diagonal * vector,
        dtype=np.float64,
    )
    total = 0.0
    for axis in range(values.shape[1]):
        centered = values[:, axis] - values[:, axis].mean()
        solution, info = spla.cg(
            operator, centered, M=preconditioner,
            rtol=rtol, atol=atol, maxiter=maxiter,
        )
        if info != 0:
            raise RuntimeError(f"Laplacian conjugate-gradient solve failed (info={info})")
        quadratic = float(centered @ solution)
        if quadratic < -1e-8:
            raise RuntimeError(f"PSD Laplacian produced negative quadratic form {quadratic}")
        total += max(quadratic, 0.0)
    return float(total / values.shape[1])


def polarization_from_laplacian(
    laplacian: sp.csr_matrix,
    coordinates: np.ndarray,
    rtol: float = 1e-7,
    atol: float = 1e-10,
    maxiter: int = 10_000,
) -> float:
    """Compute sqrt(mean_t x_t^T L^dagger x_t) with a sparse solve."""
    energy = quadratic_energy_from_laplacian(
        laplacian, coordinates, rtol=rtol, atol=atol, maxiter=maxiter,
    )
    return float(np.sqrt(max(energy, 0.0)))


def polarization(
    graph: pd.DataFrame,
    coordinates: np.ndarray,
    positive_conductance: float = 1.0,
    negative_conductance: float = 0.1,
    antagonistic_weight: float = 0.05,
) -> float:
    """Compute the canonical signed EPM score.

    The structural and antagonistic signals share the signed conductance
    geometry ``L_eta``.  Setting ``antagonistic_weight=0`` recovers the
    structural-only formulation without maintaining a separate legacy path.
    """
    if antagonistic_weight < 0:
        raise ValueError("antagonistic_weight must be nonnegative")
    components = signed_energy_components(
        graph, coordinates, positive_conductance, negative_conductance,
    )
    return float(np.sqrt(max(
        components["structural_energy"]
        + antagonistic_weight * components["antagonistic_energy"],
        0.0,
    )))


def score_from_components(
    structural_energy: float,
    antagonistic_energy: float,
    antagonistic_weight: float,
) -> float:
    """Combine already computed canonical component energies."""
    if antagonistic_weight < 0:
        raise ValueError("antagonistic_weight must be nonnegative")
    return float(np.sqrt(max(
        structural_energy + antagonistic_weight * antagonistic_energy, 0.0,
    )))


def negative_edge_laplacian(graph: pd.DataFrame, num_nodes: int) -> sp.csr_matrix:
    """Return the unweighted PSD Laplacian of the negative physical layer."""
    physical = canonical_undirected(graph)
    negative = physical.loc[physical.sign < 0]
    source = negative.source.to_numpy(dtype=np.int64)
    target = negative.target.to_numpy(dtype=np.int64)
    if not len(source):
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float64)
    adjacency = sp.coo_matrix(
        (np.ones(2 * len(source), dtype=np.float64),
         (np.concatenate([source, target]), np.concatenate([target, source]))),
        shape=(num_nodes, num_nodes), dtype=np.float64,
    ).tocsr()
    return (sp.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency).tocsr()


def signed_energy_components(
    graph: pd.DataFrame,
    coordinates: np.ndarray,
    positive_conductance: float = 1.0,
    negative_conductance: float = 0.1,
) -> dict[str, float | int]:
    """Return the structural and coherent-antagonistic EPM energies."""
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("coordinates must be a two-dimensional array")
    physical = canonical_undirected(graph)
    laplacian = build_weighted_laplacian(
        physical, len(values), positive_conductance, negative_conductance,
    )
    antagonistic_signal = negative_edge_laplacian(physical, len(values)) @ values
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
        "positive_conductance": float(positive_conductance),
        "negative_conductance": float(negative_conductance),
        "structural_energy": structural_energy,
        "structural_polarization": float(np.sqrt(max(structural_energy, 0.0))),
        "antagonistic_energy": antagonistic_energy,
        "antagonistic_polarization": float(np.sqrt(max(antagonistic_energy, 0.0))),
    }


def load_node_state(path: Path) -> np.ndarray:
    import torch
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected a tensor at {path}")
    return value.detach().cpu().numpy()


def node_state_fingerprint(state: np.ndarray) -> str:
    """Content fingerprint preventing reuse across retrained representations."""
    values = np.ascontiguousarray(np.asarray(state))
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def measure_run(
    node_state_path: Path,
    edge_path: Path,
    output_dir: Path,
    k: int,
    positive_conductance: float = 1.0,
    negative_conductance: float = 0.1,
    antagonistic_weight: float = 0.05,
) -> dict:
    if antagonistic_weight < 0:
        raise ValueError("antagonistic_weight must be nonnegative")
    state = load_node_state(node_state_path)
    graph = pd.read_csv(edge_path)
    coordinates, singular_values = opinion_coordinates(state, k)
    components = signed_energy_components(
        graph, coordinates, positive_conductance, negative_conductance,
    )
    score = float(np.sqrt(max(
        components["structural_energy"]
        + antagonistic_weight * components["antagonistic_energy"],
        0.0,
    )))
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "opinion_coordinates.npy", coordinates)
    result = {
        "schema_version": 2,
        "measure_version": "signed_coherent_v1",
        "node_state_path": str(node_state_path),
        "node_state_fingerprint": node_state_fingerprint(state),
        "edge_path": str(edge_path),
        "graph_fingerprint": graph_fingerprint(graph, directed=False),
        "num_nodes": int(state.shape[0]),
        "node_state_dimension": int(state.shape[1]),
        "k": int(k),
        "pca_singular_values": singular_values.tolist(),
        "nodewise_normalization": "l2",
        "positive_conductance": float(positive_conductance),
        "negative_conductance": float(negative_conductance),
        "antagonistic_weight": float(antagonistic_weight),
        "antagonistic_signal": "Y = L^- Z",
        "score_formula": "sqrt(structural_energy + alpha * antagonistic_energy)",
        **components,
        "polarization": score,
    }
    (output_dir / "measurement.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure EPM polarization for one learned run")
    parser.add_argument("--node-state-path", type=Path, required=True)
    parser.add_argument("--edge-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--positive-conductance", type=float, default=1.0)
    parser.add_argument("--negative-conductance", type=float, default=0.1)
    parser.add_argument("--antagonistic-weight", type=float, default=0.05)
    args = parser.parse_args()
    result = measure_run(
        args.node_state_path, args.edge_path, args.output_dir, args.k,
        args.positive_conductance, args.negative_conductance,
        args.antagonistic_weight,
    )
    print(f"polarization={result['polarization']:.10f}")


if __name__ == "__main__":
    main()
