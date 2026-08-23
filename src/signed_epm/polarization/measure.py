from __future__ import annotations

import argparse
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


def polarization_from_laplacian(
    laplacian: sp.csr_matrix,
    coordinates: np.ndarray,
    rtol: float = 1e-7,
    atol: float = 1e-10,
    maxiter: int = 10_000,
) -> float:
    """Compute sqrt(mean_t x_t^T L^dagger x_t) with a sparse solve."""
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
    return float(np.sqrt(total / values.shape[1]))


def polarization(
    graph: pd.DataFrame,
    coordinates: np.ndarray,
    positive_conductance: float = 1.0,
    negative_conductance: float = 0.1,
) -> float:
    laplacian = build_weighted_laplacian(
        graph, len(coordinates), positive_conductance, negative_conductance,
    )
    return polarization_from_laplacian(laplacian, coordinates)


def load_node_state(path: Path) -> np.ndarray:
    import torch
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected a tensor at {path}")
    return value.detach().cpu().numpy()


def measure_run(
    node_state_path: Path,
    edge_path: Path,
    output_dir: Path,
    k: int,
    positive_conductance: float = 1.0,
    negative_conductance: float = 0.1,
) -> dict:
    state = load_node_state(node_state_path)
    graph = pd.read_csv(edge_path)
    coordinates, singular_values = opinion_coordinates(state, k)
    score = polarization(graph, coordinates, positive_conductance, negative_conductance)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "opinion_coordinates.npy", coordinates)
    result = {
        "schema_version": 1,
        "node_state_path": str(node_state_path),
        "edge_path": str(edge_path),
        "graph_fingerprint": graph_fingerprint(graph, directed=False),
        "num_nodes": int(state.shape[0]),
        "node_state_dimension": int(state.shape[1]),
        "k": int(k),
        "pca_singular_values": singular_values.tolist(),
        "nodewise_normalization": "l2",
        "positive_conductance": float(positive_conductance),
        "negative_conductance": float(negative_conductance),
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
    args = parser.parse_args()
    result = measure_run(
        args.node_state_path, args.edge_path, args.output_dir, args.k,
        args.positive_conductance, args.negative_conductance,
    )
    print(f"polarization={result['polarization']:.10f}")


if __name__ == "__main__":
    main()
