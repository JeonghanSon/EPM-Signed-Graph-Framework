from __future__ import annotations

from pathlib import Path
import os
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from signed_epm.graph import canonical_undirected, graph_fingerprint
from signed_epm.models.base import EncoderAdapter, EncoderConfig


def signed_unit_adjacency(physical: pd.DataFrame, num_nodes: int) -> sp.csr_matrix:
    """Build the symmetric +1/-1 adjacency used by SGCN TSVD."""
    graph = canonical_undirected(physical)
    source = graph["source"].to_numpy(dtype=np.int64)
    target = graph["target"].to_numpy(dtype=np.int64)
    sign = np.where(graph["sign"].to_numpy(dtype=np.int64) > 0, 1.0, -1.0)
    return sp.coo_matrix(
        (np.concatenate([sign, sign]),
         (np.concatenate([source, target]), np.concatenate([target, source]))),
        shape=(num_nodes, num_nodes), dtype=np.float64,
    ).tocsr()


def signed_unit_tsvd(
    physical: pd.DataFrame,
    num_nodes: int,
    dimension: int,
    random_state: int = 0,
):
    """Create deterministic TSVD features from a +1/-1 physical snapshot.

    The packaged SGCN helper symmetrizes its input internally.  Passing the
    adapter's already-bidirectional message edges to that helper coalesces two
    copies of each orientation and changes positive weights from +1 to +3.
    Constructing the symmetric signed adjacency directly avoids that mismatch.
    """
    import torch

    adjacency = signed_unit_adjacency(physical, num_nodes)
    decomposition = TruncatedSVD(
        n_components=dimension, n_iter=128, random_state=random_state,
    )
    features = decomposition.fit(adjacency).components_.T
    return torch.from_numpy(features).to(torch.float)


class SGCNAdapter(EncoderAdapter):
    name = "sgcn"
    directed = False

    def __init__(self, structure_lambda: float = 5.0, normalize_embeddings: bool = False):
        self.structure_lambda = structure_lambda
        self.normalize_embeddings = normalize_embeddings

    def build(self, train_graph: pd.DataFrame, num_nodes: int, config: EncoderConfig,
              device: str, cache_root: Path | None = None):
        import torch
        from torch_geometric_signed_directed.nn.signed import SGCN

        physical = canonical_undirected(train_graph)
        snapshot = physical[["source", "target", "sign"]].to_numpy(dtype=np.int64)
        messages = np.concatenate([snapshot, snapshot[:, [1, 0, 2]]], axis=0)
        edges = torch.as_tensor(messages, dtype=torch.long, device=device)
        kwargs = dict(node_num=num_nodes, edge_index_s=edges,
                      in_dim=config.input_dimension, out_dim=config.output_dimension,
                      layer_num=config.layers, lamb=self.structure_lambda,
                      norm_emb=self.normalize_embeddings)
        if cache_root is None:
            initial = signed_unit_tsvd(
                physical, num_nodes, config.input_dimension, random_state=0,
            )
            return SGCN(**kwargs, init_emb=initial.to(device)).to(device)

        fingerprint = graph_fingerprint(physical, directed=False)
        cache = (Path(cache_root) / "tsvd_signed_unit_v2" / fingerprint /
                 f"nodes_{num_nodes}__dim_{config.input_dimension}.pt")
        if cache.exists():
            initial = torch.load(cache, map_location=device, weights_only=True)
            return SGCN(**kwargs, init_emb=initial).to(device)

        initial = signed_unit_tsvd(
            physical, num_nodes, config.input_dimension, random_state=0,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        # The deterministic payload makes concurrent first writers equivalent;
        # atomic replacement prevents readers from observing a partial file.
        with tempfile.NamedTemporaryFile(
            dir=cache.parent, prefix=f".{cache.name}.", delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            torch.save(initial, temporary)
            os.replace(temporary, cache)
        finally:
            temporary.unlink(missing_ok=True)
        return SGCN(**kwargs, init_emb=initial.to(device)).to(device)
