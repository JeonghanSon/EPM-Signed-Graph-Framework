from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from signed_epm.graph import canonical_undirected, graph_fingerprint
from signed_epm.models.base import EncoderAdapter, EncoderConfig


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
            return SGCN(**kwargs).to(device)

        fingerprint = graph_fingerprint(physical, directed=False)
        cache = Path(cache_root) / "tsvd" / fingerprint / f"dim_{config.input_dimension}.pt"
        if cache.exists():
            initial = torch.load(cache, map_location=device, weights_only=True)
            return SGCN(**kwargs, init_emb=initial).to(device)
        model = SGCN(**kwargs).to(device)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.x.detach().cpu(), cache)
        return model
