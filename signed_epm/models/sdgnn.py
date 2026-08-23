from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from signed_epm.models.base import EncoderAdapter, EncoderConfig


class SDGNNAdapter(EncoderAdapter):
    name = "sdgnn"
    directed = True

    def __init__(self, direction_lambda: float = 5.0, triangle_lambda: float = 1.0):
        self.direction_lambda = direction_lambda
        self.triangle_lambda = triangle_lambda

    def build(self, train_graph: pd.DataFrame, num_nodes: int, config: EncoderConfig,
              device: str, cache_root: Path | None = None):
        del cache_root  # SDGNN owns its native initialization.
        if config.input_dimension != config.output_dimension:
            raise ValueError("packaged SDGNN requires equal input and output dimensions")
        import torch
        from torch_geometric_signed_directed.nn.signed import SDGNN

        snapshot = train_graph[["source", "target", "sign"]].to_numpy(dtype=np.int64)
        edges = torch.as_tensor(snapshot, dtype=torch.long, device=device)
        return SDGNN(node_num=num_nodes, edge_index_s=edges,
                     in_dim=config.input_dimension, out_dim=config.output_dimension,
                     layer_num=config.layers, lamb_d=self.direction_lambda,
                     lamb_t=self.triangle_lambda).to(device)
