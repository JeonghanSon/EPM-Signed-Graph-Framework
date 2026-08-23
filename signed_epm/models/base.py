from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EncoderConfig:
    input_dimension: int
    output_dimension: int
    layers: int
    learning_rate: float
    epochs: int
    weight_decay: float = 0.0


class EncoderAdapter(ABC):
    """Boundary between native signed encoders and the EPM pipeline."""

    name: str
    directed: bool

    @abstractmethod
    def build(self, train_graph: pd.DataFrame, num_nodes: int, config: EncoderConfig,
              device: str, cache_root: Path | None = None) -> Any:
        raise NotImplementedError

    @staticmethod
    def loss(model: Any):
        return model.loss()

    @staticmethod
    def node_state(model: Any) -> np.ndarray:
        import torch
        model.eval()
        with torch.no_grad():
            state = model().detach().cpu().numpy()
        if not np.isfinite(state).all():
            raise RuntimeError("encoder produced non-finite node state")
        return state

    @staticmethod
    def state_dict(model: Any) -> dict:
        return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
