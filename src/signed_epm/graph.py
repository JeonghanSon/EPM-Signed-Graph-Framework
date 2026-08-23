from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def canonical_undirected(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one deterministic row per physical pair, retaining its last event."""
    result = frame.copy()
    source = result["source"].to_numpy(dtype=np.int64)
    target = result["target"].to_numpy(dtype=np.int64)
    result["source"] = np.minimum(source, target)
    result["target"] = np.maximum(source, target)
    ordering = [column for column in ("timestamp", "event_id") if column in result]
    if ordering:
        result = result.sort_values(ordering, kind="mergesort")
    return result.drop_duplicates(["source", "target"], keep="last").sort_values(
        ["source", "target"], kind="mergesort").reset_index(drop=True)


def graph_fingerprint(frame: pd.DataFrame, directed: bool) -> str:
    """Content fingerprint used to prevent cross-graph feature-cache reuse."""
    graph = frame if directed else canonical_undirected(frame)
    values = graph[["source", "target", "sign"]].to_numpy(dtype=np.int64)
    values = values[np.lexsort((values[:, 2], values[:, 1], values[:, 0]))]
    digest = hashlib.sha256()
    digest.update(b"directed\0" if directed else b"undirected\0")
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetSplits:
    root: Path
    manifest: dict
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @property
    def num_nodes(self) -> int:
        counts = self.manifest.get("counts", self.manifest)
        return int(counts.get("num_nodes", self.manifest.get("num_nodes")))

    @classmethod
    def load(cls, root: Path, read_test: bool = True) -> "DatasetSplits":
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        validation = pd.read_csv(root / "val_events.csv")
        test = pd.read_csv(root / "test_events.csv") if read_test else validation.iloc[0:0].copy()
        return cls(root, manifest, pd.read_csv(root / "train_events.csv"), validation, test)
