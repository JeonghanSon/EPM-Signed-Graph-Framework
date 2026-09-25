from __future__ import annotations

from pathlib import Path

import pandas as pd

from signed_epm.graph import canonical_undirected


def canonical_edge(source: int, target: int) -> tuple[int, int]:
    return (source, target) if source < target else (target, source)


def model_edges(physical: pd.DataFrame, directed: bool) -> pd.DataFrame:
    base = physical[["source", "target", "sign", "weight"]].copy()
    if not directed or base.empty:
        return base
    reverse = base.rename(columns={"source": "target", "target": "source"})[
        ["source", "target", "sign", "weight"]
    ]
    return pd.concat([base, reverse], ignore_index=True)


def write_augmented_graph(
    physical_edges: list[tuple[int, int]],
    base_directed: pd.DataFrame,
    base_undirected: pd.DataFrame,
    output_dir: Path,
    directed_backbone: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    additions = [canonical_edge(int(source), int(target))
                 for source, target in physical_edges]
    if any(source == target for source, target in additions):
        raise ValueError("augmentation contains a self-loop")
    if len(set(additions)) != len(additions):
        raise ValueError("augmentation contains duplicate physical edges")
    occupied = set(map(tuple, canonical_undirected(base_undirected)[
        ["source", "target"]
    ].astype(int).to_numpy()))
    overlap = occupied.intersection(additions)
    if overlap:
        raise ValueError(f"augmentation contains {len(overlap)} existing physical edges")
    physical = pd.DataFrame(additions, columns=["source", "target"])
    physical = physical.assign(sign=1, weight=1.0)
    encoded = model_edges(physical, directed_backbone)
    directed_augmented = pd.concat([base_directed, encoded], ignore_index=True, sort=False)
    undirected_augmented = pd.concat([base_undirected, physical], ignore_index=True, sort=False)
    physical.to_csv(output_dir / "new_physical_edges.csv", index=False)
    encoded.to_csv(output_dir / "new_model_edges.csv", index=False)
    directed_augmented.to_csv(
        output_dir / "train_snapshot_directed_augmented.csv", index=False,
    )
    undirected_augmented.to_csv(
        output_dir / "train_snapshot_undirected_augmented.csv", index=False,
    )
    return physical, encoded
