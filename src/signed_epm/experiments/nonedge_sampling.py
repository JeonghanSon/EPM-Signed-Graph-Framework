from __future__ import annotations

import numpy as np


def uniform_nonedge_order(
    num_nodes: int,
    occupied: set[tuple[int, int]],
    budget: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Sample physical undirected non-edges uniformly without replacement."""
    maximum = num_nodes * (num_nodes - 1) // 2 - len(occupied)
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if budget > maximum:
        raise ValueError(
            f"requested {budget} non-edges, but only {maximum} physical non-edges exist"
        )
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2026, budget]))
    chosen: list[tuple[int, int]] = []
    chosen_set: set[tuple[int, int]] = set()
    while len(chosen) < budget:
        left, right = map(int, rng.integers(0, num_nodes, size=2))
        if left == right:
            continue
        edge = tuple(sorted((left, right)))
        if edge in occupied or edge in chosen_set:
            continue
        chosen.append(edge)
        chosen_set.add(edge)
    return chosen
