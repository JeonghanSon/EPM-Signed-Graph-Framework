from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from signed_epm.data.preprocess import connected
from signed_epm.graph import graph_fingerprint


ROOT = Path(__file__).resolve().parents[3]
LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9)


def ratio_token(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def balanced_communities(num_nodes: int, num_communities: int) -> np.ndarray:
    if num_communities < 2 or num_communities > num_nodes:
        raise ValueError("num_communities must be in [2, num_nodes]")
    sizes = np.full(num_communities, num_nodes // num_communities, dtype=int)
    sizes[: num_nodes % num_communities] += 1
    return np.repeat(np.arange(num_communities), sizes)


def candidate_pools(communities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.asarray(
        [(left, right) for left in range(len(communities))
         for right in range(left + 1, len(communities))], dtype=np.int64,
    )
    same = communities[pairs[:, 0]] == communities[pairs[:, 1]]
    return pairs[same], pairs[~same]


def take(pool: np.ndarray, count: int) -> np.ndarray:
    if count < 0 or count > len(pool):
        raise ValueError(f"requested {count} edges from a pool of {len(pool)}")
    return pool[:count].copy()


def allocate(total: int, ratio: float) -> tuple[int, int]:
    intra = int(round(total * ratio))
    return intra, total - intra


def edge_frame(positive: np.ndarray, negative: np.ndarray, seed: int) -> pd.DataFrame:
    pairs = np.vstack([positive, negative])
    signs = np.concatenate([
        np.ones(len(positive), dtype=np.int8),
        -np.ones(len(negative), dtype=np.int8),
    ])
    if len({tuple(pair) for pair in pairs}) != len(pairs):
        raise RuntimeError("positive and negative physical edge sets overlap")
    rng = np.random.default_rng(seed)
    direction = rng.integers(0, 2, size=len(pairs), dtype=np.int8).astype(bool)
    sources = np.where(direction, pairs[:, 1], pairs[:, 0])
    targets = np.where(direction, pairs[:, 0], pairs[:, 1])
    order = rng.permutation(len(pairs))
    return pd.DataFrame({
        "source": sources[order], "target": targets[order],
        "sign": signs[order], "weight": signs[order].astype(float),
        "event_id": np.arange(len(pairs), dtype=np.int64),
    })


def sample_held_out(
    train: pd.DataFrame,
    all_pairs: np.ndarray,
    count: int,
    negative_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if count == 0:
        empty = pd.DataFrame({
            "source": pd.Series(dtype=np.int64), "target": pd.Series(dtype=np.int64),
            "sign": pd.Series(dtype=np.int8), "weight": pd.Series(dtype=float),
            "event_id": pd.Series(dtype=np.int64),
        })
        return empty.copy(), empty.copy()
    used = {tuple(sorted(pair)) for pair in train[["source", "target"]].to_numpy()}
    available = np.asarray([pair for pair in all_pairs if tuple(pair) not in used])
    needed = 2 * count
    if needed > len(available):
        raise ValueError("not enough physical non-train pairs for validation and test")
    rng = np.random.default_rng(seed)
    selected = available[rng.permutation(len(available))[:needed]]
    negative = int(round(count * negative_fraction))

    def make(pairs: np.ndarray, offset: int) -> pd.DataFrame:
        signs = np.ones(count, dtype=np.int8)
        signs[:negative] = -1
        signs = signs[rng.permutation(count)]
        direction = rng.integers(0, 2, size=count).astype(bool)
        source = np.where(direction, pairs[:, 1], pairs[:, 0])
        target = np.where(direction, pairs[:, 0], pairs[:, 1])
        return pd.DataFrame({
            "source": source, "target": target, "sign": signs,
            "weight": signs.astype(float),
            "event_id": np.arange(offset, offset + count, dtype=np.int64),
        })

    return make(selected[:count], len(train)), make(selected[count:], len(train) + count)


def generated_opinion(num_nodes: int, distribution: str, seed: int,
                      exponent: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        values = rng.uniform(-1.0, 1.0, size=num_nodes)
    elif distribution == "power_law":
        if exponent <= 1:
            raise ValueError("power-law exponent must exceed one")
        magnitudes = rng.pareto(exponent, size=num_nodes) + 1.0
        magnitudes /= magnitudes.max()
        signs = np.ones(num_nodes)
        signs[: num_nodes // 2] = -1.0
        rng.shuffle(signs)
        values = signs * magnitudes
    else:
        raise ValueError(f"unknown opinion distribution: {distribution}")
    return values.astype(np.float64)


def community_aligned_opinion(num_nodes: int, distribution: str, seed: int,
                              exponent: float = 2.0) -> np.ndarray:
    """Return an exactly symmetric spectrum ordered from negative to positive."""
    if num_nodes % 2:
        raise ValueError("exact symmetric alignment currently requires an even node count")
    rng = np.random.default_rng(seed)
    half = num_nodes // 2
    if distribution == "uniform":
        magnitudes = rng.uniform(0.0, 1.0, size=half)
    elif distribution == "power_law":
        if exponent <= 1:
            raise ValueError("power-law exponent must exceed one")
        magnitudes = rng.pareto(exponent, size=half) + 1.0
        magnitudes /= magnitudes.max()
    else:
        raise ValueError(f"unknown opinion distribution: {distribution}")
    return np.sort(np.concatenate([-magnitudes, magnitudes])).astype(np.float64)


def visualization_nodes(communities: np.ndarray, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    count = int(communities.max()) + 1
    centers = np.column_stack([
        3.0 * np.cos(2 * np.pi * np.arange(count) / count),
        3.0 * np.sin(2 * np.pi * np.arange(count) / count),
    ])
    positions = centers[communities] + rng.normal(0.0, 0.35, (len(communities), 2))
    return pd.DataFrame({
        "node_id": np.arange(len(communities)), "community": communities,
        "layout_x": positions[:, 0], "layout_y": positions[:, 1],
    })


@dataclass(frozen=True)
class GeneratorConfig:
    num_nodes: int = 100
    num_communities: int = 3
    positive_edges: int = 400
    negative_edges: int = 100
    held_out_edges: int = 75
    structural_negative_intra_ratio: float = 0.5
    antagonistic_positive_intra_ratio: float = 0.8
    negative_fraction: float = 0.2


def _write_dataset(root: Path, name: str, train: pd.DataFrame, validation: pd.DataFrame,
                   test: pd.DataFrame, nodes: pd.DataFrame, config: GeneratorConfig,
                   metadata: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    train.to_csv(root / "train_events.csv", index=False)
    validation.to_csv(root / "val_events.csv", index=False)
    test.to_csv(root / "test_events.csv", index=False)
    train.to_csv(root / "train_snapshot_directed.csv", index=False)
    undirected = train.copy()
    source = np.minimum(undirected.source, undirected.target)
    target = np.maximum(undirected.source, undirected.target)
    undirected["source"], undirected["target"] = source, target
    undirected.sort_values(["source", "target"]).to_csv(
        root / "train_snapshot_undirected.csv", index=False,
    )
    nodes.to_csv(root / "nodes.csv", index=False)
    manifest = {
        "schema_version": 1, "dataset": name,
        "synthetic": {**config.__dict__, **metadata},
        "counts": {
            "num_nodes": config.num_nodes,
            "splits": {"train": len(train), "val": len(validation), "test": len(test)},
            "positive": {"train": int((train.sign > 0).sum()),
                         "val": int((validation.sign > 0).sum()),
                         "test": int((test.sign > 0).sum())},
            "negative": {"train": int((train.sign < 0).sum()),
                         "val": int((validation.sign < 0).sum()),
                         "test": int((test.sign < 0).sum())},
        },
        "validation": {
            "train_connected": connected(train, config.num_nodes),
            "positive_train_connected": connected(train[train.sign > 0], config.num_nodes),
            "pair_disjoint": True, "transductive": True,
        },
        "communities": {"selected_k": config.num_communities, "source": "ground_truth"},
        "graph_fingerprint": graph_fingerprint(train, directed=False),
    }
    if not all(manifest["validation"].values()):
        raise RuntimeError(f"synthetic validation failed: {manifest['validation']}")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def generate_replicate(output_root: Path, graph_seed: int, config: GeneratorConfig,
                       levels: tuple[float, ...] = LEVELS, max_attempts: int = 100) -> list[dict]:
    communities = balanced_communities(config.num_nodes, config.num_communities)
    intra, inter = candidate_pools(communities)
    all_pairs = np.vstack([intra, inter])
    nodes = visualization_nodes(communities, seed=10_000 + graph_seed)

    for attempt in range(max_attempts):
        rng = np.random.default_rng(graph_seed * 1009 + attempt)
        pos_intra, pos_inter = intra[rng.permutation(len(intra))], inter[rng.permutation(len(inter))]
        # Reserve negatives outside the union of every positive edge that can
        # appear in the coupled structural sequence.
        max_positive_in = max(allocate(config.positive_edges, level)[0] for level in levels)
        max_positive_out = max(allocate(config.positive_edges, level)[1] for level in levels)
        structural_neg_intra = pos_intra[max_positive_in:]
        structural_neg_inter = pos_inter[max_positive_out:]
        negative_in, negative_out = allocate(config.negative_edges, config.structural_negative_intra_ratio)
        fixed_negative = np.vstack([
            take(structural_neg_intra, negative_in),
            take(structural_neg_inter, negative_out),
        ])
        structural: list[tuple[float, pd.DataFrame]] = []
        valid = True
        for level in levels:
            positive_in, positive_out = allocate(config.positive_edges, level)
            positive = np.vstack([take(pos_intra, positive_in), take(pos_inter, positive_out)])
            frame = edge_frame(positive, fixed_negative, graph_seed * 10_000 + int(level * 100))
            if not connected(frame[frame.sign > 0], config.num_nodes):
                valid = False; break
            structural.append((level, frame))
        if not valid:
            continue

        base_in, base_out = allocate(config.positive_edges, config.antagonistic_positive_intra_ratio)
        fixed_positive = np.vstack([take(pos_intra, base_in), take(pos_inter, base_out)])
        antagonistic_neg_intra = pos_intra[base_in:]
        antagonistic_neg_inter = pos_inter[base_out:]
        antagonistic: list[tuple[float, pd.DataFrame]] = []
        for level in levels:
            negative_out = int(round(config.negative_edges * level))
            negative_in = config.negative_edges - negative_out
            negative = np.vstack([
                take(antagonistic_neg_intra, negative_in),
                take(antagonistic_neg_inter, negative_out),
            ])
            antagonistic.append((level, edge_frame(
                fixed_positive, negative, graph_seed * 20_000 + int(level * 100),
            )))
        if valid:
            break
    else:
        raise RuntimeError(f"could not generate connected collision-free replicate {graph_seed}")

    records = []
    for experiment, sequence, ratio_name in (
        ("structural", structural, "positive_intra_ratio"),
        ("antagonistic", antagonistic, "negative_inter_ratio"),
    ):
        for level, train in sequence:
            val, test = sample_held_out(
                train, all_pairs, config.held_out_edges, config.negative_fraction,
                seed=graph_seed * 30_000 + int(level * 100) + (0 if experiment == "structural" else 500),
            )
            dataset = f"{experiment}_k{config.num_communities}_g{graph_seed}_r{ratio_token(level)}"
            destination = output_root / experiment / f"k{config.num_communities}" / f"graph_seed_{graph_seed}" / f"ratio_{ratio_token(level)}"
            _write_dataset(destination, dataset, train, val, test, nodes, config, {
                "experiment": experiment, "graph_seed": graph_seed,
                ratio_name: level, "generation_attempt": attempt,
                "level_coupling": "shared randomized candidate-pool order",
            })
            records.append({"experiment": experiment, "level": level,
                            "graph_seed": graph_seed, "path": str(destination)})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate controlled synthetic signed graphs")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "data" / "synthetic" / "k2_controlled")
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--opinion-seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--num-nodes", type=int, default=100)
    parser.add_argument("--communities", type=int, default=2)
    parser.add_argument("--positive-edges", type=int, default=400)
    parser.add_argument("--negative-edges", type=int, default=100)
    parser.add_argument("--held-out-edges", type=int, default=75)
    parser.add_argument("--opinion-mode", choices=["random", "community_aligned"],
                        default="community_aligned")
    parser.add_argument("--also-random-opinions", action="store_true",
                        help="save a second random assignment on the identical graphs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output exists: {args.output_root} (use --overwrite)")
        shutil.rmtree(args.output_root)
    config = GeneratorConfig(args.num_nodes, args.communities, args.positive_edges,
                             args.negative_edges, args.held_out_edges)
    records = []
    for seed in args.graph_seeds:
        records.extend(generate_replicate(args.output_root, seed, config))
    opinion_root = args.output_root / "opinions"
    opinion_root.mkdir(parents=True, exist_ok=True)
    opinions = []
    for distribution in ("uniform", "power_law"):
        for seed in args.opinion_seeds:
            values = (generated_opinion(config.num_nodes, distribution, seed)
                      if args.opinion_mode == "random" else
                      community_aligned_opinion(config.num_nodes, distribution, seed))
            path = opinion_root / f"{distribution}_seed_{seed}.csv"
            pd.DataFrame({"node_id": np.arange(config.num_nodes), "opinion": values}).to_csv(path, index=False)
            opinions.append({"distribution": distribution, "seed": seed, "path": str(path),
                             "sha256": hashlib.sha256(values.tobytes()).hexdigest()})
    additional_random = []
    if args.also_random_opinions:
        random_root = args.output_root / "opinions_random"
        random_root.mkdir(parents=True, exist_ok=True)
        for distribution in ("uniform", "power_law"):
            for seed in args.opinion_seeds:
                values = generated_opinion(config.num_nodes, distribution, seed)
                path = random_root / f"{distribution}_seed_{seed}.csv"
                pd.DataFrame({"node_id": np.arange(config.num_nodes),
                              "opinion": values}).to_csv(path, index=False)
                additional_random.append({
                    "distribution": distribution, "seed": seed, "path": str(path),
                    "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                })
    summary = {"schema_version": 1, "config": config.__dict__, "levels": list(LEVELS),
               "opinion_mode": args.opinion_mode,
               "graphs": records, "opinions": opinions,
               "additional_random_opinions": additional_random}
    (args.output_root / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(args.output_root), "graphs": len(records),
                      "opinions": len(opinions)}, indent=2))


if __name__ == "__main__":
    main()
