from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from signed_epm.data.preprocess import DEFAULT_CONFIG, ROOT, load_json


def community_sizes(communities: dict[int, list[int]]) -> list[int]:
    return [len(nodes) for nodes in communities.values()]


def estimate_signed_louvain(dataset_dir: Path, dataset: str, minimum_size: int,
                            seeds: list[int], overwrite: bool = False) -> dict:
    """Run the preserved signed multilayer Louvain implementation on train."""
    # Transitional import: this is repository-local and removes the former
    # dependency on ../EPM-ICDM. It will move under signed_epm after parity.
    from EPM.preprocessing.community_detection.run_signed_louvain import run_signed_louvain

    train_path = dataset_dir / "train_snapshot_undirected.csv"
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    output = dataset_dir / "communities"
    summary_path = output / "signed_louvain_summary.json"
    if summary_path.exists() and not overwrite:
        return load_json(summary_path)
    output.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_path)
    signs = np.sign(train["sign" if "sign" in train else "weight"]).astype(float)
    edges = list(zip(train["source"].astype(int), train["target"].astype(int), signs))
    num_nodes = int(train[["source", "target"]].to_numpy().max()) + 1
    valid_counts = []
    for seed in seeds:
        communities, _ = run_signed_louvain(edges, num_nodes, str(output), seed=seed)
        rows = [{"community_id": int(key), "nodes": ",".join(map(str, sorted(nodes))),
                 "size": len(nodes)} for key, nodes in sorted(communities.items())]
        pd.DataFrame(rows).to_csv(output / f"communities_seed{seed}.csv", index=False)
        generic = output / "communities.csv"
        if generic.exists():
            generic.unlink()
        valid = sum(size >= minimum_size for size in community_sizes(communities))
        valid_counts.append(valid)
        print(f"{dataset} seed={seed}: valid_communities={valid}", flush=True)

    counts = Counter(valid_counts)
    # Counter preserves first occurrence for ties, matching the legacy rule.
    selected = int(counts.most_common(1)[0][0])
    summary = {
        "schema_version": 1,
        "dataset": dataset,
        "source_graph": "train_snapshot_undirected.csv",
        "method": "signed_louvain",
        "positive_layer_weight": 1.0,
        "negative_layer_weight": -1.0,
        "resolution": [1.0, 1.0],
        "minimum_community_size": minimum_size,
        "seeds": seeds,
        "valid_community_counts": valid_counts,
        "mean": float(np.mean(valid_counts)),
        "selection_rule": "mode; first occurrence breaks ties",
        "selected_k": selected,
        "k_source": "computed",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest_path = dataset_dir / "manifest.json"
    manifest = load_json(manifest_path)
    manifest["communities"].update({"selected_k": selected, "k_source": "computed"})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return summary


def write_fallback(dataset_dir: Path, dataset: str, spec: dict, seeds: list[int]) -> dict:
    output = dataset_dir / "communities"
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "dataset": dataset,
        "source_graph": "train_snapshot_undirected.csv",
        "method": "documented_fallback",
        "minimum_community_size": spec["minimum_community_size"],
        "seeds": seeds,
        "valid_community_counts": [],
        "selection_rule": "dataset configuration fallback",
        "selected_k": spec["fallback_k"],
        "k_source": "fallback",
    }
    (output / "signed_louvain_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest_path = dataset_dir / "manifest.json"
    manifest = load_json(manifest_path)
    manifest["communities"].update({"selected_k": spec["fallback_k"], "k_source": "fallback"})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate PCA k with signed Louvain on the train graph")
    parser.add_argument("--dataset", nargs="+", default=["bitcoinalpha"])
    parser.add_argument("--processed-root", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--metadata-root", type=Path, default=ROOT / "data" / "metadata" / "communities")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--fallback", action="store_true",
                        help="explicitly write the documented fallback instead of running Louvain")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    configs = load_json(args.config)
    datasets = list(configs) if args.dataset == ["all"] else args.dataset
    for dataset in datasets:
        if dataset not in configs:
            raise ValueError(f"unknown dataset: {dataset}")
        spec, dataset_dir = configs[dataset], args.processed_root / dataset
        summary = (write_fallback(dataset_dir, dataset, spec, args.seeds) if args.fallback else
                   estimate_signed_louvain(dataset_dir, dataset, spec["minimum_community_size"],
                                           args.seeds, args.overwrite))
        args.metadata_root.mkdir(parents=True, exist_ok=True)
        (args.metadata_root / f"{dataset}.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
