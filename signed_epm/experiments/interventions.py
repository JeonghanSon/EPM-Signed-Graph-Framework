from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from signed_epm.data.preprocess import ROOT, load_json
from signed_epm.mitigation.core import (
    direct_augment,
    epm_augment,
    prepare_intervention,
    random_augment,
)
from signed_epm.polarization.measure import load_node_state, measure_run
from signed_epm.graph import graph_fingerprint
from signed_epm.training.evaluate import evaluate_selection


def setting_name(tau: float, max_degree: int, gamma: float) -> str:
    return f"tau{tau:.1f}_dmax{max_degree}_gamma{gamma:.1f}"


def base_root(args: argparse.Namespace) -> Path:
    return args.run_root / args.model / args.dataset / args.task / "base"


def intervention_root(args: argparse.Namespace) -> Path:
    return args.artifact_root / args.model / args.dataset / args.task


def selected_base_runs(args: argparse.Namespace) -> pd.DataFrame:
    path = base_root(args) / "selected_runs.csv"
    if not path.exists():
        raise FileNotFoundError(f"completed base selection is required: {path}")
    frame = pd.read_csv(path).sort_values("seed")
    available = set(frame.seed.astype(int))
    if not set(args.seeds).issubset(available):
        raise RuntimeError(f"base runs do not contain requested seeds {args.seeds}: {path}")
    return frame[frame.seed.astype(int).isin(args.seeds)].sort_values("seed")


def reuse_base_measurement(
    run_dir: Path,
    edge_path: Path,
    output_dir: Path,
    k: int,
    negative_conductance: float,
) -> dict | None:
    source = run_dir / "measurement"
    metadata_path = source / "measurement.json"
    coordinates_path = source / "opinion_coordinates.npy"
    if not metadata_path.exists() or not coordinates_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    graph = pd.read_csv(edge_path)
    compatible = (
        int(metadata.get("k", -1)) == k
        and float(metadata.get("positive_conductance", -1.0)) == 1.0
        and float(metadata.get("negative_conductance", -1.0)) == negative_conductance
        and metadata.get("graph_fingerprint") == graph_fingerprint(graph, directed=False)
    )
    if not compatible:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metadata_path, output_dir / "measurement.json")
    shutil.copy2(coordinates_path, output_dir / "opinion_coordinates.npy")
    return metadata


def measurement_context(args: argparse.Namespace) -> tuple[int, Path, pd.DataFrame]:
    community_metadata = load_json(
        ROOT / "data" / "metadata" / "communities" / f"{args.dataset}.json"
    )
    k = int(community_metadata["selected_k"])
    edge_path = args.data_dir / "train_snapshot_undirected.csv"
    graph = pd.read_csv(edge_path)
    return k, edge_path, graph


def measure(args: argparse.Namespace) -> None:
    k, edge_path, _ = measurement_context(args)
    for row in selected_base_runs(args).itertuples(index=False):
        seed = int(row.seed)
        node_state_path = Path(row.run_dir) / "node_embeddings.pt"
        measurement_dir = intervention_root(args) / f"seed_{seed}" / "base_measurement"
        if (measurement_dir / "measurement.json").exists():
            print(f"SKIP measured seed={seed}", flush=True)
            continue
        result = reuse_base_measurement(
            Path(row.run_dir), edge_path, measurement_dir, k,
            args.negative_conductance,
        )
        source = "reused" if result is not None else "computed"
        if result is None:
            result = measure_run(
                node_state_path, edge_path, measurement_dir, k,
                negative_conductance=args.negative_conductance,
            )
        print(
            f"MEASURE seed={seed} source={source} "
            f"polarization={result['polarization']:.10f}", flush=True,
        )


def prepare(args: argparse.Namespace) -> None:
    dataset_config = load_json(ROOT / "configs" / "datasets.json")[args.dataset]
    k, _, graph = measurement_context(args)
    profile = str(dataset_config.get("intervention_profile", "standard"))
    minimum_size = int(dataset_config.get(
        "intervention_minimum_community_size",
        dataset_config["minimum_community_size"],
    ))
    top_pairs = dataset_config.get("intervention_top_pairs")
    top_pairs = int(top_pairs) if top_pairs is not None else None
    for row in selected_base_runs(args).itertuples(index=False):
        seed = int(row.seed)
        node_state_path = Path(row.run_dir) / "node_embeddings.pt"
        seed_root = intervention_root(args) / f"seed_{seed}"
        preparation_dir = seed_root / "preparation"
        metadata_path = preparation_dir / "preparation.json"
        if metadata_path.exists():
            state = np.asarray(load_node_state(node_state_path), dtype=np.float64)
            expected = {
                "graph_fingerprint": graph_fingerprint(graph, directed=False),
                "node_state_sha256": hashlib.sha256(
                    np.ascontiguousarray(state).tobytes()
                ).hexdigest(),
                "k": k,
                "kmeans_seed": args.kmeans_seed,
                "minimum_community_size": minimum_size,
                "negative_conductance": args.negative_conductance,
                "top_pairs": top_pairs,
            }
            actual = json.loads(metadata_path.read_text(encoding="utf-8"))
            if any(actual.get(key) != value for key, value in expected.items()):
                raise RuntimeError(
                    f"cached preparation does not match current inputs: {preparation_dir}"
                )
            print(f"SKIP prepared seed={seed} (fingerprints verified)", flush=True)
        else:
            summary = prepare_intervention(
                load_node_state(node_state_path), graph, k, minimum_size,
                preparation_dir, args.kmeans_seed, args.negative_conductance,
                top_pairs,
            )
            print(
                f"PREPARE seed={seed} profile={profile} "
                f"pairs={summary['community_pairs']}/{summary['all_community_pairs']}",
                flush=True,
            )


def generate(args: argparse.Namespace) -> None:
    directed = args.model == "sdgnn"
    for seed in args.seeds:
        seed_root = intervention_root(args) / f"seed_{seed}"
        preparation_dir = seed_root / "preparation"
        if not (preparation_dir / "preparation.json").exists():
            raise FileNotFoundError(f"preparation is missing: {preparation_dir}")
        for tau, max_degree, gamma in itertools.product(
            args.taus, args.max_degrees, args.gammas,
        ):
            setting = setting_name(tau, max_degree, gamma)
            epm_dir = seed_root / "epm" / setting
            if not (epm_dir / "summary.json").exists():
                summary = epm_augment(
                    args.data_dir, preparation_dir, epm_dir,
                    tau, max_degree, gamma, directed,
                )
                print(f"EPM seed={seed} setting={setting} budget={summary['edge_budget_cost']}", flush=True)
            reference = json.loads((epm_dir / "summary.json").read_text(encoding="utf-8"))
            if "random" in args.interventions:
                random_dir = seed_root / "random" / setting
                if not (random_dir / "summary.json").exists():
                    random_augment(args.data_dir, reference, random_dir, directed, seed)
                    print(f"RANDOM seed={seed} setting={setting}", flush=True)
            if "direct" in args.interventions:
                direct_dir = seed_root / "direct" / setting
                if not (direct_dir / "summary.json").exists():
                    direct_augment(
                        args.data_dir, preparation_dir, epm_dir, direct_dir, directed, seed,
                    )
                    print(f"DIRECT seed={seed} setting={setting}", flush=True)


def tune(args: argparse.Namespace) -> None:
    for intervention in args.interventions:
        for tau, max_degree, gamma in itertools.product(
            args.taus, args.max_degrees, args.gammas,
        ):
            setting = setting_name(tau, max_degree, gamma)
            output_root = (
                args.run_root / args.model / args.dataset / args.task /
                intervention / setting
            )
            if (output_root / "selected_configuration.json").exists():
                print(f"SKIP completed intervention={intervention} setting={setting}", flush=True)
                continue
            template = str(
                intervention_root(args) / "seed_{seed}" / intervention / setting /
                "train_snapshot_directed_augmented.csv"
            )
            command = [
                sys.executable, "-m", "signed_epm.training.tune",
                "--model", args.model, "--task", args.task,
                "--data-dir", str(args.data_dir), "--output-root", str(output_root),
                "--train-path-template", template, "--device", args.device,
                "--cache-root", str(args.cache_root),
                "--seeds", *[str(seed) for seed in args.seeds], "--prune",
            ]
            print("RUN", " ".join(command), flush=True)
            subprocess.run(command, check=True)


def evaluate(args: argparse.Namespace) -> None:
    community_metadata = load_json(
        ROOT / "data" / "metadata" / "communities" / f"{args.dataset}.json"
    )
    k = int(community_metadata["selected_k"])
    for intervention in args.interventions:
        for tau, max_degree, gamma in itertools.product(
            args.taus, args.max_degrees, args.gammas,
        ):
            setting = setting_name(tau, max_degree, gamma)
            selection_root = (
                args.run_root / args.model / args.dataset / args.task /
                intervention / setting
            )
            if not (selection_root / "selected_configuration.json").exists():
                raise FileNotFoundError(f"mitigated selection is missing: {selection_root}")
            if (selection_root / "final_summary.json").exists():
                print(f"SKIP evaluated intervention={intervention} setting={setting}", flush=True)
                continue
            graph_root = intervention_root(args) / "seed_{seed}" / intervention / setting
            evaluate_selection(
                selection_root, args.data_dir, args.model, args.task, k,
                str(graph_root / "train_snapshot_directed_augmented.csv"),
                str(graph_root / "train_snapshot_undirected_augmented.csv"),
                args.negative_conductance,
            )
            print(f"EVALUATE intervention={intervention} setting={setting}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare, generate, and tune EPM interventions")
    parser.add_argument("command", choices=["measure", "prepare", "generate", "tune", "evaluate", "all"])
    parser.add_argument("--model", choices=["sgcn", "sdgnn"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", choices=["signlink_3class", "sign_prediction_2class"], required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/interventions"))
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--taus", nargs="+", type=float, default=[0.5, 0.7, 0.9])
    parser.add_argument("--max-degrees", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--gammas", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--interventions", nargs="+", choices=["epm", "random", "direct"],
                        default=["epm", "random", "direct"])
    parser.add_argument("--negative-conductance", type=float, default=0.1)
    parser.add_argument("--kmeans-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.data_dir = args.data_dir or ROOT / "data" / "processed" / args.dataset
    if args.command in {"measure", "all"}:
        measure(args)
    if args.command in {"prepare", "all"}:
        prepare(args)
    if args.command in {"generate", "all"}:
        generate(args)
    if args.command in {"tune", "all"}:
        tune(args)
    if args.command in {"evaluate", "all"}:
        evaluate(args)


if __name__ == "__main__":
    main()
