from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from signed_epm.data.preprocess import ROOT, load_json


CONFIG_COLUMNS = ["input_dimension", "output_dimension", "layers",
                  "learning_rate", "epochs", "class_weight"]


def token(value: float | int) -> str:
    return f"{value:g}".replace(".", "p") if isinstance(value, float) else str(value)


def candidate_path(root: Path, seed: int, input_dimension: int, output_dimension: int,
                   layers: int, learning_rate: float, epochs: int, class_weight: str) -> Path:
    name = (f"in_{input_dimension}__out_{output_dimension}__layers_{layers}__"
            f"lr_{token(learning_rate)}__epochs_{epochs}__weight_{class_weight}")
    return root / f"seed_{seed}" / "candidates" / name


def select_common_configuration(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Select one configuration by mean validation Macro-F1 across seeds."""
    expected_seeds = frame["seed"].nunique()
    summary = frame.groupby(CONFIG_COLUMNS, as_index=False).agg(
        validation_macro_f1_mean=("validation_macro_f1", "mean"),
        validation_macro_f1_std=("validation_macro_f1", "std"),
        completed_seeds=("seed", "nunique"),
    )
    summary = summary[summary.completed_seeds == expected_seeds].copy()
    if summary.empty:
        raise RuntimeError("no configuration completed for every requested seed")
    summary["validation_macro_f1_std"] = summary["validation_macro_f1_std"].fillna(0.0)
    summary = summary.sort_values(
        ["validation_macro_f1_mean", "validation_macro_f1_std", "learning_rate", "epochs",
         "input_dimension", "output_dimension", "layers"],
        ascending=[False, True, True, True, True, True, True], kind="mergesort").reset_index(drop=True)
    selected_config = summary.iloc[0].to_dict()
    selected = frame.copy()
    for column in CONFIG_COLUMNS:
        selected = selected[selected[column] == selected_config[column]]
    return selected_config, selected.sort_values("seed"), summary


def run_candidate(args: argparse.Namespace, seed: int, input_dimension: int,
                  output_dimension: int, layers: int, learning_rate: float,
                  epochs: int, class_weight: str) -> dict:
    run_dir = candidate_path(args.output_root, seed, input_dimension, output_dimension,
                             layers, learning_rate, epochs, class_weight)
    metrics_path = run_dir / "metrics.json"
    if args.overwrite_candidates and run_dir.exists():
        shutil.rmtree(run_dir)
    # A terminated training process can leave a non-empty directory without
    # metrics.  Such a candidate is incomplete and safe to retry; completed
    # candidates remain resumable through their metrics file.
    if run_dir.exists() and not metrics_path.exists():
        shutil.rmtree(run_dir)
    if not metrics_path.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "-m", "signed_epm.training.train",
            "--model", args.model, "--task", args.task,
            "--data-dir", str(args.data_dir), "--output-dir", str(run_dir),
            "--cache-root", str(args.cache_root), "--input-dimension", str(input_dimension),
            "--output-dimension", str(output_dimension), "--layers", str(layers),
            "--learning-rate", str(learning_rate), "--epochs", str(epochs),
            "--class-weight", class_weight, "--seed", str(seed), "--device", args.device,
            "--validation-only",
        ]
        train_path = None
        if args.train_path_template is not None:
            train_path = Path(str(args.train_path_template).format(seed=seed))
        elif args.train_path is not None:
            train_path = args.train_path
        if train_path is not None:
            if not train_path.exists():
                raise FileNotFoundError(f"training graph does not exist: {train_path}")
            command.extend(["--train-path", str(train_path)])
        print("RUN", " ".join(command), flush=True)
        subprocess.run(command, check=True)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("test") is not None:
        raise RuntimeError(f"tuning candidate contains test results: {metrics_path}")
    return {
        "seed": seed,
        "input_dimension": input_dimension,
        "output_dimension": output_dimension,
        "layers": layers,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "class_weight": class_weight,
        "validation_macro_f1": float(metrics["validation"]["macro_f1"]),
        "validation_weighted_f1": float(metrics["validation"]["weighted_f1"]),
        "validation_accuracy": float(metrics["validation"]["accuracy"]),
        "validation_auc": metrics["validation"].get("auc"),
        "run_dir": str(run_dir),
        "graph_fingerprint": metrics["graph_fingerprint"],
    }


def prune_candidates(frame: pd.DataFrame, selected: pd.DataFrame) -> int:
    keep = {str(Path(path)) for path in selected.run_dir}
    removed = 0
    for path in sorted({str(Path(path)) for path in frame.run_dir} - keep):
        directory = Path(path)
        if directory.exists():
            shutil.rmtree(directory)
            removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Select one encoder configuration over common seeds")
    parser.add_argument("--model", choices=["sgcn", "sdgnn"], required=True)
    parser.add_argument("--task", choices=["sign_prediction_2class", "signlink_3class"], required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/cache"))
    graph = parser.add_mutually_exclusive_group()
    graph.add_argument("--train-path", type=Path, default=None)
    graph.add_argument("--train-path-template", default=None,
                       help="seed-specific path containing a literal {seed} placeholder")
    parser.add_argument("--model-config-root", type=Path, default=ROOT / "configs" / "models")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--input-dimensions", nargs="+", type=int, default=None)
    parser.add_argument("--output-dimensions", nargs="+", type=int, default=None)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--learning-rates", nargs="+", type=float, default=None)
    parser.add_argument("--epoch-grid", nargs="+", type=int, default=None)
    parser.add_argument("--class-weights", nargs="+", choices=["none", "balanced"], default=["none"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite-candidates", action="store_true")
    parser.add_argument(
        "--candidates-only", action="store_true",
        help="train this seed/config shard without writing common-selection artifacts",
    )
    parser.add_argument(
        "--aggregate-shards", action="store_true",
        help="select from existing validation_candidates_shard_*.csv files without training",
    )
    parser.add_argument("--prune", action="store_true",
                        help="remove learned artifacts for validation-rejected candidates")
    args = parser.parse_args()

    if args.aggregate_shards:
        shard_paths = sorted(args.output_root.glob("validation_candidates_shard_*.csv"))
        if not shard_paths:
            raise FileNotFoundError(f"no candidate shards found under {args.output_root}")
        frame = pd.concat([pd.read_csv(path) for path in shard_paths], ignore_index=True)
        duplicated = frame.duplicated(subset=["seed", *CONFIG_COLUMNS], keep=False)
        if duplicated.any():
            raise RuntimeError("candidate shards overlap in seed/configuration rows")
        if set(frame.seed.astype(int)) != set(args.seeds):
            raise RuntimeError(
                f"shard seeds {sorted(frame.seed.astype(int).unique())} do not match "
                f"requested seeds {sorted(args.seeds)}"
            )
        common, selected, summary = select_common_configuration(frame)
        frame.to_csv(args.output_root / "validation_candidates.csv", index=False)
        summary.to_csv(args.output_root / "validation_summary.csv", index=False)
        selected.to_csv(args.output_root / "selected_runs.csv", index=False)
        (args.output_root / "selected_configuration.json").write_text(
            json.dumps(common, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"selected_configuration": common,
                          "selected_runs": selected.to_dict("records"),
                          "source_shards": [str(path) for path in shard_paths]}, indent=2),
              flush=True)
        return

    if args.train_path_template is not None and "{seed}" not in args.train_path_template:
        raise ValueError("--train-path-template must contain a literal {seed} placeholder")

    model_config = load_json(args.model_config_root / f"{args.model}.json")
    args.input_dimensions = args.input_dimensions or model_config["input_dimensions"]
    args.output_dimensions = args.output_dimensions or model_config["output_dimensions"]
    args.layers = args.layers or model_config["layers"]
    args.learning_rates = args.learning_rates or model_config["learning_rates"]
    args.epoch_grid = args.epoch_grid or model_config["epochs"]
    if args.model == "sdgnn" and any(
            input_dimension != output_dimension
            for input_dimension in args.input_dimensions
            for output_dimension in args.output_dimensions):
        raise ValueError("every SDGNN grid candidate must have equal input/output dimensions")

    args.output_root.mkdir(parents=True, exist_ok=True)
    resolved_search = {
        "model": args.model, "task": args.task, "seeds": args.seeds,
        "input_dimensions": args.input_dimensions, "output_dimensions": args.output_dimensions,
        "layers": args.layers, "learning_rates": args.learning_rates,
        "epochs": args.epoch_grid, "class_weights": args.class_weights,
        "train_path": str(args.train_path) if args.train_path is not None else None,
        "train_path_template": args.train_path_template,
        "candidate_count": (len(args.seeds) * len(args.input_dimensions) *
                            len(args.output_dimensions) * len(args.layers) *
                            len(args.learning_rates) * len(args.epoch_grid) *
                            len(args.class_weights)),
        "model_search_profile": model_config.get("search_profile"),
    }
    (args.output_root / "resolved_search.json").write_text(
        json.dumps(resolved_search, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"resolved_search": resolved_search}, indent=2), flush=True)

    rows = []
    grid = itertools.product(args.seeds, args.input_dimensions, args.output_dimensions,
                             args.layers, args.learning_rates, args.epoch_grid, args.class_weights)
    for values in grid:
        rows.append(run_candidate(args, *values))
    frame = pd.DataFrame(rows)
    if args.candidates_only:
        shard = "_".join(str(seed) for seed in args.seeds)
        path = args.output_root / f"validation_candidates_shard_{shard}.csv"
        frame.to_csv(path, index=False)
        print(json.dumps({
            "candidate_shard": str(path),
            "completed_candidates": len(frame),
            "seeds": args.seeds,
        }, indent=2), flush=True)
        return
    frame.to_csv(args.output_root / "validation_candidates.csv", index=False)
    common, selected, summary = select_common_configuration(frame)
    summary.to_csv(args.output_root / "validation_summary.csv", index=False)
    selected.to_csv(args.output_root / "selected_runs.csv", index=False)
    (args.output_root / "selected_configuration.json").write_text(
        json.dumps(common, indent=2) + "\n", encoding="utf-8")
    removed = prune_candidates(frame, selected) if args.prune else 0
    print(json.dumps({"selected_configuration": common,
                      "selected_runs": selected.to_dict("records"),
                      "pruned_candidate_directories": removed}, indent=2), flush=True)


if __name__ == "__main__":
    main()
