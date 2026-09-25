from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from signed_epm.polarization.measure import measure_run


ROOT = Path(__file__).resolve().parents[3]


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def parse_level(value: str) -> float:
    token_value = value.removeprefix("ratio_")
    if token_value.startswith("level_"):
        return float(token_value.removeprefix("level_"))
    return float(token_value.replace("p", "."))


def main() -> None:
    parser = argparse.ArgumentParser(description="Remeasure saved synthetic embeddings at a new eta")
    parser.add_argument("--encoder-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--negative-conductance", type=float, required=True)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--levels", nargs="+", type=float, default=None)
    args = parser.parse_args()
    rows = []
    for metrics_path in sorted(args.encoder_root.glob("*/*/*/encoder_metrics.json")):
        relative = metrics_path.parent.relative_to(args.encoder_root)
        experiment, seed_part, ratio_part = relative.parts
        seed = int(seed_part.removeprefix("graph_seed_"))
        level = parse_level(ratio_part)
        if args.experiments is not None and experiment not in args.experiments:
            continue
        if args.levels is not None and not any(abs(level - value) < 1e-12
                                               for value in args.levels):
            continue
        metrics = json.loads(metrics_path.read_text())
        graph_root = Path(metrics["graph_root"])
        if not graph_root.is_absolute():
            graph_root = ROOT / graph_root
        destination = args.output_root / experiment / seed_part / ratio_part
        result = measure_run(
            metrics_path.parent / "node_embeddings.pt",
            graph_root / "train_snapshot_undirected.csv",
            destination, args.k, negative_conductance=args.negative_conductance,
        )
        rows.append({"experiment": experiment, "level": level, "graph_seed": seed,
                     "epm_polarization": result["polarization"],
                     "negative_conductance": args.negative_conductance,
                     "graph_fingerprint": result["graph_fingerprint"],
                     "embedding_path": str(metrics_path.parent / "node_embeddings.pt"),
                     "measurement_dir": str(destination)})
        print(f"DONE experiment={experiment} level={level:.1f} seed={seed} "
              f"eta={args.negative_conductance:g} epm={result['polarization']:.8f}", flush=True)
    if not rows:
        raise FileNotFoundError(f"no encoder metrics below {args.encoder_root}")
    frame = pd.DataFrame(rows).sort_values(["experiment", "graph_seed", "level"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_root / "epm_raw.csv", index=False)
    aggregate = frame.groupby(["experiment", "level"], as_index=False).agg(
        mean=("epm_polarization", "mean"), std=("epm_polarization", "std"),
        n=("epm_polarization", "size"),
    )
    aggregate.to_csv(args.output_root / "epm_aggregate.csv", index=False)
    summary = {"schema_version": 1, "encoder_root": str(args.encoder_root),
               "negative_conductance": args.negative_conductance, "k": args.k,
               "experiments": args.experiments, "levels": args.levels,
               "measurements": len(frame), "embedding_retrained": False}
    (args.output_root / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
