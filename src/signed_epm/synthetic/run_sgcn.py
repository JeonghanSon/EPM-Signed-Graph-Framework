from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from signed_epm.models import EncoderConfig, SGCNAdapter
from signed_epm.polarization.measure import measure_run


ROOT = Path(__file__).resolve().parents[3]


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve(path: str | Path, data_root: Path) -> Path:
    value = Path(path)
    # Bundled manifests may retain the original repository prefix. Prefer the
    # suffix below the extracted bundle directory, even when that old absolute
    # path happens to exist on the machine performing the reproduction.
    if data_root.name in value.parts:
        index = value.parts.index(data_root.name)
        candidate = data_root.joinpath(*value.parts[index + 1:])
        if candidate.exists():
            return candidate
    candidate = data_root / value
    if candidate.exists():
        return candidate
    candidate = ROOT / value
    if candidate.exists():
        return candidate
    if value.is_absolute() and value.exists():
        return value
    raise FileNotFoundError(f"cannot resolve bundled graph path: {path}")


def train_graph(graph_root: Path, output_dir: Path, seed: int, device: str,
                input_dimension: int, output_dimension: int, layers: int,
                learning_rate: float, epochs: int, cache_root: Path,
                negative_conductance: float) -> dict:
    import torch

    metrics_path = output_dir / "encoder_metrics.json"
    measurement_path = output_dir / "measurement" / "measurement.json"
    if metrics_path.exists() and measurement_path.exists():
        return json.loads(measurement_path.read_text())
    set_seed(seed)
    graph = pd.read_csv(graph_root / "train_events.csv")
    manifest = json.loads((graph_root / "manifest.json").read_text())
    num_nodes = int(manifest["counts"]["num_nodes"])
    config = EncoderConfig(input_dimension, output_dimension, layers,
                           learning_rate, epochs)
    adapter = SGCNAdapter()
    model = adapter.build(graph, num_nodes, config, device, cache_root)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad()
        loss = adapter.loss(model)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss for {graph_root} at epoch {epoch}")
        loss.backward(); optimizer.step()
        history.append(float(loss.detach().cpu()))
    state = adapter.node_state(model)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(model), output_dir / "encoder_state.pt")
    torch.save(torch.from_numpy(state), output_dir / "node_embeddings.pt")
    metrics = {
        "schema_version": 1, "purpose": "synthetic_measurement_only",
        "classifier": None, "validation": None, "test": None,
        "graph_root": str(graph_root), "graph_seed": seed,
        "uses_all_controlled_train_edges": True,
        "config": config.__dict__, "loss_initial": history[0],
        "loss_final": history[-1], "loss_history": history,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return measure_run(
        output_dir / "node_embeddings.pt",
        graph_root / "train_snapshot_undirected.csv",
        output_dir / "measurement", k=int(manifest["communities"]["selected_k"]),
        negative_conductance=negative_conductance,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fixed SGCN encoders for synthetic EPM validation")
    parser.add_argument("--data-root", type=Path,
                        default=ROOT / "data" / "synthetic" / "k2_controlled")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "artifacts" / "synthetic" / "k2_validation" / "epm")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "artifacts" / "cache")
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--input-dimension", type=int, default=64)
    parser.add_argument("--output-dimension", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--negative-conductance", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = json.loads((args.data_root / "generation_summary.json").read_text())
    requested = set(args.graph_seeds)
    rows = []
    for record in summary["graphs"]:
        if int(record["graph_seed"]) not in requested:
            continue
        graph_root = resolve(record["path"], args.data_root)
        token = record.get("token", f"{float(record['level']):.1f}".replace(".", "p"))
        output = (args.output_root / record["experiment"] /
                  f"graph_seed_{record['graph_seed']}" / f"ratio_{token}")
        measurement = train_graph(
            graph_root, output, int(record["graph_seed"]), args.device,
            args.input_dimension, args.output_dimension, args.layers,
            args.learning_rate, args.epochs, args.cache_root,
            args.negative_conductance,
        )
        row = {"experiment": record["experiment"], "level": float(record["level"]),
               "graph_seed": int(record["graph_seed"]),
               "epm_polarization": float(measurement["polarization"]),
               "output_dir": str(output)}
        rows.append(row)
        print(f"DONE experiment={row['experiment']} level={row['level']:g} "
              f"graph_seed={row['graph_seed']} epm={row['epm_polarization']:.8f}", flush=True)
    frame = pd.DataFrame(rows).sort_values(["experiment", "graph_seed", "level"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_root / "epm_raw.csv", index=False)
    aggregate = frame.groupby(["experiment", "level"], as_index=False).agg(
        mean=("epm_polarization", "mean"), std=("epm_polarization", "std"),
        n=("epm_polarization", "size"),
    )
    aggregate.to_csv(args.output_root / "epm_aggregate.csv", index=False)
    run_summary = {
        "schema_version": 1, "data_root": str(args.data_root),
        "opinion_used_by_sgcn_or_epm": False,
        "fixed_config": {"input_dimension": args.input_dimension,
                         "output_dimension": args.output_dimension, "layers": args.layers,
                         "learning_rate": args.learning_rate, "epochs": args.epochs},
        "negative_conductance": args.negative_conductance,
        "graph_seeds": sorted(requested), "completed_graphs": len(frame),
    }
    (args.output_root / "run_summary.json").write_text(json.dumps(run_summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
