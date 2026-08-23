from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from signed_epm.graph import DatasetSplits, graph_fingerprint
from signed_epm.models import EncoderConfig, SDGNNAdapter, SGCNAdapter
from signed_epm.tasks.signlink import protocol_for


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def adapter_for(name: str):
    if name == "sgcn":
        return SGCNAdapter()
    if name == "sdgnn":
        return SDGNNAdapter()
    raise ValueError(f"unknown model: {name}")


def save_probe(probe, path: Path) -> None:
    np.savez(path, coef=probe.coef_, intercept=probe.intercept_, classes=probe.classes_,
             feature_style=np.array("ordered"))


def train(args: argparse.Namespace) -> dict:
    import torch

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists: {args.output_dir} (use --overwrite)")
    if args.overwrite and args.output_dir.exists():
        import shutil
        shutil.rmtree(args.output_dir)
    set_seed(args.seed)
    splits = DatasetSplits.load(args.data_dir, read_test=not args.validation_only)
    base_train = splits.train
    train_graph = pd.read_csv(args.train_path) if args.train_path else base_train
    adapter = adapter_for(args.model)
    protocol = protocol_for(args.task, seed=args.seed, directed=adapter.directed,
                            class_weight=None if args.class_weight == "none" else "balanced")
    examples = protocol.examples(train_graph, splits.validation, splits.test,
                                 base_train, splits.num_nodes)
    config = EncoderConfig(
        input_dimension=args.input_dimension,
        output_dimension=args.output_dimension,
        layers=args.layers,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
    )
    model = adapter.build(train_graph, splits.num_nodes, config, args.device, args.cache_root)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = adapter.loss(model)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite encoder loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        history.append({"epoch": epoch, "encoder_loss": value})
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(f"epoch={epoch:03d} encoder_loss={value:.6f}", flush=True)

    node_state = adapter.node_state(model)
    probe = protocol.fit_probe(node_state, examples["train"])
    validation = protocol.evaluate(probe, node_state, examples["validation"])
    test = None if args.validation_only else protocol.evaluate(probe, node_state, examples["test"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(model), args.output_dir / "encoder_state.pt")
    torch.save(torch.from_numpy(node_state), args.output_dir / "node_embeddings.pt")
    save_probe(probe, args.output_dir / "logistic_classifier.npz")

    result = {
        "schema_version": 1,
        "model": args.model,
        "task": args.task,
        "seed": args.seed,
        "graph_fingerprint": graph_fingerprint(train_graph, directed=adapter.directed),
        "graph_policy": "directed_as_observed" if adapter.directed else
                        "canonical_physical_edges; bidirectional_encoder_messages",
        "nonedge_policy": "strict_history" if args.task == "signlink_3class" else None,
        "classifier": ("multinomial_logistic_regression" if args.task == "signlink_3class"
                       else "binary_logistic_regression"),
        "config": {
            "input_dimension": config.input_dimension,
            "output_dimension": config.output_dimension,
            "layers": config.layers,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "weight_decay": config.weight_decay,
            "class_weight": protocol.class_weight,
        },
        "counts": {name: frame["label"].value_counts().sort_index().to_dict()
                   for name, frame in examples.items()},
        "validation": validation,
        "test": test,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"validation_macro_f1={validation['macro_f1']:.6f}", flush=True)
    if test is not None:
        print(f"test_macro_f1={test['macro_f1']:.6f}", flush=True)
    print(f"saved={args.output_dir}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one signed-graph encoder and sign-link probe")
    parser.add_argument("--model", choices=["sgcn", "sdgnn"], required=True)
    parser.add_argument("--task", choices=["sign_prediction_2class", "signlink_3class"],
                        default="signlink_3class")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--input-dimension", type=int, default=64)
    parser.add_argument("--output-dimension", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
