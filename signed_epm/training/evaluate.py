from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from signed_epm.graph import DatasetSplits
from signed_epm.polarization.measure import load_node_state, measure_run
from signed_epm.tasks.signlink import protocol_for


class SavedLinearProbe:
    def __init__(self, path: Path):
        saved = np.load(path)
        self.coef_ = saved["coef"]
        self.intercept_ = saved["intercept"]
        self.classes_ = saved["classes"]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        scores = features @ self.coef_.T + self.intercept_
        if len(self.classes_) == 2 and scores.shape[1] == 1:
            positive = 1.0 / (1.0 + np.exp(-scores[:, 0]))
            return np.column_stack([1.0 - positive, positive])
        scores = scores - scores.max(axis=1, keepdims=True)
        exponential = np.exp(scores)
        return exponential / exponential.sum(axis=1, keepdims=True)

    def predict(self, features: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(features)
        return self.classes_[np.argmax(probabilities, axis=1)]


def evaluate_selection(
    selection_root: Path,
    data_dir: Path,
    model: str,
    task: str,
    k: int,
    train_path_template: str | None = None,
    measurement_edge_template: str | None = None,
    negative_conductance: float = 0.1,
    measurement_json_template: str | None = None,
) -> pd.DataFrame:
    selected = pd.read_csv(selection_root / "selected_runs.csv").sort_values("seed")
    splits = DatasetSplits.load(data_dir, read_test=True)
    rows = []
    for row in selected.itertuples(index=False):
        seed = int(row.seed)
        run_dir = Path(row.run_dir)
        train_path = (Path(train_path_template.format(seed=seed))
                      if train_path_template else data_dir / "train_events.csv")
        edge_path = (Path(measurement_edge_template.format(seed=seed))
                     if measurement_edge_template else data_dir / "train_snapshot_undirected.csv")
        train_graph = pd.read_csv(train_path)
        node_state = load_node_state(run_dir / "node_embeddings.pt")
        probe = SavedLinearProbe(run_dir / "logistic_classifier.npz")
        protocol = protocol_for(task, seed, directed=model == "sdgnn")
        examples = protocol.examples(
            train_graph, splits.validation, splits.test, splits.train, splits.num_nodes,
        )
        test = protocol.evaluate(probe, node_state, examples["test"])
        if measurement_json_template:
            measurement_path = Path(measurement_json_template.format(seed=seed))
            measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
            if int(measurement["k"]) != k:
                raise ValueError(f"measurement k mismatch: {measurement_path}")
            if float(measurement["negative_conductance"]) != negative_conductance:
                raise ValueError(
                    f"measurement negative conductance mismatch: {measurement_path}"
                )
        else:
            measurement = measure_run(
                run_dir / "node_embeddings.pt", edge_path, run_dir / "measurement",
                k, negative_conductance=negative_conductance,
            )
        result = {
            "schema_version": 1, "seed": seed, "model": model, "task": task,
            "train_path": str(train_path), "measurement_edge_path": str(edge_path),
            "test": test, "polarization": measurement["polarization"],
        }
        (run_dir / "final_evaluation.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8",
        )
        rows.append({
            "seed": seed, "test_macro_f1": test["macro_f1"],
            "test_weighted_f1": test["weighted_f1"], "test_accuracy": test["accuracy"],
            "test_auc": test.get("auc"), "polarization": measurement["polarization"],
            "run_dir": str(run_dir),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(selection_root / "final_evaluation.csv", index=False)
    summary = {
        "schema_version": 1, "model": model, "task": task,
        "completed_seeds": len(frame),
        "test_macro_f1_mean": float(frame.test_macro_f1.mean()),
        "test_macro_f1_std": float(frame.test_macro_f1.std()),
        "test_weighted_f1_mean": float(frame.test_weighted_f1.mean()),
        "test_weighted_f1_std": float(frame.test_weighted_f1.std()),
        "test_accuracy_mean": float(frame.test_accuracy.mean()),
        "test_accuracy_std": float(frame.test_accuracy.std()),
        "test_auc_mean": (
            float(frame.test_auc.mean()) if frame.test_auc.notna().any() else None
        ),
        "test_auc_std": (
            float(frame.test_auc.std()) if frame.test_auc.notna().any() else None
        ),
        "polarization_mean": float(frame.polarization.mean()),
        "polarization_std": float(frame.polarization.std()),
    }
    (selection_root / "final_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Test and measure a validation-selected run set")
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["sgcn", "sdgnn"], required=True)
    parser.add_argument("--task", choices=["signlink_3class", "sign_prediction_2class"], required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--train-path-template", default=None)
    parser.add_argument("--measurement-edge-template", default=None)
    parser.add_argument("--negative-conductance", type=float, default=0.1)
    parser.add_argument("--measurement-json-template", default=None)
    args = parser.parse_args()
    frame = evaluate_selection(
        args.selection_root, args.data_dir, args.model, args.task, args.k,
        args.train_path_template, args.measurement_edge_template,
        args.negative_conductance,
        args.measurement_json_template,
    )
    print(f"test_macro_f1={frame.test_macro_f1.mean():.6f} ")
    print(f"polarization={frame.polarization.mean():.10f}")


if __name__ == "__main__":
    main()
