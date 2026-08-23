from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from signed_epm.graph import canonical_undirected


SIGNLINK_CLASS_NAMES = ("positive", "negative", "non_edge")
SIGN_CLASS_NAMES = ("negative", "positive")


def pair_set(frame: pd.DataFrame) -> set[tuple[int, int]]:
    return {(min(int(u), int(v)), max(int(u), int(v)))
            for u, v in frame[["source", "target"]].itertuples(index=False)}


def signed_examples(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["source", "target"]].copy()
    result["label"] = np.where(frame["sign"].to_numpy() > 0, 0, 1).astype(np.int64)
    return result


def sample_non_edges(seed: int, split: str, num_nodes: int, count: int,
                     forbidden: set[tuple[int, int]]) -> pd.DataFrame:
    offset = {"train": 0, "validation": 10_000_000, "test": 20_000_000}[split]
    rng, sampled = np.random.default_rng(seed + offset), set()
    while len(sampled) < count:
        source, target = int(rng.integers(num_nodes)), int(rng.integers(num_nodes))
        if source == target:
            continue
        pair = (min(source, target), max(source, target))
        if pair not in forbidden and pair not in sampled:
            sampled.add(pair)
    rows = []
    for left, right in sorted(sampled):
        if bool(rng.integers(0, 2)):
            left, right = right, left
        rows.append((left, right, 2))
    return pd.DataFrame(rows, columns=["source", "target", "label"])


def endpoint_features(node_state: np.ndarray, examples: pd.DataFrame) -> np.ndarray:
    source = examples["source"].to_numpy(dtype=np.int64)
    target = examples["target"].to_numpy(dtype=np.int64)
    return np.concatenate([node_state[source], node_state[target]], axis=1)


@dataclass
class SignLinkProtocol:
    seed: int
    directed: bool
    class_weight: str | None = None

    def examples(self, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame,
                 base_train: pd.DataFrame, num_nodes: int) -> dict[str, pd.DataFrame]:
        del base_train  # Future edges are not consulted; the actual train graph is authoritative.
        if not self.directed:
            train, validation, test = map(canonical_undirected, (train, validation, test))
        forbidden = {
            "train": pair_set(train),
            "validation": pair_set(train) | pair_set(validation),
            "test": pair_set(train) | pair_set(validation) | pair_set(test),
        }
        frames = {"train": train, "validation": validation, "test": test}
        result = {}
        for split, frame in frames.items():
            examples = pd.concat([
                signed_examples(frame),
                sample_non_edges(self.seed, split, num_nodes, len(frame), forbidden[split]),
            ], ignore_index=True)
            if not self.directed:
                source, target = examples["source"].to_numpy(), examples["target"].to_numpy()
                examples["source"], examples["target"] = np.minimum(source, target), np.maximum(source, target)
            result[split] = examples
        return result

    def fit_probe(self, node_state: np.ndarray, examples: pd.DataFrame) -> LogisticRegression:
        probe = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=self.seed,
                                   class_weight=self.class_weight)
        probe.fit(endpoint_features(node_state, examples), examples["label"].to_numpy())
        return probe

    @staticmethod
    def evaluate(probe: LogisticRegression, node_state: np.ndarray,
                 examples: pd.DataFrame) -> dict:
        truth = examples["label"].to_numpy(dtype=np.int64)
        prediction = probe.predict(endpoint_features(node_state, examples))
        per_class = f1_score(truth, prediction, labels=[0, 1, 2], average=None, zero_division=0)
        return {
            "accuracy": float(accuracy_score(truth, prediction)),
            "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
            "per_class_f1": dict(zip(SIGNLINK_CLASS_NAMES, map(float, per_class))),
            "confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1, 2]).tolist(),
        }


@dataclass
class SignPredictionProtocol:
    """Binary sign prediction over observed positive and negative edges."""

    seed: int
    directed: bool
    class_weight: str | None = None

    @staticmethod
    def _signed_examples(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame[["source", "target"]].copy()
        # Preserve the established binary convention: negative=0, positive=1.
        result["label"] = (frame["sign"].to_numpy() > 0).astype(np.int64)
        return result

    def examples(self, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame,
                 base_train: pd.DataFrame, num_nodes: int) -> dict[str, pd.DataFrame]:
        del base_train, num_nodes  # No non-edge construction is needed.
        if not self.directed:
            train, validation, test = map(canonical_undirected, (train, validation, test))
        return {name: self._signed_examples(frame) for name, frame in
                (("train", train), ("validation", validation), ("test", test))}

    def fit_probe(self, node_state: np.ndarray, examples: pd.DataFrame) -> LogisticRegression:
        probe = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=self.seed,
                                   class_weight=self.class_weight)
        probe.fit(endpoint_features(node_state, examples), examples["label"].to_numpy())
        return probe

    @staticmethod
    def evaluate(probe: LogisticRegression, node_state: np.ndarray,
                 examples: pd.DataFrame) -> dict:
        truth = examples["label"].to_numpy(dtype=np.int64)
        features = endpoint_features(node_state, examples)
        prediction = probe.predict(features)
        per_class = f1_score(truth, prediction, labels=[0, 1], average=None, zero_division=0)
        result = {
            "accuracy": float(accuracy_score(truth, prediction)),
            "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
            "per_class_f1": dict(zip(SIGN_CLASS_NAMES, map(float, per_class))),
            "confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1]).tolist(),
        }
        if len(np.unique(truth)) == 2:
            result["auc"] = float(roc_auc_score(truth, probe.predict_proba(features)[:, 1]))
        return result


def protocol_for(task: str, seed: int, directed: bool,
                 class_weight: str | None = None) -> SignLinkProtocol | SignPredictionProtocol:
    if task == "signlink_3class":
        return SignLinkProtocol(seed=seed, directed=directed, class_weight=class_weight)
    if task == "sign_prediction_2class":
        return SignPredictionProtocol(seed=seed, directed=directed, class_weight=class_weight)
    raise ValueError(f"unknown downstream task: {task}")
