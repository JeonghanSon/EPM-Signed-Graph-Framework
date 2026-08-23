import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

from signed_epm.mitigation.core import (
    centroid_pair_candidates,
    direct_augment,
    model_edges,
    random_augment,
)


class MitigationTests(unittest.TestCase):
    def test_centroid_pair_pruning_is_deterministic(self):
        coordinates = np.asarray([
            [0.0, 0.0], [0.0, 0.2],
            [1.0, 0.0], [1.0, 0.2],
            [4.0, 0.0], [4.0, 0.2],
        ])
        communities = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
        selected = centroid_pair_candidates(coordinates, communities, top_pairs=2)
        self.assertEqual([(left, right) for left, right, _ in selected], [(0, 2), (1, 2)])
        self.assertEqual(len(centroid_pair_candidates(coordinates, communities, None)), 3)

    def test_directed_model_edges_consume_two_budget_units(self):
        physical = pd.DataFrame({
            "source": [0, 2], "target": [1, 3], "sign": [1, 1], "weight": [1.0, 1.0],
        })
        directed = model_edges(physical, directed=True)
        self.assertEqual(len(directed), 4)
        self.assertEqual(set(map(tuple, directed[["source", "target"]].to_numpy())),
                         {(0, 1), (1, 0), (2, 3), (3, 2)})

    def test_random_and_direct_match_epm_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"; data.mkdir()
            directed = pd.DataFrame({
                "source": [0, 1, 2, 3], "target": [1, 2, 3, 4],
                "sign": [1, 1, 1, 1], "weight": [1.0] * 4,
            })
            undirected = directed.copy()
            directed.to_csv(data / "train_snapshot_directed.csv", index=False)
            undirected.to_csv(data / "train_snapshot_undirected.csv", index=False)
            (data / "manifest.json").write_text(json.dumps({"num_nodes": 8}))
            reference = {"tau": 0.5, "max_degree": 2, "gamma": 1.0,
                         "physical_edges_added": 2}
            random_summary = random_augment(data, reference, root / "random", True, seed=4)
            self.assertEqual(random_summary["physical_edges_added"], 2)
            self.assertEqual(random_summary["edge_budget_cost"], 4)

            half_summary = random_augment(
                data, {**reference, "physical_edges_added": 4},
                root / "random_half", True, seed=4, budget_fraction=0.5,
            )
            full_summary = random_augment(
                data, {**reference, "physical_edges_added": 4},
                root / "random_full", True, seed=4, budget_fraction=1.0,
            )
            half_edges = set(map(tuple, pd.read_csv(
                root / "random_half/new_physical_edges.csv"
            )[["source", "target"]].to_numpy()))
            full_edges = set(map(tuple, pd.read_csv(
                root / "random_full/new_physical_edges.csv"
            )[["source", "target"]].to_numpy()))
            self.assertTrue(half_edges < full_edges)
            self.assertEqual(half_summary["physical_edges_added"], 2)
            self.assertEqual(full_summary["physical_edges_added"], 4)

            preparation = root / "preparation"; preparation.mkdir()
            pd.DataFrame([
                {"community_id": 0, "nodes": str([0, 1]), "size": 2},
                {"community_id": 1, "nodes": str([4, 5]), "size": 2},
            ]).to_csv(preparation / "kmeans_communities.csv", index=False)
            epm = root / "epm"; epm.mkdir()
            (epm / "summary.json").write_text(json.dumps(reference))
            pd.DataFrame([{
                "community_1": 0, "community_2": 1,
                "gray_gray_edges": 1, "gray_community_edges": 1,
            }]).to_csv(epm / "selected_pairs.csv", index=False)
            direct_summary = direct_augment(data, preparation, epm, root / "direct", True, seed=4)
            self.assertEqual(direct_summary["physical_edges_added"], 2)
            self.assertEqual(direct_summary["edge_budget_cost"], 4)


if __name__ == "__main__":
    unittest.main()
