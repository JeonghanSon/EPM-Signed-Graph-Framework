import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from signed_epm.mitigation.candidates import sample_unique_gray_union
from signed_epm.mitigation.linear import (
    grounded_factor,
    rank_one_inverse_action_update,
    solve_grounded,
)
from signed_epm.mitigation.materialize import model_edges, write_augmented_graph
from signed_epm.mitigation.prepare import prepare_intervention


class MitigationTests(unittest.TestCase):
    def test_preparation_is_score_free_and_all_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = np.asarray([
                [-3., 0., .1, -.1], [-2.8, .1, 0., .2],
                [-1., 1., -.2, .1], [-.9, 1.2, .2, 0.],
                [1., -1., .1, .2], [1.2, -.8, -.1, 0.],
                [3., 0., .2, -.2], [2.8, -.1, 0., .1],
            ])
            graph = pd.DataFrame({
                "source": range(7), "target": range(1, 8),
                "sign": [1] * 7, "weight": [1.] * 7,
            })
            summary = prepare_intervention(state, graph, 4, 1, root)
            pairs = pd.read_csv(root / "community_pairs.csv")
            self.assertEqual(len(pairs), 6)
            self.assertNotIn("delta", pairs.columns)
            self.assertEqual(summary["pair_selection"], "all_retained_pairs")

    @staticmethod
    def _preparation(root: Path) -> Path:
        preparation = root / "preparation"
        rankings = preparation / "gray_rankings"
        rankings.mkdir(parents=True)
        pd.DataFrame({
            "node_id": range(8),
            "community": [0, 0, 1, 1, 2, 2, 3, 3],
        }).to_csv(preparation / "kmeans_nodes.csv", index=False)
        pd.DataFrame([
            {"community_id": label, "nodes": str([2 * label, 2 * label + 1]), "size": 2}
            for label in range(4)
        ]).to_csv(preparation / "kmeans_communities.csv", index=False)
        pairs = []
        for left in range(4):
            for right in range(left + 1, 4):
                pairs.append({"community_1": left, "community_2": right})
                excluded = {2 * left, 2 * left + 1, 2 * right, 2 * right + 1}
                nodes = [node for node in range(8) if node not in excluded]
                pd.DataFrame({"node_id": nodes, "score": range(len(nodes))}).to_csv(
                    rankings / f"pair_{left}_{right}.csv", index=False,
                )
        pd.DataFrame(pairs).to_csv(preparation / "community_pairs.csv", index=False)
        return preparation

    def test_union_sampling_is_seeded_unique_and_not_id_ordered(self):
        with tempfile.TemporaryDirectory() as directory:
            preparation = self._preparation(Path(directory))
            occupied = {(0, 1), (2, 3), (4, 5), (6, 7)}
            sample, pairs = sample_unique_gray_union(preparation, occupied, 12, 17)
            repeated, _ = sample_unique_gray_union(preparation, occupied, 12, 17)
            edges = list(map(tuple, sample[["source", "target"]].to_numpy()))
            self.assertEqual(sample.to_dict("records"), repeated.to_dict("records"))
            self.assertEqual(len(edges), len(set(edges)))
            self.assertTrue(set(edges).isdisjoint(occupied))
            self.assertNotEqual(edges, sorted(edges))
            self.assertEqual(len(pairs), 6)

    def test_materialization_rejects_duplicates(self):
        base = pd.DataFrame({
            "source": [0], "target": [1], "sign": [1], "weight": [1.],
        })
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "duplicate physical"):
                write_augmented_graph(
                    [(2, 3), (3, 2)], base, base, Path(directory), False,
                )

    def test_directed_encoding_uses_two_model_edges(self):
        physical = pd.DataFrame({
            "source": [0], "target": [1], "sign": [1], "weight": [1.],
        })
        directed = model_edges(physical, directed=True)
        self.assertEqual(set(map(tuple, directed[["source", "target"]].to_numpy())),
                         {(0, 1), (1, 0)})

    def test_rank_one_update_matches_recomputation(self):
        adjacency = sp.csr_matrix(np.asarray([
            [0., 1., 0., 0.], [1., 0., 1., 0.],
            [0., 1., 0., 1.], [0., 0., 1., 0.],
        ]))
        laplacian = sp.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
        signals = np.asarray([[-1., .2], [-.4, -.1], [.3, -.2], [1.1, .1]])
        signals -= signals.mean(axis=0, keepdims=True)
        source, target = 0, 2
        factor = grounded_factor(laplacian.tocsr())
        action = solve_grounded(factor, signals)
        incidence = np.zeros(4); incidence[source], incidence[target] = 1., -1.
        inverse_incidence = solve_grounded(factor, incidence)
        denominator = 1 + inverse_incidence[source] - inverse_incidence[target]
        updated = rank_one_inverse_action_update(
            action, inverse_incidence, source, target, denominator,
        )
        edge = sp.csr_matrix(([-1., -1.], ([source, target], [target, source])),
                             shape=(4, 4))
        edge += sp.diags([1. if node in {source, target} else 0. for node in range(4)])
        recomputed = solve_grounded(
            grounded_factor((laplacian + edge).tocsr()), signals,
        )
        np.testing.assert_allclose(updated, recomputed, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
