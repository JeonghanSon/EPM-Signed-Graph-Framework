import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from signed_epm.synthetic.generate import (
    GeneratorConfig,
    community_aligned_opinion,
    generated_opinion,
    generate_replicate,
)
from signed_epm.synthetic.validate import legacy_er
from signed_epm.synthetic.generate_unsigned_sbm import (
    P_OUT_LEVELS,
    SBMConfig,
    hohmann_opinion,
    within_probability,
)
from signed_epm.synthetic.generate_signed_sbm import (
    INTER_NEGATIVE_LEVELS,
    negative_probabilities,
)


class SyntheticGeneratorTests(unittest.TestCase):
    def test_fixed_counts_connectivity_and_coupling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = generate_replicate(root, 0, GeneratorConfig(), levels=(0.5, 0.9))
            self.assertEqual(len(records), 4)
            graphs = {}
            for record in records:
                path = Path(record["path"])
                manifest = json.loads((path / "manifest.json").read_text())
                self.assertEqual(manifest["counts"]["positive"]["train"], 400)
                self.assertEqual(manifest["counts"]["negative"]["train"], 100)
                self.assertTrue(manifest["validation"]["train_connected"])
                self.assertTrue(manifest["validation"]["positive_train_connected"])
                graphs[(record["experiment"], record["level"])] = pd.read_csv(
                    path / "train_snapshot_undirected.csv"
                )

            def pairs(frame, sign):
                return set(map(tuple, frame.loc[frame.sign == sign, ["source", "target"]].to_numpy()))

            self.assertEqual(pairs(graphs[("structural", 0.5)], -1),
                             pairs(graphs[("structural", 0.9)], -1))
            self.assertEqual(pairs(graphs[("antagonistic", 0.5)], 1),
                             pairs(graphs[("antagonistic", 0.9)], 1))

    def test_generated_opinions_are_reproducible_and_bounded(self):
        for distribution in ("uniform", "power_law"):
            left = generated_opinion(100, distribution, 7)
            right = generated_opinion(100, distribution, 7)
            np.testing.assert_array_equal(left, right)
            self.assertLessEqual(np.abs(left).max(), 1.0)
            self.assertTrue((left < 0).any() and (left > 0).any())

    def test_aligned_opinion_is_symmetric_and_sign_separated(self):
        values = community_aligned_opinion(100, "uniform", 3)
        np.testing.assert_allclose(values, -values[::-1])
        self.assertTrue((values[:50] < 0).all())
        self.assertTrue((values[50:] > 0).all())

    def test_legacy_er_ignores_negative_edge_placement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = generate_replicate(root, 1, GeneratorConfig(), levels=(0.5, 0.9))
            opinion_path = root / "opinion.csv"
            pd.DataFrame({"node_id": np.arange(100),
                          "opinion": generated_opinion(100, "uniform", 2)}).to_csv(
                              opinion_path, index=False)
            scores = []
            for record in records:
                if record["experiment"] == "antagonistic":
                    scores.append(legacy_er(
                        Path(record["path"]) / "train_snapshot_undirected.csv",
                        opinion_path, 100,
                    ))
            self.assertAlmostEqual(scores[0], scores[1], places=12)

    def test_unsigned_sbm_preserves_expected_edge_count(self):
        config = SBMConfig()
        communities = np.repeat(np.arange(8), 125)
        intra_pairs = sum(np.count_nonzero(communities == block) * 124 // 2
                          for block in range(8))
        total_pairs = config.num_nodes * (config.num_nodes - 1) // 2
        target = config.baseline_probability * total_pairs
        for p_out in P_OUT_LEVELS:
            p_in = within_probability(config, p_out)
            expected = p_in * intra_pairs + p_out * (total_pairs - intra_pairs)
            self.assertAlmostEqual(expected, target, places=10)

    def test_hohmann_opinion_is_reproducible_symmetric_and_aligned(self):
        left = hohmann_opinion(1000, 4)
        right = hohmann_opinion(1000, 4)
        np.testing.assert_array_equal(left, right)
        np.testing.assert_allclose(left, -left[::-1])
        self.assertTrue((left[:500] <= 0).all())
        self.assertTrue((left[500:] >= 0).all())
        self.assertLessEqual(np.abs(left).max(), 1.0)

    def test_signed_sbm_probabilities_match_requested_expectations(self):
        available_intra, available_inter = 58_000, 437_000
        for fraction in INTER_NEGATIVE_LEVELS:
            q_in, q_out = negative_probabilities(
                available_intra, available_inter, 1000, fraction,
            )
            expected_in = available_intra * q_in
            expected_out = available_inter * q_out
            self.assertAlmostEqual(expected_in + expected_out, 1000)
            self.assertAlmostEqual(expected_out / (expected_in + expected_out), fraction)


if __name__ == "__main__":
    unittest.main()
