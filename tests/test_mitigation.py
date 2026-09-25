import tempfile
import unittest
from pathlib import Path

import pandas as pd

from signed_epm.mitigation.core import model_edges, write_augmented_graph


class MitigationTests(unittest.TestCase):
    def test_directed_encoding_and_materialization(self):
        physical = pd.DataFrame({
            "source": [0], "target": [1], "sign": [1], "weight": [1.0],
        })
        encoded = model_edges(physical, directed=True)
        self.assertEqual(
            set(map(tuple, encoded[["source", "target"]].to_numpy())),
            {(0, 1), (1, 0)},
        )

        with tempfile.TemporaryDirectory() as directory:
            base = pd.DataFrame({
                "source": [0], "target": [2], "sign": [1], "weight": [1.0],
            })
            added, model_added = write_augmented_graph(
                [(1, 2)], base, base, Path(directory), directed_backbone=True,
            )
            self.assertEqual(len(added), 1)
            self.assertEqual(len(model_added), 2)


if __name__ == "__main__":
    unittest.main()
